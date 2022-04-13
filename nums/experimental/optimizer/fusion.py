from typing import Union, Optional, List, Tuple, Set
from functools import partial
import hashlib

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.array.base import Block
from nums.core.grid.grid import Device
from nums.experimental.optimizer.clusterstate import ClusterState

from nums.experimental.optimizer.graph import (
    TreeNode,
    BinaryOp,
    Leaf,
    UnaryOp,
    ReduceAxis,
    FunctionNode,
)

from nums.experimental.optimizer.reduction_ops import TreeReductionOp
from nums.core.kernel.kernel_manager import KernelManager


class FuseGraph(object):
    def __init__(self, root: TreeNode, km: KernelManager, max_args=2, max_scalars=100):
        self.root = root
        self.km = km
        self.max_args = max_args
        self.max_scalars = max_scalars

    def traverse(self, node: TreeNode, fuseable_nodes):
        """
        Recursively traverse this node and return the number of unique blocks.
        If <= max_args, then it's a fusion candidate.
        """
        if isinstance(node, Leaf):
            if node.is_scalar():
                # Don't count scalars toward max_args.
                return [], set(), 1
            return [node], {node.block.id}, 0
        node_leafs = []
        node_block_id_set = set()
        node_num_scalars = 0
        fuseable_children = []
        for child in node.get_children():
            child_leafs, child_block_id_set, child_num_scalars = self.traverse(
                child, fuseable_nodes
            )
            if child_leafs is None:
                # This branch has been pruned.
                return None, None, None
            if len(child_block_id_set) <= self.max_args:
                # If it's a frontier node, then there's no point in fusing it.
                # If it's a leaf, we still want to count toward node_leafs,
                # but is itself not a fuseable node.
                if not child.is_frontier() and not isinstance(child, Leaf):
                    fuseable_children.append(child)
            node_leafs += child_leafs
            node_block_id_set |= child_block_id_set
            node_num_scalars += child_num_scalars
        if len(node_block_id_set) <= self.max_args:
            # This is fuseable. We keep going.
            return node_leafs, node_block_id_set, node_num_scalars
        else:
            # This branch is no longer fuseable.
            # We prune here and update fuseable_nodes with child nodes.
            fuseable_nodes += fuseable_children
            return None, None, 0

    def __call__(self):
        fuseable_nodes = []
        root_copy = self.root.copy(self.root.cluster_state, parent=None, new_ids=False)
        leafs, leaf_id_set, num_scalars = self.traverse(root_copy, fuseable_nodes)
        if num_scalars > self.max_scalars:
            raise Exception("Large number of scalars: %s" % num_scalars)
        # If leafs is None, then pruning occurred.
        # If fuseable_nodes is empty, then either the entire tree is fuseable,
        # or there is nothing to fuse.
        if len(fuseable_nodes) == 0 and leaf_id_set is None:
            # Nothing is fuseable.
            return self.root
        if len(fuseable_nodes) == 0:
            # The entire tree is fuseable, so add the root node to fuseable_nodes.
            assert len(leaf_id_set) <= self.max_args
            fuseable_nodes.append(root_copy)
        if leaf_id_set is None:
            # There are some fuseable nodes, but there was pruning, so we can leave out the root node.
            assert len(fuseable_nodes) > 0
        return self.update_graph(root_copy, fuseable_nodes)

    def update_graph(self, root, fuseable_nodes):
        if len(fuseable_nodes) == 1 and fuseable_nodes[0] is root:
            assert root.parent is None
            return self.fuse_node(root)
        for node in fuseable_nodes:
            node: TreeNode = node
            assert node.parent is not None
            fused_node = self.fuse_node(node)
            fused_node.parent = node.parent
            node.parent.update_child(old_children=[node], new_children=[fused_node])
        return root

    def fuse_node(self, node: TreeNode):
        result = FunctionNode(node.cluster_state)
        result.op_func, result.children = node.fuse(result, self.km)
        result.set_shape(node.shape())
        result.set_grid_entry(node.grid_entry())
        result.set_grid_shape(node.grid_shape())
        result.set_dtype(node.dtype())

        expression_str = node.expression()
        result.set_expression(expression_str)
        result.op_hash = self.hash_str(expression_str)
        result.finalize(self.km)
        return result

    def hash_str(self, val: str):
        return "fused-%s" % hashlib.sha1(val.encode("utf-8")).hexdigest()
