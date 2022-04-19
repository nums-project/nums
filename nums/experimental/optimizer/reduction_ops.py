# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Set
import copy

import numpy as np

from nums.core.settings import sync_nnz
from nums.core.array.base import Block
from nums.core.grid.grid import Device
from nums.core.kernel.kernel_manager import KernelManager
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.graph import TreeNode, Leaf
import nums.core.array.utils as array_utils


class TreeReductionOp(TreeNode):
    def __init__(self, cluster_state: ClusterState, tree_node_id=None, seed=1337):
        super().__init__(cluster_state, tree_node_id)
        self.op_name = None
        # For sampling pairs of leafs in get_actions.
        self.rs = np.random.RandomState(seed)
        self.children_dict: dict = {}
        self.leafs_dict: dict = {}
        # List of actions generated upon first invocation to get_actions.
        # Not a great pattern (side effect), but it's faster than regenerating every time.
        self.action_leaf_q = []

    def __repr__(self):
        return "Reduc(id=%s, op=%s, in=%d)" % (
            str(self.tree_node_id),
            self.op_name,
            len(self.children_dict),
        )

    def get_children(self):
        return [self.children_dict[key] for key in sorted(self.children_dict.keys())]

    def num_nodes(self):
        r = 1
        for _, child in self.children_dict.items():
            r += child.num_nodes()
        return r

    def copy(self, cluster_state, parent=None, new_ids=False):
        rop: TreeReductionOp = TreeReductionOp(
            cluster_state, None if new_ids else self.tree_node_id
        )
        assert rop.tree_node_id is not None and (
            new_ids or rop.tree_node_id == self.tree_node_id
        )
        rop.parent = parent
        rop.op_name = self.op_name
        rop._shape = self._shape
        rop._grid_entry = self._grid_entry
        rop._grid_shape = self._grid_shape
        rop._dtype = self._dtype
        rop.copy_on_op = self.copy_on_op
        # This is just a list ids (integers); copy it directly.
        rop.action_leaf_q = copy.deepcopy(self.action_leaf_q)
        for child_id, child in self.children_dict.items():
            child_copy: TreeNode = child.copy(
                cluster_state=cluster_state, parent=rop, new_ids=new_ids
            )
            assert child_copy.tree_node_id is not None and (
                new_ids or child_copy.tree_node_id == child_id
            )
            rop.children_dict[child_copy.tree_node_id] = child_copy
            if child.tree_node_id in self.leafs_dict:
                rop.leafs_dict[child_copy.tree_node_id] = child_copy
        # TODO (hme): How do we properly copy random state?
        return rop

    def add_child(self, child: TreeNode):
        assert child not in self.children_dict
        self.children_dict[child.tree_node_id] = child
        if isinstance(child, Leaf):
            self.leafs_dict[child.tree_node_id] = child

    def test_integrity(self):
        # This is expensive and only used for testing.
        for leaf_id, leaf in self.leafs_dict.items():
            assert leaf_id == leaf.tree_node_id
        for child_id, child in self.children_dict.items():
            assert child_id == child.tree_node_id
            if isinstance(child, Leaf):
                assert child.tree_node_id in self.leafs_dict

    def update_child(self, old_children, new_children):
        # TODO: Remove integrity checks.
        # self.test_integrity()
        for old_child in old_children:
            assert old_child.tree_node_id in self.children_dict, (
                "Failed to update child: Old " "child isn't a child of this node."
            )
            del self.children_dict[old_child.tree_node_id]
            if old_child.tree_node_id in self.leafs_dict:
                del self.leafs_dict[old_child.tree_node_id]
        for new_child in new_children:
            self.children_dict[new_child.tree_node_id] = new_child
            if isinstance(new_child, Leaf):
                self.leafs_dict[new_child.tree_node_id] = new_child
        # self.test_integrity()

    def get_leafs(self):
        leafs = []
        for child_id, child in self.children_dict.items():
            leafs += child.get_leafs()
        return leafs

    def is_frontier(self):
        # This is a frontier if all children are computed.
        # This is a stronger constraint than just 2 leafs, but allows
        # for better pairing of operations during action selction.
        return len(self.leafs_dict) == len(self.children_dict)

    def get_frontier(self):
        # This poses an interesting generalization to our prior assumptions about frontiers.
        # We can now have this node be a frontier, as there are actions we can perform on it.
        # It may also contain children that are also frontiers, so collect those.
        # We generate the set of actions from these frontier nodes using their
        # respective actions methods.
        frontier_nodes = []
        if self.is_frontier():
            frontier_nodes.append(self)
        for child_id, child in self.children_dict.items():
            frontier_nodes += child.get_frontier()
        return frontier_nodes

    def _sort_leafs(self) -> List[Leaf]:
        # Sort leafs by node, then by device.
        result = []
        for leaf in self.leafs_dict.values():
            assert isinstance(leaf, Leaf)
            device: Device = self.cluster_state.get_block_devices(leaf.block.id)[0]
            result.append((leaf, device))
        result = sorted(result, key=lambda v: (v[1].node_id, v[1].device))
        leafs, _ = zip(*result)
        return leafs

    def _get_actions(self, leaf_ids, **kwargs):
        assert len(leaf_ids) == 2
        use_all_devices = kwargs.get("use_all_devices", False)
        actions = []
        if use_all_devices:
            devices = self.cluster_state.devices
        else:
            # Restrict node ids to the nodes on which the leafs already reside.
            left: Leaf = self.leafs_dict[leaf_ids[0]]
            right: Leaf = self.leafs_dict[leaf_ids[1]]
            devices = self.cluster_state.union_devices(left.block.id, right.block.id)
        for device in devices:
            actions.append(
                (self.tree_node_id, {"device": device, "leaf_ids": leaf_ids})
            )
        return actions

    def get_actions(self, **kwargs):
        """
        Returns a list of actions.
        An action is a tuple: First entry is a cluster node id, second is a pair of leaf_ids.
        """
        if self.is_frontier():
            if len(self.action_leaf_q) == 0:
                # This is called multiple times.
                # Only compute action_leaf_q once.
                if len(self.leafs_dict) == 1:
                    # The ReductionOp should have returned the last leaf upon executing
                    # the last pair of leaves.
                    raise Exception("Unexpected state.")
                sorted_leafs: List[Leaf] = self._sort_leafs()
                for leaf in sorted_leafs:
                    self.action_leaf_q.append(leaf.tree_node_id)
            leaf_id_pair = tuple(self.action_leaf_q[:2])
            return self._get_actions(leaf_id_pair)
        return []

    def final_action_check(self):
        assert self.is_frontier()
        if len(self.action_leaf_q) == 0:
            assert len(self.leafs_dict) == 2
            self.action_leaf_q = []
            for tnode_id in self.leafs_dict:
                self.action_leaf_q.append(tnode_id)

    def simulate_on(self, device: Device, leaf_ids=None) -> np.ndarray:
        assert len(leaf_ids) == 2
        leafs = self.leafs_dict[leaf_ids[0]], self.leafs_dict[leaf_ids[1]]
        left, right = leafs
        assert isinstance(left, Leaf) and isinstance(right, Leaf)
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_op(
            self._mem_cost(leafs), left.block.id, right.block.id, device, resources
        )
        return resources

    def execute_on(self, device: Device, leaf_ids=None) -> TreeNode:
        assert len(leaf_ids) == 2
        leafs = self.leafs_dict[leaf_ids[0]], self.leafs_dict[leaf_ids[1]]
        left, right = leafs
        assert isinstance(left, Leaf) and isinstance(right, Leaf)
        result = self._collapse(device, left, right)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]

        # Update action leaf queue.
        assert set(leaf_ids) == {self.action_leaf_q.pop(0), self.action_leaf_q.pop(0)}
        self.action_leaf_q.append(new_leaf.tree_node_id)

        # This updates load on nodes and channels.
        # This also updates block states to indicate that they now reside on the provided nodes.
        # Update the cluster state after computing the leaf, so that transfer costs are properly
        # captured by leaf node computations.
        self.cluster_state.commit_op(
            self._mem_cost(leafs), left.block.id, right.block.id, device
        )
        # Update cluster state with new block.
        # self.cluster_state.add_block(new_block.id, new_block.size(), [device])
        self.cluster_state.add_block(
            new_block.id, new_leaf.tree_node_size.nbytes, [device]
        )
        if not self.cluster_state.created_on_only:
            assert self.cluster_state.blocks_local(left.block.id, right.block.id)
            assert self.cluster_state.blocks_local(left.block.id, new_leaf.block.id)
        # The following are mutating operations.
        # Set the new leaf's parent to this node.
        new_leaf.parent = self
        # Update this node's children: We've collapsed two child leafs by performing
        # the reduction operation, so remove those leafs and replace them with the new leaf.
        self.update_child(leafs, [new_leaf])
        if len(self.children_dict) == 1:
            assert tuple(self.children_dict.values())[0] is new_leaf
            # This was constructed as a reduction with two children,
            # otherwise the reduction would have been transformed into a binary op.
            # We can return the leaf,
            # but we need to perform some mutations to remove this node from the graph.
            # Remove the node from parent reference.
            if self.parent is not None:
                self.parent.update_child([self], [new_leaf])
            # Remove the node as new_leaf's parent.
            new_leaf.parent = self.parent
            return new_leaf
        return self

    def _collapse(self, device: Device, left: Leaf, right: Leaf):
        lblock: Block = left.block
        rblock: Block = right.block
        if self.op_name == "matmul":
            raise ValueError("matmul is not a supported reduction operator.")
        op_name, args = self.op_name, {}
        assert lblock.shape == rblock.shape
        block: Block = lblock.copy()
        block.transposed = False
        block.dtype = array_utils.get_reduce_output_type(self.op_name, lblock.dtype)
        block.oid = lblock._km.bop_reduce(
            op_name,
            lblock.oid,
            rblock.oid,
            lblock.transposed,
            rblock.transposed,
            syskwargs={"device": device},
        )
        block._device = device

        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block = block
        leaf.tree_node_size = left.tree_node_size.bop(
            op_name,
            right.tree_node_size,
            **args,
        )
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self, leafs):
        # Computes the memory required to perform this operation.
        # We approximate by just computing the memory required to store the result.
        # assert leafs is not None and len(leafs) > 0
        assert leafs is not None and len(leafs) == 2
        shape = None
        for leaf in leafs:
            assert leaf.tree_node_id in self.leafs_dict
            leaf_block: Block = leaf.block
            if shape is None:
                shape = leaf_block.shape
            else:
                assert leaf_block.shape == shape
        # leaf_block: Block = leafs[0].block
        # return leaf_block.size()
        if sync_nnz > 1:
            leafs[0].tree_node_size.nnz = leafs[0].block.nnz  # Blocking fetch
            leafs[1].tree_node_size.nnz = leafs[1].block.nnz  # Blocking fetch
        return (
            leafs[0]
            .tree_node_size.bop(
                self.op_name,
                leafs[1].tree_node_size,
            )
            .nbytes
        )

    def _sample_child(self) -> TreeNode:
        for _, leaf in self.leafs_dict.items():
            return leaf
        for _, tnode in self.children_dict.items():
            return tnode

    def shape(self):
        if self._shape is None:
            self._shape = self._sample_child().shape()
        return self._shape

    def set_grid_entry(self, val):
        self._grid_entry = val

    def grid_entry(self):
        return self._grid_entry

    def set_grid_shape(self, val):
        self._grid_shape = val

    def grid_shape(self):
        return self._grid_shape

    def dtype(self):
        if self._dtype is None:
            self._dtype = array_utils.get_reduce_output_type(
                self.op_name, self._sample_child().dtype()
            )
        return self._dtype

    def expression(self):
        if self._expression is None:
            # Include id as part of expression string.
            # This will force a different hash for large fused reductions,
            # if this is, for whatever reason, needed.
            size = len(self.children_dict)
            self._expression = "TreeReductionOp(op=%s, size=%s, id=%s)" % (
                self.op_name,
                str(size),
                self.tree_node_id,
            )
        return self._expression

    def fuse(self, func_node, km: KernelManager):
        # This is fuseable, but it seems impractical.
        raise NotImplementedError()
