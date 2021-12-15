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


from typing import Union, Optional, List, Tuple, Set
from functools import partial

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.array.base import Block
from nums.core.grid.grid import DeviceID
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.core.compute.compute_manager import ComputeManager


def subsample(total_items, max_items, rs: np.random.RandomState):
    perms = rs.permutation(total_items)
    if total_items < max_items:
        return perms
    return perms[:max_items]


class TreeNode(object):
    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        # A deterministic identifier that's preserved across copies.
        # label each node as grid_entry, i, where i \in 0, ..., num nodes,
        # incremented top-down and left-to-right.
        self.cluster_state: ClusterState = cluster_state
        self.tree_node_id = (
            self.cluster_state.counter() if tree_node_id is None else tree_node_id
        )
        self.parent: TreeNode = None
        self.copy_on_op = True
        self._shape = None
        self._grid_entry = None
        self._grid_shape = None
        self._dtype = None
        self._expression = None

    def get_root(self):
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_children(self):
        raise NotImplementedError()

    def num_nodes(self):
        raise NotImplementedError()

    def copy(self, cluster_state, parent=None, new_ids=False):
        raise NotImplementedError()

    def update_child(self, old_children, new_children):
        raise NotImplementedError()

    def get_leafs(self):
        raise NotImplementedError()

    def is_frontier(self):
        raise NotImplementedError()

    def get_frontier(self):
        raise NotImplementedError()

    def get_actions(self, **kwargs):
        raise NotImplementedError()

    def simulate_on(self, device_id: DeviceID, leaf_ids=None) -> np.ndarray:
        raise NotImplementedError()

    def execute_on(self, device_id: DeviceID, leaf_ids=None):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

    def grid_entry(self):
        raise NotImplementedError()

    def grid_shape(self):
        raise NotImplementedError()

    def dtype(self):
        raise NotImplementedError()

    def expression(self):
        raise NotImplementedError()

    def fuse(self, func_node, cm: ComputeManager):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)

    def make_bop(self, op_name, other, args=None):
        assert isinstance(other, TreeNode)
        bop: BinaryOp = BinaryOp(self.cluster_state)
        bop.op_name = op_name
        bop.args = args
        assert self.copy_on_op == other.copy_on_op
        bop.copy_on_op = self.copy_on_op
        # Need to copy here in case self and other are used in other operations.
        if self.copy_on_op:
            bop.left = self.copy(bop.cluster_state, parent=bop, new_ids=True)
            bop.right = other.copy(bop.cluster_state, parent=bop, new_ids=True)
        else:
            assert self.parent is None and other.parent is None
            bop.left, bop.right = self, other
            bop.left.parent, bop.right.parent = bop, bop
        return bop

    def tensordot(self, other, axes):
        return self.make_bop("tensordot", other, args={"axes": axes})

    def __matmul__(self, other):
        return self.make_bop("matmul", other)

    def __add__(self, other):
        return self.make_bop("add", other)

    def __sub__(self, other):
        return self.make_bop("sub", other)

    def __mul__(self, other):
        return self.make_bop("mul", other)

    def __truediv__(self, other):
        return self.make_bop("truediv", other)

    def __pow__(self, other):
        return self.make_bop("pow", other)


class Leaf(TreeNode):
    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        # The leaf abstraction enables the same block to be a part of multiple computations,
        # evolving its state across all leafs holding a reference to the block.
        super().__init__(cluster_state, tree_node_id)
        self.block = None

    def get_children(self):
        return []

    def __repr__(self):
        return "Leaf(id=%s, bid=%s)" % (str(self.tree_node_id), str(self.block.id))

    def num_nodes(self):
        return 1

    def copy(self, cluster_state, parent=None, new_ids=False):
        leaf: Leaf = Leaf(cluster_state, None if new_ids else self.tree_node_id)
        assert leaf.tree_node_id is not None and (
            new_ids or leaf.tree_node_id == self.tree_node_id
        )
        leaf._shape = self._shape
        leaf._grid_entry = self._grid_entry
        leaf._grid_shape = self._grid_shape
        leaf._dtype = self._dtype
        leaf.parent = parent
        leaf.block = self.block
        leaf.copy_on_op = self.copy_on_op
        return leaf

    def get_leafs(self):
        return [self]

    def is_frontier(self):
        return False

    def get_frontier(self):
        return []

    def get_actions(self, **kwargs):
        return []

    def shape(self):
        if self._shape is None:
            self._shape = self.block.shape
        return self._shape

    def grid_entry(self):
        if self._grid_entry is None:
            self._grid_entry = self.block.grid_entry
        return self._grid_entry

    def grid_shape(self):
        if self._grid_shape is None:
            self._grid_shape = self.block.grid_shape
        return self._grid_shape

    def dtype(self):
        if self._dtype is None:
            self._dtype = self.block.dtype
        return self._dtype

    def expression(self):
        if self._expression is None:
            self._expression = str(self.block)
        return self._expression

    def fuse(self, func_node, cm: ComputeManager):
        f = cm.get_fuseable("identity")
        return f, [self.copy(self.cluster_state, func_node)]

    def is_scalar(self):
        return self.block.size() == 1


class UnaryOp(TreeNode):
    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        super().__init__(cluster_state, tree_node_id)
        self.child: TreeNode = None
        self.op_name = None

    def get_children(self):
        return [self.child]

    def copy(self, cluster_state, parent=None, new_ids=False):
        uop: UnaryOp = UnaryOp(cluster_state, None if new_ids else self.tree_node_id)
        assert uop.tree_node_id is not None and (
            new_ids or uop.tree_node_id == self.tree_node_id
        )
        uop._shape = self._shape
        uop._grid_entry = self._grid_entry
        uop._grid_shape = self._grid_shape
        uop._dtype = self._dtype
        uop.parent = parent
        uop.child = self.child.copy(cluster_state, parent=uop, new_ids=new_ids)
        uop.op_name = self.op_name
        uop.copy_on_op = self.copy_on_op
        return uop

    def update_child(self, old_children, new_children):
        assert len(old_children) == len(new_children) == 1
        old_child, new_child = old_children[0], new_children[0]
        self.child = new_child

    def get_leafs(self):
        return self.child.get_leafs()

    def is_frontier(self):
        return isinstance(self.child, Leaf)

    def get_frontier(self):
        if self.is_frontier():
            return [self]
        else:
            return self.child.get_frontier()

    def num_nodes(self):
        return self.child.num_nodes() + 1

    def get_actions(self, **kwargs):
        actions = []
        if self.is_frontier():
            use_all_devices = kwargs.get("use_all_devices", False)
            if use_all_devices:
                device_ids = self.cluster_state.device_ids
            else:
                # Restrict device ids to the nodes on which the leafs already reside.
                device_ids = self.cluster_state.get_block_device_ids(
                    self.child.block.id
                )
            for device_id in device_ids:
                actions.append((self.tree_node_id, {"device_id": device_id}))
        return actions

    def simulate_on(self, device_id: DeviceID, leaf_ids=None) -> np.ndarray:
        assert leaf_ids is None
        assert isinstance(self.child, Leaf)
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_uop(
            self._mem_cost(), self.child.block.id, device_id, resources
        )
        return resources

    def execute_on(self, device_id: DeviceID, leaf_ids=None) -> Leaf:
        assert leaf_ids is None
        assert isinstance(self.child, Leaf)
        result = self._collapse(device_id)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]
        self.cluster_state.commit_uop(self._mem_cost(), self.child.block.id, device_id)
        self.cluster_state.add_block(new_block.id, new_block.size(), [device_id])
        if not self.cluster_state.created_on_only:
            assert self.cluster_state.blocks_local(
                self.child.block.id, new_leaf.block.id
            )
        new_leaf.parent = self.parent
        if self.parent is not None:
            self.parent.update_child([self], [new_leaf])
        return new_leaf

    def _collapse(self, device_id: DeviceID):
        assert isinstance(self.child, Leaf)
        block: Block = self.child.block
        op_name, args = self.op_name, {}
        if op_name == "transpose":
            block: Block = block.transpose(defer=True)
        else:
            block: Block = block.ufunc(op_name, device_id=device_id)
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block = block
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self):
        assert isinstance(self.child, Leaf)
        block: Block = self.child.block
        return np.product(block.shape)

    def shape(self):
        if self._shape is None:
            child_shape = self.child.shape()
            if self.op_name == "transpose":
                self._shape = tuple(reversed(child_shape))
            else:
                self._shape = child_shape
        return self._shape

    def grid_entry(self):
        if self._grid_entry is None:
            child_grid_entry = self.child.grid_entry()
            if self.op_name == "transpose":
                self._grid_entry = tuple(reversed(child_grid_entry))
            else:
                self._grid_entry = child_grid_entry
        return self._grid_entry

    def grid_shape(self):
        if self._grid_shape is None:
            child_grid_shape = self.child.grid_shape()
            if self.op_name == "transpose":
                self._grid_shape = tuple(reversed(child_grid_shape))
            else:
                self._grid_shape = child_grid_shape
        return self._grid_shape

    def dtype(self):
        if self._dtype is None:
            self._dtype = array_utils.get_uop_output_type(
                self.op_name, self.child.dtype()
            )
        return self._dtype

    def expression(self):
        if self._expression is None:
            self._expression = "UnaryOp(op=%s, x=%s)" % (
                self.op_name,
                self.child.expression(),
            )
        return self._expression

    def fuse(self, func_node, cm: ComputeManager):
        child_op, child_args = self.child.fuse(func_node, cm)
        if self.op_name == "transpose":
            self_op = cm.get_fuseable("transpose")
        else:
            self_op = cm.get_fuseable("map_uop")
            self_op = partial(self_op, op_name=self.op_name, args=(), kwargs={})

        def fused(*args):
            return self_op(arr=child_op(*args))

        return fused, child_args


class ReduceAxis(UnaryOp):
    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        super().__init__(cluster_state, tree_node_id)
        self.axis = None
        self.keepdims = None

    def copy(self, cluster_state, parent=None, new_ids=False):
        ra: ReduceAxis = ReduceAxis(
            cluster_state, None if new_ids else self.tree_node_id
        )
        assert ra.tree_node_id is not None and (
            new_ids or ra.tree_node_id == self.tree_node_id
        )
        ra._shape = self._shape
        ra._grid_entry = self._grid_entry
        ra._grid_shape = self._grid_shape
        ra._dtype = self._dtype
        ra.parent = parent
        ra.child = self.child.copy(cluster_state, parent=ra, new_ids=new_ids)
        ra.op_name = self.op_name
        ra.axis = self.axis
        ra.keepdims = self.keepdims
        ra.copy_on_op = self.copy_on_op
        return ra

    def _collapse(self, device_id: DeviceID):
        assert isinstance(self.child, Leaf)
        child_block: Block = self.child.block
        op_name, args = self.op_name, {}

        block = Block(
            grid_entry=self.grid_entry(),
            grid_shape=self.grid_shape(),
            shape=self.shape(),
            dtype=self.dtype(),
            transposed=False,
            cm=child_block._cm,
        )
        block.oid = child_block._cm.reduce_axis(
            op_name=op_name,
            arr=child_block.oid,
            axis=self.axis,
            keepdims=self.keepdims,
            transposed=child_block.transposed,
            syskwargs={"device_id": device_id},
        )
        block.device_id = device_id
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block = block
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self):
        assert isinstance(self.child, Leaf)
        return np.product(self.shape())

    def update_tuple_property(self, val, keep_dim_val: Union[int, tuple] = 1):
        result = []
        for curr_axis in range(len(val)):
            axis_entry = val[curr_axis]
            if curr_axis == self.axis or self.axis is None:
                if self.keepdims:
                    axis_entry = keep_dim_val
                else:
                    continue
            result.append(axis_entry)
        return tuple(result)

    def shape(self):
        if self._shape is None:
            self._shape = self.update_tuple_property(self.child.shape(), keep_dim_val=1)
        return self._shape

    def grid_entry(self):
        if self._grid_entry is None:
            self._grid_entry = self.update_tuple_property(
                self.child.grid_entry(), keep_dim_val=0
            )
        return self._grid_entry

    def grid_shape(self):
        if self._grid_shape is None:
            self._grid_shape = self.update_tuple_property(
                self.child.grid_shape(), keep_dim_val=1
            )
        return self._grid_shape

    def dtype(self):
        if self._dtype is None:
            self._dtype = array_utils.get_reduce_output_type(
                self.op_name, self.child.dtype()
            )
        return self._dtype

    def expression(self):
        if self._expression is None:
            self._expression = "ReduceAxis(op=%s, x=%s, axis=%s, keepdims=%s)" % (
                self.op_name,
                self.child.expression(),
                str(self.axis),
                str(self.keepdims),
            )
        return self._expression

    def fuse(self, func_node, cm: ComputeManager):
        child_op, child_args = self.child.fuse(func_node, cm)

        self_op = cm.get_fuseable("reduce_axis")
        kwargs = {
            "op_name": self.op_name,
            "axis": self.axis,
            "keepdims": self.keepdims,
            # When fusing, transposed should always be False.
            "transposed": False,
        }
        self_op = partial(self_op, **kwargs)

        def fused(*args):
            return self_op(arr=child_op(*args))

        return fused, child_args


class BinaryOp(TreeNode):
    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        super().__init__(cluster_state, tree_node_id)
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.op_name = None
        self.args = None

    def __repr__(self):
        bop_symbol = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "truediv": "/",
            "matmul": "@",
            "tensordot": "@",
        }[self.op_name]
        return "BOp(id=%s, op=%s%s%s)" % (
            self.tree_node_id,
            str(self.left.tree_node_id),
            bop_symbol,
            str(self.right.tree_node_id),
        )

    def get_children(self):
        return [self.left, self.right]

    def num_nodes(self):
        return self.left.num_nodes() + self.right.num_nodes() + 1

    def copy(self, cluster_state, parent=None, new_ids=False):
        bop = BinaryOp(cluster_state, None if new_ids else self.tree_node_id)
        assert bop.tree_node_id is not None and (
            new_ids or bop.tree_node_id == self.tree_node_id
        )
        bop._shape = self._shape
        bop._grid_entry = self._grid_entry
        bop._grid_shape = self._grid_shape
        bop._dtype = self._dtype
        bop.parent = parent
        bop.op_name = self.op_name
        bop.args = None if self.args is None else self.args.copy()
        bop.left = self.left.copy(cluster_state, bop, new_ids=new_ids)
        bop.right = self.right.copy(cluster_state, bop, new_ids=new_ids)
        bop.copy_on_op = self.copy_on_op
        return bop

    def update_child(self, old_children, new_children):
        assert len(old_children) == len(new_children) == 1
        old_child, new_child = old_children[0], new_children[0]
        if old_child == self.left:
            self.left = new_child
        elif old_child == self.right:
            self.right = new_child
        else:
            raise Exception(
                "Failed to update child: Old child doesn't this nodes children."
            )

    def get_leafs(self):
        return self.left.get_leafs() + self.right.get_leafs()

    def is_frontier(self):
        return isinstance(self.left, Leaf) and isinstance(self.right, Leaf)

    def get_frontier(self):
        if self.is_frontier():
            # This is a frontier node.
            return [self]
        return self.left.get_frontier() + self.right.get_frontier()

    def get_actions(self, **kwargs):
        """
        Returns a list of actions.
        An action is a tuple: First entry is a function. Second entry is kwargs.
        Invoked actions return a new node without mutating the tree,
        which is always a leaf for BinaryOp.
        """
        actions = []
        if self.is_frontier():
            use_all_devices = kwargs.get("use_all_devices", False)
            if use_all_devices:
                device_ids = self.cluster_state.device_ids
            else:
                # Restrict node ids to the nodes on which the leafs already reside.
                device_ids = self.cluster_state.union_devices(
                    self.left.block.id, self.right.block.id
                )
            for device_id in device_ids:
                actions.append((self.tree_node_id, {"device_id": device_id}))
        return actions

    def simulate_on(self, device_id: DeviceID, leaf_ids=None) -> np.ndarray:
        assert leaf_ids is None
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_op(
            self._mem_cost(),
            self.left.block.id,
            self.right.block.id,
            device_id,
            resources,
        )
        return resources

    def execute_on(self, device_id: DeviceID, leaf_ids=None) -> Leaf:
        """
        Update cluster state to reflect the cluster's load after computing this node.
        We generate a leaf node for BinaryOp, updating the leaf node's computation
        time based on object transfer costs, etc.
        """
        assert leaf_ids is None
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        result = self._collapse(device_id)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]
        # This updates load on nodes and channels.
        # This also updates block states to indicate that they now reside on the provided nodes.
        # Update the cluster state after computing the leaf, so that transfer costs are properly
        # captured by leaf node computations.
        self.cluster_state.commit_op(
            self._mem_cost(), self.left.block.id, self.right.block.id, device_id
        )
        # Update cluster state with new block.
        self.cluster_state.add_block(new_block.id, new_block.size(), [device_id])
        if not self.cluster_state.created_on_only:
            assert self.cluster_state.blocks_local(
                self.left.block.id, self.right.block.id
            )
            assert self.cluster_state.blocks_local(
                self.left.block.id, new_leaf.block.id
            )
        # These are mutating operations.
        # Eliminate references to this node and replace them with leaf.
        new_leaf.parent = self.parent
        if self.parent is not None:
            self.parent.update_child([self], [new_leaf])
        return new_leaf

    def _collapse(self, device_id: DeviceID):
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        lblock: Block = self.left.block
        rblock: Block = self.right.block
        if self.op_name == "matmul":
            op_name, args = "tensordot", {"axes": 1}
        elif self.op_name == "tensordot":
            op_name, args = "tensordot", self.args
        else:
            op_name, args = self.op_name, {}
            assert array_utils.can_broadcast_shapes(lblock.shape, rblock.shape)
        block: Block = lblock.bop(op_name, rblock, args=args, device_id=device_id)
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block = block
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self):
        # Computes the memory required to perform this operation.
        # We approximate by just computing the memory required to store the result.
        return np.product(self.shape())

    def _compute_block_params(self):
        left_shape = self.left.shape()
        right_shape = self.right.shape()
        left_grid_entry = self.left.grid_entry()
        right_grid_entry = self.right.grid_entry()
        left_grid_shape = self.left.grid_shape()
        right_grid_shape = self.right.grid_shape()
        if self.op_name == "matmul" or self.op_name == "tensordot":
            assert isinstance(self.args, dict)
            axes = self.args.get("axes", 1)
            this_sum_axes = left_shape[-axes:]
            other_sum_axes = right_shape[:axes]
            assert this_sum_axes == other_sum_axes
            (
                self._shape,
                self._grid_entry,
                self._grid_shape,
            ) = array_utils.get_tensordot_block_params(
                left_shape,
                left_grid_entry,
                left_grid_shape,
                right_shape,
                right_grid_entry,
                right_grid_shape,
                axes,
            )
        else:
            (
                self._shape,
                self._grid_entry,
                self._grid_shape,
            ) = array_utils.get_elementwise_bop_block_params(
                left_shape,
                left_grid_entry,
                left_grid_shape,
                right_shape,
                right_grid_entry,
                right_grid_shape,
            )
            assert self._shape == array_utils.broadcast_shape(left_shape, right_shape)

    def shape(self):
        if self._shape is None:
            self._compute_block_params()
        return self._shape

    def grid_entry(self):
        if self._grid_entry is None:
            self._compute_block_params()
        return self._grid_entry

    def grid_shape(self):
        if self._grid_shape is None:
            self._compute_block_params()
        return self._grid_shape

    def dtype(self):
        if self._dtype is None:
            self._dtype = array_utils.get_bop_output_type(
                self.op_name, self.left.dtype(), self.right.dtype()
            )
        return self._dtype

    def expression(self):
        if self._expression is None:
            if self.op_name == "matmul" or self.op_name == "tensordot":
                axes = self.args.get("axes", 1)
                self._expression = "BinaryOp(op=%s, x=%s, y=%s, axes=%s)" % (
                    self.op_name,
                    self.left.expression(),
                    self.right.expression(),
                    axes,
                )
            self._expression = "BinaryOp(op=%s, x=%s, y=%s)" % (
                self.op_name,
                self.left.expression(),
                self.right.expression(),
            )
        return self._expression

    def fuse(self, func_node, cm: ComputeManager):
        left_op, left_args = self.left.fuse(func_node, cm)
        right_op, right_args = self.right.fuse(func_node, cm)

        self_op = cm.get_fuseable("bop")

        axes = 1 if self.args is None else self.args.get("axes", 1)
        self_op = partial(self_op, op=self.op_name, a1_T=False, a2_T=False, axes=axes)
        num_left = len(left_args)
        # Combine the left and right args.
        args = left_args + right_args

        def fused(*args):
            # The fused op returned splits the input args
            # into the original args
            # for the left and right operators.
            # We can inductively prove that this is correct
            # for arbitrary compositions of unary and binary operators.
            args1 = args[:num_left]
            args2 = args[num_left:]
            return self_op(a1=left_op(*args1), a2=right_op(*args2))

        return fused, args


class FunctionNode(TreeNode):
    """
    A pure function of Leaf nodes.
    No extra args or kwargs supported.
    This is mainly used for fusion,
    but we allow the child type to be something other than a Leaf node to
    enable graphs of the form C = sum(X, axis=0); D = C + Y,
    where X requires a reduction, but C + Y is element-wise at most 2 blocks.
    """

    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        super().__init__(cluster_state, tree_node_id)
        self.op_hash = None
        self.op_func = None
        self.children: List[TreeNode] = None

    def finalize(self, cm: ComputeManager):
        assert isinstance(self._shape, tuple)
        assert isinstance(self._grid_entry, tuple)
        assert isinstance(self._grid_shape, tuple)
        assert self._dtype is not None
        assert isinstance(self._expression, str)
        assert isinstance(self.op_hash, str)
        assert callable(self.op_func)
        assert self.children is not None
        cm.register(self.op_hash, self.op_func, {})

    def __repr__(self):
        return "Function(id=%s, op=%s, args=%s" % (
            self.tree_node_id,
            self.op_hash,
            len(self.children),
        )

    def get_children(self):
        return self.children

    def num_nodes(self):
        # Count self.
        num_nodes = 1
        for child in self.children:
            num_nodes += child.num_nodes()
        return num_nodes

    def copy(self, cluster_state, parent=None, new_ids=False):
        fnode = FunctionNode(cluster_state, None if new_ids else self.tree_node_id)
        assert fnode.tree_node_id is not None and (
            new_ids or fnode.tree_node_id == self.tree_node_id
        )
        fnode._shape = self._shape
        fnode._grid_entry = self._grid_entry
        fnode._grid_shape = self._grid_shape
        fnode._dtype = self._dtype
        fnode.parent = parent
        fnode.op_hash = self.op_hash
        fnode.op_func = self.op_func
        fnode.op_expression = self.op_expression
        fnode.children = [
            child.copy(cluster_state, fnode, new_ids=new_ids) for child in self.children
        ]
        fnode.copy_on_op = self.copy_on_op
        return fnode

    def update_child(self, old_children, new_children):
        assert len(old_children) == len(new_children) == 1
        old_child, new_child = old_children[0], new_children[0]
        new_children = []
        for child in self.children:
            if child.tree_node_id == old_child.tree_node_id:
                new_children.append(new_children)
            else:
                new_children.append(child)
        self.children = new_children

    def get_leafs(self):
        leafs = []
        for child in self.children:
            leafs += child.get_leafs()
        return leafs

    def is_frontier(self):
        _is_frontier = True
        for child in self.children:
            _is_frontier &= isinstance(child, Leaf)
        return _is_frontier

    def get_frontier(self):
        if self.is_frontier():
            # This is a frontier node.
            return [self]
        frontier = []
        for child in self.children:
            frontier += child.get_frontier()
        return frontier

    def get_actions(self, **kwargs):
        """
        Returns a list of actions.
        An action is a tuple: First entry is a function. Second entry is kwargs.
        Invoked actions return a new node without mutating the tree,
        which is always a leaf for BinaryOp.
        """
        actions = []
        if self.is_frontier():
            use_all_devices = kwargs.get("use_all_devices", False)
            if use_all_devices:
                device_ids = self.cluster_state.device_ids
            else:
                # Restrict node ids to the nodes on which the leafs already reside.
                device_ids = set()
                for child in self.children:
                    leaf: Leaf = child
                    device_ids |= set(
                        self.cluster_state.get_block_device_ids(leaf.block.id)
                    )
            device_ids = list(device_ids)
            for device_id in device_ids:
                actions.append((self.tree_node_id, {"device_id": device_id}))
        return actions

    def simulate_on(self, device_id: DeviceID, leaf_ids=None) -> np.ndarray:
        assert leaf_ids is None
        assert self.is_frontier()
        block_ids = [child.block.id for child in self.children]
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_nary_op(
            self._mem_cost(),
            block_ids,
            device_id,
            resources,
        )
        return resources

    def execute_on(self, device_id: DeviceID, leaf_ids=None) -> Leaf:
        """
        Update cluster state to reflect the cluster's load after computing this node.
        We generate a leaf node for BinaryOp, updating the leaf node's computation
        time based on object transfer costs, etc.
        """
        assert leaf_ids is None
        assert self.is_frontier()
        block_ids = [child.block.id for child in self.children]
        result = self._collapse(device_id)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]
        # This updates load on nodes and channels.
        # This also updates block states to indicate that they now reside on the provided nodes.
        # Update the cluster state after computing the leaf, so that transfer costs are properly
        # captured by leaf node computations.
        self.cluster_state.commit_nary_op(self._mem_cost(), block_ids, device_id)
        # Update cluster state with new block.
        self.cluster_state.add_block(new_block.id, new_block.size(), [device_id])
        if not self.cluster_state.created_on_only:
            for block_id in block_ids:
                assert self.cluster_state.blocks_local(block_id, new_leaf.block.id)
        # These are mutating operations.
        # Eliminate references to this node and replace them with leaf.
        new_leaf.parent = self.parent
        if self.parent is not None:
            self.parent.update_child([self], [new_leaf])
        return new_leaf

    def _collapse(self, device_id: DeviceID):
        cm: ComputeManager = None
        block_oids = []
        for child in self.children:
            assert isinstance(child, Leaf)
            block_oids.append(child.block.oid)
            if cm is None:
                cm = child.block._cm
        block: Block = Block(
            self.grid_entry, self.grid_shape, self._shape, self.dtype, False, cm
        )
        block.device_id = device_id
        block.oid = block._cm.call(
            self.op_hash, *block_oids, syskwargs={"device_id": device_id}
        )
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block = block
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self):
        return np.product(self._shape)

    def set_shape(self, val):
        self._shape = val

    def set_grid_entry(self, val):
        self._grid_entry = val

    def set_grid_shape(self, val):
        self._grid_shape = val

    def set_dtype(self, val):
        self._dtype = val

    def set_expression(self, val):
        self._expression = val

    def shape(self):
        return self._shape

    def grid_entry(self):
        return self._grid_entry

    def grid_shape(self):
        return self._grid_shape

    def dtype(self):
        return self._dtype

    def expression(self):
        return self._expression

    def fuse(self, func_node, cm: ComputeManager):
        raise NotImplementedError()
