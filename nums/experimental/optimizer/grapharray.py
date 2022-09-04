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


import itertools
import functools
from typing import Union

import numpy as np
import opt_einsum as oe

from nums.core.settings import sync_nnz
from nums.core.array import utils as array_utils
from nums.core.array.base import BlockBase, BlockArrayBase
from nums.core.array.blockarray import BlockArray
from nums.core.array.sparse import SparseBlockArray
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.grid.grid import Device
from nums.core.storage.storage import ArrayGrid
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.graph import (
    TreeNode,
    Leaf,
    UnaryOp,
    ReduceAxis,
    FunctionNode,
    Einsum,
)
from nums.experimental.optimizer.reduction_ops import TreeReductionOp
from nums.experimental.optimizer.fusion import FuseGraph
from nums.experimental.optimizer.fusion_utils import set_using_marker, traverse_marker
from nums.experimental.optimizer.node_meta import LeafMeta


class GraphArray(object):
    @staticmethod
    def graphs_from_ba(
        ba: BlockArrayBase, cluster_state: ClusterState, copy_on_op
    ) -> np.ndarray:
        graphs = np.empty(shape=ba.grid.grid_shape, dtype=np.object)
        for grid_entry in ba.grid.get_entry_iterator():
            block: BlockBase = ba.blocks[grid_entry]
            # Allocate the block to the node on which it's created.
            km: KernelManager = ba.km
            device: Device = km.device_grid.get_device(
                block.true_grid_entry(), block.true_grid_shape()
            )
            # cluster_state.add_block(block.id, block.size(), devices=[device])
            # cluster_state.init_mem_load(device, block.id)

            # Create the leaf representing this block for future computations.
            leaf: Leaf = Leaf(cluster_state)
            leaf.block = block
            if sync_nnz > 0 and not block.is_dense:
                nnz = block.nnz  # Blocking fetch
            else:
                nnz = np.prod(block.shape)
            leaf.tree_node_meta = LeafMeta(
                block.shape,
                nnz,
                block.dtype,
                block.is_dense,
            )
            leaf.dense_kernel = block.is_dense
            leaf.copy_on_op = copy_on_op
            graphs[grid_entry] = leaf

            cluster_state.add_block(
                block.id, leaf.tree_node_meta.nbytes, devices=[device]
            )
            cluster_state.init_mem_load(device, block.id)
        return graphs

    @classmethod
    def to_ga(cls, val, cluster_state, km, copy_on_op=True):
        if isinstance(val, GraphArray):
            return val
        elif isinstance(val, BlockArrayBase):
            return GraphArray.from_ba(val, cluster_state, copy_on_op)
        else:
            from nums.core.array.blockarray import BlockArray

            ba: BlockArrayBase = BlockArray.to_block_array(val, km)
            return GraphArray.from_ba(ba, cluster_state, copy_on_op)

    @classmethod
    def from_ba(cls, ba: BlockArrayBase, cluster_state: ClusterState, copy_on_op=True):
        return GraphArray(
            ba.grid,
            cluster_state,
            GraphArray.graphs_from_ba(ba, cluster_state, copy_on_op),
            ba.km,
            copy_on_op=copy_on_op,
        )

    def to_ba(self):
        sample_node = self.graphs[tuple(0 for _ in self.graphs.shape)]
        assert isinstance(
            sample_node, Leaf
        ), "Cannot convert unsolved GraphArray to BlockArray."
        if sample_node.block.is_dense:
            return BlockArray(self.grid.copy(), self.km, self.to_blocks())
        return SparseBlockArray(self.grid.copy(), self.km, self.to_blocks())

    def __init__(
        self,
        grid: ArrayGrid,
        cluster_state: ClusterState,
        graphs: np.ndarray,
        km: KernelManager,
        copy_on_op=True,
    ):
        # The ArrayGrid corresponding to the output of this GraphArray.
        self.grid = grid
        # The (shared) cluster state with which this GraphArray is associated.
        self.cluster_state = cluster_state
        # The shape of the output corresponding to this GraphArray.
        self.shape = self.grid.shape
        # The block_shape of the output corresponding to this GraphArray.
        self.block_shape = self.grid.block_shape
        # The grid_shape of the output corresponding to this GraphArray.
        self.grid_shape = grid.grid_shape
        self.dtype = self.grid.dtype
        # The graphs this data structure is comprised of.
        self.graphs = graphs
        self.km = km
        # Whether the graph array is copied whenever an operation is performed.
        # See _add_uop for example.
        self.copy_on_op = copy_on_op

    def __repr__(self):
        return str(self.graphs)

    def copy(self, new_ids=False):
        new_cluster = self.cluster_state.copy()
        graphs_copy = np.empty(shape=self.grid.grid_shape, dtype=np.object)
        for grid_entry in self.grid.get_entry_iterator():
            old_tree_node: TreeNode = self.graphs[grid_entry]
            # The recursive copy should go through without issue,
            # since nodes only hold reference to cluster_state and block ids.
            graphs_copy[grid_entry] = old_tree_node.copy(
                cluster_state=new_cluster, new_ids=new_ids
            )
        return GraphArray(self.grid, new_cluster, graphs_copy, self.km)

    def iterator(self):
        # Yields a breadth first ordered list for each entry.
        for grid_entry in self.grid.get_entry_iterator():
            root: TreeNode = self.graphs[grid_entry]
            q = [root]
            while len(q) > 0:
                node: TreeNode = q.pop(0)
                q += node.get_children()
                yield node

    def to_blocks(self) -> np.ndarray:
        blocks: np.ndarray = np.empty(self.grid.grid_shape, dtype=BlockBase)
        for grid_entry in self.grid.get_entry_iterator():
            leaf: TreeNode = self.graphs[grid_entry]
            assert isinstance(leaf, Leaf), "%s,%s" % (str(leaf), type(leaf))
            blocks[grid_entry] = leaf.block
        return blocks

    def other_to_ga(self, other):
        return GraphArray.to_ga(other, self.cluster_state, self.km, self.copy_on_op)

    @staticmethod
    def tensordot(a, b, axes=2):
        b = a.other_to_ga(b)
        # TODO: Reuse BlockArrayBase tensordot operator.
        a_axes = a.grid.grid_shape[:-axes]
        a_sum_axes = a.grid.grid_shape[-axes:]
        b_axes = b.grid.grid_shape[axes:]
        b_sum_axes = b.grid.grid_shape[:axes]
        assert a_sum_axes == b_sum_axes
        result_shape = tuple(a.shape[:-axes] + b.shape[axes:])
        result_block_shape = tuple(a.block_shape[:-axes] + b.block_shape[axes:])
        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=a.dtype.__name__,
        )
        assert result_grid.grid_shape == tuple(a_axes + b_axes)
        result_graphs = np.empty(shape=result_grid.grid_shape, dtype=np.object)
        a_dims = list(itertools.product(*map(range, a_axes)))
        b_dims = list(itertools.product(*map(range, b_axes)))
        sum_dims = list(itertools.product(*map(range, a_sum_axes)))
        for i in a_dims:
            for j in b_dims:
                # A \in \R^{I \times K}
                # B \in \R^{K \times J}
                # C \in \R^{I \times J}
                # C[i, j] = sum_k^K { A[i, k] * B[k, j] }
                grid_entry = tuple(i + j)
                if len(sum_dims) == 1:
                    k = sum_dims[0]
                    a_node: TreeNode = a.graphs[tuple(i + k)]
                    b_node: TreeNode = b.graphs[tuple(k + j)]
                    dot_node: TreeNode = a_node.tensordot(b_node, axes=axes)
                    result_graphs[grid_entry] = dot_node
                else:
                    rop = TreeReductionOp(a.cluster_state)
                    rop.set_grid_entry(grid_entry)
                    rop.set_grid_shape(result_grid.grid_shape)
                    rop.op_name = "sum"
                    rop.copy_on_op = a.copy_on_op
                    for k in sum_dims:
                        a_node: TreeNode = a.graphs[tuple(i + k)]
                        b_node: TreeNode = b.graphs[tuple(k + j)]
                        dot_node: TreeNode = a_node.tensordot(b_node, axes=axes)
                        # Explicitly add parent here, since sum depends on prod.
                        # Not needed for other ops; make_bop takes care of it.
                        # We don't need to copy the node here since the local
                        # tree structure here is never exposed.
                        dot_node.parent = rop
                        rop.add_child(dot_node)
                    rop.tree_node_meta = dot_node.tree_node_meta
                    # Assume children are all dense or all sparse.
                    rop.dense_kernel = dot_node.tree_node_meta.is_dense
                    result_graphs[grid_entry] = rop

        return GraphArray(
            result_grid,
            a.cluster_state,
            result_graphs,
            a.km,
            copy_on_op=a.copy_on_op,
        )

    def __matmul__(self, other):
        return self.tensordot(self, other, axes=1)

    def ga_from_arr(self, arr: Union[TreeNode, np.ndarray], result_shape: tuple):
        if isinstance(arr, TreeNode):
            sample_node: TreeNode = arr
            assert result_shape == ()
        else:
            sample_node: TreeNode = arr[tuple(0 for dim in arr.shape)]
        result_block_shape = sample_node.shape()
        result_dtype_str = self.grid.dtype.__name__
        result_grid = ArrayGrid(
            shape=result_shape, block_shape=result_block_shape, dtype=result_dtype_str
        )
        assert arr.shape == result_grid.grid_shape
        return GraphArray(
            result_grid, self.cluster_state, arr, self.km, copy_on_op=self.copy_on_op
        )

    def __add__(self, other):
        other = self.other_to_ga(other)
        return self.ga_from_arr(
            self.graphs + other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __sub__(self, other):
        other = self.other_to_ga(other)
        return self.ga_from_arr(
            self.graphs - other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __mul__(self, other):
        other = self.other_to_ga(other)
        return self.ga_from_arr(
            self.graphs * other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __truediv__(self, other):
        other = self.other_to_ga(other)
        return self.ga_from_arr(
            self.graphs / other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __pow__(self, other):
        other = self.other_to_ga(other)
        return self.ga_from_arr(
            self.graphs**other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __imatmul__ = __matmul__
    __itruediv__ = __truediv__
    __ipow__ = __pow__

    __radd__ = __add__

    def __rsub__(self, other):
        other = self.other_to_ga(other)
        return other - self

    __rmul__ = __mul__

    def __rmatmul__(self, other):
        other = self.other_to_ga(other)
        return other @ self

    def __rtruediv__(self, other):
        other = self.other_to_ga(other)
        return other / self

    def __rpow__(self, other):
        other = self.other_to_ga(other)
        return other**self

    # Unary operators.
    def __neg__(self):
        return self.ufunc("negative")

    def __pos__(self):
        return self.ufunc("positive")

    def ufunc(self, op_name):
        result_grid = self.grid.copy()
        result_graphs = np.empty(shape=result_grid.grid_shape, dtype=np.object)
        for grid_entry in self.grid.get_entry_iterator():
            self._add_uop(op_name, grid_entry, self.graphs, result_graphs)
        return GraphArray(
            result_grid,
            self.cluster_state,
            result_graphs,
            self.km,
            copy_on_op=self.copy_on_op,
        )

    def __getattr__(self, item):
        if item != "T":
            raise NotImplementedError(item)
        metaT = self.grid.to_meta()
        metaT["shape"] = tuple(reversed(metaT["shape"]))
        metaT["block_shape"] = tuple(reversed(metaT["block_shape"]))
        result_grid: ArrayGrid = ArrayGrid.from_meta(metaT)
        result_graphs = np.copy(self.graphs.T)
        for grid_entry in result_grid.get_entry_iterator():
            self._add_uop("transpose", grid_entry, result_graphs, result_graphs)
        return GraphArray(
            result_grid,
            self.cluster_state,
            result_graphs,
            self.km,
            copy_on_op=self.copy_on_op,
        )

    def _add_uop(self, op_name, grid_entry, old_arr, new_arr):
        uop: UnaryOp = UnaryOp(self.cluster_state)
        uop.copy_on_op = self.copy_on_op
        uop.op_name = op_name
        # Do this in case old_arr == new_arr.
        old_root: TreeNode = old_arr[grid_entry]
        assert old_root.parent is None
        if self.copy_on_op:
            # Need to copy here, in case old_root is used in other operations.
            # We could eliminate this requirement by maintaining multiple parents,
            # but this breaks a lot of assumptions.
            uop.child = old_root.copy(uop.cluster_state, parent=uop, new_ids=True)
        else:
            assert old_root.parent is None
            uop.child = old_root
            old_root.parent = uop
        uop.tree_node_meta = old_root.tree_node_meta.uop_partial(op_name)
        uop.dense_kernel = old_root.tree_node_meta.is_dense
        new_arr[grid_entry] = uop

    def reduce_axis(self, op_name, axis, keepdims):
        if not (axis is None or isinstance(axis, (int, np.int32, np.int64))):
            raise NotImplementedError("Only integer axis is currently supported.")
        if 0 in self.shape:
            raise ValueError("Attempted reduce on array with dimension of 0.")

        # Generate reduced tree nodes.
        reduced_graphs = np.empty_like(self.graphs, dtype=np.object)
        for grid_entry in self.grid.get_entry_iterator():
            tnode: TreeNode = self.graphs[grid_entry]
            reduced_tnode: ReduceAxis = ReduceAxis(self.cluster_state)
            reduced_tnode.child = tnode
            assert tnode.parent is None
            tnode.parent = reduced_tnode
            reduced_tnode.op_name = op_name
            reduced_tnode.axis = axis
            reduced_tnode.keepdims = keepdims
            reduced_tnode.tree_node_meta = tnode.tree_node_meta.reduce_axis_partial(
                op_name,
                axis,
                keepdims,
            )
            reduced_tnode.dense_kernel = tnode.tree_node_meta.is_dense
            reduced_graphs[grid_entry] = reduced_tnode

        # Compute output GraphArray properties.
        result_shape = []
        result_block_shape = []
        for curr_axis in range(len(self.shape)):
            axis_size, axis_block_size = (
                self.shape[curr_axis],
                self.block_shape[curr_axis],
            )
            if curr_axis == axis or axis is None:
                if keepdims:
                    axis_size, axis_block_size = 1, 1
                else:
                    continue
            result_shape.append(axis_size)
            result_block_shape.append(axis_block_size)
        result_shape = tuple(result_shape)
        result_block_shape = tuple(result_block_shape)
        result_dtype = array_utils.get_reduce_output_type(op_name, self.dtype)
        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=result_dtype.__name__,
        )

        result_graphs = np.empty(shape=result_grid.grid_shape, dtype=np.object)
        if reduced_graphs.size == 1:
            # Just a single entry, nothing to do.
            if result_grid.shape == ():
                # keepdims = False.
                result_graphs[()] = reduced_graphs.item()
            else:
                # keepdims = True.
                result_graphs[:] = reduced_graphs.item()
            return GraphArray(
                grid=result_grid,
                cluster_state=self.cluster_state,
                graphs=result_graphs,
                km=self.km,
                copy_on_op=self.copy_on_op,
            )

        # Compute output GraphArray.
        if axis is None:
            rop = TreeReductionOp(self.cluster_state)
            rop.set_grid_entry(tuple([0] * len(result_grid.grid_shape)))
            rop.set_grid_shape(result_grid.grid_shape)
            rop.op_name = op_name
            rop.copy_on_op = self.copy_on_op
            for child in reduced_graphs.flatten().tolist():
                rop.add_child(child)
                assert child.parent is None
                child.parent = rop
            rop.tree_node_meta = child.tree_node_meta
            rop.dense_kernel = child.tree_node_meta.is_dense
            if result_grid.shape == ():
                # keepdims = False.
                result_graphs[()] = rop
            else:
                # keepdims = True.
                result_graphs[:] = rop
        else:
            for result_grid_entry in result_grid.get_entry_iterator():
                rop = TreeReductionOp(self.cluster_state)
                rop.set_grid_entry(result_grid_entry)
                rop.set_grid_shape(result_grid.grid_shape)
                rop.op_name = op_name
                rop.copy_on_op = self.copy_on_op

                for reduce_axis_entry in range(reduced_graphs.shape[axis]):
                    grid_entry = list(result_grid_entry)
                    if keepdims:
                        grid_entry[axis] = reduce_axis_entry
                    else:
                        grid_entry = (
                            grid_entry[:axis] + [reduce_axis_entry] + grid_entry[axis:]
                        )
                    grid_entry = tuple(grid_entry)
                    child = reduced_graphs[grid_entry]
                    rop.add_child(child)
                    assert child.parent is None
                    child.parent = rop
                rop.tree_node_meta = child.tree_node_meta
                rop.dense_kernel = child.tree_node_meta.is_dense
                result_graphs[result_grid_entry] = rop

        return GraphArray(
            grid=result_grid,
            cluster_state=self.cluster_state,
            graphs=result_graphs,
            km=self.km,
            copy_on_op=self.copy_on_op,
        )

    def sum(self, axis=None, keepdims=False):
        return self.reduce_axis("sum", axis, keepdims)

    def compile(self, max_args: int):
        result_graphs = np.empty_like(self.graphs, dtype=self.graphs.dtype)
        counter = 0
        first_grid_entry = (0,) * len(self.grid.shape)
        for grid_entry in self.grid.get_entry_iterator():
            graph = self.graphs[grid_entry]
            _, leaf_inputs = traverse_marker(graph, 0)
            if grid_entry == first_grid_entry:
                fused_graph = FuseGraph(graph, self.km, max_args=max_args)()
                if not isinstance(fused_graph, FunctionNode):
                    # Stopgap as this function currently assumes root is FunctionNode.
                    return self
                result_graphs[grid_entry] = fused_graph
                # result_graphs[grid_entry] = FuseGraph(
                #     graph, self.km, max_args=max_args
                # )()
                # fused_graph = result_graphs[grid_entry]
            else:
                # TODO: support subtree fusion in this section. FuseGraph already does.
                fused_graph_copy = fused_graph.copy(self.cluster_state, new_ids=True)
                fused_graph_copy = set_using_marker(fused_graph_copy, leaf_inputs)
                fused_graph_copy.set_grid_entry(grid_entry)
                result_graphs[grid_entry] = fused_graph_copy
        return GraphArray(self.grid.copy(), self.cluster_state, result_graphs, self.km)

    @staticmethod
    def einsum(
        cluster_state: ClusterState,
        km: KernelManager,
        copy_on_op,
        subscript: str,
        *operands
    ):
        input_strings, output_string, operands = oe.parser.parse_einsum_input(
            (subscript,) + operands
        )
        input_strings = input_strings.split(",")
        all_vars = set(functools.reduce(lambda x, s: set(x) | set(s), input_strings))

        output_vars = set(output_string)
        assert len(output_vars) == len(
            output_string
        ), "Repeated vars in output not supported."
        sum_vars = all_vars - output_vars
        sorted_vars = sorted(list(output_vars)) + sorted(list(sum_vars))
        var_to_idx = {v: i for i, v in enumerate(sorted_vars)}

        # Type check.
        axis_dims = {}
        axis_block_dims = {}
        axis_grid_dims = {}
        for i, ga in enumerate(operands):
            assert isinstance(ga, GraphArray)
            input_string = input_strings[i]
            for j, char in enumerate(input_string):
                if char not in axis_dims:
                    axis_dims[char] = ga.shape[j]
                    axis_block_dims[char] = ga.block_shape[j]
                    axis_grid_dims[char] = ga.grid_shape[j]
                else:
                    assert axis_dims[char] == ga.shape[j]
                    assert axis_block_dims[char] == ga.block_shape[j]
                    assert axis_grid_dims[char] == ga.grid_shape[j]

        # How many blocks are we summing over?
        num_sum_blocks = 1
        for char in sum_vars:
            num_sum_blocks *= axis_grid_dims[char]

        # Construct output ArrayGrid.
        output_shape = []
        output_block_shape = []
        output_variable_to_axis = {}
        for i, char in enumerate(output_string):
            output_shape.append(axis_dims[char])
            output_block_shape.append(axis_block_dims[char])
            output_variable_to_axis[char] = i
        output_shape = tuple(output_shape)
        output_block_shape = tuple(output_block_shape)

        # Just sample a dtype for now.
        dtype = operands[0].dtype
        grid: ArrayGrid = ArrayGrid(output_shape, output_block_shape, dtype.__name__)

        # Construct iteration space.
        grid_idx_iterator = itertools.product(
            *[range(axis_grid_dims[char]) for char in sorted_vars]
        )
        # Perform einsum kernel operations.
        result_graphs = np.empty(shape=grid.grid_shape, dtype=np.object)
        for grid_idx in grid_idx_iterator:

            # Map input block grid entries.
            input_subgraphs = []
            for i, input_string in enumerate(input_strings):
                input_grid_entry = []
                for char in input_string:
                    input_grid_entry.append(grid_idx[var_to_idx[char]])
                input_grid_entry = tuple(input_grid_entry)
                input_subgraphs.append(operands[i].graphs[input_grid_entry])

            # Map output block grid entry.
            output_grid_entry = []
            for char in output_string:
                output_grid_entry.append(grid_idx[var_to_idx[char]])
            output_grid_entry = tuple(output_grid_entry)

            # TODO: Do we need a reduce node? Can we use a single einsum? Can we use a binary op?
            einsum_node: Einsum = Einsum(cluster_state)
            einsum_node.subscript = subscript
            einsum_node.children = input_subgraphs
            einsum_node.set_dtype(grid.dtype)
            # Output shape of einsum operation = shape of reduce output.
            einsum_node.set_shape(grid.get_block_shape(output_grid_entry))
            # This depends on multiple inputs,
            # so set the grid entry equal to grid entry of reduce node.
            einsum_node.set_grid_entry(output_grid_entry)
            einsum_node.set_grid_shape(grid.grid_shape)
            if num_sum_blocks == 1:
                # There is only one block along the sum dims.
                assert result_graphs[output_grid_entry] is None
                result_graphs[output_grid_entry] = einsum_node
            else:
                # We have multiple blocks along sum dims,
                # so create a reduce node and add einsum nodes to it.
                if result_graphs[output_grid_entry] is None:
                    # Construct reduce node.
                    rop = TreeReductionOp(cluster_state)
                    rop.op_name = "sum"
                    rop.copy_on_op = copy_on_op
                    rop.set_grid_entry(output_grid_entry)
                    rop.set_grid_shape(grid.grid_shape)
                    rop.dtype = grid.dtype
                    result_graphs[output_grid_entry] = rop
                rop = result_graphs[output_grid_entry]
                einsum_node.parent = rop
                rop.add_child(einsum_node)

        return GraphArray(
            grid,
            cluster_state,
            result_graphs,
            km,
            copy_on_op=copy_on_op,
        )
