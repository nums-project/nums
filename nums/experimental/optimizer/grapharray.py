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
from typing import Union

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.array.base import BlockArrayBase, Block
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import DeviceID
from nums.core.storage.storage import ArrayGrid
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.graph import TreeNode, Leaf, UnaryOp
from nums.experimental.optimizer.reduction_ops import TreeReductionOp


class GraphArray(object):
    @staticmethod
    def graphs_from_ba(
        ba: BlockArrayBase, cluster_state: ClusterState, copy_on_op
    ) -> np.ndarray:
        graphs = np.empty(shape=ba.grid.grid_shape, dtype=np.object)
        for grid_entry in ba.grid.get_entry_iterator():
            block: Block = ba.blocks[grid_entry]
            # Allocate the block to the node on which it's created.
            cm: ComputeManager = ComputeManager.instance
            device_id: DeviceID = cm.device_grid.get_device_id(
                block.true_grid_entry(), block.true_grid_shape()
            )
            cluster_state.add_block(block.id, block.size(), device_ids=[device_id])
            cluster_state.init_mem_load(device_id, block.id)

            # Create the leaf representing this block for future computations.
            leaf: Leaf = Leaf(cluster_state)
            leaf.block = block
            leaf.copy_on_op = copy_on_op
            graphs[grid_entry] = leaf
        return graphs

    @classmethod
    def from_ba(cls, ba: BlockArrayBase, cluster_state: ClusterState, copy_on_op=True):
        return GraphArray(
            ba.grid,
            cluster_state,
            GraphArray.graphs_from_ba(ba, cluster_state, copy_on_op),
            copy_on_op=copy_on_op,
        )

    def __init__(
        self,
        grid: ArrayGrid,
        cluster_state: ClusterState,
        graphs: np.ndarray,
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
        self.dtype = self.grid.dtype
        # The graphs this data structure is comprised of.
        self.graphs = graphs
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
        return GraphArray(self.grid, new_cluster, graphs_copy)

    def to_blocks(self) -> np.ndarray:
        blocks: np.ndarray = np.empty(self.grid.grid_shape, dtype=Block)
        for grid_entry in self.grid.get_entry_iterator():
            leaf: TreeNode = self.graphs[grid_entry]
            assert isinstance(leaf, Leaf), "%s,%s" % (str(leaf), type(leaf))
            blocks[grid_entry] = leaf.block
        return blocks

    def other_to_ba(self, other):
        if isinstance(other, GraphArray):
            return other
        return self.from_ba(other, self.cluster_state)

    def tensordot(self, other, axes=2):
        other = self.other_to_ba(other)
        # TODO: Reuse BlockArrayBase tensordot operator.
        this_axes = self.grid.grid_shape[:-axes]
        this_sum_axes = self.grid.grid_shape[-axes:]
        other_axes = other.grid.grid_shape[axes:]
        other_sum_axes = other.grid.grid_shape[:axes]
        assert this_sum_axes == other_sum_axes
        result_shape = tuple(self.shape[:-axes] + other.shape[axes:])
        result_block_shape = tuple(self.block_shape[:-axes] + other.block_shape[axes:])
        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=self.dtype.__name__,
        )
        assert result_grid.grid_shape == tuple(this_axes + other_axes)
        result_graphs = np.empty(shape=result_grid.grid_shape, dtype=np.object)
        this_dims = list(itertools.product(*map(range, this_axes)))
        other_dims = list(itertools.product(*map(range, other_axes)))
        sum_dims = list(itertools.product(*map(range, this_sum_axes)))
        for i in this_dims:
            for j in other_dims:
                # A \in \R^{I \times K}
                # B \in \R^{K \times J}
                # C \in \R^{I \times J}
                # C[i, j] = sum_k^K { A[i, k] * B[k, j] }
                grid_entry = tuple(i + j)
                if len(sum_dims) == 1:
                    k = sum_dims[0]
                    self_node: TreeNode = self.graphs[tuple(i + k)]
                    other_node: TreeNode = other.graphs[tuple(k + j)]
                    dot_node: TreeNode = self_node.tensordot(other_node, axes=axes)
                    result_graphs[grid_entry] = dot_node
                else:
                    rop = TreeReductionOp(self.cluster_state)
                    rop.op_name = "add"
                    rop.copy_on_op = self.copy_on_op
                    for k in sum_dims:
                        self_node: TreeNode = self.graphs[tuple(i + k)]
                        other_node: TreeNode = other.graphs[tuple(k + j)]
                        dot_node: TreeNode = self_node.tensordot(other_node, axes=axes)
                        # Explicitly add parent here, since sum depends on prod.
                        # Not needed for other ops; make_bop takes care of it.
                        # We don't need to copy the node here since the local
                        # tree structure here is never exposed.
                        dot_node.parent = rop
                        rop.add_child(dot_node)
                    result_graphs[grid_entry] = rop

        return GraphArray(
            result_grid, self.cluster_state, result_graphs, copy_on_op=self.copy_on_op
        )

    def block_sum(self, axis=None):
        assert axis is None, "Only complete reductions are currently supported."
        # # Note that this does not sum within each block.
        # # To do this, we need a new node type.
        result_node = TreeReductionOp(self.cluster_state)
        result_node.op_name = "add"
        result_node.copy_on_op = self.copy_on_op
        shape = None
        for grid_entry in self.grid.get_entry_iterator():
            node: TreeNode = self.graphs[grid_entry]
            if shape is None:
                shape = node.shape()
            else:
                assert shape == node.shape(), "shape mismatch for block_sum"
            node.parent = result_node
            result_node.add_child(node)

        result_grid = ArrayGrid(
            shape=shape, block_shape=shape, dtype=self.dtype.__name__
        )
        result_graphs = np.empty(shape=result_grid.grid_shape, dtype=np.object)
        if shape == ():
            result_graphs[()] = result_node
        else:
            result_graphs[:] = result_node
        return GraphArray(
            result_grid, self.cluster_state, result_graphs, copy_on_op=self.copy_on_op
        )

    def __matmul__(self, other):
        return self.tensordot(other, axes=1)

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
            result_grid, self.cluster_state, arr, copy_on_op=self.copy_on_op
        )

    def __add__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(
            self.graphs + other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __sub__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(
            self.graphs - other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __mul__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(
            self.graphs * other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __truediv__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(
            self.graphs / other.graphs,
            array_utils.broadcast_shape(self.shape, other.shape),
        )

    def __pow__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(
            self.graphs ** other.graphs,
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
        other = self.other_to_ba(other)
        return other - self

    __rmul__ = __mul__

    def __rmatmul__(self, other):
        other = self.other_to_ba(other)
        return other @ self

    def __rtruediv__(self, other):
        other = self.other_to_ba(other)
        return other / self

    def __rpow__(self, other):
        other = self.other_to_ba(other)
        return other ** self

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
            result_grid, self.cluster_state, result_graphs, copy_on_op=self.copy_on_op
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
            result_grid, self.cluster_state, result_graphs, copy_on_op=self.copy_on_op
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
        new_arr[grid_entry] = uop

    def ordered_node_list(self, root: TreeNode):
        result = []
        q = [root]
        while len(q) > 0:
            node: TreeNode = q.pop(0)
            q += node.get_children()
            result.append(node)
        return result

    def iterator(self):
        # Yields a breadth first ordered list for each entry.
        for grid_entry in self.grid.get_entry_iterator():
            root: TreeNode = self.graphs[grid_entry]
            q = [root]
            while len(q) > 0:
                node: TreeNode = q.pop(0)
                q += node.get_children()
                yield node
