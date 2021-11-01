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


import warnings
import itertools

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.array.base import BlockArrayBase, Block
from nums.core.array.view import ArrayView
from nums.core.grid.grid import ArrayGrid
from nums.core.compute.compute_manager import ComputeManager


# pylint: disable=too-many-lines


class BlockArray(BlockArrayBase):
    @classmethod
    def empty(cls, shape, block_shape, dtype, cm: ComputeManager):
        return BlockArray.create("empty", shape, block_shape, dtype, cm)

    @classmethod
    def create(cls, create_op_name, shape, block_shape, dtype, cm: ComputeManager):
        grid = ArrayGrid(shape=shape, block_shape=block_shape, dtype=dtype.__name__)
        grid_meta = grid.to_meta()
        arr = BlockArray(grid, cm)
        for grid_entry in grid.get_entry_iterator():
            arr.blocks[grid_entry].oid = cm.new_block(
                create_op_name,
                grid_entry,
                grid_meta,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
        return arr

    @classmethod
    def from_scalar(cls, val, cm):
        if not array_utils.is_scalar(val):
            raise ValueError("%s is not a scalar." % val)
        return BlockArray.from_np(np.array(val), block_shape=(), copy=False, cm=cm)

    @classmethod
    def from_oid(cls, oid, shape, dtype, cm):
        block_shape = shape
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        ba = BlockArray(grid, cm)
        for i, grid_entry in enumerate(grid.get_entry_iterator()):
            assert i == 0
            ba.blocks[grid_entry].oid = oid
        return ba

    @classmethod
    def from_np(cls, arr, block_shape, copy, cm):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = BlockArray(grid, cm)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = arr[grid_slice]
            if copy:
                block = np.copy(block)
            rarr.blocks[grid_entry].oid = cm.put(block)
            rarr.blocks[grid_entry].dtype = getattr(np, dtype_str)
        return rarr

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, cm):
        sample_idx = tuple(0 for dim in arr.shape)
        if isinstance(arr, Block):
            sample_block = arr
            result_shape = ()
        else:
            sample_block = arr[sample_idx]
            if result_shape is None:
                result_shape = array_utils.shape_from_block_array(arr)
        result_block_shape = sample_block.shape
        result_dtype_str = sample_block.dtype.__name__
        result_grid = ArrayGrid(
            shape=result_shape, block_shape=result_block_shape, dtype=result_dtype_str
        )
        assert arr.shape == result_grid.grid_shape
        result = BlockArray(result_grid, cm)
        for grid_entry in result_grid.get_entry_iterator():
            if isinstance(arr, Block):
                block: Block = arr
            else:
                block: Block = arr[grid_entry]
            result.blocks[grid_entry] = block
        return result

    def copy(self):
        grid_copy = self.grid.from_meta(self.grid.to_meta())
        rarr_copy = BlockArray(grid_copy, self.cm)
        for grid_entry in grid_copy.get_entry_iterator():
            rarr_copy.blocks[grid_entry] = self.blocks[grid_entry].copy()
        return rarr_copy

    def touch(self):
        """
        "Touch" an array. This is an efficient distributed "wait" operation.
        """
        oids = []
        for grid_entry in self.grid.get_entry_iterator():
            block: Block = self.blocks[grid_entry]
            oids.append(
                self.cm.touch(
                    block.oid,
                    syskwargs={
                        "grid_entry": block.grid_entry,
                        "grid_shape": block.grid_shape,
                    },
                )
            )
        self.cm.get(oids)
        return self

    def reshape(self, *shape, **kwargs):
        block_shape = kwargs.get("block_shape", None)
        if array_utils.is_int(shape):
            shape = (shape,)
        elif len(shape) == 0:
            shape = self.shape
        elif isinstance(shape[0], (tuple, list)):
            assert len(shape) == 1
            shape = shape[0]
        else:
            assert all(np.issubdtype(type(n), int) for n in shape)
        shape = Reshape.compute_shape(self.shape, shape)
        if block_shape is None:
            if shape == self.shape:
                # This is a noop.
                block_shape = self.block_shape
            else:
                block_shape = self.cm.get_block_shape(shape, self.dtype)
        return Reshape()(self, shape, block_shape)

    def expand_dims(self, axis):
        """
        This function refers to the numpy implementation of expand_dims.
        """
        if type(axis) not in (tuple, list):
            axis = (axis,)
        out_ndim = len(axis) + self.ndim
        axis = np.core.numeric.normalize_axis_tuple(axis, out_ndim)

        shape_it = iter(self.shape)
        block_shape_it = iter(self.block_shape)
        shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
        block_shape = [
            1 if ax in axis else next(block_shape_it) for ax in range(out_ndim)
        ]
        return self.reshape(shape, block_shape=block_shape)

    def squeeze(self):
        shape = self.shape
        block_shape = self.block_shape
        new_shape = []
        new_block_shape = []
        for s, b in zip(shape, block_shape):
            if s == 1:
                assert b == 1
                continue
            new_shape.append(s)
            new_block_shape.append(b)
        return self.reshape(new_shape, block_shape=new_block_shape)

    def swapaxes(self, axis1, axis2):
        meta_swap = self.grid.to_meta()
        shape = list(meta_swap["shape"])
        block_shape = list(meta_swap["block_shape"])
        dim = len(shape)
        if axis1 >= dim or axis2 >= dim:
            raise ValueError("axis is larger than the array dimension")
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
        block_shape[axis1], block_shape[axis2] = block_shape[axis2], block_shape[axis1]
        meta_swap["shape"] = tuple(shape)
        meta_swap["block_shape"] = tuple(block_shape)
        grid_swap = ArrayGrid.from_meta(meta_swap)
        rarr_src = np.ndarray(self.blocks.shape, dtype="O")

        for grid_entry in self.grid.get_entry_iterator():
            rarr_src[grid_entry] = self.blocks[grid_entry].swapaxes(axis1, axis2)
        rarr_src = rarr_src.swapaxes(axis1, axis2)

        rarr_swap = BlockArray(grid_swap, self.cm, rarr_src)
        return rarr_swap

    def transpose(self, defer=False, redistribute=False):
        """
        Transpose this matrix. Only use defer with arithmetic operations.
        Setting redistribute to True may significantly impact performance.
        :param defer: When true, the transpose operation will be applied
        with the next arithmetic operation.
        :param redistribute: If defer is false, setting this to true will
        redistribute the data according to the device grid (data placement policy).
        This parameter has no effect when defer is true.
        :return: The transposed matrix.
        """
        if defer and redistribute:
            warnings.warn("defer is True, redistribute=True will be ignored.")
        metaT = self.grid.to_meta()
        metaT["shape"] = tuple(reversed(metaT["shape"]))
        metaT["block_shape"] = tuple(reversed(metaT["block_shape"]))
        gridT = ArrayGrid.from_meta(metaT)
        rarrT = BlockArray(gridT, self.cm)
        rarrT.blocks = np.copy(self.blocks.T)
        for grid_entry in rarrT.grid.get_entry_iterator():
            rarrT.blocks[grid_entry] = rarrT.blocks[grid_entry].transpose(
                defer, redistribute
            )
        return rarrT

    def __getattr__(self, item):
        if item == "__array_priority__" or item == "__array_struct__":
            # This is triggered by a numpy array on the LHS.
            raise ValueError("Unable to covert numpy array to block array.")
        elif item == "ndim":
            return len(self.shape)
        elif item == "T":
            return self.transpose()
        else:
            raise NotImplementedError(item)

    def _preprocess_subscript(self, item, allow_indexes=False):
        if not isinstance(item, tuple):
            ss = (item,)
        else:
            ss = item
        # We need to fetch any block arrays.
        tmp = []
        for entry in ss:
            if isinstance(entry, BlockArray):
                tmp.append(entry.get())
            else:
                tmp.append(entry)
        ss = tuple(tmp)
        is_handled_advanced = False
        array_encountered = False
        axis = None

        # Check if this is a supported advanced indexing operation.
        for i, entry in enumerate(ss):
            if isinstance(entry, slice) and entry.start is None and entry.stop is None:
                continue
            elif allow_indexes and (
                array_utils.is_int(entry) or array_utils.is_uint(entry)
            ):
                continue
            elif array_utils.is_array_like(entry):
                if array_encountered:
                    raise NotImplementedError(
                        "Advanced indexing is only supported " "along a single axis."
                    )
                is_handled_advanced = True
                array_encountered = True
                axis = i
            else:
                if array_encountered:
                    raise NotImplementedError(
                        "Advanced indexing is only supported "
                        "with full slices along other axes."
                    )
                is_handled_advanced = False
                break

        return ss, is_handled_advanced, axis

    def __getitem__(self, item):
        ss, is_handled_advanced, axis = self._preprocess_subscript(item)

        if is_handled_advanced:
            # Treat this as a shuffle.
            return self._advanced_single_array_select(ss, axis=axis)

        av: ArrayView = ArrayView.from_block_array(self)
        # TODO (hme): We don't have to create, but do so for now until we need to optimize.
        return av[item].create(BlockArray)

    def _advanced_single_array_select(
        self, ss: tuple, block_size: int = None, axis: int = 0
    ):
        # Create output array along the axis of the selection operation.
        # We don't allocate zeros for output array. Instead, we let the update kernel
        # create the initial set of zeros to save some memory.
        array = ss[axis]
        assert len(array.shape) == 1
        if array_utils.is_bool(array.dtype, type_test=True):
            array = np.arange(len(array))[array]
        assert np.all(0 <= array) and np.all(array < self.shape[axis])
        ss = list(ss)
        ss[axis] = array
        ss = tuple(ss)

        if block_size is None:
            block_size = self.block_shape[axis]
        axis_dim = len(array)
        shape = tuple(
            list(self.shape[:axis]) + [axis_dim] + list(self.shape[axis + 1 :])
        )
        block_shape = tuple(
            list(self.block_shape[:axis])
            + [block_size]
            + list(self.block_shape[axis + 1 :])
        )
        dst_arr = BlockArray(
            ArrayGrid(shape=shape, block_shape=block_shape, dtype=self.dtype.__name__),
            cm=self.cm,
        )
        # Along axis, we don't know which destination blocks depend on which source blocks.
        # For every destination block,
        # apply sel to every source block.
        # If the destination block depends on the source block, apply the changes and return a
        # new array, otherwise return the unchanged array.
        # For k blocks along axis, this has worst case complexity of generating k^2 copies of
        # the destination block.
        src_arr = self
        dst_grid_shape = dst_arr.grid.grid_shape
        src_grid_shape = src_arr.grid.grid_shape
        ss = self.cm.put(ss)
        for i in range(dst_grid_shape[axis]):
            for dst_grid_entry in dst_arr.grid.get_entry_iterator():
                if dst_grid_entry[axis] != i:
                    # Compute the value of dest_arr for each block along axis.
                    # e.g. for a 2 dim array, we fix the row and compute the column blocks.
                    continue
                dst_block: Block = dst_arr.blocks[dst_grid_entry]
                dst_coord: tuple = dst_arr.grid.get_entry_coordinates(dst_grid_entry)
                for j in range(src_grid_shape[axis]):
                    # Apply sel from each block along axis of src_arr.
                    # e.g. for 2 dim array, we fix the column blocks
                    # given by dst_grid_entry, and iterate over the rows.
                    src_grid_entry = tuple(
                        list(dst_grid_entry[:axis])
                        + [j]
                        + list(dst_grid_entry[axis + 1 :])
                    )
                    src_block: Block = src_arr.blocks[src_grid_entry]
                    src_coord: tuple = src_arr.grid.get_entry_coordinates(
                        src_grid_entry
                    )
                    if dst_block.oid is None:
                        dst_arg = (dst_block.shape, dst_block.dtype)
                    else:
                        dst_arg = dst_block.oid
                    dst_block.oid = self.cm.update_block_along_axis(
                        dst_arg,
                        src_block.oid,
                        ss,
                        axis,
                        dst_coord,
                        src_coord,
                        syskwargs={
                            "grid_entry": dst_grid_entry,
                            "grid_shape": dst_arr.grid.grid_shape,
                        },
                    )
        return dst_arr

    def __setitem__(self, key, value):
        value: BlockArray = BlockArray.to_block_array(value, self.cm)
        ss, is_handled_advanced, axis = self._preprocess_subscript(
            key, allow_indexes=True
        )
        if is_handled_advanced:
            return self._advanced_single_array_assign(ss, value, axis)
        av: ArrayView = ArrayView.from_block_array(self)
        av[key] = value

    def _advanced_single_array_assign(
        self,
        ss: tuple,
        value,
        axis: int,
    ):
        array = ss[axis]
        assert len(array.shape) == 1
        if array_utils.is_bool(array.dtype, type_test=True):
            array = np.arange(len(array))[array]
        assert np.all(0 <= array) and np.all(array < self.shape[axis])
        ss = list(ss)
        ss[axis] = array
        ss = tuple(ss)

        # The subscript contains a single array. We therefore know one of two things is true:
        # 1. value is the same shape as self along axes != axis.
        # 2. value is scalar or 1-dimensional.
        # We currently don't support the case where value may broadcasted if it has more dims.
        # This should be a straight-forward future task.
        value: BlockArray = value
        mode = None
        if len(value.shape) == 0:
            # subscripted value per block will broadcast to other dimensions.
            mode = "scalar"
        elif len(value.shape) == 1:
            # assert len(value.shape) == len(ss)
            mode = "single-dim"
            # Can broadcast if trailing dim matches.
            assert len(ss[axis]) == value.shape[0]

            for i in range(len(self.shape)):
                if i == axis:
                    assert len(ss[i]) == value.shape[0]
                elif i < axis:
                    # Nothing to check here.
                    # These entries are : or integer.
                    pass
                else:
                    if i < len(ss):
                        if not isinstance(ss[i], slice):
                            # ss[i] is an integer.
                            continue
                    # If we're here, then the rest of the subscript operator
                    # will resolve to :, which is not broadcastable.
                    raise ValueError(
                        "Cannot broadcast input array "
                        "from shape %s into shape %s"
                        % (value.shape, tuple([value.shape[0]] + list(self.shape[i:])))
                    )
        elif len(value.shape) == len(self.shape):
            mode = "multi-dim"
            new_block_shape = []
            for i in range(len(self.shape)):
                if i == axis:
                    new_block_shape.append(value.block_shape[i])
                elif i < len(ss) and (
                    array_utils.is_int(ss[i]) or array_utils.is_uint(ss[i])
                ):
                    # These entries are : or integer.
                    # assert array_utils.is_int(ss[i]) or array_utils.is_uint(ss[i])
                    assert value.shape[i] == 1
                    new_block_shape.append(1)
                else:
                    assert value.shape[i] == self.shape[i], "Shape mismatch."
                    new_block_shape.append(self.block_shape[i])
            if new_block_shape != value.block_shape:
                warnings.warn(
                    "Assigned value does not match block shape of assignee. "
                    "Applying reshape to assigned value."
                )
                value = value.reshape(block_shape=new_block_shape)

        # Like select, iterate over destination blocks along the axis being updated.
        # e.g. if self is 2-dim and axis=0, then fix the row and iterate over the columns.
        # If value has the same shape as self, then for each destination block,
        # iterate over the blocks in value along axis.
        # e.g. if self is 2-dim and axis=0, then for the given column, iterate over the rows
        # of value.
        # If value is scalar, then attempt to assign it to every destination block.
        # If value is 1-dim, the just iterate over the dim and assign accordingly.

        dst_arr = self
        src_arr = value
        src_grid_shape = src_arr.grid.grid_shape
        np_ss = ss
        ss = self.cm.put(ss)
        for dst_grid_entry in dst_arr.grid.get_entry_iterator():
            dst_block: Block = dst_arr.blocks[dst_grid_entry]
            dst_coord: tuple = dst_arr.grid.get_entry_coordinates(dst_grid_entry)

            # Make sure index values in subscript are within bounds of dst_arr.
            # We don't need to check src_arr:
            # 1) The block shapes of dst_arr and src_arr are the same except along axis
            #    and indices in ss. We are not concerned with axes the indices in ss correspond to,
            #    because they are of size 1 in src_arr => we only need to check that indices
            #    fall within bounds of dst_arr.
            # 2) For each dst_arr, we test the values
            #    to assign to dst_arr by traverse the src_arr along axis.
            #    Thus, size along all other axes are equal or broadcasted.
            skip = False
            for curr_axis in range(len(np_ss)):
                if curr_axis == axis or isinstance(np_ss[curr_axis], slice):
                    continue
                if not (
                    dst_coord[curr_axis]
                    <= np_ss[curr_axis]
                    < dst_coord[curr_axis] + dst_block.shape[curr_axis]
                ):
                    skip = True
                    break
            if skip:
                continue

            if mode == "scalar":
                src_block: Block = src_arr.blocks.item()
                src_coord: tuple = src_arr.grid.get_entry_coordinates(
                    src_block.grid_entry
                )
                dst_block.oid = self.cm.update_block_along_axis(
                    dst_block.oid,
                    src_block.oid,
                    ss,
                    axis,
                    dst_coord,
                    src_coord,
                    is_assignment=True,
                    syskwargs={
                        "grid_entry": dst_grid_entry,
                        "grid_shape": dst_arr.grid.grid_shape,
                    },
                )
            elif mode == "single-dim":
                for src_grid_entry in src_arr.grid.get_entry_iterator():
                    src_block: Block = src_arr.blocks[src_grid_entry]
                    src_coord: tuple = src_arr.grid.get_entry_coordinates(
                        src_grid_entry
                    )
                    dst_block.oid = self.cm.update_block_along_axis(
                        dst_block.oid,
                        src_block.oid,
                        ss,
                        axis,
                        dst_coord,
                        src_coord,
                        is_assignment=True,
                        syskwargs={
                            "grid_entry": dst_grid_entry,
                            "grid_shape": dst_arr.grid.grid_shape,
                        },
                    )
            elif mode == "multi-dim":
                for j in range(src_grid_shape[axis]):
                    # Apply sel from each block along axis of src_arr.
                    # e.g. for 2 dim array, we fix the column blocks
                    # given by dst_grid_entry, and iterate over the rows.
                    src_grid_entry = tuple(
                        list(dst_grid_entry[:axis])
                        + [j]
                        + list(dst_grid_entry[axis + 1 :])
                    )
                    src_block: Block = src_arr.blocks[src_grid_entry]
                    src_coord: tuple = src_arr.grid.get_entry_coordinates(
                        src_grid_entry
                    )
                    dst_block.oid = self.cm.update_block_along_axis(
                        dst_block.oid,
                        src_block.oid,
                        ss,
                        axis,
                        dst_coord,
                        src_coord,
                        is_assignment=True,
                        syskwargs={
                            "grid_entry": dst_grid_entry,
                            "grid_shape": dst_arr.grid.grid_shape,
                        },
                    )
        return dst_arr

    @staticmethod
    def to_block_array(obj, cm: ComputeManager, block_shape=None):
        if isinstance(obj, BlockArray):
            return obj
        if isinstance(obj, np.ndarray):
            np_array = obj
        elif isinstance(obj, list):
            np_array = np.array(obj)
        elif array_utils.is_scalar(obj):
            return BlockArray.from_scalar(obj, cm)
        else:
            raise Exception("Unsupported type %s" % type(obj))
        if block_shape is None:
            block_shape = cm.get_block_shape(np_array.shape, np_array.dtype)
        return BlockArray.from_np(np_array, block_shape, False, cm)

    def check_or_convert_other(self, other, compute_block_shape=False):
        block_shape = None if compute_block_shape else self.block_shape
        return BlockArray.to_block_array(other, self.cm, block_shape=block_shape)

    def ufunc(self, op_name):
        result = self.copy()
        for grid_entry in self.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].ufunc(op_name)
        return result

    def _tree_reduce(
        self, op_name, blocks_or_oids, result_grid_entry, result_grid_shape
    ):
        """
        Basic tree reduce imp.
        Schedules op on same node as left operand.
        :param op_name: The reduction op.
        :param blocks_or_oids: A list of type Block or a list of tuples.
                               Tuples must be of the form
                               (oid, grid_entry, grid_shape, transposed)
        :param result_grid_entry: The grid entry of the result block. This will be used
                                  to compute the final reduction step.
        :param result_grid_shape: The grid entry of the result block. This will be used
                                  to compute the final reduction step.
        :return: The oid of the result.
        """
        oid_list = blocks_or_oids
        if isinstance(blocks_or_oids[0], Block):
            oid_list = [
                (b.oid, b.grid_entry, b.grid_shape, b.transposed)
                for b in blocks_or_oids
            ]
        if len(oid_list) == 1:
            return oid_list[0][0]
        q = oid_list
        while len(q) > 1:
            a_oid, a_ge, a_gs, a_T = q.pop(0)
            b_oid, _, _, b_T = q.pop(0)
            ge, gs = (
                (result_grid_entry, result_grid_shape) if len(q) == 0 else (a_ge, a_gs)
            )
            c_oid = self.cm.bop_reduce(
                op_name,
                a_oid,
                b_oid,
                a_T,
                b_T,
                syskwargs={
                    "grid_entry": ge,
                    "grid_shape": gs,
                },
            )
            q.append((c_oid, ge, gs, False))
        r_oid, r_ge, r_gs, _ = q.pop(0)
        assert r_ge == result_grid_entry
        assert r_gs == result_grid_shape
        return r_oid

    def reduce_axis(self, op_name, axis, keepdims=False):
        if not (axis is None or isinstance(axis, (int, np.int32, np.int64))):
            raise NotImplementedError("Only integer axis is currently supported.")
        block_reduced_oids = np.empty_like(self.blocks, dtype=tuple)
        for grid_entry in self.grid.get_entry_iterator():
            block = self.blocks[grid_entry]
            block_oid = self.cm.reduce_axis(
                op_name=op_name,
                arr=block.oid,
                axis=axis,
                keepdims=keepdims,
                transposed=block.transposed,
                syskwargs={
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                },
            )
            block_reduced_oids[grid_entry] = (
                block_oid,
                block.grid_entry,
                block.grid_shape,
                False,
            )
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
        result = BlockArray(result_grid, self.cm)

        if axis is None:
            if result.shape == ():
                result_block: Block = result.blocks[()]
            else:
                result_block: Block = result.blocks[:].item()
            result_block.oid = self._tree_reduce(
                op_name,
                block_reduced_oids.flatten().tolist(),
                result_block.grid_entry,
                result_block.grid_shape,
            )
        else:
            for result_grid_entry in result_grid.get_entry_iterator():
                block_reduced_oids_axis = []
                for sum_dim in range(self.grid.grid_shape[axis]):
                    grid_entry = list(result_grid_entry)
                    if keepdims:
                        grid_entry[axis] = sum_dim
                    else:
                        grid_entry = grid_entry[:axis] + [sum_dim] + grid_entry[axis:]
                    grid_entry = tuple(grid_entry)
                    block_reduced_oids_axis.append(block_reduced_oids[grid_entry])
                result_block: Block = result.blocks[result_grid_entry]
                result_block.oid = self._tree_reduce(
                    op_name,
                    block_reduced_oids_axis,
                    result_block.grid_entry,
                    result_block.grid_shape,
                )
        return result

    def __matmul__(self, other):
        if len(self.shape) > 2:
            # TODO (bcp): NumPy's implementation does a stacked matmul, which is not supported yet.
            raise NotImplementedError(
                "Matrix multiply for tensors of rank > 2 not supported yet."
            )
        else:
            return self.tensordot(other, 1)

    def _compute_tensordot_syskwargs(self, self_block: Block, other_block: Block):
        # Schedule on larger block.
        if np.product(self_block.shape) >= np.product(other_block.shape):
            return self_block.true_grid_entry(), self_block.true_grid_shape()
        else:
            return other_block.true_grid_entry(), other_block.true_grid_shape()

    def tensordot(self, other, axes=2):
        if not isinstance(other, BlockArray):
            raise ValueError(
                "Cannot automatically construct BlockArray for tensor operations."
            )

        if isinstance(axes, int):
            pass
        elif array_utils.is_array_like(axes):
            raise NotImplementedError("Non-integer axes is currently not supported.")
        else:
            raise TypeError(f"Unexpected axes type '{type(axes).__name__}'")

        if array_utils.np_tensordot_param_test(
            self.shape, self.ndim, other.shape, other.ndim, axes
        ):
            raise ValueError("shape-mismatch for sum")

        other = self.check_or_convert_other(other, compute_block_shape=True)

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
            dtype=array_utils.get_bop_output_type(
                "tensordot", self.dtype, other.dtype
            ).__name__,
        )
        assert result_grid.grid_shape == tuple(this_axes + other_axes)
        result = BlockArray(result_grid, self.cm)
        this_dims = list(itertools.product(*map(range, this_axes)))
        other_dims = list(itertools.product(*map(range, other_axes)))
        sum_dims = list(itertools.product(*map(range, this_sum_axes)))
        for i in this_dims:
            for j in other_dims:
                grid_entry = tuple(i + j)
                result_block: Block = result.blocks[grid_entry]
                sum_oids = []
                for k in sum_dims:
                    self_block: Block = self.blocks[tuple(i + k)]
                    other_block: Block = other.blocks[tuple(k + j)]
                    dot_grid_args = self._compute_tensordot_syskwargs(
                        self_block, other_block
                    )
                    dotted_oid = self.cm.bop(
                        "tensordot",
                        self_block.oid,
                        other_block.oid,
                        self_block.transposed,
                        other_block.transposed,
                        axes=axes,
                        syskwargs={
                            "grid_entry": dot_grid_args[0],
                            "grid_shape": dot_grid_args[1],
                        },
                    )
                    sum_oids.append(
                        (dotted_oid, dot_grid_args[0], dot_grid_args[1], False)
                    )
                result_block.oid = self._tree_reduce(
                    "sum", sum_oids, result_block.grid_entry, result_block.grid_shape
                )
        return result

    def _fast_element_wise(self, op_name, other):
        """
        Implements fast scheduling for basic element-wise operations.
        """
        dtype = array_utils.get_bop_output_type(op_name, self.dtype, other.dtype)
        # Schedule the op first.
        blocks = np.empty(shape=self.grid.grid_shape, dtype=Block)
        for grid_entry in self.grid.get_entry_iterator():
            self_block: Block = self.blocks[grid_entry]
            other_block: Block = other.blocks[grid_entry]
            blocks[grid_entry] = block = Block(
                grid_entry=grid_entry,
                grid_shape=self_block.grid_shape,
                rect=self_block.rect,
                shape=self_block.shape,
                dtype=dtype,
                transposed=False,
                cm=self.cm,
            )
            block.oid = self.cm.bop(
                op_name,
                self_block.oid,
                other_block.oid,
                self_block.transposed,
                other_block.transposed,
                axes={},
                syskwargs={
                    "grid_entry": grid_entry,
                    "grid_shape": self.grid.grid_shape,
                },
            )
        return BlockArray(
            ArrayGrid(self.shape, self.block_shape, dtype.__name__),
            self.cm,
            blocks=blocks,
        )

    def __elementwise__(self, op_name, other):
        other = self.check_or_convert_other(other)
        if self.shape == other.shape and self.block_shape == other.block_shape:
            return self._fast_element_wise(op_name, other)
        blocks_op = self.blocks.__getattribute__("__%s__" % op_name)
        return BlockArray.from_blocks(
            blocks_op(other.blocks), result_shape=None, cm=self.cm
        )

    def __add__(self, other):
        return self.__elementwise__("add", other)

    def __sub__(self, other):
        return self.__elementwise__("sub", other)

    def __mul__(self, other):
        return self.__elementwise__("mul", other)

    def __truediv__(self, other):
        return self.__elementwise__("truediv", other)

    def __pow__(self, other):
        return self.__elementwise__("pow", other)

    def __invert__(self):
        return self.ufunc("invert")

    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __imatmul__ = __matmul__
    __itruediv__ = __truediv__
    __ipow__ = __pow__

    # TODO (hme): Type check bool ops.
    def __bool__(self):
        # pylint: disable=no-member
        dtype = self.dtype
        if isinstance(dtype, type):
            # TODO (hme): Fix this strange issue.
            dtype = dtype()
        if isinstance(dtype, (bool, np.bool)) and np.sum(self.shape) == len(self.shape):
            return self.get().__bool__()
        return True

    def __inequality__(self, op, other):
        other = self.check_or_convert_other(other)
        assert (
            other.shape == () or other.shape == self.shape
        ), "Currently supports comparison with scalars only."
        shape = array_utils.broadcast(self.shape, other.shape).shape
        block_shape = array_utils.broadcast_block_shape(
            self.shape, other.shape, self.block_shape
        )
        dtype = bool.__name__
        grid = ArrayGrid(shape, block_shape, dtype)
        result = BlockArray(grid, self.cm)
        for grid_entry in result.grid.get_entry_iterator():
            if other.shape == ():
                other_block: Block = other.blocks.item()
            else:
                other_block: Block = other.blocks[grid_entry]
            result.blocks[grid_entry] = self.blocks[grid_entry].bop(
                op, other_block, args={}
            )

        return result

    def __ge__(self, other):
        return self.__inequality__("ge", other)

    def __gt__(self, other):
        return self.__inequality__("gt", other)

    def __le__(self, other):
        return self.__inequality__("le", other)

    def __lt__(self, other):
        return self.__inequality__("lt", other)

    def __eq__(self, other):
        return self.__inequality__("eq", other)

    def __ne__(self, other):
        return self.__inequality__("ne", other)

    __radd__ = __add__

    def __rsub__(self, other):
        other = self.check_or_convert_other(other)
        return other - self

    __rmul__ = __mul__

    def __rmatmul__(self, other):
        other = self.check_or_convert_other(other)
        return other @ self

    def __rtruediv__(self, other):
        other = self.check_or_convert_other(other)
        return other / self

    def __rpow__(self, other):
        other = self.check_or_convert_other(other)
        return other ** self

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def astype(self, dtype):
        grid = ArrayGrid(self.shape, self.block_shape, dtype.__name__)
        result = BlockArray(grid, self.cm)
        for grid_entry in result.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].astype(dtype)
        return result

    def flattened_oids(self):
        oids = []
        for grid_entry in self.grid.get_entry_iterator():
            oid = self.blocks[grid_entry].oid
            oids.append(oid)
        return oids


class Reshape(object):
    @staticmethod
    def compute_shape(shape, input_shape):
        size = np.product(shape)
        if -1 in input_shape:
            new_shape = []
            other_dim_prod = 1
            negative_one_seen = False
            for dim in input_shape:
                if dim == -1:
                    if negative_one_seen:
                        raise Exception("Only one -1 permitted in reshape.")
                    negative_one_seen = True
                    continue
                other_dim_prod *= dim
            if size % other_dim_prod != 0:
                raise Exception("Invalid shape.")
            for dim in input_shape:
                if dim == -1:
                    new_shape.append(size // other_dim_prod)
                else:
                    new_shape.append(dim)
        else:
            new_shape = input_shape
        assert size == np.product(new_shape)
        return new_shape

    def _group_index_lists_by_block(
        self, dst_slice_tuples, src_grid: ArrayGrid, dst_index_list, src_index_list
    ):
        # TODO(hme): Keep this function here until it's needed for greater support of
        #  selection/assignment operations.
        # Block grid entries needed to write to given dst_slice_selection.
        src_blocks = {}
        dst_slice_np = np.array(dst_slice_tuples).T
        dst_index_arr = np.array(dst_index_list)
        src_index_arr = np.array(src_index_list)
        # Pick the smallest type to represent indices.
        # A set of these indices may be transmitted over the network,
        # so we want to pick the smallest encoding possible.
        index_types = [
            (2 ** 8, np.uint8),
            (2 ** 16, np.uint16),
            (2 ** 32, np.uint32),
            (2 ** 64, np.uint64),
        ]
        index_type = None
        for bound, curr_index_type in index_types:
            if np.all(np.array(src_grid.block_shape) < bound) and np.all(
                dst_slice_np[1] < bound
            ):
                index_type = curr_index_type
                break
        if index_type is None:
            raise Exception("Unable to encode block indices, blocks are too large.")
        for grid_entry in src_grid.get_entry_iterator():
            src_slice_np = np.array(src_grid.get_slice_tuples(grid_entry)).T
            index_pairs = []
            for i in range(src_index_arr.shape[0]):
                src_index = src_index_arr[i]
                dst_index = dst_index_arr[i]
                if np.all(
                    (src_slice_np[0] <= src_index) & (src_index < src_slice_np[1])
                ):
                    index_pair = (
                        (dst_index - dst_slice_np[0]).astype(index_type),
                        (src_index - src_slice_np[0]).astype(index_type),
                    )
                    index_pairs.append(index_pair)
            if len(index_pairs) > 0:
                src_blocks[grid_entry] = index_pairs
        return src_blocks

    def _arbitrary_reshape(self, arr: BlockArray, shape, block_shape) -> BlockArray:
        # This is the worst-case scenario.
        # Generate index mappings per block, and group source indices to minimize
        # RPCs and generation of new objects.
        cm = arr.cm
        dst_arr = BlockArray.empty(
            shape=shape, block_shape=block_shape, dtype=arr.dtype, cm=cm
        )
        for dst_grid_entry in dst_arr.grid.get_entry_iterator():
            dst_block: Block = dst_arr.blocks[dst_grid_entry]
            dst_slice_selection = dst_arr.grid.get_slice(dst_grid_entry)
            dst_index_list = array_utils.slice_sel_to_index_list(dst_slice_selection)
            src_index_list = array_utils.translate_index_list(
                dst_index_list, shape, arr.shape
            )
            src_blocks = self._group_index_lists_by_block(
                dst_arr.grid.get_slice_tuples(dst_grid_entry),
                arr.grid,
                dst_index_list,
                src_index_list,
            )
            for src_grid_entry in src_blocks:
                src_block: Block = arr.blocks[src_grid_entry]
                index_pairs = src_blocks[src_grid_entry]
                syskwargs = {
                    "grid_entry": dst_grid_entry,
                    "grid_shape": dst_arr.grid.grid_shape,
                }
                dst_block.oid = cm.update_block_by_index(
                    dst_block.oid, src_block.oid, index_pairs, syskwargs=syskwargs
                )
        return dst_arr

    def _block_shape_reshape(self, arr, block_shape):
        rarr: BlockArray = BlockArray.empty(arr.shape, block_shape, arr.dtype, arr.cm)
        for grid_entry in rarr.grid.get_entry_iterator():
            grid_entry_slice = rarr.grid.get_slice(grid_entry)
            # TODO (hme): This could be less costly.
            rarr[grid_entry_slice] = arr[grid_entry_slice]
        return rarr

    def _strip_ones(self, shape):
        return tuple(filter(lambda x: x != 1, shape))

    def _check_positions_ones(self, shape, block_shape):
        # If a position in the shape is 1, then the corresponding
        # position in block_shape should also be 1.
        for i in range(len(shape)):
            if shape[i] == 1:
                if shape[i] != block_shape[i]:
                    return False
        return True

    def _is_simple_reshape(self, arr: BlockArray, shape, block_shape):
        # Is the reshape a difference of factors of 1?
        # Strip out 1s and compare.
        # If a position in the shape is 1, then the corresponding
        # position in block_shape should also be 1.

        # If source shape and dest shape are the same or source block_shape and dest block_shape
        # are same, this is not a simple reshape.
        if shape == arr.shape or block_shape == arr.block_shape:
            return False

        # Checks if source shape and dest shape are same & source block_shape and dest
        # block_shape are same after stripping ones.
        if not (
            self._strip_ones(shape) == self._strip_ones(arr.shape)
            and self._strip_ones(block_shape) == self._strip_ones(arr.block_shape)
        ):
            return False
        if not self._check_positions_ones(shape, block_shape):
            return False
        return True

    def _simple_reshape(self, arr, shape, block_shape):
        # Reshape the array of blocks only.
        # This is only used when the difference in shape are factors of 1s,
        # and the ordering of other factors are maintained.

        # Check assumptions.
        assert len(self._strip_ones(arr.shape)) == len(self._strip_ones(shape))

        # Create new grid, and perform reshape on blocks
        # to simplify access to source blocks.
        grid = ArrayGrid(shape, block_shape, dtype=arr.dtype.__name__)
        src_blocks = arr.blocks.reshape(grid.grid_shape)
        rarr = BlockArray(grid, arr.cm)
        for grid_entry in grid.get_entry_iterator():
            src_block: Block = src_blocks[grid_entry]
            dst_block: Block = rarr.blocks[grid_entry]
            syskwargs = {"grid_entry": grid_entry, "grid_shape": grid.grid_shape}
            dst_block.oid = arr.cm.reshape(
                src_block.oid, dst_block.shape, syskwargs=syskwargs
            )
        return rarr

    def _validate(self, arr, shape, block_shape):
        assert -1 not in shape
        assert -1 not in block_shape
        assert len(shape) == len(block_shape)
        assert np.product(arr.shape) == np.product(shape)

    def __call__(self, arr: BlockArray, shape, block_shape):
        self._validate(arr, shape, block_shape)
        if arr.shape == shape and arr.block_shape == block_shape:
            return arr
        elif self._is_simple_reshape(arr, shape, block_shape):
            return self._simple_reshape(arr, shape, block_shape)
        elif arr.shape == shape and arr.block_shape != block_shape:
            return self._block_shape_reshape(arr, block_shape)
        elif arr.shape != shape and arr.block_shape == block_shape:
            # Just do full reshape for this case as well.
            # Though there may be a better solution, we generally expect
            # the block shape to change with array shape.
            return self._arbitrary_reshape(arr, shape, block_shape)
        else:
            assert arr.shape != shape and arr.block_shape != block_shape
            return self._arbitrary_reshape(arr, shape, block_shape)
