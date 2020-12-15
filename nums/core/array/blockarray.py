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

import numpy as np

from nums.core import settings
from nums.core.storage.storage import ArrayGrid
from nums.core.array import utils as array_utils
from nums.core.array.base import BlockArrayBase, Block
from nums.core.array.view import ArrayView
from nums.core.array import selection


class BlockArray(BlockArrayBase):

    @classmethod
    def empty(cls, shape, block_shape, dtype, system):
        grid = ArrayGrid(shape=shape,
                         block_shape=block_shape,
                         dtype=dtype.__name__)
        grid_meta = grid.to_meta()
        arr = BlockArray(grid, system)
        for grid_entry in grid.get_entry_iterator():
            arr.blocks[grid_entry].oid = system.empty(grid_entry, grid_meta,
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": grid.grid_shape
                                                      })
        return arr

    @classmethod
    def from_scalar(cls, val, system):
        if isinstance(val, int):
            dtype = np.int
        elif isinstance(val, float):
            dtype = np.float
        else:
            assert isinstance(val, (np.int32, np.int64, np.float32, np.float64))
            dtype = None
        return BlockArray.from_np(np.array(val, dtype=dtype),
                                  block_shape=(),
                                  copy=False,
                                  system=system)

    @classmethod
    def from_oid(cls, oid, shape, dtype, system):
        block_shape = shape
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        ba = BlockArray(grid, system)
        for i, grid_entry in enumerate(grid.get_entry_iterator()):
            assert i == 0
            ba.blocks[grid_entry].oid = oid
        return ba

    @classmethod
    def from_np(cls, arr, block_shape, copy, system):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = BlockArray(grid, system)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = arr[grid_slice]
            if copy:
                block = np.copy(block)
            rarr.blocks[grid_entry].oid = system.put(block)
            rarr.blocks[grid_entry].dtype = getattr(np, dtype_str)
        return rarr

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, system):
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
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=result_dtype_str)
        assert arr.shape == result_grid.grid_shape
        result = BlockArray(result_grid, system)
        for grid_entry in result_grid.get_entry_iterator():
            if isinstance(arr, Block):
                block: Block = arr
            else:
                block: Block = arr[grid_entry]
            result.blocks[grid_entry] = block
        return result

    def copy(self):
        grid_copy = self.grid.from_meta(self.grid.to_meta())
        rarr_copy = BlockArray(grid_copy, self.system)
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
            oids.append(self.system.touch(block.oid, syskwargs=block.syskwargs()))
        self.system.get(oids)
        return self

    def reshape(self, shape=None, **kwargs):
        block_shape = kwargs.get("block_shape", None)
        if array_utils.is_int(shape):
            shape = (shape,)
        elif shape is None:
            shape = self.shape
        shape = Reshape.compute_shape(self.shape, shape)
        if block_shape is None:
            if shape == self.shape:
                # This is a noop.
                block_shape = self.block_shape
            else:
                block_shape = self._get_and_register_block_shape(shape)
        return Reshape()(self, shape, block_shape)

    # TODO (hme): Remove this during engine/sys refactor.
    # Temporary hack to obtain block_shape for reshape invocations.
    def _get_and_register_block_shape(self, shape):
        # pylint: disable=import-outside-toplevel
        # Only allow this to be used if app manager is maintaining an app instance.
        import nums.core.application_manager as am
        assert am.is_initialized(), "Unexpected application state: " \
                                    "application instance doesn't exist."
        app = am.instance()
        return app.get_block_shape(shape, self.dtype)

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
        block_shape = [1 if ax in axis else next(block_shape_it) for ax in range(out_ndim)]
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

    def __getattr__(self, item):
        if item == "__array_priority__" or item == "__array_struct__":
            # This is triggered by a numpy array on the LHS.
            raise ValueError("Unable to covert numpy array to block array.")
        elif item == "ndim":
            return len(self.shape)
        elif item == "T":
            metaT = self.grid.to_meta()
            metaT["shape"] = tuple(reversed(metaT["shape"]))
            metaT["block_shape"] = tuple(reversed(metaT["block_shape"]))
            gridT = ArrayGrid.from_meta(metaT)
            rarrT = BlockArray(gridT, self.system)
            rarrT.blocks = np.copy(self.blocks.T)
            for grid_entry in rarrT.grid.get_entry_iterator():
                rarrT.blocks[grid_entry] = rarrT.blocks[grid_entry].transpose()
            return rarrT
        else:
            raise NotImplementedError(item)

    def __getitem__(self, item):
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
        is_handled_advanced = True
        if len(ss) > 1:
            # Check if all entries are full slices except the last entry.
            for entry in ss[:-1]:
                is_handled_advanced = is_handled_advanced and (isinstance(entry, slice)
                                                               and entry.start is None
                                                               and entry.stop is None)
        if is_handled_advanced and array_utils.is_array_like(ss[-1]):
            # Treat this as a shuffle.
            return self._advanced_single_array_subscript(sel=(ss[-1],), axis=len(ss)-1)

        av: ArrayView = ArrayView.from_block_array(self)
        # TODO (hme): We don't have to create, but do so for now until we need to optimize.
        return av[item].create(BlockArray)

    def _advanced_single_array_subscript(self, sel: tuple, block_size=None, axis=0):

        def group_by_block(dst_grid_entry,
                           dst_slice_tuples,
                           src_grid,
                           dst_index_list,
                           src_index_list,
                           axis=0):
            # Block grid entries needed to write to given dst_slice_selection.
            src_blocks = {}
            dst_slice_np = np.array(dst_slice_tuples).T
            dst_index_arr = np.array(dst_index_list)
            src_index_arr = np.array(src_index_list)
            # Pick the smallest type to represent indices.
            # A set of these indices may be transmitted over the network,
            # so we want to pick the smallest encoding possible.
            index_types = [(2 ** 8, np.uint8), (2 ** 16, np.uint16),
                           (2 ** 32, np.uint32), (2 ** 64, np.uint64)]
            index_type = None
            for bound, curr_index_type in index_types:
                if np.all(np.array(src_grid.block_shape[axis]) < bound) and np.all(
                        dst_slice_np[1][axis] < bound):
                    index_type = curr_index_type
                    break
            if index_type is None:
                raise Exception("Unable to encode block indices, blocks are too large.")
            dst_entry_test = list(dst_grid_entry[:axis]) + list(dst_grid_entry[axis + 1:])
            num_pairs_check = 0
            for grid_entry in src_grid.get_entry_iterator():
                # Must match on every entry except axis.
                src_entry_test = list(grid_entry[:axis]) + list(grid_entry[axis+1:])
                if dst_entry_test != src_entry_test:
                    # Skip this block.
                    continue
                src_slice_np = np.array(src_grid.get_slice_tuples(grid_entry)).T
                index_pairs = []
                for i in range(src_index_arr.shape[0]):
                    src_index = src_index_arr[i]
                    dst_index = dst_index_arr[i]
                    if np.all((src_slice_np[0][axis] <= src_index)
                              & (src_index < src_slice_np[1][axis])):
                        index_pair = (np.array(dst_index - dst_slice_np[0][axis], dtype=index_type),
                                      np.array(src_index - src_slice_np[0][axis], dtype=index_type))
                        index_pairs.append(index_pair)
                        num_pairs_check += 1
                if len(index_pairs) > 0:
                    src_blocks[grid_entry] = index_pairs
            assert num_pairs_check == len(dst_index_list)
            return src_blocks

        array = sel[0]
        assert len(array.shape) == 1
        assert np.all(0 <= array) and np.all(array < self.shape[axis])
        if block_size is None:
            block_size = self.block_shape[axis]
        axis_dim = len(array)
        shape = tuple(list(self.shape[:axis]) + [axis_dim] + list(self.shape[axis+1:]))
        block_shape = tuple(list(self.block_shape[:axis])
                            + [block_size]
                            + list(self.block_shape[axis+1:]))
        dst_arr = BlockArray.empty(shape=shape,  block_shape=block_shape,
                                   dtype=self.dtype,  system=self.system)

        for dst_grid_entry in dst_arr.grid.get_entry_iterator():
            dst_block: Block = dst_arr.blocks[dst_grid_entry]
            dst_slice_selection = dst_arr.grid.get_slice(dst_grid_entry)
            dst_index_array = selection.slice_to_range(dst_slice_selection[axis], shape[axis])
            src_index_array = array[dst_slice_selection[axis]]
            assert len(dst_index_array) == len(src_index_array)
            # Can this be sped up by grouping all src blocks outside of this loop?
            src_blocks = group_by_block(
                dst_grid_entry,
                dst_arr.grid.get_slice_tuples(dst_grid_entry),
                self.grid,
                dst_index_array,
                src_index_array,
                axis
            )
            for src_grid_entry in src_blocks:
                src_block: Block = self.blocks[src_grid_entry]
                index_pairs = src_blocks[src_grid_entry]
                syskwargs = {"grid_entry": dst_grid_entry, "grid_shape": dst_arr.grid.grid_shape}
                dst_block.oid = self.system.update_block_along_axis(dst_block.oid,
                                                                    src_block.oid,
                                                                    index_pairs,
                                                                    axis,
                                                                    syskwargs=syskwargs)
        return dst_arr

    def __setitem__(self, key, value):
        av: ArrayView = ArrayView.from_block_array(self)
        av[key] = value

    def check_or_convert_other(self, other):
        if isinstance(other, BlockArray):
            return other
        if isinstance(other, np.ndarray):
            return self.from_np(other, self.block_shape, False, self.system)
        if isinstance(other, list):
            other = np.array(other)
            return self.from_np(other, self.block_shape, False, self.system)
        if isinstance(other, (np.int32, np.int64, np.float32, np.float64, int, float)):
            return self.from_scalar(other, self.system)
        if isinstance(other, (np.bool, np.bool_, bool)):
            other = np.array(other)
            return self.from_np(other, self.block_shape, False, self.system)
        raise Exception("Unsupported type %s" % type(other))

    def ufunc(self, op_name):
        result = self.copy()
        for grid_entry in self.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].ufunc(op_name)
        return result

    def reduce_axis(self, op_name, axis, keepdims=False):
        if not (axis is None or isinstance(axis, (int, np.int32, np.int64))):
            raise NotImplementedError("Only integer axis is currently supported.")
        result_blocks = np.empty_like(self.blocks, dtype=Block)
        for grid_entry in self.grid.get_entry_iterator():
            result_blocks[grid_entry] = self.blocks[grid_entry].reduce_axis(op_name,
                                                                            axis,
                                                                            keepdims=keepdims)
        result_shape = []
        result_block_shape = []
        for curr_axis in range(len(self.shape)):
            axis_size, axis_block_size = self.shape[curr_axis], self.block_shape[curr_axis]
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
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=result_dtype.__name__)
        result = BlockArray(result_grid, self.system)

        if op_name in settings.np_pairwise_reduction_map:
            # Do a pairwise reduction with the pairwise reduction op.
            pairwise_op_name = settings.np_pairwise_reduction_map.get(op_name, op_name)
            if axis is None:
                reduced_block: Block = None
                for grid_entry in self.grid.get_entry_iterator():
                    if reduced_block is None:
                        reduced_block = result_blocks[grid_entry]
                        continue
                    next_block = result_blocks[grid_entry]
                    reduced_block = reduced_block.bop(pairwise_op_name, next_block, {})
                if result.shape == ():
                    result.blocks[()] = reduced_block
                else:
                    result.blocks[:] = reduced_block

            else:
                for result_grid_entry in result_grid.get_entry_iterator():
                    reduced_block: Block = None
                    for sum_dim in range(self.grid.grid_shape[axis]):
                        grid_entry = list(result_grid_entry)
                        if keepdims:
                            grid_entry[axis] = sum_dim
                        else:
                            grid_entry = grid_entry[:axis] + [sum_dim] + grid_entry[axis:]
                        grid_entry = tuple(grid_entry)
                        next_block: Block = result_blocks[grid_entry]
                        if reduced_block is None:
                            reduced_block = next_block
                        else:
                            reduced_block = reduced_block.bop(pairwise_op_name, next_block, {})
                    result.blocks[result_grid_entry] = reduced_block
        else:
            op_func = np.__getattribute__(op_name)
            if result.shape == ():
                result.blocks[()] = op_func(result_blocks, axis=axis, keepdims=keepdims)
            else:
                result.blocks = op_func(result_blocks, axis=axis, keepdims=keepdims)
        return result

    def __matmul__(self, other):
        if len(self.shape) > 2:
            return self.tensordot(other, 2)
        else:
            return self.tensordot(other, 1)

    def tensordot(self, other, axes=2):
        if not isinstance(other, BlockArray):
            raise ValueError("Cannot automatically construct BlockArray for tensor operations.")

        def basic_vector(ba: BlockArray, axis):
            if len(ba.shape) == 0:
                return False
            if len(ba.shape) == 1:
                return True
            size = ba.shape[axis]
            rest = list(ba.shape[:axis]) + list(ba.shape[axis + 1:])
            return np.sum(rest) == len(rest) <= 1 < size

        other = self.check_or_convert_other(other)
        if basic_vector(self, len(self.shape) - 1) and basic_vector(other, 0):
            return self._vecdot(other)
        elif len(self.shape) == 2 and (len(other.shape) == 1
                                       or (len(other.shape) == 2 and other.shape[1] == 1)):
            # Optimized matrix vector multiply.
            return self._matvec(other)
        else:
            return self._tensordot(other, axes)

    def _tensordot(self, other, axes):
        this_axes = self.grid.grid_shape[:-axes]
        this_sum_axes = self.grid.grid_shape[-axes:]
        other_axes = other.grid.grid_shape[axes:]
        other_sum_axes = other.grid.grid_shape[:axes]
        assert this_sum_axes == other_sum_axes
        result_shape = tuple(self.shape[:-axes] + other.shape[axes:])
        result_block_shape = tuple(self.block_shape[:-axes] + other.block_shape[axes:])
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=array_utils.get_bop_output_type("tensordot",
                                                                      self.dtype,
                                                                      other.dtype).__name__)
        assert result_grid.grid_shape == tuple(this_axes + other_axes)
        result = BlockArray(result_grid, self.system)
        this_dims = list(itertools.product(*map(range, this_axes)))
        other_dims = list(itertools.product(*map(range, other_axes)))
        sum_dims = list(itertools.product(*map(range, this_sum_axes)))
        for i in this_dims:
            for j in other_dims:
                grid_entry = tuple(i + j)
                result_block = None
                for k in sum_dims:
                    self_block: Block = self.blocks[tuple(i + k)]
                    other_block: Block = other.blocks[tuple(k + j)]
                    dotted_block = self_block.tensordot(other_block, axes=axes)
                    if result_block is None:
                        result_block = dotted_block
                    else:
                        result_block += dotted_block
                result.blocks[grid_entry] = result_block
        return result

    def _vecdot(self, other):
        assert self.shape[-1] == other.shape[0], str((self.shape[1], other.shape[0]))
        result_shape = tuple(self.shape[:-1] + other.shape[1:])
        result_block_shape = tuple(self.block_shape[:-1] + other.block_shape[1:])
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=self.dtype.__name__)
        result = BlockArray(result_grid, self.system)
        self_num_axes = len(self.grid.grid_shape)
        other_num_axes = len(other.grid.grid_shape)
        oids = []
        for i in range(self.grid.grid_shape[-1]):
            self_grid_entry = tuple(i if axis == self_num_axes-1 else 0
                                    for axis in range(self_num_axes))
            other_grid_entry = tuple(i if axis == 0 else 0 for axis in range(other_num_axes))
            self_block: Block = self.blocks[self_grid_entry]
            other_block: Block = other.blocks[other_grid_entry]
            if self_block.transposed != other_block.transposed:
                # The vectors are aligned if their transpositions satisfy the xor relation.
                if self_block.transposed:
                    # Use other grid entry for dot,
                    # because physically,
                    # other block is located on same node as self block.
                    sch_grid_entry = other_grid_entry
                    sch_grid_shape = other.grid.grid_shape
                elif other_block.transposed:
                    # Use self grid entry for dot.
                    sch_grid_entry = self_grid_entry
                    sch_grid_shape = self.grid.grid_shape
                else:
                    raise Exception("Impossible.")
            else:
                # They're either both transposed or not.
                # Either way, one will need to be transmitted, so transmit other.
                sch_grid_entry = self_grid_entry
                sch_grid_shape = self.grid.grid_shape
            dot_oid = self.system.bop("tensordot",
                                      a1=self_block.oid,
                                      a2=other_block.oid,
                                      a1_shape=self_block.shape,
                                      a2_shape=other_block.shape,
                                      a1_T=self_block.transposed,
                                      a2_T=other_block.transposed,
                                      axes=1,
                                      syskwargs={
                                          "grid_entry": sch_grid_entry,
                                          "grid_shape": sch_grid_shape
                                      })
            oids.append(dot_oid)
        result_grid_entry = tuple(0 for _ in range(len(result.grid.grid_shape)))
        result_oid = self.system.sum_reduce(*oids,
                                            syskwargs={
                                                "grid_entry": result_grid_entry,
                                                "grid_shape": result.grid.grid_shape
                                            })
        result.blocks[result_grid_entry].oid = result_oid
        return result

    def _matvec(self, other):
        # Schedule block matmult on existing block nodes of the matrix.
        # This is cheaper than moving matrix and vec blocks to result node.
        assert self.shape[1] == other.shape[0], str((self.shape[1], other.shape[0]))
        result_shape = tuple(self.shape[:1] + other.shape[1:])
        result_block_shape = tuple(self.block_shape[:1] + other.block_shape[1:])
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=self.dtype.__name__)
        result = BlockArray(result_grid, self.system)
        for i in range(self.grid.grid_shape[0]):
            row = []
            for j in range(self.grid.grid_shape[1]):
                grid_entry = (i, j)
                self_block: Block = self.blocks[grid_entry]
                if len(other.shape) == 2:
                    other_block: Block = other.blocks[(grid_entry[1], 0)]
                    result_grid_entry = (i, 0)
                else:
                    other_block: Block = other.blocks[grid_entry[1]]
                    result_grid_entry = (i,)
                if self_block.transposed:
                    # Reverse grid shape and entry to obtain virtual layout of matrix blocks.
                    sch_grid_shape = tuple(reversed(self.grid.grid_shape))
                    sch_grid_entry = tuple(reversed(grid_entry))
                else:
                    sch_grid_shape = self.grid.grid_shape
                    sch_grid_entry = grid_entry
                dot_oid = self.system.bop("tensordot",
                                          a1=self_block.oid,
                                          a2=other_block.oid,
                                          a1_shape=self_block.shape,
                                          a2_shape=other_block.shape,
                                          a1_T=self_block.transposed,
                                          a2_T=other_block.transposed,
                                          axes=1,
                                          syskwargs={
                                              "grid_entry": sch_grid_entry,
                                              "grid_shape": sch_grid_shape
                                          })
                row.append(dot_oid)
            result_oid = self.system.sum_reduce(*row,
                                                syskwargs={
                                                    "grid_entry": result_grid_entry,
                                                    "grid_shape": result.grid.grid_shape
                                                })
            result.blocks[result_grid_entry].oid = result_oid
        return result

    def __add__(self, other):
        other = self.check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks + other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __sub__(self, other):
        other = self.check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks - other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __mul__(self, other):
        other = self.check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks * other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __truediv__(self, other):
        other = self.check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks / other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __pow__(self, other):
        other = self.check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks ** other.blocks,
                                      result_shape=None,
                                      system=self.system)

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
        assert other.shape == () or other.shape == self.shape, \
            "Currently supports comparison with scalars only."
        shape = array_utils.broadcast(self.shape, other.shape).shape
        block_shape = array_utils.broadcast_block_shape(self.shape, other.shape, self.block_shape)
        dtype = np.bool.__name__
        grid = ArrayGrid(shape, block_shape, dtype)
        result = BlockArray(grid, self.system)
        for grid_entry in result.grid.get_entry_iterator():
            if other.shape == ():
                other_block: Block = other.blocks.item()
            else:
                other_block: Block = other.blocks[grid_entry]
            result.blocks[grid_entry] = self.blocks[grid_entry].bop(op,
                                                                    other_block,
                                                                    args={})

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
        result = BlockArray(grid, self.system)
        for grid_entry in result.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].astype(dtype)
        return result


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
                    new_shape.append(size//other_dim_prod)
                else:
                    new_shape.append(dim)
        else:
            new_shape = input_shape
        assert size == np.product(new_shape)
        return new_shape

    def _group_index_lists_by_block(self, dst_slice_tuples,
                                    src_grid: ArrayGrid, dst_index_list,
                                    src_index_list):
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
        index_types = [(2**8, np.uint8), (2**16, np.uint16),
                       (2**32, np.uint32), (2**64, np.uint64)]
        index_type = None
        for bound, curr_index_type in index_types:
            if np.all(np.array(src_grid.block_shape) < bound) and np.all(dst_slice_np[1] < bound):
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
                if np.all((src_slice_np[0] <= src_index) & (src_index < src_slice_np[1])):
                    index_pair = ((dst_index - dst_slice_np[0]).astype(index_type),
                                  (src_index - src_slice_np[0]).astype(index_type))
                    index_pairs.append(index_pair)
            if len(index_pairs) > 0:
                src_blocks[grid_entry] = index_pairs
        return src_blocks

    def _arbitrary_reshape(self, arr: BlockArray, shape, block_shape) -> BlockArray:
        # This is the worst-case scenario.
        # Generate index mappings per block, and group source indices to minimize
        # RPCs and generation of new objects.
        system = arr.system
        dst_arr = BlockArray.empty(shape=shape, block_shape=block_shape,
                                   dtype=arr.dtype, system=system)
        for dst_grid_entry in dst_arr.grid.get_entry_iterator():
            dst_block: Block = dst_arr.blocks[dst_grid_entry]
            dst_slice_selection = dst_arr.grid.get_slice(dst_grid_entry)
            dst_index_list = array_utils.slice_sel_to_index_list(dst_slice_selection)
            src_index_list = array_utils.translate_index_list(dst_index_list, shape, arr.shape)
            src_blocks = self._group_index_lists_by_block(
                dst_arr.grid.get_slice_tuples(dst_grid_entry),
                arr.grid,
                dst_index_list,
                src_index_list
            )
            for src_grid_entry in src_blocks:
                src_block: Block = arr.blocks[src_grid_entry]
                index_pairs = src_blocks[src_grid_entry]
                syskwargs = {"grid_entry": dst_grid_entry, "grid_shape": dst_arr.grid.grid_shape}
                dst_block.oid = system.update_block_by_index(dst_block.oid,
                                                             src_block.oid,
                                                             index_pairs,
                                                             syskwargs=syskwargs)
        return dst_arr

    def _block_shape_reshape(self, arr, block_shape):
        rarr: BlockArray = BlockArray.empty(arr.shape, block_shape, arr.dtype, arr.system)
        for grid_entry in rarr.grid.get_entry_iterator():
            grid_entry_slice = rarr.grid.get_slice(grid_entry)
            # TODO (hme): This could be less costly.
            rarr[grid_entry_slice] = arr[grid_entry_slice]
        return rarr

    def _strip_ones(self, shape):
        return tuple(filter(lambda x: x != 1, shape))

    def _is_simple_reshape(self, arr: BlockArray, shape, block_shape):
        # Is the reshape a difference of factors of 1?
        # Strip out 1s and compare.
        return (self._strip_ones(shape) == self._strip_ones(arr.shape) and
                self._strip_ones(block_shape) == self._strip_ones(arr.block_shape))

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
        rarr = BlockArray(grid, arr.system)
        for grid_entry in grid.get_entry_iterator():
            src_block: Block = src_blocks[grid_entry]
            dst_block: Block = rarr.blocks[grid_entry]
            syskwargs = {"grid_entry": grid_entry, "grid_shape": grid.grid_shape}
            dst_block.oid = arr.system.reshape(src_block.oid,
                                               dst_block.shape,
                                               syskwargs=syskwargs)
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
