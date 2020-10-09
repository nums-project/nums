# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import itertools

import numpy as np

from nums.core.storage.storage import ArrayGrid
from nums.core.array import utils as array_utils
from nums.core.array.base import BlockArrayBase, Block
from nums.core.array.view import ArrayView


class BlockArray(BlockArrayBase):

    # TODO (hme): Add block_shape constraints.

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

    def reshape(self, shape=None, block_shape=None):
        # TODO (hme): Add support for arbitrary reshape.
        if shape is None:
            shape = self.shape
        if block_shape is None:
            block_shape = self.block_shape
        if shape == self.shape and block_shape == self.block_shape:
            return self

        temp_shape = shape
        temp_block_shape = block_shape
        shape = []
        block_shape = []
        negative_one = False
        for i, dim in enumerate(temp_shape):
            if dim == -1:
                assert len(self.shape) == 1
                if negative_one:
                    raise Exception("Only one -1 permitted in reshape.")
                negative_one = True
                shape.append(self.shape[i])
                assert temp_block_shape[i] == -1
                block_shape.append(self.block_shape[0])
            else:
                shape.append(dim)
                block_shape.append(temp_block_shape[i])
        del temp_shape
        shape = tuple(shape)
        block_shape = tuple(block_shape)

        assert np.product(shape) == np.product(self.shape)
        # Make sure the difference is either a preceding or succeeding one.
        if len(shape) > len(self.shape):
            if shape[0] == 1:
                grid_entry_op = "shift"
                assert shape[1:] == self.shape
            elif shape[-1] == 1:
                grid_entry_op = "pop"
                assert shape[:-1] == self.shape
            else:
                raise Exception()
        elif len(shape) < len(self.shape):
            if self.shape[0] == 1:
                grid_entry_op = "prep"
                assert self.shape[1:] == shape
            elif self.shape[-1] == 1:
                grid_entry_op = "app"
                assert self.shape[:-1] == shape
            else:
                raise Exception()
        else:
            grid_entry_op = "none"
            assert self.shape == shape

        grid = ArrayGrid(shape=shape,
                         block_shape=block_shape,
                         dtype=self.grid.dtype.__name__)
        grid_meta = grid.to_meta()
        rarr = BlockArray(grid, self.system)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = self.system.empty(grid_entry, grid_meta,
                                                            syskwargs={
                                                                "grid_entry": grid_entry,
                                                                "grid_shape": grid.grid_shape
                                                            })
            grid_entry_slice = grid.get_slice(grid_entry)
            if grid_entry_op == "shift":
                grid_entry_slice = tuple([0] + list(grid_entry_slice)[1:])
                self_grid_entry_slice = self.grid.get_slice(grid_entry[1:])
            elif grid_entry_op == "pop":
                grid_entry_slice = tuple(list(grid_entry_slice)[:-1] + [0])
                self_grid_entry_slice = self.grid.get_slice(grid_entry[:-1])
            elif grid_entry_op == "prep":
                self_grid_entry_slice = self.grid.get_slice(tuple([0] + list(grid_entry)))
            elif grid_entry_op == "prep":
                self_grid_entry_slice = self.grid.get_slice(tuple(list(grid_entry) + [0]))
            else:
                assert grid_entry_op == "none"
                self_grid_entry_slice = grid_entry_slice

            # TODO (hme): This is costly.
            rarr[grid_entry_slice] = self[self_grid_entry_slice]
        return rarr

    def __getattr__(self, item):
        if item == "__array_priority__" or item == "__array_struct__":
            # This is triggered by a numpy array on the LHS.
            raise ValueError("Unable to covert numpy array to block array.")
        if item != "T":
            raise NotImplementedError(item)
        metaT = self.grid.to_meta()
        metaT["shape"] = tuple(reversed(metaT["shape"]))
        metaT["block_shape"] = tuple(reversed(metaT["block_shape"]))
        gridT = ArrayGrid.from_meta(metaT)
        rarrT = BlockArray(gridT, self.system)
        rarrT.blocks = np.copy(self.blocks.T)
        for grid_entry in rarrT.grid.get_entry_iterator():
            rarrT.blocks[grid_entry] = rarrT.blocks[grid_entry].transpose()
        return rarrT

    def __getitem__(self, item):
        av: ArrayView = ArrayView.from_block_array(self)
        # TODO (hme): We don't have to create, but do so for now until we need to optimize.
        return av[item].create(BlockArray)

    def __setitem__(self, key, value):
        av: ArrayView = ArrayView.from_block_array(self)
        av[key] = value

    def _check_or_convert_other(self, other):
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
        result_blocks = np.empty_like(self.blocks, dtype=Block)
        for grid_entry in self.grid.get_entry_iterator():
            result_blocks[grid_entry] = self.blocks[grid_entry].reduce_axis(op_name,
                                                                            axis,
                                                                            keepdims=keepdims)
        result_shape = []
        result_block_shape = []
        for curr_axis in range(len(self.shape)):
            axis_size, axis_block_size = self.shape[curr_axis], self.block_shape[curr_axis]
            if curr_axis == axis:
                if keepdims:
                    axis_size, axis_block_size = 1, 1
                else:
                    continue
            result_shape.append(axis_size)
            result_block_shape.append(axis_block_size)
        result_shape = tuple(result_shape)
        result_block_shape = tuple(result_block_shape)
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=self.dtype.__name__)
        result = BlockArray(result_grid, self.system)
        op_func = np.__getattribute__(op_name)
        reduced_blocks = op_func(result_blocks, axis=axis, keepdims=keepdims)
        if result.shape == ():
            result.blocks[()] = reduced_blocks
        else:
            result.blocks = reduced_blocks
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

        other = self._check_or_convert_other(other)
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
        other = self._check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks + other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __sub__(self, other):
        other = self._check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks - other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __mul__(self, other):
        other = self._check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks * other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __truediv__(self, other):
        other = self._check_or_convert_other(other)
        return BlockArray.from_blocks(self.blocks / other.blocks,
                                      result_shape=None,
                                      system=self.system)

    def __pow__(self, other):
        other = self._check_or_convert_other(other)
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
        other = self._check_or_convert_other(other)
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
                                                                    args={},
                                                                    bool_op=True)

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
        other = self._check_or_convert_other(other)
        return other - self

    __rmul__ = __mul__

    def __rmatmul__(self, other):
        other = self._check_or_convert_other(other)
        return other @ self

    def __rtruediv__(self, other):
        other = self._check_or_convert_other(other)
        return other / self

    def __rpow__(self, other):
        other = self._check_or_convert_other(other)
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
