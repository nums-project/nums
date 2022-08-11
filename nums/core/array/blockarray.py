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
from nums.core.array.base import Block, BlockArrayBase
from nums.core.array.view import ArrayView
from nums.core.grid.grid import ArrayGrid
from nums.core.kernel.kernel_manager import KernelManager


# pylint: disable=too-many-lines


class BlockArray(BlockArrayBase):
    def __init__(self, grid: ArrayGrid, km: KernelManager, blocks: np.ndarray = None):
        if blocks is not None:
            assert blocks.dtype == Block, "BlockArray must be initialized with Blocks"
        super().__init__(grid, km, blocks)
        try:
            self.nbytes = self.grid.nbytes()
        except ValueError as _:
            self.nbytes = None
        if self.blocks is None:
            # TODO (hme): Subclass np.ndarray for self.blocks instances,
            #  and override key methods to better integrate with NumPy's ufuncs.
            self.blocks = np.empty(shape=self.grid.grid_shape, dtype=Block)
            for grid_entry in self.grid.get_entry_iterator():
                self.blocks[grid_entry] = Block(
                    grid_entry=grid_entry,
                    grid_shape=self.grid.grid_shape,
                    shape=self.grid.get_block_shape(grid_entry),
                    dtype=self.dtype,
                    transposed=False,
                    km=self.km,
                )

    @classmethod
    def empty(cls, shape, block_shape, dtype, km: KernelManager):
        return BlockArray.create("empty", shape, block_shape, dtype, km)

    @classmethod
    def create(cls, create_op_name, shape, block_shape, dtype, km: KernelManager):
        grid = ArrayGrid(shape=shape, block_shape=block_shape, dtype=dtype.__name__)
        grid_meta = grid.to_meta()
        arr = BlockArray(grid, km)
        for grid_entry in grid.get_entry_iterator():
            arr.blocks[grid_entry].oid = km.new_block(
                create_op_name,
                grid_entry,
                grid_meta,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
        return arr

    @classmethod
    def from_scalar(cls, val, km):
        if not array_utils.is_scalar(val):
            raise ValueError("%s is not a scalar." % val)
        return BlockArray.from_np(np.array(val), block_shape=(), copy=False, km=km)

    @classmethod
    def from_oid(cls, oid, shape, dtype, km):
        block_shape = shape
        dtype = array_utils.to_dtype_cls(dtype)
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        ba = BlockArray(grid, km)
        for i, grid_entry in enumerate(grid.get_entry_iterator()):
            assert i == 0
            ba.blocks[grid_entry].oid = oid
        return ba

    @classmethod
    def from_np(cls, arr, block_shape, copy, km):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = BlockArray(grid, km)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = arr[grid_slice]
            if copy:
                block = np.copy(block)
            rarr.blocks[grid_entry].oid = km.put(
                block,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
            rarr.blocks[grid_entry].dtype = getattr(np, dtype_str)
        return rarr

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, km):
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
        result = BlockArray(result_grid, km)
        for grid_entry in result_grid.get_entry_iterator():
            if isinstance(arr, Block):
                block: Block = arr
            else:
                block: Block = arr[grid_entry]
            result.blocks[grid_entry] = block
        return result

    def copy(self):
        grid_copy = self.grid.from_meta(self.grid.to_meta())
        rarr_copy = BlockArray(grid_copy, self.km)
        for grid_entry in grid_copy.get_entry_iterator():
            rarr_copy.blocks[grid_entry] = self.blocks[grid_entry].copy()
        return rarr_copy

    def astype(self, dtype):
        grid = ArrayGrid(self.shape, self.block_shape, dtype.__name__)
        result = BlockArray(grid, self.km)
        for grid_entry in result.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].astype(dtype)
        return result

    def is_single_block(self):
        return self.blocks.size == 1

    def to_single_block(self, replicate=False):
        res: BlockArray = self.reshape(*self.shape, block_shape=self.shape)
        if replicate:
            block: Block = res.blocks.item()
            num_devices: int = len(self.km.devices())
            for i in range(num_devices):
                self.km.touch(
                    block.oid,
                    syskwargs={
                        "grid_entry": (i,),
                        "grid_shape": (num_devices,),
                    },
                )
        return res

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

        rarr_swap = BlockArray(grid_swap, self.km, rarr_src)
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
        rarrT = BlockArray(gridT, self.km)
        rarrT.blocks = np.copy(self.blocks.T)
        for grid_entry in rarrT.grid.get_entry_iterator():
            rarrT.blocks[grid_entry] = rarrT.blocks[grid_entry].transpose(
                defer, redistribute
            )
        return rarrT

    @staticmethod
    def to_block_array(obj, km: KernelManager, block_shape=None):
        if isinstance(obj, BlockArray):
            return obj
        if isinstance(obj, np.ndarray):
            np_array = obj
        elif isinstance(obj, list):
            np_array = np.array(obj)
        elif array_utils.is_scalar(obj):
            return BlockArray.from_scalar(obj, km)
        else:
            raise Exception("Unsupported type %s" % type(obj))
        if block_shape is None:
            block_shape = km.get_block_shape(np_array.shape, np_array.dtype)
        return BlockArray.from_np(np_array, block_shape, False, km)

    def check_or_convert_other(self, other, compute_block_shape=False):
        block_shape = None if compute_block_shape else self.block_shape
        return BlockArray.to_block_array(other, self.km, block_shape=block_shape)

    def _check_bop_implemented(self, other):
        if isinstance(other, (BlockArray, np.ndarray, list)) or array_utils.is_scalar(
            other
        ):
            return True
        return False

    def ufunc(self, op_name):
        result = self.copy()
        for grid_entry in self.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].ufunc(op_name)
        return result

    def reduce_axis(self, op_name, axis, keepdims=False):
        if not (axis is None or isinstance(axis, (int, np.int32, np.int64))):
            raise NotImplementedError("Only integer axis is currently supported.")
        if 0 in self.shape:
            return BlockArray.create("zeros", (), (), float, self.km)
        block_reduced_oids = np.empty_like(self.blocks, dtype=tuple)
        for grid_entry in self.grid.get_entry_iterator():
            block = self.blocks[grid_entry]
            block_oid = self.km.reduce_axis(
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
        result = BlockArray(result_grid, self.km)

        if axis is None:
            if result.shape == ():
                result_block: Block = result.blocks[()]
            else:
                result_block: Block = result.blocks[:].item()
            result_block.oid = self.tree_reduce(
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
                result_block.oid = self.tree_reduce(
                    op_name,
                    block_reduced_oids_axis,
                    result_block.grid_entry,
                    result_block.grid_shape,
                )
        return result

    #################
    # Arithmetic
    #################

    @staticmethod
    def elementwise(op_name, a, b):
        if isinstance(a, BlockArray):
            b = a.check_or_convert_other(b)
        elif isinstance(b, BlockArray):
            a = b.check_or_convert_other(a)
        else:
            raise NotImplementedError()

        if a.shape == b.shape and a.block_shape == b.block_shape:
            return BlockArray._fast_elementwise(op_name, a, b)
        blocks_op = a.blocks.__getattribute__("__%s__" % op_name)
        return BlockArray.from_blocks(blocks_op(b.blocks), result_shape=None, km=a.km)

    @staticmethod
    def _fast_elementwise(op_name, a, b):
        """
        Implements fast scheduling for basic element-wise operations.
        """
        dtype = array_utils.get_bop_output_type(op_name, a.dtype, b.dtype)
        # Schedule the op first.
        blocks = np.empty(shape=a.grid.grid_shape, dtype=Block)
        for grid_entry in a.grid.get_entry_iterator():
            a_block: Block = a.blocks[grid_entry]
            b_block: Block = b.blocks[grid_entry]
            blocks[grid_entry] = block = Block(
                grid_entry=grid_entry,
                grid_shape=a_block.grid_shape,
                shape=a_block.shape,
                dtype=dtype,
                transposed=False,
                km=a.km,
            )
            block.oid = a.km.bop(
                op_name,
                a_block.oid,
                b_block.oid,
                a_block.transposed,
                b_block.transposed,
                axes={},
                syskwargs={
                    "grid_entry": grid_entry,
                    "grid_shape": a.grid.grid_shape,
                },
            )
        return BlockArray(
            ArrayGrid(a.shape, a.block_shape, dtype.__name__),
            a.km,
            blocks=blocks,
        )

    #################
    # Linear Algebra
    #################

    @staticmethod
    def tensordot(a, b, axes=2):
        if isinstance(axes, int):
            pass
        elif array_utils.is_array_like(axes):
            raise NotImplementedError("Non-integer axes is currently not supported.")
        else:
            raise TypeError(f"Unexpected axes type '{type(axes).__name__}'")

        if isinstance(a, BlockArray):
            b = a.check_or_convert_other(b, compute_block_shape=True)
        elif isinstance(b, BlockArray):
            a = b.check_or_convert_other(a, compute_block_shape=True)
        else:
            raise NotImplementedError()

        if array_utils.np_tensordot_param_test(a.shape, a.ndim, b.shape, b.ndim, axes):
            raise ValueError("shape-mismatch for sum")

        if axes > 0:
            a_axes = a.grid.grid_shape[:-axes]
            a_sum_axes = a.grid.grid_shape[-axes:]
            b_axes = b.grid.grid_shape[axes:]
            b_sum_axes = b.grid.grid_shape[:axes]
            assert a_sum_axes == b_sum_axes
            result_shape = tuple(a.shape[:-axes] + b.shape[axes:])
            result_block_shape = tuple(a.block_shape[:-axes] + b.block_shape[axes:])
        else:
            a_axes = a.grid.grid_shape
            b_axes = b.grid.grid_shape
            a_sum_axes = ()
            result_shape = tuple(a.shape + b.shape)
            result_block_shape = tuple(a.block_shape + b.block_shape)

        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=array_utils.get_bop_output_type(
                "tensordot", a.dtype, b.dtype
            ).__name__,
        )
        assert result_grid.grid_shape == tuple(a_axes + b_axes)
        result = BlockArray(result_grid, a.km)
        a_dims = list(itertools.product(*map(range, a_axes)))
        b_dims = list(itertools.product(*map(range, b_axes)))
        sum_dims = list(itertools.product(*map(range, a_sum_axes)))
        for i in a_dims:
            for j in b_dims:
                grid_entry = tuple(i + j)
                result_block: Block = result.blocks[grid_entry]
                sum_oids = []
                for k in sum_dims:
                    a_block: Block = a.blocks[tuple(i + k)]
                    b_block: Block = b.blocks[tuple(k + j)]
                    dot_grid_args = a._compute_tensordot_syskwargs(a_block, b_block)
                    dotted_oid = a.km.bop(
                        "tensordot",
                        a_block.oid,
                        b_block.oid,
                        a_block.transposed,
                        b_block.transposed,
                        axes=axes,
                        syskwargs={
                            "grid_entry": dot_grid_args[0],
                            "grid_shape": dot_grid_args[1],
                        },
                    )
                    sum_oids.append(
                        (dotted_oid, dot_grid_args[0], dot_grid_args[1], False)
                    )
                result_block.oid = a.tree_reduce(
                    "sum", sum_oids, result_block.grid_entry, result_block.grid_shape
                )
        return result

    #################
    # Inequalities
    #################

    def __inequality__(self, op, other):
        other = self.check_or_convert_other(other)
        if other is NotImplemented:
            return NotImplemented
        assert (
            other.shape == () or other.shape == self.shape
        ), "Currently supports comparison with scalars only."
        shape = array_utils.broadcast(self.shape, other.shape).shape
        block_shape = array_utils.broadcast_block_shape(
            self.shape, other.shape, self.block_shape
        )
        dtype = bool.__name__
        grid = ArrayGrid(shape, block_shape, dtype)
        result = BlockArray(grid, self.km)
        for grid_entry in result.grid.get_entry_iterator():
            if other.shape == ():
                other_block: Block = other.blocks.item()
            else:
                other_block: Block = other.blocks[grid_entry]
            result.blocks[grid_entry] = self.blocks[grid_entry].bop(
                op, other_block, args={}
            )
        return result
