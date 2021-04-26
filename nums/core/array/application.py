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


from typing import List, Union

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.random import NumsRandomState
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid
from nums.core.grid.grid import DeviceID
from nums.core.storage.storage import StoredArray, StoredArrayS3
from nums.core.systems.filesystem import FileSystem


# pylint: disable = too-many-lines


class ArrayApplication(object):

    def __init__(self, cm: ComputeManager, fs: FileSystem):
        self.cm: ComputeManager = cm
        self._fs: FileSystem = fs
        self._array_grids: (str, ArrayGrid) = {}
        self.random = self.random_state()

        self.one_half = self.scalar(.5)
        self.two = self.scalar(2.0)
        self.one = self.scalar(1.0)
        self.zero = self.scalar(0.0)

    def compute_block_shape(self,
                            shape: tuple,
                            dtype: Union[type, np.dtype],
                            cluster_shape=None,
                            num_cores=None):
        return self.cm.compute_block_shape(shape, dtype, cluster_shape, num_cores)

    def get_block_shape(self, shape, dtype):
        return self.cm.get_block_shape(shape, dtype)

    def _get_array_grid(self, filename: str, stored_array_cls) -> ArrayGrid:
        if filename not in self._array_grids:
            store_inst: StoredArray = stored_array_cls(filename)
            self._array_grids[filename] = store_inst.get_grid()
        return self._array_grids[filename]

    ######################################
    # Filesystem API
    ######################################

    def write_fs(self, ba: BlockArray, filename: str):
        res = self._write(ba, filename, self._fs.write_block_fs)
        self._fs.write_meta_fs(ba, filename)
        return res

    def read_fs(self, filename: str):
        meta = self._fs.read_meta_fs(filename)
        addresses = meta["addresses"]
        grid_meta = meta["grid_meta"]
        grid = ArrayGrid.from_meta(grid_meta)
        ba: BlockArray = BlockArray(grid, self.cm)
        for grid_entry in addresses:
            device_id: DeviceID = DeviceID.from_str(addresses[grid_entry])
            ba.blocks[grid_entry].oid = self._fs.read_block_fs(filename,
                                                               grid_entry,
                                                               grid_meta,
                                                               syskwargs={
                                                                   "device_id": device_id
                                                               })
        return ba

    def delete_fs(self, filename: str):
        meta = self._fs.read_meta_fs(filename)
        addresses = meta["addresses"]
        grid_meta = meta["grid_meta"]
        grid = ArrayGrid.from_meta(grid_meta)
        result_grid = ArrayGrid(grid.grid_shape,
                                tuple(np.ones_like(grid.shape, dtype=np.int)),
                                dtype=dict.__name__)
        rarr = BlockArray(result_grid, self.cm)
        for grid_entry in addresses:
            device_id: DeviceID = DeviceID.from_str(addresses[grid_entry])
            rarr.blocks[grid_entry].oid = self._fs.delete_block_fs(filename,
                                                                   grid_entry,
                                                                   grid_meta,
                                                                   syskwargs={
                                                                       "device_id": device_id
                                                                   })
        self._fs.delete_meta_fs(filename)
        return rarr

    def write_s3(self, ba: BlockArray, filename: str):
        grid_entry = tuple(np.zeros_like(ba.shape, dtype=np.int))
        result = self._fs.write_meta_s3(filename,
                                        grid_meta=ba.grid.to_meta(),
                                        syskwargs={
                                            "grid_entry": grid_entry,
                                            "grid_shape": ba.grid.grid_shape
                                        })
        assert "ETag" in self.cm.get(result).item(), "Metadata write failed."
        return self._write(ba, filename, self._fs.write_block_s3)

    def _write(self, ba: BlockArray, filename, remote_func):
        grid = ba.grid
        result_grid = ArrayGrid(grid.grid_shape,
                                tuple(np.ones_like(grid.shape, dtype=np.int)),
                                dtype=dict.__name__)
        rarr = BlockArray(result_grid, self.cm)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = remote_func(ba.blocks[grid_entry].oid,
                                                      filename,
                                                      grid_entry,
                                                      grid.to_meta(),
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": grid.grid_shape
                                                      })
        return rarr

    def read_s3(self, filename: str):
        store_cls, remote_func = StoredArrayS3, self._fs.read_block_s3
        grid = self._get_array_grid(filename, store_cls)
        grid_meta = grid.to_meta()
        grid_entry_iterator = grid.get_entry_iterator()
        rarr = BlockArray(grid, self.cm)
        for grid_entry in grid_entry_iterator:
            rarr.blocks[grid_entry].oid = remote_func(filename, grid_entry, grid_meta,
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": grid.grid_shape
                                                      })
        return rarr

    def delete_s3(self, filename: str):
        grid = self._get_array_grid(filename, StoredArrayS3)
        grid_entry = tuple(np.zeros_like(grid.shape, dtype=np.int))
        result = self._fs.delete_meta_s3(filename,
                                         syskwargs={
                                             "grid_entry": grid_entry,
                                             "grid_shape": grid.grid_shape
                                         })
        deleted_key = self.cm.get(result).item()["Deleted"][0]["Key"]
        assert deleted_key == StoredArrayS3(filename, grid).get_meta_key()
        results: BlockArray = self._delete(filename,
                                           StoredArrayS3,
                                           self._fs.delete_block_s3)
        return results

    def _delete(self, filename, store_cls, remote_func):
        grid = self._get_array_grid(filename, store_cls)
        result_grid = ArrayGrid(grid.grid_shape,
                                tuple(np.ones_like(grid.shape, dtype=np.int)),
                                dtype=dict.__name__)
        rarr = BlockArray(result_grid, self.cm)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = remote_func(filename, grid_entry, grid.to_meta(),
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": grid.grid_shape
                                                      })
        return rarr

    def read_csv(self, filename, dtype=float, delimiter=',', has_header=False, num_workers=None):
        if num_workers is None:
            num_workers = self.cm.num_cores_total()
        arrays: list = self._fs.read_csv(filename, dtype, delimiter, has_header,
                                         num_workers)
        shape = np.zeros(len(arrays[0].shape), dtype=int)
        for array in arrays:
            shape += np.array(array.shape, dtype=int)
        shape = tuple(shape)
        block_shape = self.cm.get_block_shape(shape, dtype)
        result = self.concatenate(arrays, axis=0, axis_block_size=block_shape[0])
        # Release references immediately, in case we need to do another reshape.
        del arrays
        if result.block_shape[1] != block_shape[1]:
            result = result.reshape(block_shape=block_shape)
        return result

    def loadtxt(self, fname, dtype=float, comments='# ', delimiter=' ',
                converters=None, skiprows=0, usecols=None, unpack=False,
                ndmin=0, encoding='bytes', max_rows=None, num_workers=None) -> BlockArray:
        if num_workers is None:
            num_workers = self.cm.num_cores_total()
        return self._fs.loadtxt(
            fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows,
            usecols=usecols, unpack=unpack, ndmin=ndmin,
            encoding=encoding, max_rows=max_rows, num_workers=num_workers)

    ######################################
    # Array Operations API
    ######################################

    def scalar(self, value):
        return BlockArray.from_scalar(value, self.cm)

    def array(self, array: np.ndarray, block_shape: tuple = None):
        assert len(array.shape) == len(block_shape)
        return BlockArray.from_np(array,
                                  block_shape=block_shape,
                                  copy=False,
                                  cm=self.cm)

    def zeros(self, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        return self._new_array("zeros", shape, block_shape, dtype)

    def ones(self, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        return self._new_array("ones", shape, block_shape, dtype)

    def empty(self, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        return self._new_array("empty", shape, block_shape, dtype)

    def _new_array(self, op_name: str, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        assert len(shape) == len(block_shape)
        if dtype is None:
            dtype = np.float64
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        grid_meta = grid.to_meta()
        rarr = BlockArray(grid, self.cm)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = self.cm.new_block(op_name,
                                                            grid_entry,
                                                            grid_meta,
                                                            syskwargs={
                                                                "grid_entry": grid_entry,
                                                                "grid_shape": grid.grid_shape
                                                            })
        return rarr

    def concatenate(self, arrays: List, axis: int, axis_block_size: int = None):
        num_arrs = len(arrays)
        assert num_arrs > 1
        first_arr: BlockArray = arrays[0]
        num_axes = len(first_arr.shape)
        # Check assumptions and define result shapes and block shapes.
        for i in range(num_arrs):
            curr_ba: BlockArray = arrays[i]
            assert num_axes == len(curr_ba.shape), "Unequal num axes."
            assert curr_ba.dtype == first_arr.dtype, "Incompatible dtypes " \
                                                     "%s, %s" % (curr_ba.dtype, first_arr.dtype)
            for curr_axis in range(num_axes):
                first_block_size = first_arr.block_shape[curr_axis]
                block_size = curr_ba.block_shape[curr_axis]
                if first_block_size == block_size:
                    continue
                elif axis == curr_axis:
                    assert axis_block_size is not None, "block axis size is required " \
                                                        "when block shapes are neq."
                else:
                    raise ValueError("Other axis shapes and block shapes must be equal.")

        # Compute result shapes.
        result_shape = []
        result_block_shape = []
        for curr_axis in range(num_axes):
            if curr_axis == axis:
                if axis_block_size is None:
                    # They are all equal.
                    axis_block_size = first_arr.block_shape[curr_axis]
                result_block_size = axis_block_size
                result_size = 0
                for i in range(num_arrs):
                    curr_ba: BlockArray = arrays[i]
                    size = curr_ba.shape[curr_axis]
                    result_size += size
            else:
                result_size = first_arr.shape[curr_axis]
                result_block_size = first_arr.block_shape[curr_axis]
            result_shape.append(result_size)
            result_block_shape.append(result_block_size)
        result_shape, result_block_shape = tuple(result_shape), tuple(result_block_shape)
        result_ba = self.empty(result_shape, result_block_shape, first_arr.dtype)

        # Write result blocks.
        # TODO (hme): This can be optimized by updating blocks directly.
        pos = 0
        for arr in arrays:
            delta = arr.shape[axis]
            axis_slice = slice(pos, pos + delta)
            result_selector = tuple([slice(None, None) for _ in range(axis)] + [axis_slice, ...])
            result_ba[result_selector] = arr
            pos += delta
        return result_ba

    def eye(self, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        assert len(shape) == len(block_shape) == 2
        if dtype is None:
            dtype = np.float64
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        grid_meta = grid.to_meta()
        rarr = BlockArray(grid, self.cm)
        for grid_entry in grid.get_entry_iterator():
            syskwargs = {"grid_entry": grid_entry, "grid_shape": grid.grid_shape}
            if np.all(np.diff(grid_entry) == 0):
                # This is a diagonal block.
                rarr.blocks[grid_entry].oid = self.cm.new_block("eye",
                                                                grid_entry,
                                                                grid_meta,
                                                                syskwargs=syskwargs)
            else:
                rarr.blocks[grid_entry].oid = self.cm.new_block("zeros",
                                                                grid_entry,
                                                                grid_meta,
                                                                syskwargs=syskwargs)
        return rarr

    def diag(self, X: BlockArray) -> BlockArray:
        if len(X.shape) == 1:
            shape = X.shape[0], X.shape[0]
            block_shape = X.block_shape[0], X.block_shape[0]
            grid = ArrayGrid(shape, block_shape, X.dtype.__name__)
            grid_meta = grid.to_meta()
            rarr = BlockArray(grid, self.cm)
            for grid_entry in grid.get_entry_iterator():
                syskwargs = {"grid_entry": grid_entry, "grid_shape": grid.grid_shape}
                if np.all(np.diff(grid_entry) == 0):
                    # This is a diagonal block.
                    rarr.blocks[grid_entry].oid = self.cm.diag(X.blocks[grid_entry[0]].oid,
                                                               syskwargs=syskwargs)
                else:
                    rarr.blocks[grid_entry].oid = self.cm.new_block("zeros",
                                                                    grid_entry,
                                                                    grid_meta,
                                                                    syskwargs=syskwargs)
        elif len(X.shape) == 2:
            assert X.shape[0] == X.shape[1], "X must be a square array."
            assert X.block_shape[0] == X.block_shape[1], "block_shape must be square."
            shape = X.shape[0],
            block_shape = X.block_shape[0],
            grid = ArrayGrid(shape, block_shape, X.dtype.__name__)
            rarr = BlockArray(grid, self.cm)
            for grid_entry in X.grid.get_entry_iterator():
                out_grid_entry = grid_entry[:1]
                out_grid_shape = grid.grid_shape[:1]
                syskwargs = {"grid_entry": out_grid_entry, "grid_shape": out_grid_shape}
                if np.all(np.diff(grid_entry) == 0):
                    # This is a diagonal block.
                    rarr.blocks[out_grid_entry].oid = self.cm.diag(X.blocks[grid_entry].oid,
                                                                   syskwargs=syskwargs)
        else:
            raise ValueError("X must have 1 or 2 axes.")
        return rarr

    def arange(self, start_in, shape, block_shape, step=1, dtype=None) -> BlockArray:
        assert step == 1
        if dtype is None:
            dtype = np.__getattribute__(str(np.result_type(start_in, shape[0] + start_in)))

        # Generate ranges per block.
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        rarr = BlockArray(grid, self.cm)
        for _, grid_entry in enumerate(grid.get_entry_iterator()):
            syskwargs = {"grid_entry": grid_entry, "grid_shape": grid.grid_shape}
            start = start_in + block_shape[0] * grid_entry[0]
            entry_shape = grid.get_block_shape(grid_entry)
            stop = start + entry_shape[0]
            rarr.blocks[grid_entry].oid = self.cm.arange(start,
                                                         stop,
                                                         step,
                                                         dtype,
                                                         syskwargs=syskwargs)
        return rarr

    def linspace(self, start, stop, shape, block_shape, endpoint, retstep, dtype, axis):
        assert axis == 0
        assert endpoint is True
        assert retstep is False

        step_size = (stop - start) / (shape[0] - 1)
        result = self.arange(0, shape, block_shape)
        result = start + result * step_size

        if dtype is not None and dtype != result.dtype:
            result = result.astype(dtype)
        return result

    def log(self, X: BlockArray):
        return X.ufunc("log")

    def exp(self, X: BlockArray):
        return X.ufunc("exp")

    def abs(self, X: BlockArray):
        return X.ufunc("abs")

    def min(self, X: BlockArray, axis=None, keepdims=False):
        return self.reduce("min", X, axis, keepdims)

    def max(self, X: BlockArray, axis=None, keepdims=False):
        return self.reduce("max", X, axis, keepdims)

    def argmin(self, X: BlockArray, axis=None):
        pass

    def sum(self, X: BlockArray, axis=None, keepdims=False, dtype=None):
        return self.reduce("sum", X, axis, keepdims, dtype)

    def reduce(self, op_name: str, X: BlockArray, axis=None, keepdims=False, dtype=None):
        res = X.reduce_axis(op_name, axis, keepdims=keepdims)
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def mean(self, X: BlockArray, axis=None, keepdims=False, dtype=None):
        if X.dtype not in (float, np.float32, np.float64):
            X = X.astype(np.float64)
        num_summed = np.product(X.shape) if axis is None else X.shape[axis]
        res = self.sum(X, axis=axis, keepdims=keepdims) / num_summed
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def var(self, X: BlockArray, axis=None, ddof=0, keepdims=False, dtype=None):
        mean = self.mean(X, axis=axis, keepdims=True)
        ss = self.sum((X - mean) ** self.two, axis=axis, keepdims=keepdims)
        num_summed = (np.product(X.shape) if axis is None else X.shape[axis]) - ddof
        res = ss / num_summed
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def std(self, X: BlockArray, axis=None, ddof=0, keepdims=False, dtype=None):
        res = self.sqrt(self.var(X, axis, ddof, keepdims))
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def argop(self, op_name: str, arr: BlockArray, axis=None):
        if len(arr.shape) > 1:
            raise NotImplementedError("%s currently supports one-dimensional arrays." % op_name)
        if axis is None:
            axis = 0
        assert axis == 0
        grid = ArrayGrid(shape=(), block_shape=(), dtype=np.int64.__name__)
        result = BlockArray(grid, self.cm)
        reduction_result = None, None
        for grid_entry in arr.grid.get_entry_iterator():
            block_slice: slice = arr.grid.get_slice(grid_entry)[0]
            block: Block = arr.blocks[grid_entry]
            syskwargs = {
                "grid_entry": grid_entry,
                "grid_shape": arr.grid.grid_shape,
                "options": {"num_returns": 2},
            }
            reduction_result = self.cm.arg_op(op_name,
                                              block.oid,
                                              block_slice,
                                              *reduction_result,
                                              syskwargs=syskwargs)
        argoptima, _ = reduction_result
        result.blocks[()].oid = argoptima
        return result

    def sqrt(self, X):
        if X.dtype not in (float, np.float32, np.float64):
            X = X.astype(np.float64)
        return X.ufunc("sqrt")

    def norm(self, X):
        return self.sqrt(X.T @ X)

    def xlogy(self, x: BlockArray, y: BlockArray) -> BlockArray:
        if x.dtype not in (float, np.float32, np.float64):
            x = x.astype(np.float64)
        if x.dtype not in (float, np.float32, np.float64):
            y = y.astype(np.float64)
        return self.map_bop("xlogy", x, y)

    def where(self,
              condition: BlockArray,
              x: BlockArray = None,
              y: BlockArray = None):
        result_oids = []
        shape_oids = []
        num_axes = max(1, len(condition.shape))
        # Stronger constraint than necessary, but no reason for anything stronger.
        if x is not None or y is not None:
            assert x is not None and y is not None
            assert condition.shape == x.shape == y.shape
            assert condition.block_shape == x.block_shape == y.block_shape
            assert x.dtype == y.dtype
            result = BlockArray(x.grid.copy(), self.cm)
            for grid_entry in condition.grid.get_entry_iterator():
                cond_oid = condition.blocks[grid_entry].oid
                x_oid = x.blocks[grid_entry].oid
                y_oid = y.blocks[grid_entry].oid
                r_oid = self.cm.where(cond_oid, x_oid, y_oid, None,
                                      syskwargs={
                                          "grid_entry": grid_entry,
                                          "grid_shape": condition.grid.grid_shape,
                                          "options": {"num_returns": 1}
                                      })
                result.blocks[grid_entry].oid = r_oid
            return result
        else:
            for grid_entry in condition.grid.get_entry_iterator():
                block: Block = condition.blocks[grid_entry]
                block_slice_tuples = condition.grid.get_slice_tuples(grid_entry)
                roids = self.cm.where(block.oid, None, None,
                                      block_slice_tuples,
                                      syskwargs={
                                          "grid_entry": grid_entry,
                                          "grid_shape": condition.grid.grid_shape,
                                          "options": {"num_returns": num_axes + 1}
                                      })
                block_oids, shape_oid = roids[:-1], roids[-1]
                shape_oids.append(shape_oid)
                result_oids.append(block_oids)
            shapes = self.cm.get(shape_oids)
            result_shape = (np.sum(shapes),)
            if result_shape == (0,):
                return (self.array(np.array([], dtype=np.int64), block_shape=(0,)),)
            # Remove empty shapes.
            result_shape_pair = []
            for i, shape in enumerate(shapes):
                if np.sum(shape) > 0:
                    result_shape_pair.append((result_oids[i], shape))
            result_block_shape = self.cm.compute_block_shape(result_shape, np.int64)
            result_arrays = []
            for axis in range(num_axes):
                block_arrays = []
                for i in range(len(result_oids)):
                    if shapes[i] == (0,):
                        continue
                    block_arrays.append(BlockArray.from_oid(result_oids[i][axis],
                                                            shapes[i],
                                                            np.int64,
                                                            self.cm))
                if len(block_arrays) == 1:
                    axis_result = block_arrays[0]
                else:
                    axis_result = self.concatenate(block_arrays, 0, result_block_shape[0])
                result_arrays.append(axis_result)
            return tuple(result_arrays)

    def map_uop(self,
                op_name: str,
                arr: BlockArray,
                out: BlockArray = None,
                where=True,
                args=None,
                kwargs=None) -> BlockArray:
        """A map, for unary operators, that applies to every entry of an array.

        Args:
            op_name: An element-wise unary operator.
            arr: A BlockArray.
            out: A BlockArray to which the result is written.
            where: An indicator specifying the indices to which op is applied.
            args: Args provided to op.
            kwargs: Keyword args provided to op.

        Returns:
            A BlockArray.
        """
        if where is not True:
            raise NotImplementedError("'where' argument is not yet supported.")
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        shape = arr.shape
        block_shape = arr.block_shape
        dtype = array_utils.get_uop_output_type(op_name, arr.dtype)
        assert len(shape) == len(block_shape)
        if out is None:
            grid = ArrayGrid(shape, block_shape, dtype.__name__)
            rarr = BlockArray(grid, self.cm)
        else:
            rarr = out
            grid = rarr.grid
            assert rarr.shape == arr.shape and rarr.block_shape == arr.block_shape
        for grid_entry in grid.get_entry_iterator():
            # TODO(hme): Faster to create ndarray first,
            #  and instantiate block array on return
            #  to avoid instantiating blocks on BlockArray initialization.
            rarr.blocks[grid_entry] = arr.blocks[grid_entry].uop_map(op_name,
                                                                     args=args,
                                                                     kwargs=kwargs)
        return rarr

    def matmul(self,
               arr_1: BlockArray,
               arr_2: BlockArray) -> BlockArray:
        return arr_1 @ arr_2

    def tensordot(self,
                  arr_1: BlockArray,
                  arr_2: BlockArray,
                  axes: int = 2) -> BlockArray:
        return arr_1.tensordot(arr_2, axes)

    def map_bop(self,
                op_name: str,
                arr_1: BlockArray,
                arr_2: BlockArray,
                out: BlockArray = None,
                where=True,
                args=None,
                kwargs=None) -> BlockArray:
        # TODO (hme): Move this into BlockArray, and invoke on operator implementations.
        """A map for binary operators that applies element-wise to every entry of the input arrays.

        Args:
            op_name: An element-wise binary operator.
            arr_1: A BlockArray.
            arr_2: A BlockArray.
            out: A BlockArray to which the result is written.
            where: An indicator specifying the indices to which op is applied.
            args: Args provided to op.
            kwargs: Keyword args provided to op.

        Returns:
            A BlockArray.
        """
        if where is not True:
            raise NotImplementedError("'where' argument is not yet supported.")
        if args is not None:
            raise NotImplementedError("'args' is not yet supported.")
        if not (kwargs is None or len(kwargs) == 0):
            raise NotImplementedError("'kwargs' is not yet supported.")

        try:
            ufunc = np.__getattribute__(op_name)
            if (op_name.endswith("max") or op_name == "maximum"
                    or op_name.endswith("min") or op_name == "minimum"
                    or op_name.startswith("logical")):
                rarr = self._broadcast_bop(op_name, arr_1, arr_2)
            else:
                result_blocks: np.ndarray = ufunc(arr_1.blocks, arr_2.blocks)
                rarr = BlockArray.from_blocks(result_blocks,
                                              result_shape=None,
                                              cm=self.cm)
        except Exception as _:
            rarr = self._broadcast_bop(op_name, arr_1, arr_2)
        if out is not None:
            assert out.grid.grid_shape == rarr.grid.grid_shape
            assert out.shape == rarr.shape
            assert out.block_shape == rarr.block_shape
            out.blocks[:] = rarr.blocks[:]
            rarr = out
        return rarr

    def _broadcast_bop(self, op_name, arr_1, arr_2) -> BlockArray:
        """We want to avoid invoking this op whenever possible; NumPy's imp is faster.

        Args:
            op_name: Name of binary operation.
            arr_1: A BlockArray.
            arr_2: A BlockArray.

        Returns:
            A BlockArray.
        """
        if arr_1.shape != arr_2.shape:
            output_grid_shape = array_utils.broadcast_shape(arr_1.grid.grid_shape,
                                                            arr_2.grid.grid_shape)
            arr_1 = arr_1.broadcast_to(output_grid_shape)
            arr_2 = arr_2.broadcast_to(output_grid_shape)
        dtype = array_utils.get_bop_output_type(op_name,
                                                arr_1.dtype,
                                                arr_2.dtype)
        grid = ArrayGrid(arr_1.shape, arr_1.block_shape, dtype.__name__)
        rarr = BlockArray(grid, self.cm)
        for grid_entry in rarr.grid.get_entry_iterator():
            block_1: Block = arr_1.blocks[grid_entry]
            block_2: Block = arr_2.blocks[grid_entry]
            rarr.blocks[grid_entry] = block_1.bop(op_name, block_2, {})
        return rarr

    def get(self, *arrs):
        if len(arrs) == 1:
            if isinstance(arrs[0], BlockArray):
                return arrs[0].get()
            else:
                return arrs[0]
        else:
            r = []
            for item in arrs:
                if isinstance(item, BlockArray):
                    r.append(item.get())
                else:
                    r.append(item)
            return r

    def array_compare(self, func_name, a: BlockArray, b: BlockArray, *args):
        assert a.shape == b.shape and a.block_shape == b.block_shape
        bool_list = []
        grid_shape = a.grid.grid_shape
        for grid_entry in a.grid.get_entry_iterator():
            a_block, b_block = a.blocks[grid_entry].oid, b.blocks[grid_entry].oid
            bool_list.append(self.cm.array_compare(func_name, a_block, b_block, args,
                                                   syskwargs={
                                                       "grid_entry": grid_entry,
                                                       "grid_shape": grid_shape
                                                   }))
        oid = self.cm.logical_and(*bool_list,
                                  syskwargs={"grid_entry": (0, 0), "grid_shape": (1, 1)})
        return BlockArray.from_oid(oid, (), np.bool, self.cm)

    def array_equal(self, a: BlockArray, b: BlockArray):
        return self.array_compare("array_equal", a, b)

    def array_equiv(self, a: BlockArray, b: BlockArray):
        return self.array_compare("array_equiv", a, b)

    def allclose(self, a: BlockArray, b: BlockArray, rtol=1.e-5, atol=1.e-8):
        return self.array_compare("allclose", a, b, rtol, atol)

    def qr(self, X: BlockArray):
        return self.indirect_tsqr(X)

    def indirect_tsr(self, X: BlockArray, reshape_output=True):
        assert len(X.shape) == 2
        # TODO (hme): This assertion is temporary and ensures returned
        #  shape of qr of block is correct.
        assert X.block_shape[0] >= X.shape[1]
        # Compute R for each block.
        grid = X.grid
        grid_shape = grid.grid_shape
        shape = X.shape
        block_shape = X.block_shape
        R_oids = []
        # Assume no blocking along second dim.
        for i in range(grid_shape[0]):
            # Select a row according to block_shape.
            row = []
            for j in range(grid_shape[1]):
                row.append(X.blocks[i, j].oid)
            R_oids.append(self.cm.qr(*row,
                                     mode="r",
                                     axis=1,
                                     syskwargs={
                                         "grid_entry": (i, 0),
                                         "grid_shape": (grid_shape[0], 1),
                                         "options": {"num_returns": 1}
                                     })
                          )

        # Construct R by summing over R blocks.
        # TODO (hme): Communication may be inefficient due to redundancy of data.
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        tsR = BlockArray(ArrayGrid(shape=R_shape,
                                   block_shape=R_shape,
                                   dtype=X.dtype.__name__),
                         self.cm)
        tsR.blocks[0, 0].oid = self.cm.qr(*R_oids,
                                          mode="r",
                                          axis=0,
                                          syskwargs={
                                              "grid_entry": (0, 0),
                                              "grid_shape": (1, 1),
                                              "options": {"num_returns": 1}
                                          })
        # If blocking is "tall-skinny," then we're done.
        if R_shape != R_block_shape:
            if reshape_output:
                R = tsR.reshape(R_shape, block_shape=R_block_shape)
            else:
                R = tsR
        else:
            R = tsR
        return R

    def indirect_tsqr(self, X: BlockArray, reshape_output=True):
        shape = X.shape
        block_shape = X.block_shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        tsR = self.indirect_tsr(X, reshape_output=False)

        # Compute inverse of R.
        tsR_inverse = self.inv(tsR)
        # If blocking is "tall-skinny," then we're done.
        if R_shape != R_block_shape:
            R_inverse = tsR_inverse.reshape(R_shape, block_shape=R_block_shape)
            if reshape_output:
                R = tsR.reshape(R_shape, block_shape=R_block_shape)
            else:
                R = tsR
        else:
            R_inverse = tsR_inverse
            R = tsR

        Q = X @ R_inverse
        return Q, R

    def direct_tsqr(self, X, reshape_output=True):
        assert len(X.shape) == 2

        # Compute R for each block.
        shape = X.shape
        grid = X.grid
        grid_shape = grid.grid_shape
        block_shape = X.block_shape
        Q_oids = []
        R_oids = []
        QR_dims = []
        Q2_shape = [0, shape[1]]
        for i in range(grid_shape[0]):
            # Select a row according to block_shape.
            row = []
            for j in range(grid_shape[1]):
                row.append(X.blocks[i, j].oid)
            # We invoke "reduced", so q, r is returned with dimensions (M, K), (K, N), K = min(M, N)
            M = grid.get_block_shape((i, 0))[0]
            N = shape[1]
            K = min(M, N)
            QR_dims.append(((M, K), (K, N)))
            Q2_shape[0] += K
            # Run each row on separate nodes along first axis.
            # This maintains some data locality.
            Q_oid, R_oid = self.cm.qr(*row,
                                      mode="reduced",
                                      axis=1,
                                      syskwargs={
                                          "grid_entry": (i, 0),
                                          "grid_shape": (grid_shape[0], 1),
                                          "options": {"num_returns": 2}
                                      })
            R_oids.append(R_oid)
            Q_oids.append(Q_oid)

        # TODO (hme): This pulls several order N^2 R matrices on a single node.
        #  A solution is the recursive extension to direct TSQR.
        Q2_oid, R2_oid = self.cm.qr(*R_oids,
                                    mode="reduced",
                                    axis=0,
                                    syskwargs={
                                        "grid_entry": (0, 0),
                                        "grid_shape": (1, 1),
                                        "options": {"num_returns": 2}
                                    })

        Q2_shape = tuple(Q2_shape)
        Q2_block_shape = (QR_dims[0][1][0], shape[1])
        Q2 = self._vec_from_oids([Q2_oid],
                                 shape=Q2_shape,
                                 block_shape=Q2_block_shape,
                                 dtype=X.dtype)
        # The resulting Q's from this operation are N^2 (same size as above R's).
        Q2_oids = list(map(lambda block: block.oid, Q2.blocks.flatten()))

        # Construct Q.
        Q = self.zeros(shape=shape,
                       block_shape=(block_shape[0], shape[1]),
                       dtype=X.dtype)
        for i, grid_entry in enumerate(Q.grid.get_entry_iterator()):
            Q.blocks[grid_entry].oid = self.cm.bop("tensordot", Q_oids[i], Q2_oids[i],
                                                   a1_T=False, a2_T=False, axes=1,
                                                   syskwargs={"grid_entry": grid_entry,
                                                              "grid_shape": Q.grid.grid_shape})

        # Construct R.
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        tsR = self._vec_from_oids([R2_oid], shape=R_shape, block_shape=R_shape, dtype=X.dtype)
        # If blocking is "tall-skinny," then we're done.
        if R_shape == R_block_shape or not reshape_output:
            R = tsR
        else:
            R = tsR.reshape(R_shape, block_shape=R_block_shape)

        if Q.shape != block_shape or not reshape_output:
            Q = Q.reshape(shape, block_shape=block_shape)

        return Q, R

    def svd(self, X):
        # TODO(hme): Optimize by merging with direct qr to compute U directly,
        #  to avoid wasting space storing intermediate Q.
        #  This may not really help until we have operator fusion.
        assert len(X.shape) == 2
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        Q, R = self.direct_tsqr(X, reshape_output=False)
        assert R.shape == R.block_shape
        R_U, S, VT = self.cm.svd(R.blocks[(0, 0)].oid,
                                 syskwargs={"grid_entry": (0, 0),
                                            "grid_shape": (1, 1)})
        R_U: BlockArray = self._vec_from_oids([R_U], R_shape, R_block_shape, X.dtype)
        S: BlockArray = self._vec_from_oids([S], R_shape[:1], R_block_shape[:1], X.dtype)
        VT = self._vec_from_oids([VT], R_shape, R_block_shape, X.dtype)
        U = Q @ R_U

        return U, S, VT

    def inv(self, X: BlockArray):
        return self._inv(self.cm.inv, {}, X)

    def _inv(self, remote_func, kwargs, X: BlockArray):
        # TODO (hme): Implement scalable version.
        block_shape = X.block_shape
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
        if single_block:
            result = X.copy()
        else:
            result = X.reshape(block_shape=X.shape)
        result.blocks[0, 0].oid = remote_func(result.blocks[0, 0].oid,
                                              **kwargs,
                                              syskwargs={
                                                  "grid_entry": (0, 0),
                                                  "grid_shape": (1, 1)
                                              })
        if not single_block:
            result = result.reshape(block_shape=block_shape)
        return result

    def cholesky(self, X: BlockArray):
        # TODO (hme): Implement scalable version.
        # Note:
        # A = Q, R
        # A.T @ A = R.T @ R
        # A.T @ A = L @ L.T
        # => R == L.T
        block_shape = X.block_shape
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
        if single_block:
            result = X.copy()
        else:
            result = X.reshape(block_shape=X.shape)
        result.blocks[0, 0].oid = self.cm.cholesky(result.blocks[0, 0].oid,
                                                   syskwargs={
                                                       "grid_entry": (0, 0),
                                                       "grid_shape": (1, 1)
                                                   })
        if not single_block:
            result = result.reshape(block_shape=block_shape)
        return result

    def fast_linear_regression(self, X: BlockArray, y: BlockArray):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        Q, R = self.indirect_tsqr(X, reshape_output=False)
        R_inv = self.inv(R)
        if R_shape != R_block_shape:
            R_inv = R_inv.reshape(R_shape, block_shape=R_block_shape)
        theta = R_inv @ (Q.T @ y)
        return theta

    def linear_regression(self, X: BlockArray, y: BlockArray):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        Q, R = self.direct_tsqr(X, reshape_output=False)
        # Invert R.
        R_inv = self.inv(R)
        if R_shape != R_block_shape:
            R_inv = R_inv.reshape(R_shape, block_shape=R_block_shape)
        theta = R_inv @ (Q.T @ y)
        return theta

    def ridge_regression(self, X: BlockArray, y: BlockArray, lamb: float):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert lamb >= 0
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        R = self.indirect_tsr(X)
        lamb_vec = self.array(lamb * np.eye(R_shape[0]), block_shape=R_block_shape)
        # TODO (hme): A better solution exists, which inverts R by augmenting X and y.
        #  See Murphy 7.5.2.
        theta = self.inv(lamb_vec + R.T @ R) @ (X.T @ y)
        return theta

    def _vec_from_oids(self, oids, shape, block_shape, dtype):
        arr = BlockArray(ArrayGrid(shape=shape,
                                   block_shape=shape,
                                   dtype=dtype.__name__),
                         self.cm)
        # Make sure resulting grid shape is a vector (1 dimensional).
        assert np.sum(arr.grid.grid_shape) == (max(arr.grid.grid_shape)
                                               + len(arr.grid.grid_shape) - 1)
        for i, grid_entry in enumerate(arr.grid.get_entry_iterator()):
            arr.blocks[grid_entry].oid = oids[i]
        if block_shape != shape:
            return arr.reshape(block_shape=block_shape)
        return arr

    def random_state(self, seed=None):
        return NumsRandomState(self.cm, seed)

    def nanmean(self, a: BlockArray, axis=None, keepdims=False, dtype=None):
        if not array_utils.is_float(a):
            a = a.astype(np.float64)

        num_summed = self.sum(~a.ufunc("isnan"), axis=axis, dtype=a.dtype, keepdims=keepdims)

        if num_summed.ndim == 0 and num_summed == 0:
            return self.scalar(np.nan)

        if num_summed.ndim > 0:
            num_summed = self.where(num_summed == 0,
                                    self.empty(num_summed.shape, num_summed.block_shape) * np.nan,
                                    num_summed)

        res = self.reduce("nansum", a, axis=axis, dtype=dtype, keepdims=keepdims) / num_summed

        if dtype is not None:
            res = res.astype(dtype)
        return res

    def nanvar(self, a: BlockArray, axis=None, ddof=0, keepdims=False, dtype=None):
        mean = self.nanmean(a, axis=axis, keepdims=True)
        ss = self.reduce("nansum", (a - mean)**self.two, axis=axis, dtype=dtype, keepdims=keepdims)
        num_summed = self.sum(~a.ufunc("isnan"), axis=axis, dtype=a.dtype, keepdims=keepdims) - ddof
        res = ss / num_summed
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def nanstd(self, a: BlockArray, axis=None, ddof=0, keepdims=False, dtype=None):
        res = self.sqrt(self.nanvar(a, axis, ddof, keepdims))
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def atleast_1d(self, *arys):
        # TODO (hme): Refactor this to use check_or_convert_other
        res = []
        for ary in arys:
            if not isinstance(ary, BlockArray):
                ary = np.array(ary)
                block_shape = self.cm.compute_block_shape(ary.shape, ary.dtype)
                ary = self.array(ary, block_shape)
            if ary.ndim == 0:
                result = ary.reshape(1)
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def atleast_2d(self, *arys):
        # TODO (hme): Refactor this to use check_or_convert_other
        res = []
        for ary in arys:
            if not isinstance(ary, BlockArray):
                ary = np.array(ary)
                block_shape = self.cm.compute_block_shape(ary.shape, ary.dtype)
                ary = self.array(ary, block_shape)
            if ary.ndim == 0:
                result = ary.reshape(1, 1)

            # TODO (MWE): Implement this using newaxis when supported
            # This is because ary.base needs to stay consistent,
            # which reshape will not accomplish
            elif ary.ndim == 1:
                result = ary.reshape(1, ary.shape[0])
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def atleast_3d(self, *arys):
        # TODO (hme): Refactor this to use check_or_convert_other
        res = []
        for ary in arys:
            if not isinstance(ary, BlockArray):
                ary = np.array(ary)
                block_shape = self.cm.compute_block_shape(ary.shape, ary.dtype)
                ary = self.array(ary, block_shape)
            if ary.ndim == 0:
                result = ary.reshape(1, 1, 1)

            # TODO (MWE): Implement this using newaxis when supported
            # This is because ary.base needs to stay consistent,
            # which reshape will not accomplish
            elif ary.ndim == 1:
                result = ary.reshape(1, ary.shape[0], 1)
            elif ary.ndim == 2:
                result = ary.reshape(ary.shape[0], ary.shape[1], 1)
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res
