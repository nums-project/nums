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

import os
import itertools
import pickle
from typing import Tuple, List, Any, Iterator

import numpy as np
import boto3

from nums.core.storage.utils import Batch


class ArrayGrid(object):
    # TODO: Add negative val support.
    # TODO: Move to array module.

    @classmethod
    def from_meta(cls, d: dict):
        return cls(**d)

    def __init__(self, shape: Tuple, block_shape: Tuple, dtype: str):
        self.shape = tuple(shape)
        self.block_shape = tuple(np.min([shape, block_shape], axis=0))
        self.dtype = dict if dtype == "dict" else getattr(np, dtype)
        self.grid_shape = []
        self.grid_slices = []
        for i in range(len(self.shape)):
            dim = self.shape[i]
            block_dim = block_shape[i]
            axis_slices = Batch(dim, block_dim).batches
            self.grid_slices.append(axis_slices)
            self.grid_shape.append(len(axis_slices))
        self.grid_shape = tuple(self.grid_shape)

    def to_meta(self) -> dict:
        return {
            "shape": self.shape,
            "block_shape": self.block_shape,
            "dtype": self.dtype.__name__
        }

    def copy(self):
        return self.from_meta(self.to_meta())

    def get_entry_iterator(self) -> Iterator[Tuple]:
        if 0 in self.shape:
            return []
        return itertools.product(*map(range, self.grid_shape))

    def get_slice(self, grid_entry):
        slices = []
        for axis, slice_index in enumerate(grid_entry):
            slices.append(slice(*self.grid_slices[axis][slice_index]))
        return tuple(slices)

    def get_slice_tuples(self, grid_entry: Tuple) -> List[Tuple[slice]]:
        slice_tuples = []
        for axis, slice_index in enumerate(grid_entry):
            slice_tuples.append(tuple(self.grid_slices[axis][slice_index]))
        return slice_tuples

    def get_block_shape(self, grid_entry: Tuple):
        slice_tuples = self.get_slice_tuples(grid_entry)
        block_shape = []
        for slice_tuple in slice_tuples:
            block_shape.append(slice_tuple[1] - slice_tuple[0])
        return tuple(block_shape)

    def get_size_of(self, units: str, num_entries: int) -> str:
        entry_bytes = self.dtype().itemsize
        num_bytes = entry_bytes * num_entries
        units_mag = 1
        if units == "KB":
            units_mag = 3
        if units == "MB":
            units_mag = 6
        if units == "GB":
            units_mag = 9
        return "%.2f %s" % (num_bytes / 10**units_mag, units)

    def get_block_size(self, units: str) -> str:
        return self.get_size_of(units, np.product(self.block_shape))

    def get_size(self, units: str) -> str:
        return self.get_size_of(units, np.product(self.shape))

    def select(self, slice_tuples: List[Tuple]) -> List[List]:
        """
        :param slice_tuples: An array of slice tuples.
               e.g. [(100, 200), (0, 100)]
        :return: An array of [grid_entry, slice_tuples]
                 pairs required to perform the selection.
        """
        grid_axis_intervals = []
        for axis, (i_1, i_2) in enumerate(slice_tuples):
            assert 0 <= i_1 < i_2 <= self.shape[axis]
            bs = self.block_shape[axis]
            grid_i_1 = i_1 // bs
            grid_i_2 = (i_2 + bs - 1) // bs
            # Collect required intervals along each grid coordinate
            # in order to perform the selection.
            grid_axis_intervals.append(list(range(grid_i_1, grid_i_2)))

        selection_superset = []
        iterator = itertools.product(*grid_axis_intervals)
        for grid_entry in iterator:
            block_slice_tuples = []
            for axis, grid_index in enumerate(grid_entry):
                # The length of grid_entry is equal to the number of axes.
                # For each axis, append the block slice corresponding to the grid entry.
                block_slice_tuples.append(tuple(self.grid_slices[axis][grid_index]))
            selection_superset.append([grid_entry, block_slice_tuples])
        return selection_superset


class StoredArray(object):

    def __init__(self, filename: str, grid: ArrayGrid):
        self.filename = filename
        self.dirname, self.array_name = os.path.split(self.filename)
        self.grid = grid

    def init_grid(self):
        self.grid = self.get_grid()

    def get_key(self, grid_entry: Tuple):
        index_str = "_".join(map(str, grid_entry))
        return "%s_%s" % (self.array_name, index_str)

    def get_meta_key(self):
        return "%s_meta" % self.array_name

    def put(self, grid_entry: Tuple, block: np.ndarray) -> Any:
        raise NotImplementedError()

    def get(self, grid_entry: Tuple) -> np.ndarray:
        raise NotImplementedError()

    def delete(self, grid_entry: Tuple) -> Any:
        raise NotImplementedError()

    def get_grid(self) -> ArrayGrid:
        raise NotImplementedError()

    def put_grid(self, array_grid: ArrayGrid) -> Any:
        raise NotImplementedError()

    def delete_grid(self) -> Any:
        raise NotImplementedError()

    def del_array(self) -> Any:
        raise NotImplementedError()

    def put_array(self, arr: np.ndarray):
        grid_entry_iterator = self.grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = self.grid.get_slice(grid_entry)
            block = arr[grid_slice]
            self.put(grid_entry, block)

    def get_array(self):
        grid_shape = self.grid.grid_shape
        result = np.zeros(shape=self.grid.shape)
        iterator = list(itertools.product(*map(range, grid_shape)))
        block_shape = np.array(self.grid.block_shape, dtype=np.int)
        for grid_entry in iterator:
            start = block_shape * grid_entry
            entry_shape = np.array(self.grid.get_block_shape(grid_entry), dtype=np.int)
            end = start + entry_shape
            slices = tuple(map(lambda item: slice(*item), zip(*(start, end))))
            result[slices] = self.get(grid_entry)
        return result


class StoredArrayS3(StoredArray):

    def __init__(self, filename: str, grid: ArrayGrid = None):
        self.client = boto3.client('s3')
        super(StoredArrayS3, self).__init__(filename, grid)
        if self.filename[0] == "/":
            raise Exception("Leading / in s3 filename: %s" % filename)
        fileparts = self.filename.split("/")
        self.container_name = fileparts[0]
        self.array_name = "/".join(fileparts[1:])

    def put(self, grid_entry: Tuple, block: np.ndarray) -> Any:
        block_bytes = block.tobytes()
        response = self.client.put_object(
            Bucket=self.container_name,
            Key=self.get_key(grid_entry),
            Body=block_bytes,
        )
        return response

    def get(self, grid_entry: Tuple) -> np.ndarray:
        try:
            response = self.client.get_object(
                Bucket=self.container_name,
                Key=self.get_key(grid_entry),
            )
        except Exception as e:
            print("[Error] StoredArrayS3: Failed to get",
                  self.container_name,
                  self.get_key(grid_entry))
            raise e
        block_bytes = response['Body'].read()
        dtype = self.grid.dtype
        shape = self.grid.get_block_shape(grid_entry)
        try:
            block = np.frombuffer(block_bytes, dtype=dtype).reshape(shape)
        except Exception as e:
            print("[Error] StoredArrayS3: Failed to read from buffer",
                  self.container_name,
                  self.get_key(grid_entry))
            raise e
        return block

    def delete(self, grid_entry: Tuple) -> Any:
        objects = [{"Key": self.get_key(grid_entry)}]
        response = self.client.delete_objects(
            Bucket=self.container_name,
            Delete={
                'Objects': objects,
            },
        )
        return response

    def delete_grid(self) -> Any:
        objects = [{"Key": self.get_meta_key()}]
        response = self.client.delete_objects(
            Bucket=self.container_name,
            Delete={
                'Objects': objects,
            },
        )
        return response

    def put_grid(self, array_grid: ArrayGrid) -> Any:
        self.grid = array_grid
        body = pickle.dumps(self.grid.to_meta())
        response = self.client.put_object(
            Bucket=self.container_name,
            Key=self.get_meta_key(),
            Body=body,
        )
        return response

    def get_grid(self) -> ArrayGrid:
        try:
            response = self.client.get_object(Bucket=self.container_name,
                                              Key=self.get_meta_key())
            meta_dict = pickle.loads(response['Body'].read())
            return ArrayGrid.from_meta(meta_dict)
        except Exception as _:
            return None

    def del_array(self):
        objects = []
        grid_entry_iterator = self.grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            objects.append({"Key": self.get_key(grid_entry)})
        response = self.client.delete_objects(
            Bucket=self.container_name,
            Delete={
                'Objects': objects,
            },
        )
        return response


class StoredArrayDFS(StoredArray):

    def __init__(self, filename: str, grid: ArrayGrid = None):
        super(StoredArrayDFS, self).__init__(filename, grid)

    def get(self, grid_entry: Tuple) -> np.ndarray:
        raise NotImplementedError()

    def put(self, grid_entry: Tuple, block: np.ndarray) -> Any:
        raise NotImplementedError()

    def get_grid(self) -> ArrayGrid:
        raise NotImplementedError()

    def put_grid(self, array_grid: ArrayGrid) -> Any:
        raise NotImplementedError()

    def del_array(self) -> Any:
        raise NotImplementedError()


class StoredArrayFS(StoredArray):

    def __init__(self, filename: str, grid: ArrayGrid = None):
        super(StoredArrayFS, self).__init__(filename, grid)

    def get(self, grid_entry: Tuple) -> np.ndarray:
        raise NotImplementedError()

    def put(self, grid_entry: Tuple, block: np.ndarray) -> Any:
        raise NotImplementedError()

    def get_grid(self) -> ArrayGrid:
        raise NotImplementedError()

    def put_grid(self, array_grid: ArrayGrid) -> Any:
        raise NotImplementedError()

    def del_array(self) -> Any:
        raise NotImplementedError()


class BimodalGaussian(object):

    @classmethod
    def get_dataset(cls, n, d, p=0.9, seed=1, dtype=np.float64, theta=None):
        return cls(10, 2, 30, 4, dim=d, seed=seed, dtype=dtype).sample(n, p=p, theta=theta)

    def __init__(self, mu1, sigma1, mu2, sigma2, dim=2, seed=1337, dtype=np.float64):
        self.dtype = dtype
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        self.dim = dim
        self.mu1 = self.to_arr(mu1, 1)
        self.sigma1 = self.to_arr(sigma1, 1)
        self.mu2 = self.to_arr(mu2, 1)
        self.sigma2 = self.to_arr(sigma2, 1)

    def to_arr(self, sigma, num_axes):
        assert num_axes == 1 or num_axes == 2
        sigma_arr = sigma
        if not isinstance(sigma, np.ndarray):
            # Assume it's not diag.
            sigma_arr = np.empty(self.dim, dtype=self.dtype)
            sigma_arr[:] = sigma
        if num_axes == 2:
            if len(sigma_arr.shape) == 1:
                sigma_arr = np.diag(sigma_arr).astype(self.dtype)
                assert len(sigma_arr.shape) == 2
                assert sigma_arr.shape[0] == sigma_arr.shape[1]
        else:
            assert len(sigma_arr.shape) == num_axes
        return sigma_arr

    def sample(self, n, p=0.9, theta=None):
        # Larger p => more samples of first Gaussian.
        # Pass theta to sample for regression.
        n1 = int(n * p)
        n2 = n - n1
        X1 = self.rs.randn(n1, self.dim).astype(self.dtype) * self.sigma1.T + self.mu1.T
        X2 = self.rs.randn(n2, self.dim).astype(self.dtype) * self.sigma2.T + self.mu2.T
        if theta is None:
            y1 = np.ones(n1, dtype=self.dtype)
            y2 = np.zeros(n2, dtype=self.dtype)
        else:
            y1 = X1 @ theta
            y2 = X2 @ theta
        X = np.concatenate((X1, X2), axis=0).astype(self.dtype)
        y = np.concatenate((y1, y2), axis=0).astype(self.dtype)
        idx = self.rs.permutation(n)
        X = X[idx]
        y = y[idx]
        return X, y
