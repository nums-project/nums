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
import logging
import os
import pickle
from typing import Tuple, Any

import boto3
import numpy as np

from nums.core.grid.grid import ArrayGrid


class StoredArray(object):
    # TODO (hme): This is no longer a useful abstraction.

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
        self.client = boto3.client("s3")
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
            logging.getLogger(__name__).error(
                "[Error] StoredArrayS3: Failed to get %s %s",
                self.container_name,
                self.get_key(grid_entry),
            )
            raise e
        block_bytes = response["Body"].read()
        dtype = self.grid.dtype
        shape = self.grid.get_block_shape(grid_entry)
        try:
            block = np.frombuffer(block_bytes, dtype=dtype).reshape(shape)
        except Exception as e:
            logging.getLogger(__name__).error(
                "[Error] StoredArrayS3: " "Failed to read from buffer %s %s",
                self.container_name,
                self.get_key(grid_entry),
            )
            raise e
        return block

    def delete(self, grid_entry: Tuple) -> Any:
        objects = [{"Key": self.get_key(grid_entry)}]
        response = self.client.delete_objects(
            Bucket=self.container_name,
            Delete={
                "Objects": objects,
            },
        )
        return response

    def delete_grid(self) -> Any:
        objects = [{"Key": self.get_meta_key()}]
        response = self.client.delete_objects(
            Bucket=self.container_name,
            Delete={
                "Objects": objects,
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
            response = self.client.get_object(
                Bucket=self.container_name, Key=self.get_meta_key()
            )
            meta_dict = pickle.loads(response["Body"].read())
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
                "Objects": objects,
            },
        )
        return response


class BimodalGaussian(object):
    @classmethod
    def get_dataset(cls, n, d, p=0.9, seed=1, dtype=np.float64, theta=None):
        return cls(10, 2, 30, 4, dim=d, seed=seed, dtype=dtype).sample(
            n, p=p, theta=theta
        )

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
