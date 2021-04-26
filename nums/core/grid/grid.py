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
from typing import Tuple, Iterator, List

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.storage.utils import Batch


class ArrayGrid(object):

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
            if dim == 0:
                # Special case of empty array.
                axis_slices = []
            else:
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

    def nbytes(self):
        if array_utils.is_float(self.dtype, type_test=True):
            dtype = np.finfo(self.dtype).dtype
        elif array_utils.is_int(self.dtype, type_test=True) \
                or array_utils.is_uint(self.dtype, type_test=True):
            dtype = np.iinfo(self.dtype).dtype
        elif array_utils.is_complex(self.dtype, type_test=True):
            dtype = np.dtype(self.dtype)
        elif self.dtype in (bool, np.bool_):
            dtype = np.dtype(np.bool_)
        else:
            raise ValueError("dtype %s not supported" % str(self.dtype))

        dtype_nbytes = dtype.alignment
        nbytes = np.product(self.shape) * dtype_nbytes
        return nbytes


class DeviceID(object):

    def __init__(self, node_id, device_type, device_id):
        self.node_id = node_id
        self.device_type = device_type
        self.device_id = device_id

    def __str__(self):
        return "node:%s/%s:%s" % (self.node_id, self.device_type, self.device_id)


class DeviceGrid(object):

    def __init__(self, grid_shape, device_type):
        # TODO (hme): Work out what this becomes in the multi-node multi-device setting.
        self.grid_shape = grid_shape
        self.device_type = device_type

    def get_device_id(self, agrid_entry, agrid_shape):
        raise NotImplementedError()

    def get_entry_iterator(self) -> Iterator[Tuple]:
        return itertools.product(*map(range, self.grid_shape))


class CyclicDeviceGrid(DeviceGrid):

    def get_device_id(self, agrid_entry, agrid_shape):
        dgrid_entry = tuple(np.array(agrid_entry) % np.array(self.grid_shape))
        return dgrid_entry
