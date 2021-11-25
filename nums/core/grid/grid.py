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
            "dtype": self.dtype.__name__,
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

    def get_entry_coordinates(self, grid_entry) -> Tuple[int]:
        coordinates = []
        for axis, slice_index in enumerate(grid_entry):
            coordinates.append(self.grid_slices[axis][slice_index][0])
        return tuple(coordinates)

    def get_block_shape(self, grid_entry: Tuple):
        slice_tuples = self.get_slice_tuples(grid_entry)
        block_shape = []
        for slice_tuple in slice_tuples:
            block_shape.append(slice_tuple[1] - slice_tuple[0])
        return tuple(block_shape)

    def nbytes(self):
        if array_utils.is_float(self.dtype, type_test=True):
            dtype = np.finfo(self.dtype).dtype
        elif array_utils.is_int(self.dtype, type_test=True) or array_utils.is_uint(
            self.dtype, type_test=True
        ):
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
    @classmethod
    def from_str(cls, s: str):
        a, b = s.split("/")
        node_id, node_addr = a.split("=")
        device_type, device_id = b.split(":")
        return DeviceID(int(node_id), node_addr, device_type, int(device_id))

    def __init__(self, node_id: int, node_addr: str, device_type: str, device_id: int):
        self.node_id: int = node_id
        self.node_addr: str = node_addr
        self.device_type: str = device_type
        self.device_id: int = device_id

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return "%s=%s/%s:%s" % (
            self.node_id,
            self.node_addr,
            self.device_type,
            self.device_id,
        )

    def __eq__(self, other):
        return str(self) == str(other)


class DeviceGrid(object):
    def __init__(self, grid_shape, device_type, device_ids, workers_per_node=None):
        # TODO (hme): Work out what this becomes in the multi-node multi-device setting.
        self.grid_shape = grid_shape
        self.device_type = device_type
        self.device_ids: List[DeviceID] = device_ids
        self.device_grid: np.ndarray = np.empty(shape=self.grid_shape, dtype=object)
        self.workers_per_node = workers_per_node
        # For nested layouts.
        self.node_grid_shape = None
        self.num_nodes = None

        if self.workers_per_node is not None:
            self.node_grid_shape = []
            for dim in self.grid_shape:
                if dim == 1:
                    # Special case. Ignore this dim.
                    self.node_grid_shape.append(1)
                else:
                    assert dim % self.workers_per_node == 0
                    self.node_grid_shape.append(dim // self.workers_per_node)
            self.node_grid_shape = tuple(self.node_grid_shape)
            self.num_nodes = np.product(self.node_grid_shape)
            logging.getLogger(__name__).info("node_grid %s", str(self.node_grid_shape))
            assert self.num_nodes == len(self.device_ids) // self.workers_per_node
        else:
            logging.getLogger(__name__).info("no node_grid.")

        unique_node_ids = set()
        node_addr_check = set()
        # TODO: For nested cyclic, when given workers,
        #  order device list so that it cycles nodes.
        #  This will enable nested cyclic layouts over arbitrary number of dimensions.
        for i, cluster_entry in enumerate(self.get_cluster_entry_iterator()):
            device_id: DeviceID = self.device_ids[i]
            # Check some assumptions if workers_per_node is given.
            unique_node_ids.add(device_id.node_id)
            node_addr_check.add(device_id.node_addr)
            if self.workers_per_node is not None:
                assert device_id.node_id == i // self.workers_per_node
                assert device_id.device_id == i % self.workers_per_node
            self.device_grid[cluster_entry] = device_id

        assert len(node_addr_check) == len(unique_node_ids)
        if self.workers_per_node is not None:
            assert len(node_addr_check) == self.num_nodes
        logging.getLogger(__name__).info("device_grid %s", str(self.grid_shape))

    def get_cluster_entry_iterator(self):
        return itertools.product(*map(range, self.grid_shape))

    def get_device_id(self, agrid_entry, agrid_shape):
        raise NotImplementedError()

    def get_entry_iterator(self) -> Iterator[Tuple]:
        return itertools.product(*map(range, self.grid_shape))


class CyclicDeviceGrid(DeviceGrid):
    def get_device_id(self, agrid_entry, agrid_shape):
        cluster_entry = self.get_cluster_entry(agrid_entry, agrid_shape)
        return self.device_grid[cluster_entry]

    def get_cluster_entry(self, agrid_entry, agrid_shape):
        # pylint: disable = unused-argument
        cluster_entry = []
        num_grid_entry_axes = len(agrid_entry)
        num_cluster_axes = len(self.grid_shape)
        for cluster_axis in range(num_cluster_axes):
            if cluster_axis < num_grid_entry_axes:
                cluster_dim = self.grid_shape[cluster_axis]
                grid_entry_dim = agrid_entry[cluster_axis]
                cluster_entry.append(grid_entry_dim % cluster_dim)
            else:
                # When array has fewer axes than cluster.
                cluster_entry.append(0)
            # Ignore trailing array axes, as these are "cycled" to 0 by assuming
            # the dimension of those cluster axes is 1.
        return tuple(cluster_entry)


class NestedCyclicDeviceGrid(CyclicDeviceGrid):
    # Distribute the data such that each array grid axis
    # is first cycled over nodes, and then cycled over workers.
    # For example, if we have 3 workers per node and 2 nodes,
    # map the blocks for array grid (7,) as follows:
    # i, node_idx, worker_idx
    # 0, 0, 0
    # 1, 1, 0
    # 2, 2, 0
    # 3, 0, 1
    # 4, 1, 1
    # 5, 2, 1
    # 6, 0, 2

    def get_cluster_entry_axis_index(self, axis, agrid_entry):
        # For some value 0 <= i < agrid_shape[axis]
        # Compute the entry axis index in device_grid coordinates.
        axis_num_nodes = self.node_grid_shape[axis]
        if axis_num_nodes == 1:
            # Special case. Map to 0.
            # This is fine since we're expecting the work to be saturated by the other axes.
            return 0
        i = agrid_entry[axis]
        node_idx = i % axis_num_nodes
        worker_idx = (i // max(1, axis_num_nodes)) % self.workers_per_node
        # So far, we have computed node_idx and worker_idx as described in the example given above.
        # Now flatten to device_grid coordinates.
        # The device grid is laid out as
        # node_id, device_id
        # 0, 0
        # 0, 1
        # 0, 2
        # 1, 0
        # 1, 1
        # 1, 2
        # etc.
        # We want to jump node_idx * self.workers_per_node workers,
        # moving us to the segment of devices corresponding to node_idx.
        return node_idx * self.workers_per_node + worker_idx

    def get_cluster_entry(self, agrid_entry, agrid_shape):
        # pylint: disable = unused-argument
        cluster_entry = []
        num_grid_entry_axes = len(agrid_entry)
        num_cluster_axes = len(self.grid_shape)
        for cluster_axis in range(num_cluster_axes):
            if cluster_axis < num_grid_entry_axes:
                cluster_entry_axis = self.get_cluster_entry_axis_index(
                    cluster_axis, agrid_entry
                )
                cluster_entry.append(cluster_entry_axis)
            else:
                # When array has fewer axes than cluster.
                cluster_entry.append(0)
            # Ignore trailing array axes, as these are "cycled" to 0 by assuming
            # the dimension of those cluster axes is 1.
        return tuple(cluster_entry)


class PackedDeviceGrid(DeviceGrid):
    # For the nested case, where number of devices = number of workers, this is equivalent to unnested cyclic.
    def get_device_id(self, agrid_entry, agrid_shape):
        cluster_entry = self.get_cluster_entry(agrid_entry, agrid_shape)
        return self.device_grid[cluster_entry]

    def get_cluster_entry(self, agrid_entry, agrid_shape):
        cluster_entry = []
        num_grid_entry_axes = len(agrid_entry)
        num_cluster_axes = len(self.grid_shape)
        for cluster_axis in range(num_cluster_axes):
            if cluster_axis < num_grid_entry_axes:
                cluster_entry.append(
                    self.compute_cluster_entry_axis(
                        axis=cluster_axis,
                        ge_axis_val=agrid_entry[cluster_axis],
                        gs_axis_val=agrid_shape[cluster_axis],
                        cs_axis_val=self.grid_shape[cluster_axis],
                    )
                )
            else:
                cluster_entry.append(0)
        return tuple(cluster_entry)

    def compute_cluster_entry_axis(self, axis, ge_axis_val, gs_axis_val, cs_axis_val):
        if ge_axis_val >= gs_axis_val:
            raise ValueError(
                "Array grid_entry is not < grid_shape along axis %s." % axis
            )
        return int(ge_axis_val / gs_axis_val * cs_axis_val)
