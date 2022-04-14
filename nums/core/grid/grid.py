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


from dataclasses import dataclass
import itertools
import logging
from typing import Tuple, Iterator, List

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.storage.utils import Batch


class ArrayGrid:
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


@dataclass(frozen=True)
class Device:
    node_id: int
    node_addr: str
    device_type: str
    device: int


class DeviceGrid:
    def __init__(self, grid_shape, device_type, devices):
        self.grid_shape = grid_shape
        self.device_type = device_type
        self.device_grid: np.ndarray = np.empty(shape=self.grid_shape, dtype=object)

        self._check_devices(devices)
        # Delegate device ordering to subclasses,
        # so that proper ordering is achieved when given multiple devices/workers per node.
        devices: List[Device] = self._order_devices(devices)

        for i, cluster_entry in enumerate(self.get_grid_entry_iterator()):
            device: Device = devices[i]
            self.device_grid[cluster_entry] = device
            print(cluster_entry, device)

        logging.getLogger(__name__).info("device_grid %s", str(self.grid_shape))

    def _check_devices(self, devices: List[Device]):
        node_map = {}
        node_addr_check = set()
        for device in devices:
            node_addr_check.add(device.node_addr)
            if device.node_id not in node_map:
                node_map[device.node_id] = 0
            node_map[device.node_id] += 1
        assert len(node_addr_check) == len(node_map)
        devices_per_node = None
        for _, device_count in node_map.items():
            if devices_per_node is None:
                devices_per_node = device_count
            assert devices_per_node == device_count
        num_nodes = len(node_map)
        return num_nodes, devices_per_node

    def _order_devices(self, devices: List[Device]):
        raise NotImplementedError()

    def get_grid_entry_iterator(self):
        return itertools.product(*map(range, self.grid_shape))

    def get_device(self, agrid_entry, agrid_shape):
        raise NotImplementedError()


class CyclicDeviceGrid(DeviceGrid):
    def _order_devices(self, devices: List[Device]):
        # We order by device id first, then by node id, which ensures data layout
        # cycles over nodes first, then workers.
        return sorted(devices, key=lambda device: (device.device, device.node_id))

    def get_device(self, agrid_entry, agrid_shape):
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


class PackedDeviceGrid(DeviceGrid):
    # Places adjacent blocks on the same nodes.
    # Only useful on Ray.
    def _order_devices(self, devices: List[Device]):
        return sorted(devices, key=lambda device: (device.node_id, device.device))

    def _check_devices(self, devices: List[Device]):
        _, devices_per_node = super()._check_devices(devices)
        assert devices_per_node == 1

    def get_device(self, agrid_entry, agrid_shape):
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
