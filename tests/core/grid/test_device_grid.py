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

from typing import List
import itertools

import pytest
import numpy as np

from nums.core.grid.grid import DeviceID, ArrayGrid, CyclicDeviceGrid, PackedDeviceGrid


def mock_device_ids(num_nodes):
    r = []
    for node_id in range(num_nodes):
        did = DeviceID(node_id, "node%s" % node_id, "cpu", 1)
        r.append(did)
    return r


def test_bounds():
    grid: ArrayGrid = ArrayGrid(shape=(2, 6, 10), block_shape=(1, 2, 5), dtype="float32")
    for cluster_shape in [(1,), (1, 1), (1, 1, 1), (1, 1, 1, 1)]:
        cyclic_grid: CyclicDeviceGrid = CyclicDeviceGrid(cluster_shape, "cpu", mock_device_ids(1))
        packed_grid: PackedDeviceGrid = PackedDeviceGrid(cluster_shape, "cpu", mock_device_ids(1))
        for grid_entry in grid.get_entry_iterator():
            cluster_entry = cyclic_grid.get_cluster_entry(grid_entry, grid.grid_shape)
            assert cluster_entry == tuple([0]*len(cyclic_grid.grid_shape))
            cluster_entry = packed_grid.get_cluster_entry(grid_entry, grid.grid_shape)
            assert cluster_entry == tuple([0]*len(packed_grid.grid_shape))


def test_computations():
    grid: ArrayGrid = ArrayGrid(shape=(2, 6, 10), block_shape=(1, 2, 5), dtype="float32")
    cluster_shapes = list(itertools.product(list(range(1, 5)),
                                            list(range(1, 7)),
                                            list(range(1, 11))))
    for cluster_shape in cluster_shapes:
        device_ids = mock_device_ids(int(np.product(cluster_shape)))
        cyclic_grid: CyclicDeviceGrid = CyclicDeviceGrid(cluster_shape, "cpu", device_ids)
        for grid_entry in grid.get_entry_iterator():
            cluster_entry = cyclic_grid.get_cluster_entry(grid_entry, grid.grid_shape)
            assert cluster_entry == tuple(np.array(grid_entry) % np.array(cluster_shape))

    def true_packed_entry(grid_entry, grid_shape, cluster_shape):
        grid_entry = np.array(grid_entry)
        grid_shape = np.array(grid_shape)
        cluster_shape = np.array(cluster_shape)
        r = grid_entry / grid_shape * cluster_shape
        # r = np.min(cluster_shape-1, r, axis=1)
        return tuple(r.astype(int).tolist())

    for cluster_shape in cluster_shapes:
        device_ids = mock_device_ids(int(np.product(cluster_shape)))
        packed_grid: PackedDeviceGrid = PackedDeviceGrid(cluster_shape, "cpu", device_ids)
        for grid_entry in grid.get_entry_iterator():
            cluster_entry = packed_grid.get_cluster_entry(grid_entry, grid.grid_shape)
            assert cluster_entry == true_packed_entry(grid_entry, grid.grid_shape, cluster_shape)


def test_errors():
    cluster_shape = (1, 2, 3)
    device_ids = mock_device_ids(int(np.product(cluster_shape)))
    grid: ArrayGrid = ArrayGrid(shape=(8, 20, 12), block_shape=(2, 5, 3), dtype="float32")
    packed_grid: PackedDeviceGrid = PackedDeviceGrid(cluster_shape, "cpu", device_ids)

    grid_shape = grid.grid_shape
    grid_entry = tuple(np.array(grid_shape, dtype=int) - 1)
    assert packed_grid.get_cluster_entry(grid_entry, grid_shape) == (0, 1, 2)

    grid_entry = np.array(grid_shape, dtype=int) - 1
    grid_entry[0] += 1
    grid_entry = tuple(grid_entry)
    with pytest.raises(ValueError):
        # Out of bounds grid entry.
        packed_grid.get_cluster_entry(grid_entry, grid_shape)


def test_device_id():
    cluster_shape = (1, 2, 3)
    device_ids = mock_device_ids(int(np.product(cluster_shape)))
    grid: ArrayGrid = ArrayGrid(shape=(8, 20, 12), block_shape=(2, 5, 3), dtype="float32")

    # A basic smoke test.
    device_ids: List[DeviceID] = mock_device_ids(int(np.product(cluster_shape)))
    cyclic_grid: CyclicDeviceGrid = CyclicDeviceGrid(cluster_shape, "cpu", device_ids)

    touched_devices = set()
    for grid_entry in grid.get_entry_iterator():
        touched_devices.add(cyclic_grid.get_device_id(grid_entry, grid.grid_shape))
    assert len(touched_devices) == len(device_ids)

    packed_grid: PackedDeviceGrid = PackedDeviceGrid(cluster_shape, "cpu", device_ids)
    touched_devices = set()
    for grid_entry in grid.get_entry_iterator():
        touched_devices.add(packed_grid.get_device_id(grid_entry, grid.grid_shape))
    assert len(touched_devices) == len(device_ids)

if __name__ == "__main__":
    test_bounds()
    test_computations()
    test_errors()
    test_device_id()
