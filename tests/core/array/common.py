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


import numpy as np

from nums.core.array.application import ArrayApplication, BlockArray
from nums.core.compute import numpy_compute
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import DeviceGrid, CyclicDeviceGrid
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.systems import SystemInterface, RaySystem


def check_block_integrity(arr: BlockArray):
    for grid_entry in arr.grid.get_entry_iterator():
        assert arr.blocks[grid_entry].grid_entry == grid_entry
        assert arr.blocks[grid_entry].rect == arr.grid.get_slice_tuples(grid_entry)
        assert arr.blocks[grid_entry].shape == arr.grid.get_block_shape(grid_entry)


class MockMultiNodeDeviceGrid(CyclicDeviceGrid):
    # pylint: disable=abstract-method, bad-super-call

    def __init__(self, grid_shape, device_type, device_ids):
        # Replicate available devices to satisfy cluster requirements.
        assert len(device_ids) == 1
        mock_device_ids = device_ids * np.prod(self.grid_shape)
        super(CyclicDeviceGrid, self).__init__(grid_shape, device_type, mock_device_ids)


def mock_cluster(cluster_shape):
    system: SystemInterface = RaySystem(use_head=True)
    system.init()
    device_grid: DeviceGrid = MockMultiNodeDeviceGrid(cluster_shape, "cpu", system.devices())
    cm = ComputeManager.create(system, numpy_compute, device_grid)
    fs = FileSystem(cm)
    return ArrayApplication(cm, fs)
