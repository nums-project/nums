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

from nums.core.compute import numpy_compute
from nums.core.compute.compute_manager import ComputeManager
from nums.core.systems.systems import SystemInterface, RaySystem
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.schedulers import RayScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication, BlockArray


def check_block_integrity(arr: BlockArray):
    for grid_entry in arr.grid.get_entry_iterator():
        assert arr.blocks[grid_entry].grid_entry == grid_entry
        assert arr.blocks[grid_entry].rect == arr.grid.get_slice_tuples(grid_entry)
        assert arr.blocks[grid_entry].shape == arr.grid.get_block_shape(grid_entry)


class MockMultiNodeScheduler(BlockCyclicScheduler):
    # pylint: disable=abstract-method, bad-super-call

    def init(self):
        # Intentionally calling init of grandparent class.
        super(BlockCyclicScheduler, self).init()
        # Replicate available nodes to satisfy cluster requirements.
        assert len(self.available_nodes) == 1

        self.available_nodes = self.available_nodes * np.prod(self.cluster_shape)
        for i, cluster_entry in enumerate(self.get_cluster_entry_iterator()):
            self.cluster_grid[cluster_entry] = self.available_nodes[i]


def mock_cluster(cluster_shape):
    scheduler: RayScheduler = MockMultiNodeScheduler(cluster_shape=cluster_shape,
                                                     use_head=True)
    system: SystemInterface = RaySystem(scheduler=scheduler)
    system.init()

    cm = ComputeManager.create(system, numpy_compute)
    fs = FileSystem(cm)
    return ArrayApplication(cm, fs)
