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


import numpy as np

from nums.core.storage.storage import StoredArray
from nums.core.systems import numpy_compute
from nums.core.systems.systems import System, RaySystem
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.schedulers import RayScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication, BlockArray


def serial_read(container_name, name, store_cls):
    sa: StoredArray = store_cls(container_name, name)
    sa.init_grid()
    return sa.get_array()


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
        print("cluster_shape", self.cluster_shape)
        print("cluster_grid", self.cluster_grid)


def mock_cluster(cluster_shape):
    scheduler: RayScheduler = MockMultiNodeScheduler(compute_module=numpy_compute,
                                                     cluster_shape=cluster_shape,
                                                     use_head=True)
    system: System = RaySystem(compute_module=numpy_compute,
                               scheduler=scheduler)
    system.init()
    return ArrayApplication(system=system, filesystem=FileSystem(system))
