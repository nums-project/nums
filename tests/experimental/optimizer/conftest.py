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
import pytest

from nums.core.array.application import ArrayApplication, BlockArray
from nums.core.compute import numpy_compute
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import DeviceGrid, CyclicDeviceGrid, DeviceID
from nums.core.storage.storage import StoredArray
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.systems import RaySystem
from nums.experimental.optimizer.grapharray import GraphArray
from nums.experimental.optimizer.tree_search import RandomTS

rs = np.random.RandomState(1337)


def compute_graph_array(ga: GraphArray) -> BlockArray:
    result_ga: GraphArray = RandomTS(
        seed=rs, max_samples_per_step=1, max_reduction_pairs=1, force_final_action=True
    ).solve(ga)
    result_ga.grid, result_ga.to_blocks()
    return BlockArray(result_ga.grid, ComputeManager.instance, result_ga.to_blocks())


def collapse_graph_array(ga: GraphArray) -> GraphArray:
    return RandomTS(
        seed=rs, max_samples_per_step=1, max_reduction_pairs=1, force_final_action=True
    ).solve(ga)


def serial_read(container_name, name, store_cls):
    sa: StoredArray = store_cls(container_name, name)
    sa.init_grid()
    return sa.get_array()


def check_block_integrity(arr: BlockArray):
    for grid_entry in arr.grid.get_entry_iterator():
        assert arr.blocks[grid_entry].grid_entry == grid_entry
        assert arr.blocks[grid_entry].rect == arr.grid.get_slice_tuples(grid_entry)
        assert arr.blocks[grid_entry].shape == arr.grid.get_block_shape(grid_entry)


class MockMultiNodeSystem(RaySystem):
    def mock_devices(self, num_nodes):
        assert len(self._available_nodes) == 1
        src_node = self._available_nodes[0]
        src_node_key = self._node_key(src_node)
        self._num_nodes = num_nodes
        self._devices = []
        for node_id in range(self._num_nodes):
            # Generate distinct device ids, but map them all to the same actual node.
            # When the function is invoked, the device id will map to the actual node.
            did = DeviceID(node_id, src_node_key, "cpu", 1)
            self._devices.append(did)
            self._device_to_node[did] = src_node


def mock_cluster(cluster_shape):
    system: MockMultiNodeSystem = MockMultiNodeSystem(use_head=True)
    system.init()
    system.mock_devices(np.product(cluster_shape))
    device_grid: DeviceGrid = CyclicDeviceGrid(cluster_shape, "cpu", system.devices())
    cm = ComputeManager.create(system, numpy_compute, device_grid)
    fs = FileSystem(cm)
    return ArrayApplication(cm, fs)


def destroy_mock_cluster(app: ArrayApplication):
    app.cm.system.shutdown()
    ComputeManager.destroy()


@pytest.fixture(scope="function", params=[(1, 1)])
def app_inst_mock_none(request):
    app_inst = mock_cluster(request.param)
    yield app_inst
    destroy_mock_cluster(app_inst)


@pytest.fixture(scope="function", params=[(10, 1)])
def app_inst_mock_big(request):
    app_inst = mock_cluster(request.param)
    yield app_inst
    destroy_mock_cluster(app_inst)


@pytest.fixture(scope="function", params=[(4, 1)])
def app_inst_mock_small(request):
    app_inst = mock_cluster(request.param)
    yield app_inst
    destroy_mock_cluster(app_inst)
