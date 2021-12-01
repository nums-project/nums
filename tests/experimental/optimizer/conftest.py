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
from nums.experimental.nums_dask.dask_system import DaskSystem
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


class MockMultiNodeRaySystem(RaySystem):
    def mock_devices(self, num_nodes):
        assert len(self._available_nodes) == 1
        src_node = self._available_nodes[0]
        src_node_key = self._node_key(src_node)
        self._num_nodes = num_nodes
        self._devices = []
        for node_id in range(self._num_nodes):
            # Generate distinct device ids, but map them all to the same actual node.
            # When the function is invoked, the device id will map to the actual node.
            did = DeviceID(node_id, src_node_key, "cpu", 0)
            self._devices.append(did)
            self._device_to_node[did] = src_node


class MockMultiNodeDaskSystem(DaskSystem):
    def mock_devices(self, num_workers, workers_per_node):
        assert len(self._node_addresses) == 1
        self._num_devices = num_workers
        assert num_workers % workers_per_node == 0
        num_nodes = num_workers // workers_per_node
        assert workers_per_node == len(self._worker_addresses)
        self.workers_per_node = workers_per_node

        worker_addresses = sorted(self._worker_addresses)
        node_addr = self._node_addresses[0]
        self._node_addresses = []
        self._worker_addresses = []
        self._node_to_worker = {}
        self._devices = []
        for node_id in range(num_nodes):
            mock_node_addr = "mock." + node_addr
            self._node_addresses.append(mock_node_addr)
            self._worker_addresses += worker_addresses
            self._node_to_worker[mock_node_addr] = {"workers": list(worker_addresses)}
            for worker_id, worker_addr in enumerate(worker_addresses):
                # What matters for DaskSystem is that worker_id maps to a valid worker_addr.
                did = DeviceID(node_id, mock_node_addr, "cpu", worker_id)
                self._devices.append(did)


class MockCyclicDeviceGrid(CyclicDeviceGrid):
    def __init__(self, grid_shape, device_type, device_ids):
        self.grid_shape = grid_shape
        self.device_type = device_type
        self.device_grid: np.ndarray = np.empty(shape=self.grid_shape, dtype=object)

        # Skip tests.
        device_ids = self._order_device_ids(device_ids)

        for i, cluster_entry in enumerate(self.get_grid_entry_iterator()):
            device_id: DeviceID = device_ids[i]
            self.device_grid[cluster_entry] = device_id


def mock_dask_cluster(cluster_shape):
    workers_per_node = 4
    for dim in cluster_shape:
        assert dim == 1 or dim % workers_per_node == 0
    system: MockMultiNodeDaskSystem = MockMultiNodeDaskSystem(num_devices=workers_per_node,
                                                              num_cpus=workers_per_node)
    system.init()
    system.mock_devices(np.product(cluster_shape), workers_per_node=workers_per_node)
    device_grid: DeviceGrid = MockCyclicDeviceGrid(cluster_shape, "cpu", system.devices())
    cm = ComputeManager.create(system, numpy_compute, device_grid)
    fs = FileSystem(cm)
    return ArrayApplication(cm, fs)


def mock_ray_cluster(cluster_shape):
    system: MockMultiNodeRaySystem = MockMultiNodeRaySystem(use_head=True)
    system.init()
    system.mock_devices(np.product(cluster_shape))
    device_grid: DeviceGrid = MockCyclicDeviceGrid(cluster_shape, "cpu", system.devices())
    cm = ComputeManager.create(system, numpy_compute, device_grid)
    fs = FileSystem(cm)
    return ArrayApplication(cm, fs)


def destroy_mock_cluster(app: ArrayApplication):
    app.cm.system.shutdown()
    ComputeManager.destroy()


@pytest.fixture(scope="function", params=[(1, 1)])
def app_inst_mock_none(request):
    app_inst = mock_ray_cluster(request.param)
    yield app_inst
    destroy_mock_cluster(app_inst)


@pytest.fixture(scope="function", params=[(10, 1)])
def app_inst_mock_big(request):
    app_inst = mock_ray_cluster(request.param)
    yield app_inst
    destroy_mock_cluster(app_inst)


@pytest.fixture(scope="function", params=[(4, 1)])
def app_inst_mock_small(request):
    app_inst = mock_ray_cluster(request.param)
    yield app_inst
    destroy_mock_cluster(app_inst)


@pytest.fixture(scope="function", params=[(8, 1)])
def app_inst_mock_dask(request):
    app_inst = mock_dask_cluster(request.param)
    yield app_inst
    destroy_mock_cluster(app_inst)
