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

import numpy as np

from nums.core.grid.grid import DeviceID


class Counter(object):
    def __init__(self):
        self.n = -1

    def __call__(self):
        self.n += 1
        return self.n

    def copy(self):
        c_copy = Counter()
        c_copy.n = self.n
        return c_copy


class ClusterState(object):
    def __init__(
        self,
        device_ids: List[DeviceID],
        counter: Counter = None,
        created_on_only=False,
        local_transfer_coeff=0.1,
    ):
        self.created_on_only = created_on_only
        # Intra-node transfers are faster than inter-node transfers.
        # It's critical we account for this when objects need to be transferred within a node.
        # The value below is the fraction of network load to use for local transfers.
        # By default, assume an order of magnitude speedup for local transfers.
        self.local_transfer_coeff = local_transfer_coeff
        if counter is None:
            self.counter = Counter()
        else:
            self.counter = counter
        self.device_ids: List[DeviceID] = device_ids
        self.num_devices: int = len(self.device_ids)
        self.node_ids: List[int] = sorted(
            list(set([device_id.node_id for device_id in self.device_ids]))
        )
        self.num_nodes: int = len(self.node_ids)
        self.workers_per_node: int = self.num_devices // self.num_nodes
        # Check assumptions.
        workers_per_node = {node_id: 0 for node_id in self.node_ids}
        for device_id in self.device_ids:
            workers_per_node[device_id.node_id] += 1
        for node_id, count in workers_per_node.items():
            assert self.workers_per_node == count

        # The system instance on which we perform operations.
        # This is exposed for use by nodes only.
        # 3 matrices: mem, net_in, net_out.
        self.mem_idx, self.net_in_idx, self.net_out_idx = 0, 1, 2
        self.resources: np.ndarray = np.zeros(shape=(3, self.num_devices), dtype=float)
        # Dict from block id to block size.
        self.block_sizes: [int, int] = {}
        # Dict from block id to list of device id.
        self.block_devices: [int, List[DeviceID]] = {}

    def copy(self):
        new_cluster = ClusterState(self.device_ids, self.counter.copy())
        # Copy resources.
        new_cluster.resources = self.resources.copy()
        # Copy blocks.
        for block_id in self.block_sizes:
            # Don't copy blocks themselves. Updating references is enough.
            new_cluster.block_sizes[block_id] = self.block_sizes[block_id]
            new_cluster.block_devices[block_id] = list(self.block_devices[block_id])
        return new_cluster

    def add_resource_load(
        self, resources: np.ndarray, resource_idx: int, device_id: DeviceID, value
    ):
        device_idx = self.workers_per_node * device_id.node_id + device_id.device_id
        resources[resource_idx][device_idx] += value
        return resources

    # Block Ops.

    def add_block(self, block_id: int, block_size: int, device_ids: List[DeviceID]):
        # This is a strong assertion and may not make sense once this class is fully integrated.
        assert block_id not in self.block_sizes and block_id not in self.block_devices
        self.block_sizes[block_id] = block_size
        self.block_devices[block_id] = device_ids

    def _get_block_size(self, block_id: int) -> int:
        return self.block_sizes[block_id]

    def get_block_device_ids(self, block_id: int):
        return self.block_devices[block_id]

    def union_devices(self, block_id_a: int, block_id_b: int):
        block_a_device_ids = self.get_block_device_ids(block_id_a)
        block_b_device_ids = self.get_block_device_ids(block_id_b)
        return list(set(block_a_device_ids).union(set(block_b_device_ids)))

    def mutual_devices(self, block_id_a: int, block_id_b: int):
        block_a_device_ids = self.get_block_device_ids(block_id_a)
        block_b_device_ids = self.get_block_device_ids(block_id_b)
        return list(set(block_a_device_ids).intersection(set(block_b_device_ids)))

    def blocks_local(self, block_id_a: int, block_id_b: int):
        return len(self.mutual_devices(block_id_a, block_id_b)) > 0

    def init_mem_load(self, device_id: DeviceID, block_id: int):
        size: int = self._get_block_size(block_id)
        block_device_ids: list = self.get_block_device_ids(block_id)
        assert device_id in block_device_ids
        self.add_resource_load(self.resources, self.mem_idx, device_id, size)
        # self.resources[self.mem_idx][device_id.node_id] += size

    def simulate_copy_block(
        self, block_id: int, to_device_id: DeviceID, resources: np.ndarray
    ):
        size: int = self._get_block_size(block_id)
        device_ids: List[DeviceID] = self.get_block_device_ids(block_id)
        if to_device_id in device_ids:
            return resources
        # Pick the first device. This is the worst-case assumption,
        # since it imposes the greatest load (w.r.t. cost function) on the network,
        # though we really don't have control over this.
        from_device_id: DeviceID = device_ids[0]
        # Update load.
        transfer_cost = size
        if from_device_id.node_id == to_device_id.node_id:
            # This is a local (intra-node) object transfer.
            # We account for the speedup by reducing the cost
            # by a factor of local_transfer_coeff.
            transfer_cost *= self.local_transfer_coeff
        self.add_resource_load(
            resources, self.net_out_idx, from_device_id, transfer_cost
        )
        self.add_resource_load(resources, self.net_in_idx, to_device_id, transfer_cost)
        self.add_resource_load(resources, self.mem_idx, to_device_id, size)
        # resources[self.net_out_idx][from_device_id.device_id] += size
        # resources[self.net_in_idx][to_device_id.device_id] += size
        # resources[self.mem_idx][to_device_id.device_id] += size
        return resources

    def simulate_op(
        self,
        op_mem: int,
        block_id_a: int,
        block_id_b: int,
        device_id: DeviceID,
        resources: np.ndarray,
    ):
        if device_id not in self.get_block_device_ids(block_id_a):
            resources = self.simulate_copy_block(block_id_a, device_id, resources)
        if device_id not in self.get_block_device_ids(block_id_b):
            resources = self.simulate_copy_block(block_id_b, device_id, resources)
        self.add_resource_load(resources, self.mem_idx, device_id, op_mem)
        # resources[self.mem_idx][device_id.device_id] += op_mem
        return resources

    def commit_copy_block(self, block_id: int, to_device_id: DeviceID):
        self.resources = self.simulate_copy_block(
            block_id, to_device_id, self.resources
        )
        # Update node location.
        if not self.created_on_only:
            block_device_ids: List[DeviceID] = self.get_block_device_ids(block_id)
            block_device_ids.append(to_device_id)

    def commit_op(
        self, op_mem: int, block_id_a: int, block_id_b: int, device_id: DeviceID
    ):
        if device_id not in self.get_block_device_ids(block_id_a):
            self.commit_copy_block(block_id_a, device_id)
        if device_id not in self.get_block_device_ids(block_id_b):
            self.commit_copy_block(block_id_b, device_id)
        self.add_resource_load(self.resources, self.mem_idx, device_id, op_mem)
        # self.resources[self.mem_idx][device_id.device_id] += op_mem

    def simulate_uop(
        self, op_mem: int, block_id: int, device_id: DeviceID, resources: np.ndarray
    ):
        if device_id not in self.get_block_device_ids(block_id):
            resources = self.simulate_copy_block(block_id, device_id, resources)
        self.add_resource_load(resources, self.mem_idx, device_id, op_mem)
        # resources[self.mem_idx][device_id.device_id] += op_mem
        return resources

    def commit_uop(self, op_mem: int, block_id: int, device_id: DeviceID):
        if device_id not in self.get_block_device_ids(block_id):
            self.commit_copy_block(block_id, device_id)
        self.add_resource_load(self.resources, self.mem_idx, device_id, op_mem)
        # self.resources[self.mem_idx][device_id.device_id] += op_mem
