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

from nums.core.grid.grid import Device


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
        devices: List[Device],
        counter: Counter = None,
        created_on_only=False,
        local_transfer_coeff=1e-4,
    ):
        self.num_ops = 0
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
        self.devices: List[Device] = devices
        self.num_devices: int = len(self.devices)
        self.node_ids: List[int] = sorted(
            list(set([device.node_id for device in self.devices]))
        )
        self.num_nodes: int = len(self.node_ids)
        self.workers_per_node: int = self.num_devices // self.num_nodes
        # Check assumptions.
        workers_per_node = {node_id: 0 for node_id in self.node_ids}
        for device in self.devices:
            workers_per_node[device.node_id] += 1
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
        self.block_devices: [int, List[Device]] = {}

    def copy(self):
        new_cluster = ClusterState(self.devices, self.counter.copy())
        # Copy resources.
        new_cluster.resources = self.resources.copy()
        # Copy blocks.
        for block_id in self.block_sizes:
            # Don't copy blocks themselves. Updating references is enough.
            new_cluster.block_sizes[block_id] = self.block_sizes[block_id]
            new_cluster.block_devices[block_id] = list(self.block_devices[block_id])
        return new_cluster

    def add_resource_load(
        self, resources: np.ndarray, resource_idx: int, device: Device, value
    ):
        device_idx = self.workers_per_node * device.node_id + device.device
        resources[resource_idx][device_idx] += value
        return resources

    # Block Ops.

    def add_block(self, block_id: int, block_size: int, devices: List[Device]):
        # This is a strong assertion and may not make sense once this class is fully integrated.
        assert block_id not in self.block_sizes and block_id not in self.block_devices
        self.block_sizes[block_id] = block_size
        self.block_devices[block_id] = devices

    def _get_block_size(self, block_id: int) -> int:
        return self.block_sizes[block_id]

    def get_block_devices(self, block_id: int):
        return self.block_devices[block_id]

    def union_devices(self, block_id_a: int, block_id_b: int):
        block_a_devices = self.get_block_devices(block_id_a)
        block_b_devices = self.get_block_devices(block_id_b)
        return list(set(block_a_devices).union(set(block_b_devices)))

    def mutual_devices(self, block_id_a: int, block_id_b: int):
        block_a_devices = self.get_block_devices(block_id_a)
        block_b_devices = self.get_block_devices(block_id_b)
        return list(set(block_a_devices).intersection(set(block_b_devices)))

    def blocks_local(self, block_id_a: int, block_id_b: int):
        return len(self.mutual_devices(block_id_a, block_id_b)) > 0

    def init_mem_load(self, device: Device, block_id: int):
        size: int = self._get_block_size(block_id)
        block_devices: list = self.get_block_devices(block_id)
        assert device in block_devices
        self.add_resource_load(self.resources, self.mem_idx, device, size)
        # self.resources[self.mem_idx][device.node_id] += size

    def simulate_copy_block(
        self, block_id: int, to_device: Device, resources: np.ndarray
    ):
        size: int = self._get_block_size(block_id)
        devices: List[Device] = self.get_block_devices(block_id)
        if to_device in devices:
            return resources

        # Pick the device which minimizes cost.
        # We can prevent this from happening by setting created_on_only.
        from_device: Device = devices[0]
        for device in devices:
            if device.node_id == to_device.node_id:
                from_device = device
                break
        for device in devices:
            if (
                device.node_id == to_device.node_id
                and device.device == to_device.device
            ):
                from_device = device
                break

        # Update load.
        transfer_cost = size
        if from_device.node_id == to_device.node_id:
            # This is a local (intra-node) object transfer.
            # We account for the speedup by reducing the cost
            # by a factor of local_transfer_coeff.
            transfer_cost *= self.local_transfer_coeff
        self.add_resource_load(resources, self.net_out_idx, from_device, transfer_cost)
        self.add_resource_load(resources, self.net_in_idx, to_device, transfer_cost)
        self.add_resource_load(resources, self.mem_idx, to_device, size)
        # resources[self.net_out_idx][from_device.device] += size
        # resources[self.net_in_idx][to_device.device] += size
        # resources[self.mem_idx][to_device.device] += size
        return resources

    def simulate_op(
        self,
        op_mem: int,
        block_id_a: int,
        block_id_b: int,
        device: Device,
        resources: np.ndarray,
    ):
        if device not in self.get_block_devices(block_id_a):
            resources = self.simulate_copy_block(block_id_a, device, resources)
        if device not in self.get_block_devices(block_id_b):
            resources = self.simulate_copy_block(block_id_b, device, resources)
        self.add_resource_load(resources, self.mem_idx, device, op_mem)
        # resources[self.mem_idx][device.device] += op_mem
        return resources

    def commit_copy_block(self, block_id: int, to_device: Device):
        self.resources = self.simulate_copy_block(block_id, to_device, self.resources)
        # Update node location.
        if not self.created_on_only:
            block_devices: List[Device] = self.get_block_devices(block_id)
            block_devices.append(to_device)

    def commit_op(self, op_mem: int, block_id_a: int, block_id_b: int, device: Device):
        self.num_ops += 1
        if device not in self.get_block_devices(block_id_a):
            self.commit_copy_block(block_id_a, device)
        if device not in self.get_block_devices(block_id_b):
            self.commit_copy_block(block_id_b, device)
        self.add_resource_load(self.resources, self.mem_idx, device, op_mem)
        # self.resources[self.mem_idx][device.device] += op_mem

    def simulate_uop(
        self, op_mem: int, block_id: int, device: Device, resources: np.ndarray
    ):
        if device not in self.get_block_devices(block_id):
            resources = self.simulate_copy_block(block_id, device, resources)
        self.add_resource_load(resources, self.mem_idx, device, op_mem)
        # resources[self.mem_idx][device.device] += op_mem
        return resources

    def commit_uop(self, op_mem: int, block_id: int, device: Device):
        self.num_ops += 1
        if device not in self.get_block_devices(block_id):
            self.commit_copy_block(block_id, device)
        self.add_resource_load(self.resources, self.mem_idx, device, op_mem)
        # self.resources[self.mem_idx][device.device] += op_mem

    def simulate_nary_op(
        self,
        op_mem: int,
        block_ids: List[int],
        device: Device,
        resources: np.ndarray,
    ):
        for block_id in block_ids:
            if device not in self.get_block_devices(block_id):
                resources = self.simulate_copy_block(block_id, device, resources)
        resources[self.mem_idx][device.node_id] += op_mem
        return resources

    def commit_nary_op(self, op_mem: int, block_ids: List[int], device: Device):
        self.num_ops += 1
        for block_id in block_ids:
            if device not in self.get_block_devices(block_id):
                self.commit_copy_block(block_id, device)
        self.add_resource_load(self.resources, self.mem_idx, device, op_mem)
        # self.resources[self.mem_idx][device.node_id] += op_mem
