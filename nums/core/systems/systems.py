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


import logging
from types import FunctionType
from typing import Any, Union, List, Dict

import ray

from nums.core.grid.grid import DeviceID
from nums.core.systems.system_interface import SystemInterface
from nums.core.systems.utils import get_private_ip, get_num_cores
from nums.core import settings


class SerialSystem(SystemInterface):
    def __init__(self):
        self._remote_functions: dict = {}

    def init(self):
        pass

    def shutdown(self):
        pass

    def put(self, value: Any):
        return value

    def get(self, object_ids: Union[Any, List]):
        return object_ids

    def remote(self, function: FunctionType, remote_params: dict):
        return function

    def devices(self):
        return [DeviceID(0, "localhost", "cpu", 0)]

    def register(self, name: str, func: callable, remote_params: dict = None):
        if name in self._remote_functions:
            return
        if remote_params is None:
            remote_params = {}
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device_id: DeviceID, options: Dict):
        return self._remote_functions[name](*args, **kwargs)

    def num_cores_total(self):
        return int(get_num_cores())


class RaySystem(SystemInterface):
    # pylint: disable=abstract-method
    """
    Implements SystemInterface for Ray.
    """

    def __init__(self, use_head=False, num_nodes=None):
        self._use_head = use_head
        self._num_nodes = num_nodes
        self._manage_ray = True
        self._remote_functions = {}
        self._available_nodes = []
        self._head_node = None
        self._worker_nodes = []
        self._devices: List[DeviceID] = []
        self._device_to_node: Dict[DeviceID, Dict] = {}

    def init(self):
        if ray.is_initialized():
            self._manage_ray = False
        if self._manage_ray:
            ray.init()
        # Compute available nodes, based on CPU resource.
        if settings.head_ip is None:
            # TODO (hme): Have this be a class argument vs. using what's set in settings directly.
            logging.getLogger(__name__).info("Using driver node ip as head node.")
            head_ip = get_private_ip()
        else:
            head_ip = settings.head_ip
        total_cpus = 0
        nodes = ray.nodes()
        for node in nodes:
            node_ip = self._node_ip(node)
            if head_ip == node_ip:
                logging.getLogger(__name__).info("head node %s", node_ip)
                self._head_node = node
            elif self._has_cpu_resources(node):
                logging.getLogger(__name__).info("worker node %s", node_ip)
                total_cpus += node["Resources"]["CPU"]
                self._worker_nodes.append(node)
                self._available_nodes.append(node)
        if self._head_node is None:
            if self._use_head:
                logging.getLogger(__name__).warning(
                    "Failed to determine which node is the head."
                    " The head node will be used even though"
                    " nums.core.settings.use_head = False."
                )
        elif self._use_head and self._has_cpu_resources(self._head_node):
            total_cpus += self._head_node["Resources"]["CPU"]
            self._available_nodes.append(self._head_node)
        logging.getLogger(__name__).info("total cpus %s", total_cpus)

        if self._num_nodes is None:
            self._num_nodes = len(self._available_nodes)
        assert self._num_nodes <= len(self._available_nodes)

        self.init_devices()

    def init_devices(self):
        self._devices = []
        for node_id in range(self._num_nodes):
            node = self._available_nodes[node_id]
            did = DeviceID(node_id, self._node_key(node), "cpu", 1)
            self._devices.append(did)
            self._device_to_node[did] = node

    def _has_cpu_resources(self, node):
        return self._node_cpu_resources(node) > 0.0

    def _node_cpu_resources(self, node):
        return node["Resources"]["CPU"] if "CPU" in node["Resources"] else 0.0

    def _node_key(self, node):
        node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
        assert len(node_key) == 1
        return node_key[0]

    def _node_ip(self, node):
        return self._node_key(node).split(":")[1]

    def shutdown(self):
        if self._manage_ray:
            ray.shutdown()

    def warmup(self, n: int):
        # Quick warm-up. Useful for quick and more accurate testing.
        if n > 0:
            assert n < 10 ** 6

            def warmup_func(n):
                # pylint: disable=import-outside-toplevel
                import random

                r = ray.remote(num_cpus=1)(lambda x, y: x + y).remote
                for _ in range(n):
                    _a = random.randint(0, 1000)
                    _b = random.randint(0, 1000)
                    _v = self.get(r(self.put(_a), self.put(_b)))

            warmup_func(n)

    def put(self, value):
        return ray.put(value)

    def get(self, object_ids):
        return ray.get(object_ids)

    def remote(self, function: FunctionType, remote_params: dict):
        r = ray.remote(num_cpus=1, **remote_params)
        return r(function)

    def register(self, name: str, func: callable, remote_params: dict = None):
        if name in self._remote_functions:
            return
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device_id: DeviceID, options: Dict):
        if device_id is not None:
            node = self._device_to_node[device_id]
            node_key = self._node_key(node)
            if "resources" in options:
                assert node_key not in options
            options["resources"] = {node_key: 1.0 / 10 ** 4}
        return self._remote_functions[name].options(**options).remote(*args, **kwargs)

    def devices(self):
        return self._devices

    def num_cores_total(self):
        num_cores = sum(
            map(lambda n: n["Resources"]["CPU"], self._device_to_node.values())
        )
        return int(num_cores)


class RaySystemStockScheduler(RaySystem):
    """
    An implementation of the Ray system which ignores scheduling commands given
    by the caller. For testing only.
    """

    def call(self, name: str, args, kwargs, device_id: DeviceID, options: Dict):
        if device_id is not None:
            node = self._device_to_node[device_id]
            node_key = self._node_key(node)
            if "resources" in options:
                assert node_key not in options
        return self._remote_functions[name].options(**options).remote(*args, **kwargs)
