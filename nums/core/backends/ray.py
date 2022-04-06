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
import warnings
import logging
from types import FunctionType
from typing import Any, List, Dict, Optional

import ray

from nums.core import settings
from nums.core.grid.grid import Device

from .base import Backend
from .utils import get_private_ip, get_num_cores


class RayBackend(Backend):
    # pylint: disable=abstract-method
    """
    Implements backend for Ray.
    """

    def __init__(
        self,
        address: Optional[str] = None,
        use_head: bool = False,
        num_nodes: Optional[int] = None,
        num_cpus: Optional[int] = None,
    ):
        self._address: str = address
        self._use_head = use_head
        self._num_nodes = num_nodes
        self.num_cpus = int(get_num_cores()) if num_cpus is None else num_cpus
        self._manage_ray = True
        self._remote_functions = {}
        self._actors: dict = {}
        self._actor_node_index = 0
        self._available_nodes = []
        self._head_node = None
        self._worker_nodes = []
        self._devices: List[Device] = []
        self._device_to_node: Dict[Device, Dict] = {}

    def init(self):
        if ray.is_initialized():
            self._manage_ray = False
        if self._manage_ray:
            if self._address is None:
                ray.init(num_cpus=self.num_cpus)
            else:
                ray.init(address=self._address)
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
        self._device_to_node = {}
        for node_id in range(self._num_nodes):
            node = self._available_nodes[node_id]
            did = Device(node_id, self._node_key(node), "cpu", 1)
            self._devices.append(did)
            self._device_to_node[did] = node

    def _has_cpu_resources(self, node: dict) -> bool:
        return self._node_cpu_resources(node) > 0.0

    def _node_cpu_resources(self, node: dict) -> float:
        return node["Resources"]["CPU"] if "CPU" in node["Resources"] else 0.0

    def _node_key(self, node: dict) -> str:
        node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
        assert len(node_key) == 1
        return node_key[0]

    def _node_ip(self, node: dict) -> str:
        return self._node_key(node).split(":")[1]

    def shutdown(self):
        if self._manage_ray:
            ray.shutdown()

    def warmup(self, n: int):
        # Quick warm-up. Useful for quick and more accurate testing.
        if n > 0:
            assert n < 10**6

            def warmup_func(n):
                # pylint: disable=import-outside-toplevel
                import random

                r = ray.remote(num_cpus=1)(lambda x, y: x + y).remote

                num_devices = len(self._devices)
                for i in range(n):
                    _a = random.randint(0, 1000)
                    _b = random.randint(0, 1000)
                    d0 = i % num_devices
                    d1 = (i + 1) % num_devices
                    _v = self.get(
                        r(
                            self.put(_a, self._devices[d0]),
                            self.put(_b, self._devices[d1]),
                        )
                    )

            warmup_func(n)

    def put(self, value: Any, device: Device):
        return self.call("identity", [value], {}, device, {})

    def get(self, object_ids):
        return ray.get(object_ids)

    def remote(self, function: FunctionType, remote_params: dict):
        r = ray.remote(num_cpus=1, **remote_params)
        return r(function)

    def register(self, name: str, func: callable, remote_params: dict = None):
        if name in self._remote_functions:
            return
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device: Device, options: Dict):
        if device is not None:
            node = self._device_to_node[device]
            node_key = self._node_key(node)
            if "resources" in options:
                assert node_key not in options
            options["resources"] = {node_key: 1.0 / 10**4}
        return self._remote_functions[name].options(**options).remote(*args, **kwargs)

    def devices(self) -> List[Device]:
        return self._devices

    def num_cores_total(self) -> int:
        num_cores = sum(
            map(lambda n: n["Resources"]["CPU"], self._device_to_node.values())
        )
        return int(num_cores)

    def register_actor(self, name: str, cls: type):
        if name in self._actors:
            warnings.warn(
                "Actor %s has already been registered. "
                "Overwriting with %s." % (name, cls.__name__)
            )
            return
        self._actors[name] = ray.remote(cls)

    def make_actor(self, name: str, *args, device: Device = None, **kwargs):
        # Distribute actors round-robin over devices.
        if device is None:
            device = self._devices[self._actor_node_index]
            self._actor_node_index = (self._actor_node_index + 1) % len(self._devices)
        actor = self._actors[name]
        node = self._device_to_node[device]
        node_key = self._node_key(node)
        options = {"resources": {node_key: 1.0 / 10**4}}
        return actor.options(**options).remote(*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        return getattr(actor, method).remote(*args, **kwargs)


class RayBackendStockScheduler(RayBackend):
    """
    An implementation of the Ray backend which ignores scheduling commands given
    by the caller. For testing only.
    """

    def call(self, name: str, args, kwargs, device: Device, options: Dict):
        if device is not None:
            node = self._device_to_node[device]
            node_key = self._node_key(node)
            if "resources" in options:
                assert node_key not in options["resources"]
        return self._remote_functions[name].options(**options).remote(*args, **kwargs)

    def make_actor(
        self, name: str, *args, device: Device = None, **kwargs
    ):  # pylint: disable=unused-argument
        actor = self._actors[name]
        return actor.remote(*args, **kwargs)
