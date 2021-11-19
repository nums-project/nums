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
import warnings
from types import FunctionType
from typing import Any, Union, List, Dict, Optional

try:
    import dask
    from dask.distributed import Client
except Exception as e:
    raise Exception(
        "Unable to import dask. Install dask with command 'pip install dask[complete]'"
    ) from e
from nums.core.grid.grid import DeviceID
from nums.core.systems.system_interface import SystemInterface
from nums.core.systems.utils import get_num_cores


# pylint: disable = unused-argument


class DaskSystem(SystemInterface):
    def __init__(
        self,
        address: Optional[str] = None,
        num_nodes: Optional[int] = None,
        num_cpus: Optional[int] = None,
    ):
        self._address: str = address
        self._num_nodes: int = num_nodes
        self.num_cpus: int = int(get_num_cores()) if num_cpus is None else num_cpus

        self._client: Client = None
        self._remote_functions: dict = {}
        self._actors: dict = {}
        self._actor_node_index = 0
        self._worker_addresses = []
        self._node_addresses = []
        self._devices = []
        self._device_to_node: Dict[DeviceID, Dict] = {}

    def init(self):
        if self._address is None:
            self._client = Client(
                n_workers=self.num_cpus, processes=False, memory_limit=0
            )
        else:
            self._client = Client(address=self._address)
        self.init_devices()

    def init_devices(self):
        null_op = self._client.run(lambda: None)
        self._worker_addresses = sorted(
            list(map(lambda addr: addr.split("://")[-1], null_op.keys()))
        )
        self._node_addresses = sorted(
            list(set(map(lambda addr: addr.split(":")[0], self._worker_addresses)))
        )
        # Remove any trailing slashes.
        self._node_addresses = list(
            map(lambda x: x.split("/")[0], self._node_addresses)
        )

        self._devices = []
        self._device_to_node = {}
        for node_id in range(self._num_nodes):
            node_addr = self._node_addresses[node_id]
            logging.getLogger(__name__).info("worker node %s", node_addr)
            did = DeviceID(node_id, node_addr, "cpu", 1)
            self._devices.append(did)
            self._device_to_node[did] = node_addr
        logging.getLogger(__name__).info("total cpus %s", len(self._worker_addresses))

    def shutdown(self):
        if self._address is None:
            # Kill the scheduler and workers.
            self._client.shutdown()
        else:
            # Disconnect this client.
            self._client.close()
        self._client = None

    def put(self, value: Any):
        return self._client.submit(lambda x: x, value)

    def get(self, object_ids: Union[Any, List]):
        return self._client.gather(object_ids)

    def remote(self, function: FunctionType, remote_params: dict):
        return function

    def devices(self) -> List[DeviceID]:
        return self._devices

    def register(self, name: str, func: callable, remote_params: dict = None):
        if name in self._remote_functions:
            return
        if remote_params is None:
            remote_params = {}
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device_id: DeviceID, options: Dict):
        assert device_id is not None
        node_addr = self._device_to_node[device_id]
        func = self._remote_functions[name]
        if "num_returns" in options:
            dfunc = dask.delayed(func, nout=options["num_returns"])
            result = dfunc(*args, **kwargs)
            return self._client.compute(result, workers=node_addr)
        else:
            return self._client.submit(func, *args, **kwargs, workers=node_addr)

    def num_cores_total(self) -> int:
        num_cores = 0
        for node_addr in list(self._device_to_node.values()):
            for worker_addr in self._worker_addresses:
                if node_addr in worker_addr:
                    num_cores += 1
        return num_cores

    def register_actor(self, name: str, cls: type):
        if name in self._actors:
            warnings.warn(
                "Actor %s has already been registered. "
                "Overwriting with %s." % (name, cls.__name__)
            )
            return
        self._actors[name] = cls

    def make_actor(self, name: str, *args, device_id: DeviceID = None, **kwargs):
        # Distribute actors round-robin over devices.
        if device_id is None:
            device_id = self._devices[self._actor_node_index]
            self._actor_node_index = (self._actor_node_index + 1) % len(self._devices)
        actor = self._actors[name]
        node_addr = self._device_to_node[device_id]
        # Do something with class and node address...
        raise NotImplementedError("DaskSystem currently does not support Actors.")

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        return getattr(actor, method).remote(*args, **kwargs)


class DaskSystemStockScheduler(DaskSystem):
    """
    An implementation of the Dask system which ignores scheduling commands given
    by the caller. For testing only.
    """

    def call(self, name: str, args, kwargs, device_id: DeviceID, options: Dict):
        func = self._remote_functions[name]
        return self._client.submit(func, *args, **kwargs)

    def make_actor(self, name: str, *args, device_id: DeviceID = None, **kwargs):
        raise NotImplementedError(
            "DaskSystemStockScheduler currently does not support Actors."
        )
