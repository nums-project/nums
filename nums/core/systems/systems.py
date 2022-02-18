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
from typing import Any, Union, List, Dict, Optional
from itertools import repeat
from dataclasses import dataclass

import ray

from nums.core.grid.grid import DeviceID
from nums.core.systems.system_interface import SystemInterface
from nums.core.systems.utils import get_private_ip, get_num_cores
from nums.core import settings


# pylint: disable = unused-argument


class SerialSystem(SystemInterface):
    def __init__(self, num_cpus: Optional[int] = None):
        self.num_cpus = int(get_num_cores()) if num_cpus is None else num_cpus
        self._remote_functions: dict = {}
        self._actors: dict = {}

    def init(self):
        pass

    def shutdown(self):
        pass

    def put(self, value: Any, device_id: DeviceID):
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

    def register_actor(self, name: str, cls: type):
        if name in self._actors:
            warnings.warn(
                "Actor %s has already been registered. "
                "Overwriting with %s." % (name, cls.__name__)
            )
            return
        self._actors[name] = cls

    def make_actor(self, name: str, *args, device_id: DeviceID = None, **kwargs):
        return self._actors[name](*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        return getattr(actor, method)(*args, **kwargs)

    def num_cores_total(self) -> int:
        return self.num_cpus


@dataclass
class MPIRemoteObj(object):
    rank: int


@dataclass
class MPILocalObj(object):
    value: Any


class MPISystem(SystemInterface):
    """
    Implements SystemInterface for MPI.
    """

    def __init__(self):
        # pylint: disable=import-outside-toplevel c-extension-no-member
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.proc_name: str = get_private_ip()

        self._devices: List[DeviceID] = []

        self._remote_functions: dict = {}
        self._actors: dict = {}

        # This is same as number of mpi processes.
        self.num_cpus = self.size
        # self._devices = []

        # self._device_to_node: Dict[DeviceID, int] = {}
        self._device_to_rank: Dict[DeviceID, int] = {}
        self._actor_to_rank: dict = {}
        self._actor_node_index = 0

    def init(self):
        self.init_devices()

    def init_devices(self):
        # TODO: sort proc_names and don't do an all-gather for `did`. Construct `did` locally.
        proc_names = list(set(self.comm.allgather(self.proc_name)))
        did = DeviceID(
            proc_names.index(self.proc_name), self.proc_name, "cpu", self.rank
        )
        self._device_to_rank[did] = self.rank
        self._devices = self.comm.allgather(did)
        self._device_to_rank = self.comm.allgather({did: self.rank})
        self._device_to_rank = dict(
            (key, val) for k in self._device_to_rank for key, val in k.items()
        )

    def shutdown(self):
        pass

    # TODO: this is scatter. (Document this)
    def put(self, value: Any, device_id: DeviceID):
        dest_rank = self._device_to_rank[device_id]
        assert not isinstance(value, (MPILocalObj, MPIRemoteObj))
        if self.rank == dest_rank:
            return MPILocalObj(value)
        else:
            return MPIRemoteObj(dest_rank)

    def get(self, object_ids: Union[Any, List]):
        resolved_object_ids = []
        if not isinstance(object_ids, (MPIRemoteObj, MPILocalObj)):
            for obj in object_ids:
                if isinstance(obj, MPIRemoteObj):
                    dest_rank = obj.get_dest_rank()
                # This should be true for just one rank which has the data.
                else:
                    dest_rank = self.rank
                # TODO: see if all-2-all might be more efficient.
                obj = self.comm.bcast(obj, root=dest_rank)
                obj_value = obj.value
                assert not isinstance(obj_value, (MPILocalObj, MPIRemoteObj))
                resolved_object_ids.append(obj_value)
            return resolved_object_ids
        else:
            obj = object_ids
            if isinstance(obj, MPIRemoteObj):
                dest_rank = obj.get_dest_rank()
            # This should be true for just one rank which has the data.
            else:
                dest_rank = self.rank
            # TODO: see if all-2-all might be more efficient.
            obj = self.comm.bcast(obj, root=dest_rank)
            assert not isinstance(obj.value, (MPILocalObj, MPIRemoteObj))
            return obj.value

    def remote(self, function: FunctionType, remote_params: dict):
        return function, remote_params

    def devices(self) -> List[DeviceID]:
        return self._devices

    def register(self, name: str, func: callable, remote_params: dict = None):
        if name in self._remote_functions:
            return
        if remote_params is None:
            remote_params = {}
        self._remote_functions[name] = self.remote(func, remote_params)

    def _parse_call(self, name: str, options: Dict):
        func, remote_params = self._remote_functions[name]
        nout = 1
        if "num_returns" in options:
            # options has higher priority than remote_params.
            if options["num_returns"] > 1:
                nout = options["num_returns"]
        elif "num_returns" in remote_params and remote_params["num_returns"] > 1:
            nout = remote_params["num_returns"]
        return func, nout

    def call(self, name: str, args, kwargs, device_id: DeviceID, options: Dict):
        dest_rank = self._device_to_rank[device_id]
        for arg in args:
            if isinstance(arg, MPILocalObj):
                assert not isinstance(arg.value, MPILocalObj)
        resolved_args = self._resolve_args(args, dest_rank)
        resolved_kwargs = self._resolve_kwargs(kwargs, dest_rank)
        func, nout = self._parse_call(name, options)
        if dest_rank == self.rank:
            result = func(*resolved_args, **resolved_kwargs)
            if nout > 1:
                return tuple([MPILocalObj(result[i]) for i in range(nout)])
            else:
                return MPILocalObj(result)
        else:
            if nout > 1:
                return tuple(repeat(MPIRemoteObj(dest_rank), nout))
            else:
                return MPIRemoteObj(dest_rank)

    def _resolve_kwargs(self, kwargs: dict, device_rank):
        # Resolve dependencies: iterate over kwargs and figure out which ones need fetching.
        assert isinstance(kwargs, dict), str(type(kwargs))
        resolved_args = {}
        for k, v in kwargs.items():
            resolved_args[k] = self._resolve_object(v, device_rank)
        return resolved_args

    def _resolve_args(self, args: Union[list, tuple], device_rank):
        # Resolve dependencies: iterate over args and figure out which ones need fetching.
        assert isinstance(args, (list, tuple)), str(type(args))
        resolved_args = []
        for arg in args:
            resolved_arg = self._resolve_object(arg, device_rank)
            resolved_args.append(resolved_arg)
        return resolved_args

    def _resolve_object(self, obj, device_rank):
        if not isinstance(obj, (MPIRemoteObj, MPILocalObj)):
            return obj
        is_remote = isinstance(obj, MPIRemoteObj)
        device_rank_object_is_remote = self.comm.bcast(is_remote, device_rank)
        if not device_rank_object_is_remote:
            if device_rank == self.rank:
                return obj.value
            else:
                # No need to do anything on other ranks when object is already on device rank.
                return obj
        # Check if obj is remote.
        elif device_rank == self.rank:
            sender_rank = obj.get_dest_rank()
            # TODO: Try Isend and Irecv and have a switch for sync and async.
            arg_value = self.comm.recv(sender_rank)
            return arg_value
        elif isinstance(obj, MPILocalObj):
            # The obj is stored on this rank, so send it to the device on which the op will be
            # executed.
            self.comm.send(obj.value, device_rank)
            return obj
        else:
            # The obj is remote and this is not the device on which we want to invoke the op.
            # Because the obj is remote, this is not the sender.
            return obj

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
        dest_rank = self._device_to_rank[device_id]
        resolved_args = self._resolve_args(args, dest_rank)
        resolved_kwargs = self._resolve_kwargs(kwargs, dest_rank)
        if dest_rank == self.rank:
            actor_obj = actor(*resolved_args, **resolved_kwargs)
            actor_id = id(actor_obj)
            self._actor_to_rank[actor_id] = dest_rank
            return actor_obj
        else:
            actor_obj = MPIRemoteObj(dest_rank)
            actor_id = id(actor_obj)
            self._actor_to_rank[actor_id] = dest_rank
            return actor_obj

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        dest_rank = self._actor_to_rank[id(actor)]
        # Resolve args.
        resolved_args = self._resolve_args(args, dest_rank)
        resolved_kwargs = self._resolve_kwargs(kwargs, dest_rank)
        # Make sure it gets called on the correct rank.
        if not isinstance(actor, MPIRemoteObj):
            return getattr(actor, method)(*resolved_args, **resolved_kwargs)
        else:
            # Return an MPIRemoteObj corresponding to result of actor method call?
            return None

    def num_cores_total(self) -> int:
        return self.num_cpus


class RaySystem(SystemInterface):
    # pylint: disable=abstract-method
    """
    Implements SystemInterface for Ray.
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
        self._devices: List[DeviceID] = []
        self._device_to_node: Dict[DeviceID, Dict] = {}

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
            did = DeviceID(node_id, self._node_key(node), "cpu", 1)
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
            assert n < 10 ** 6

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

    def put(self, value: Any, device_id: DeviceID):
        return self.call("identity", [value], {}, device_id, {})

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
                assert node_key not in options["resources"]
            options["resources"] = {node_key: 1.0 / 10 ** 4}
        return self._remote_functions[name].options(**options).remote(*args, **kwargs)

    def devices(self) -> List[DeviceID]:
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

    def make_actor(self, name: str, *args, device_id: DeviceID = None, **kwargs):
        # Distribute actors round-robin over devices.
        if device_id is None:
            device_id = self._devices[self._actor_node_index]
            self._actor_node_index = (self._actor_node_index + 1) % len(self._devices)
        actor = self._actors[name]
        node = self._device_to_node[device_id]
        node_key = self._node_key(node)
        options = {"resources": {node_key: 1.0 / 10 ** 4}}
        return actor.options(**options).remote(*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        return getattr(actor, method).remote(*args, **kwargs)


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
                assert node_key not in options["resources"]
        return self._remote_functions[name].options(**options).remote(*args, **kwargs)

    def make_actor(self, name: str, *args, device_id: DeviceID = None, **kwargs):
        actor = self._actors[name]
        return actor.remote(*args, **kwargs)
