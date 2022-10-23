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
from dataclasses import dataclass
from itertools import repeat
from types import FunctionType
from typing import Any, List, Optional, Dict, Union
import warnings

import ray

from nums.core import settings
from nums.core.grid.grid import Device

from .utils import get_num_cores, get_num_gpus
from .base import Backend
from .utils import get_private_ip

try:
    import cupy as cp
    from cupy.cuda.nccl import (
        NcclCommunicator,
        NCCL_FLOAT64,
        NCCL_FLOAT32,
        NCCL_FLOAT16,
        NCCL_SUM,
        groupStart,
        groupEnd,
    )
    import os

    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["CUPY_TF32"] = "1"
    os.environ["CUPY_ACCELERATORS"] = "cub,cutensor"


except:
    pass

import numpy as np

DTYPES = {
    "float16": NCCL_FLOAT16,
    "float32": NCCL_FLOAT32,
    "float64": NCCL_FLOAT64,
    "int32": NCCL_FLOAT32,
    "int64": NCCL_FLOAT64,
}


### This is a serial gpu implementation (No communication)
class GPUSerialBackend(Backend):
    def __init__(self, num_cpus: Optional[int] = None, num_gpus: Optional[int] = None):
        import cupy as cp

        self.num_cpus = int(get_num_cores()) if num_cpus is None else num_cpus
        self._remote_functions: dict = {}
        self._actors: dict = {}
        self.num_gpus = 1
        self.cp = cp

    def init(self):
        pass

    def shutdown(self):
        # mempool = self.cp.get_default_memory_pool()
        # mempool.free_all_blocks()
        pass

    def put(self, value: Any, device: Device):
        """
        Put object into backend storage and force placement on the relevant node.
        """
        import cupy as cp
        import numpy as np

        if np.isscalar(value):
            return value

        return cp.array(value)

    def get(self, object_ids: Union[Any, List]):
        """
        Get object from backend storage.

        CuPy also uses .get() to copy to CPU memory to serve to user.
        """
        import cupy as cp
        import numpy as np

        cp.cuda.Device(0).synchronize()
        # TODO: some things in CuPy don't translate well to NumPy, clean this up in a helper function
        if type(object_ids[0]) == np.float64 or type(object_ids[0]) == int:
            return object_ids
        if isinstance(object_ids, list):
            return [
                a.get()
                for a in object_ids
                if type(a) is not bool and type(a) != np.ndarray
            ]  # TODO: clean up this case
        else:
            return object_ids.get()

    def remote(self, function: FunctionType, remote_params: Dict):
        """
        Return a callable remote function with remote_params.
        """
        return function

    def devices(self):
        return [Device(0, "localhost", "gpu", 0)]

    def register(self, name: str, func: callable, remote_params: Dict = None):
        if name in self._remote_functions:
            return
        if remote_params is None:
            remote_params = {}
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device: Device, options: Dict):
        return self._remote_functions[name](*args, **kwargs)

    # This is for sklearn, ignore for now
    def register_actor(self, name: str, cls: type):
        """
        :param name: Name of the actor. This should be unique.
        :param cls: The Python class to convert into an actor.
        :return: None
        """
        assert name not in self._actors
        self._actors[name] = cls

    def make_actor(self, name: str, *args, device: Device = None, **kwargs):
        """
        :param name: The name of the actor.
        :param args: args to pass to __init__.
        :param device: A device. This is captured by the system and not passed to __init__.
        :param kwargs: kwargs to pass to __init__.
        :return: An Actor.
        """
        return self._actors[name](*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        """
        :param actor: Actor instance.
        :param method: Method name.
        :param args: Method args.
        :param kwargs: Method kwargs.
        :return: Result of calling method.
        """
        return getattr(actor, method)(*args, **kwargs)

    def num_cores_total(self):
        return self.num_gpus


# This is a backend with Ray enabled with remote functions
# TODO: scrap later if Actor implementation performs better
class GPUParallelBackend(Backend):
    def __init__(
        self,
        address: Optional[str] = None,
        use_head: bool = True,  # TODO: Probably not needed for a GPU setup?
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        num_nodes: Optional[int] = None,
    ):
        # NOTES: use_head is ignored for now
        self._use_head = use_head
        self.num_cpus = int(get_num_cores()) if num_cpus is None else num_cpus
        self._remote_functions: dict = {}
        self._actors: dict = {}
        self.num_gpus = int(get_num_gpus()) if num_gpus is None else num_gpus

        # Ray attributes for attaching to a Ray cluster environment
        self._manage_ray = True
        self._address: str = address
        self._num_nodes = num_nodes
        self._available_nodes = []
        self._head_node = None
        self._worker_nodes = []
        self._devices: List[Device] = []
        self._device_to_node: Dict[Device, Dict] = {}
        import cupy as cp

        self.cp = cp

    def init(self):
        if ray.is_initialized():
            self._manage_ray = False
        if self._manage_ray:
            if self._address is None:
                ray.init(num_cpus=self.num_cpus, num_gpus=self.num_gpus)
            else:
                # Don't need to manually set the number of cpus and gpus
                ray.init(address=self._address)
        # Compute available nodes, based on CPU resource.
        # TODO: this is for inter node communication
        if settings.head_ip is None:
            # TODO (hme): Have this be a class argument vs. using what's set in settings directly.
            logging.getLogger(__name__).info("Using driver node ip as head node.")
            head_ip = get_private_ip()
        else:
            head_ip = settings.head_ip
        total_cpus = 0
        total_gpus = 0
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
            total_gpus += self._head_node["Resources"]["GPU"]
            # implement total_gpus whne inter node comunication is implemented
            self._available_nodes.append(self._head_node)
        logging.getLogger(__name__).info("total cpus %s", total_cpus)
        logging.getLogger(__name__).info("total gpus %s", total_gpus)

        if self._num_nodes is None:
            self._num_nodes = len(self._available_nodes)
        assert self._num_nodes <= len(self._available_nodes)

        self.init_devices()

    def init_devices(self):
        self._devices = []
        self._device_to_node = {}
        for node_id in range(self.num_gpus):
            # TODO: Undo this to account for multi-node systems
            node = self._available_nodes[0]  # index changed ot 0, temp hack/fix
            did = Device(node_id, self._node_key(node), "gpu", 1)
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

    def put(self, value: Any, device: Device):
        """
        Put object into backend storage and force placement on the relevant node.
        """
        return self.call("identity", [value], {}, device, {})

    def get(self, object_ids: Union[Any, List]):
        """
        Get object from backend storage.
        CuPy also uses .get() to copy to CPU memory to serve to user.
        """
        import cupy as cp

        # for i in range(8):
        #     cp.cuda.Device(i).synchronize()
        carrs = ray.get(object_ids)
        if isinstance(carrs, list):
            return [cp.asnumpy(carr) for carr in carrs]
        return cp.asnumpy(carrs)

    def remote(self, function: FunctionType, remote_params: Dict):
        """
        Return a callable remote function with remote_params.
        """
        r = ray.remote(num_gpus=1, **remote_params)
        return r(function)

    def devices(self):
        return self._devices

    def register(self, name: str, func: callable, remote_params: Dict = None):
        if name in self._remote_functions:
            return
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device: Device, options: Dict):
        if device is not None:
            node = self._device_to_node[device]
            node_key = self._node_key(node)
            if "resources" in options:
                assert node_key not in options
            options["resources"] = {node_key: 1.0 / 10 ** 4}
        return self._remote_functions[name].options(**options).remote(*args, **kwargs)

    def register_actor(self, name: str, cls: type):
        """
        :param name: Name of the actor. This should be unique.
        :param cls: The Python class to convert into an actor.
        :return: None
        """
        assert name not in self._actors
        self._actors[name] = cls

    def make_actor(self, name: str, *args, device: Device = None, **kwargs):
        """
        :param name: The name of the actor.
        :param args: args to pass to __init__.
        :param device: A device. This is captured by the system and not passed to __init__.
        :param kwargs: kwargs to pass to __init__.
        :return: An Actor.
        """
        return self._actors[name](*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        """
        :param actor: Actor instance.
        :param method: Method name.
        :param args: Method args.
        :param kwargs: Method kwargs.
        :return: Result of calling method.
        """
        return getattr(actor, method)(*args, **kwargs)

    def num_cores_total(self):
        return self.num_gpus


class GPURayActorBackend(Backend):
    def __init__(
        self,
        address: Optional[str] = None,
        use_head: bool = True,  # TODO: Probably not needed for a GPU setup?
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        num_nodes: Optional[int] = None,
    ):
        # NOTES: use_head is ignored for now
        self._use_head = use_head
        self.num_cpus = int(get_num_cores()) if num_cpus is None else num_cpus
        self._remote_functions: dict = {}
        self._actors: dict = {}
        self.num_gpus = int(get_num_gpus()) if num_gpus is None else num_gpus
        self._devices = []
        self._device_to_node = {}
        self.cp = cp
        self._available_nodes = []
        self._worker_nodes = []
        self._gpu_list = list(range(int(get_num_gpus())))
        self._comm = None
        self._cache = {}

    def init(self):
        for gpu_id in range(self.num_gpus):
            pci_bus_id = self.cp.cuda.Device(gpu_id).pci_bus_id
            # add inter node logic here when possible
            self._worker_nodes.append(pci_bus_id)
            self._available_nodes.append(pci_bus_id)

        self.init_devices()
        self._comm = NcclCommunicator.initAll(self._gpu_list)

    def init_devices(self):
        for i in range(self.num_gpus):
            pci_bus_id = self.cp.cuda.Device(i).pci_bus_id
            gpu_device = self.cp.cuda.Device.from_pci_bus_id(pci_bus_id)
            did = Device(i, self._gpu_key(pci_bus_id), "gpu", 0)
            self._devices.append(did)
            self._device_to_node[did] = gpu_device

    def _gpu_key(self, pci_bus_id):
        return "gpu:{}".format(pci_bus_id)

    def shutdown(self):
        mempool = self.cp.get_default_memory_pool()
        mempool.free_all_blocks()

    def put(self, value: Any, device: Device):
        """
        Put object into backend storage and force placement on the relevant node.
        """
        if device is not None:
            gpu = self._device_to_node[device]
            node_key = self._gpu_key(gpu)
            # if "resources" in options:
            #     assert node_key not in options
            # options["resources"] = {node_key: 1.0 / 10**4}

        import cupy as cp
        import numpy as np

        with gpu:
            if np.isscalar(value):
                return value

            return cp.array(value)

    def get(self, object_ids: Union[Any, List]):
        """
        Get object from backend storage.

        CuPy also uses .get() to copy to CPU memory to serve to user.
        """
        import cupy as cp
        import numpy as np

        for i in range(self.num_gpus):
            cp.cuda.Device(i).synchronize()

        # TODO: some things in CuPy don't translate well to NumPy, clean this up in a helper function
        if type(object_ids[0]) == np.float64 or type(object_ids[0]) == int:
            return object_ids
        if isinstance(object_ids, list):
            return [
                a.get()
                for a in object_ids
                if type(a) is not bool and type(a) != np.ndarray
            ]  # TODO: clean up this case
        else:
            return object_ids.get()

    def remote(self, function: FunctionType, remote_params: Dict):
        """
        Return a callable remote function with remote_params.
        """
        return function

    def devices(self):
        return self._devices

    def register(self, name: str, func: callable, remote_params: Dict = None):
        if name in self._remote_functions:
            return
        if remote_params is None:
            remote_params = {}
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device: Device, options: Dict):
        gpu = self._device_to_node[device]

        new_args = []
        stream = cp.cuda.Stream.null.ptr

        for arg in args:
            if isinstance(arg, self.cp.ndarray) and gpu != arg.device:

                key = str(arg.data.ptr) + str(arg.shape) + str(gpu.id)
                if key not in self._cache:
                    groupStart()
                    self._comm[arg.device.id].bcast(
                        arg.data.ptr,
                        arg.size,
                        NCCL_INT64,
                        self._comm[arg.device.id].rank_id(),
                        stream,
                    )
                    for i in range(self.num_gpus):
                        if i != arg.device.id:
                            key = str(arg.data.ptr) + str(arg.shape) + str(i)
                            with self.cp.cuda.Device(i):
                                self._cache[key] = self.cp.zeros(
                                    shape=arg.shape, dtype=arg.dtype
                                )
                            self._comm[i].bcast(
                                self._cache[key].data.ptr,
                                arg.size,
                                NCCL_INT64,
                                self._comm[arg.device.id].rank_id(),
                                stream,
                            )
                    groupEnd()

                key = str(arg.data.ptr) + str(arg.shape) + str(gpu.id)
                new_arg = self._cache[key]
                new_args.append(new_arg)

            else:
                new_args.append(arg)

        with gpu:
            return self._remote_functions[name](*new_args, **kwargs)

    # This is for sklearn, ignore for now
    def register_actor(self, name: str, cls: type):
        """
        :param name: Name of the actor. This should be unique.
        :param cls: The Python class to convert into an actor.
        :return: None
        """
        assert name not in self._actors
        self._actors[name] = cls

    def make_actor(self, name: str, *args, device: Device = None, **kwargs):
        """
        :param name: The name of the actor.
        :param args: args to pass to __init__.
        :param device: A device. This is captured by the system and not passed to __init__.
        :param kwargs: kwargs to pass to __init__.
        :return: An Actor.
        """
        return self._actors[name](*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        """
        :param actor: Actor instance.
        :param method: Method name.
        :param args: Method args.
        :param kwargs: Method kwargs.
        :return: Result of calling method.
        """
        return getattr(actor, method)(*args, **kwargs)

    def num_cores_total(self):
        return self.num_gpus


# This serves as a layer of indirection between sending stuff via actor. #TODO: Explore this later
@ray.remote(num_gpus=1)
class GPUActor(object):
    def __init__(self):
        return "This actor is allowed to use GPUs {}.".format(ray.get_gpu_ids())


class GPUIntraBackend(Backend):
    def __init__(self, num_cpus: Optional[int] = None, num_gpus: Optional[int] = None):
        import cupy as cp

        self.num_cpus = int(get_num_cores()) if num_cpus is None else num_cpus
        self._remote_functions: dict = {}
        self._actors: dict = {}
        self.num_gpus = int(get_num_gpus()) if num_gpus is None else num_gpus
        self._devices = []
        self._device_to_node = {}
        self.cp = cp
        self._available_nodes = []
        self._worker_nodes = []
        self._gpu_list = list(range(int(get_num_gpus())))
        self._comm = None
        self._streams = []
        self._recv_bufs = {}

    def init(self):
        for gpu_id in range(self.num_gpus):
            pci_bus_id = self.cp.cuda.Device(gpu_id).pci_bus_id
            # add inter node logic here when possible
            self._worker_nodes.append(pci_bus_id)
            self._available_nodes.append(pci_bus_id)

        self.init_devices()
        self._comm = NcclCommunicator.initAll(self._gpu_list)

        for i in range(self.num_gpus):
            with self.cp.cuda.Device(i):
                self._streams.append(self.cp.cuda.Stream(non_blocking=True))

    def init_devices(self):
        for i in range(self.num_gpus):
            pci_bus_id = self.cp.cuda.Device(i).pci_bus_id
            gpu_device = self.cp.cuda.Device.from_pci_bus_id(pci_bus_id)
            did = Device(i, self._gpu_key(pci_bus_id), "gpu", 0)
            self._devices.append(did)
            self._device_to_node[did] = gpu_device

    def _gpu_key(self, pci_bus_id):
        return "gpu:{}".format(pci_bus_id)

    def shutdown(self):
        mempool = self.cp.get_default_memory_pool()
        mempool.free_all_blocks()

    def put(self, value: Any, device: Device):
        """
        Put object into backend storage and force placement on the relevant node.
        """
        if device is not None:
            gpu = self._device_to_node[device]
            node_key = self._gpu_key(gpu)
            # if "resources" in options:
            #     assert node_key not in options
            # options["resources"] = {node_key: 1.0 / 10**4}

        import cupy as cp
        import numpy as np

        # print("putting")
        with self.cp.cuda.Device(self._streams[gpu.id].device_id):
            with self._streams[gpu.id]:
                if np.isscalar(value):
                    # print("scalar")
                    return value
                # print("arr")
                return cp.array(value)

    def get(self, object_ids: Union[Any, List]):
        """
        Get object from backend storage.

        CuPy also uses .get() to copy to CPU memory to serve to user.
        """
        import cupy as cp
        import numpy as np

        self._recv_bufs.clear()

        for i in range(self.num_gpus):
            cp.cuda.Device(i).synchronize()
            self._streams[i].synchronize()

        # TODO: some things in CuPy don't translate well to NumPy, clean this up in a helper function
        if type(object_ids[0]) == np.float64 or type(object_ids[0]) == int:
            return object_ids
        if isinstance(object_ids, list):
            return [
                a.get()
                for a in object_ids
                if type(a) is not bool and type(a) != np.ndarray
            ]  # TODO: clean up this case
        else:
            return object_ids.get()

    def remote(self, function: FunctionType, remote_params: Dict):
        """
        Return a callable remote function with remote_params.
        """
        return function

    def devices(self):
        return self._devices

    def register(self, name: str, func: callable, remote_params: Dict = None):
        if name in self._remote_functions:
            return
        if remote_params is None:
            remote_params = {}
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device: Device, options: Dict):
        gpu = self._device_to_node[device]

        new_args = []

        for arg in args:
            if isinstance(arg, self.cp.ndarray) and gpu != arg.device:
                send_rank = arg.device.id # probably don't need this, delete later after benchmakrking
                recv_rank = gpu.id
                # print("sending from {} to {}".format(send_rank, recv_rank))
                with self.cp.cuda.Device(self._streams[recv_rank].device_id):
                    with self._streams[recv_rank]:
                        key = str(recv_rank) + "_" +  str(arg.size)
                        if key not in self._recv_bufs:
                            # Allocate and reuse as many buffers needed for communication. Will be deleted after
                            # a call to synchronization
                            self._recv_bufs[key] = self.cp.zeros(shape=arg.shape, dtype=arg.dtype)

                        new_arg = self._recv_bufs[key]

                        # new_arg = self.cp.zeros(shape=arg.shape, dtype=arg.dtype)
                nccl_type = DTYPES[arg.dtype.name]
                groupStart()
                # self._comm[send_rank].send(
                #     arg.data.ptr, arg.size, nccl_type, self._comm[recv_rank].rank_id(), self._streams[send_rank].ptr
                # )
                # self._comm[recv_rank].recv(
                #     new_arg.data.ptr, arg.size, nccl_type, self._comm[send_rank].rank_id(), self._streams[recv_rank].ptr
                # )

                self._comm[recv_rank].send(
                    arg.data.ptr,
                    arg.size,
                    nccl_type,
                    recv_rank,
                    self._streams[recv_rank].ptr,
                )
                self._comm[recv_rank].recv(
                    new_arg.data.ptr,
                    arg.size,
                    nccl_type,
                    recv_rank,
                    self._streams[recv_rank].ptr,
                )
                groupEnd()
                new_args.append(new_arg)
            else:
                new_args.append(arg)

        with self.cp.cuda.Device(self._streams[gpu.id].device_id):
            with self._streams[gpu.id]:
                return self._remote_functions[name](*new_args, **kwargs)

    # This is for sklearn, ignore for now
    def register_actor(self, name: str, cls: type):
        """
        :param name: Name of the actor. This should be unique.
        :param cls: The Python class to convert into an actor.
        :return: None
        """
        assert name not in self._actors
        self._actors[name] = cls

    def make_actor(self, name: str, *args, device: Device = None, **kwargs):
        """
        :param name: The name of the actor.
        :param args: args to pass to __init__.
        :param device: A device. This is captured by the system and not passed to __init__.
        :param kwargs: kwargs to pass to __init__.
        :return: An Actor.
        """
        return self._actors[name](*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        """
        :param actor: Actor instance.
        :param method: Method name.
        :param args: Method args.
        :param kwargs: Method kwargs.
        :return: Result of calling method.
        """
        return getattr(actor, method)(*args, **kwargs)

    def num_cores_total(self):
        return self.num_gpus
