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
from dataclasses import dataclass
from itertools import repeat
from types import FunctionType
from typing import Any, List, Optional, Dict, Union
import warnings

import ray

from nums.core.grid.grid import Device

from .utils import get_num_cores
from .base import Backend
from .utils import get_private_ip


#TODO:
# from nums.core.kernel import cupy_kernel


### This is a serial gpu implementation (No communication)
class GPUSerialBackend(Backend):

    def __init__(self, num_cpus: Optional[int] = None):
        self.num_cpus = int(get_num_cores()) if num_cpus is None else num_cpus
        self._remote_functions: dict = {}
        self._actors: dict = {}

    def init(self):
        pass

    def shutdown(self):
        pass

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
        self.cp.cuda.Device(0).synchronize()
        if isinstance(x, list):
            return [a.get() for a in x]
        else:
            return x.get()

    def remote(self, function: FunctionType, remote_params: Dict):
        """
        Return a callable remote function with remote_params.
        """
        r = ray.remote(num_cpus=1, **remote_params)
        return r(function)

    def devices(self):
        return [Device(0, "localhost", "gpu", 0)] # Just maps to 1 GPU
        # node_id: int
        # node_addr: str
        # device_type: str
        # device: int

    def register(self, name: str, func: callable, remote_params: Dict = None):
        if name in self._remote_functions:
            return
        self._remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, args, kwargs, device: Device, options: Dict):
        raise NotImplementedError(
            "Implement RPC as e.g. " "self.remote_functions[name](*args, **new_kwargs)"
        )

    def register_actor(self, name: str, cls: type):
        """
        :param name: Name of the actor. This should be unique.
        :param cls: The Python class to convert into an actor.
        :return: None
        """
        raise NotImplementedError()

    def make_actor(self, name: str, *args, device: Device = None, **kwargs):
        """
        :param name: The name of the actor.
        :param args: args to pass to __init__.
        :param device: A device. This is captured by the system and not passed to __init__.
        :param kwargs: kwargs to pass to __init__.
        :return: An Actor.
        """
        raise NotImplementedError()

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        """
        :param actor: Actor instance.
        :param method: Method name.
        :param args: Method args.
        :param kwargs: Method kwargs.
        :return: Result of calling method.
        """
        raise NotImplementedError()

    def num_cores_total(self):
        raise NotImplementedError()
