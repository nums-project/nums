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
from types import FunctionType
from typing import Any, List, Dict, Optional, Union

from nums.core.grid.grid import DeviceID

from .base import Backend
from .utils import get_num_cores


class SerialBackend(Backend):
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
        assert name not in self._actors
        self._actors[name] = cls

    def make_actor(self, name: str, *args, device_id: DeviceID = None, **kwargs):
        return self._actors[name](*args, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        return getattr(actor, method)(*args, **kwargs)

    def num_cores_total(self) -> int:
        return self.num_cpus
