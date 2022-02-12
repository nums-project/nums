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
import abc
from types import FunctionType
from typing import Any, Dict, List, Union

from nums.core.grid.grid import DeviceID


class Backend:
    @abc.abstractmethod
    def init(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def put(self, value: Any, device_id: DeviceID):
        """
        Put object into backend storage and force placement on the relevant node.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get(self, object_ids: Union[Any, List]):
        """
        Get object from backend storage.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def remote(self, function: FunctionType, remote_params: Dict):
        """
        Return a callable remote function with remote_params.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def devices(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def register(self, name: str, func: callable, remote_params: Dict = None):
        raise NotImplementedError("Implements a way to register new remote functions.")

    def call(self, name: str, args, kwargs, device_id: DeviceID, options: Dict):
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

    def make_actor(self, name: str, *args, device_id: DeviceID = None, **kwargs):
        """
        :param name: The name of the actor.
        :param args: args to pass to __init__.
        :param device_id: A device id. This is captured by the backend and not passed to __init__.
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
