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


from types import FunctionType
from typing import Any, Union, List, Dict

from nums.core.grid.grid import DeviceID


class SystemInterface(object):
    def init(self):
        raise NotImplementedError()

    def shutdown(self):
        raise NotImplementedError()

    def put(self, value: Any):
        """
        Put object into system storage.
        """
        raise NotImplementedError()

    def get(self, object_ids: Union[Any, List]):
        """
        Get object from system storage.
        """
        raise NotImplementedError()

    def remote(self, function: FunctionType, remote_params: Dict):
        """
        Return a callable remote function with remote_params.
        """
        raise NotImplementedError()

    def devices(self):
        raise NotImplementedError()

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
        :param device_id: A device id. This is captured by the system and not passed to __init__.
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
