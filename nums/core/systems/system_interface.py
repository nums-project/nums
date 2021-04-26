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
from typing import Any, Union, List


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

    def remote(self, function: FunctionType, remote_params: dict):
        """
        Return a callable remote function with remote_params.
        """
        raise NotImplementedError()

    def nodes(self):
        raise NotImplementedError()

    def get_block_addresses(self, grid):
        """
        Maps each grid entry in a grid array to a node address.
        """
        raise NotImplementedError()

    def register(self, name: str, func: callable, remote_params: dict = None):
        raise NotImplementedError("Implements a way to register new remote functions.")

    def call(self, name: str, *args, **kwargs):
        raise NotImplementedError("Implement RPC as e.g. "
                                  "self.remote_functions[name](*args, **new_kwargs)")

    def call_with_options(self, name: str, args, kwargs, options):
        raise NotImplementedError("Implement RPC with options support.")

    def get_options(self, cluster_entry, cluster_shape):
        # TODO (hme): API needs improvements in this area.
        raise NotImplementedError()
