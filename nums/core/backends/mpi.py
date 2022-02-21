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
from typing import Any, List, Dict, Union
import warnings

from nums.core.grid.grid import DeviceID

from .base import Backend
from .utils import get_private_ip


@dataclass
class MPIRemoteObj(object):
    rank: int


@dataclass
class MPILocalObj(object):
    value: Any


class MPIBackend(Backend):
    """
    Implements backend for MPI.
    """

    def __init__(self):
        # pylint: disable=import-outside-toplevel c-extension-no-member import-error
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
                    dest_rank = obj.rank
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
                dest_rank = obj.rank
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
        if device_rank == self.rank:
            if isinstance(obj, MPILocalObj):
                # If the obj is local then just return the value.
                return obj.value
            # If the object is not local then execute a receive.
            sender_rank = obj.rank
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
