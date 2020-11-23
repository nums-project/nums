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


import inspect
from types import ModuleType, FunctionType
from typing import Any, Union, List, Dict

import ray
import numpy as np

from nums.core.storage.storage import ArrayGrid
from nums.core.systems.schedulers import RayScheduler, TaskScheduler, BlockCyclicScheduler
from nums.core.systems.interfaces import SystemInterface, ComputeInterface, RNGInterface
from nums.core.systems.utils import check_implementation, extract_functions


class System(SystemInterface, ComputeInterface):
    # pylint: disable=abstract-method

    def __init__(self, compute_module):
        self.compute_module: ModuleType = compute_module
        self.compute_imp = compute_module.ComputeCls
        self.methods: dict = {}
        # Check that all of kernel interface is implemented.
        check_implementation(ComputeInterface, self.compute_imp)
        # Collect implemented module functions.
        self.module_functions = extract_functions(self.compute_imp)
        if getattr(compute_module, "RNG", None) is None:
            raise Exception("No random number generator implemented "
                            "for compute module %s" % str(compute_module))
        self.rng_cls = compute_module.RNG

    def init(self):
        for name, _ in self.module_functions.items():
            self.methods[name] = self.get_callable(name)

    def shutdown(self):
        raise NotImplementedError()

    def get_rng(self, seed) -> RNGInterface:
        return self.rng_cls(seed)

    def get_callable(self, name: str):
        def new_func(*args, **kwargs):
            return self.call(name, *args, **kwargs)
        return new_func

    def __getattribute__(self, name: str):
        methods = object.__getattribute__(self, "methods")
        if name in methods:
            return methods[name]
        return object.__getattribute__(self, name)


class SerialSystem(System):
    # pylint: disable=abstract-method,useless-super-delegation

    def __init__(self, compute_module):
        super(SerialSystem, self).__init__(compute_module)
        self.remote_functions: dict = {}

    def init(self):
        # Collect function signatures.
        function_signatures: dict = {}
        required_methods = inspect.getmembers(ComputeInterface(), predicate=inspect.ismethod)
        for name, func in required_methods:
            function_signatures[name] = func
        for name, func in self.module_functions.items():
            func_sig = function_signatures[name]
            try:
                remote_params = func_sig.remote_params
            except Exception as _:
                remote_params = {}
            self.remote_functions[name] = self.remote(func, remote_params)
        super(SerialSystem, self).init()

    def shutdown(self):
        pass

    def put(self, value: Any):
        return value

    def get(self, object_ids: Union[Any, List]):
        return object_ids

    def remote(self, function: FunctionType, remote_params: dict):
        return function

    def nodes(self):
        return [{"Resources": {"node:0": 1.0}}]

    def register(self, name: str, func: callable, remote_params: dict = None):
        if name in self.remote_functions:
            return
        if remote_params is None:
            remote_params = {}
        self.remote_functions[name] = self.remote(func, remote_params)

    def call(self, name: str, *args, **kwargs):
        kwargs = kwargs.copy()
        if "syskwargs" in kwargs:
            del kwargs["syskwargs"]
        return self.remote_functions[name](*args, **kwargs)

    def call_with_options(self, name, args, kwargs, options):
        return self.call(name, *args, **kwargs)

    def get_options(self, cluster_entry, cluster_shape):
        node = self.nodes()[0]
        node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
        assert len(node_key) == 1
        node_key = node_key[0]
        return {
            "resources": {node_key: 1.0/10**4}
        }

    def get_block_addresses(self, grid: ArrayGrid):
        addresses: dict = {}
        nodes = self.nodes()
        index = 0
        for grid_entry in grid.get_entry_iterator():
            node = nodes[index]
            node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
            assert len(node_key) == 1
            node_key = node_key[0]
            addresses[grid_entry] = node_key
            index = (index + 1) % len(nodes)
        return addresses


class RaySystem(System):
    # pylint: disable=abstract-method
    """
    Implements SystemInterface and ComputeInterface to support static typing.
    """

    def __init__(self, compute_module, scheduler: RayScheduler):
        super(RaySystem, self).__init__(compute_module)
        self.scheduler: RayScheduler = scheduler
        self.manage_ray = True

    def init(self):
        if ray.is_initialized():
            self.manage_ray = False
        if self.manage_ray:
            ray.init()
        self.scheduler.init()
        super(RaySystem, self).init()

    def shutdown(self):
        if self.manage_ray:
            ray.shutdown()

    def warmup(self, n: int):
        # Quick warm-up. Useful for quick and more accurate testing.
        if n > 0:
            assert n < 10 ** 6

            def warmup_func(n):
                # pylint: disable=import-outside-toplevel
                import random
                r = ray.remote(num_cpus=1)(lambda x, y: x + y).remote
                for _ in range(n):
                    _a = random.randint(0, 1000)
                    _b = random.randint(0, 1000)
                    _v = self.get(r(self.put(_a), self.put(_b)))

            warmup_func(n)

    def put(self, value: Any):
        return self.scheduler.put(value)

    def get(self, object_ids: Union[Any, List]):
        return self.scheduler.get(object_ids)

    def remote(self, function: FunctionType, remote_params: dict):
        return self.scheduler.remote(function, remote_params)

    def nodes(self):
        return self.scheduler.nodes()

    def register(self, name: str, func: callable, remote_params: dict = None):
        self.scheduler.register(name, func, remote_params)

    def call(self, name: str, *args, **kwargs):
        return self.scheduler.call(name, *args, **kwargs)

    def call_with_options(self, name: str, args, kwargs, options):
        return self.scheduler.call_with_options(name, args, kwargs, options)

    def get_options(self, cluster_entry, cluster_shape):
        if isinstance(self.scheduler, BlockCyclicScheduler):
            scheduler: BlockCyclicScheduler = self.scheduler
            node: Dict = scheduler.cluster_grid[scheduler.get_cluster_entry(cluster_entry)]
            node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
            assert len(node_key) == 1
            node_key = node_key[0]
        elif isinstance(self.scheduler, TaskScheduler):
            # Just do round-robin over nodes.
            nodes = self.nodes()
            # Compute a flattened index from the cluster entry.
            strides = [np.product(cluster_shape[i:]) for i in range(1, len(cluster_shape))] + [1]
            index = sum(np.array(cluster_entry) * strides)
            node = nodes[index]
            node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
            assert len(node_key) == 1
            node_key = node_key[0]
        else:
            raise Exception()
        return {
            "resources": {node_key: 1.0/10**4}
        }

    def get_block_addresses(self, grid: ArrayGrid):
        addresses: dict = {}
        if isinstance(self.scheduler, BlockCyclicScheduler):
            scheduler: BlockCyclicScheduler = self.scheduler
            for grid_entry in grid.get_entry_iterator():
                node: Dict = scheduler.cluster_grid[scheduler.get_cluster_entry(grid_entry)]
                node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
                assert len(node_key) == 1
                node_key = node_key[0]
                addresses[grid_entry] = node_key
        elif isinstance(self.scheduler, TaskScheduler):
            # Just do round-robin over nodes.
            nodes = self.nodes()
            index = 0
            for grid_entry in grid.get_entry_iterator():
                node = nodes[index]
                node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
                assert len(node_key) == 1
                node_key = node_key[0]
                addresses[grid_entry] = node_key
                index = (index + 1) % len(nodes)
        return addresses
