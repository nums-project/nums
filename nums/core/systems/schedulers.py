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


import itertools
import logging
from types import FunctionType
from typing import Tuple
import inspect

import ray
import numpy as np

from nums.core.systems.interfaces import ComputeInterface, ComputeImp
from nums.core.systems.utils import check_implementation, extract_functions, get_private_ip


class RayScheduler(ComputeInterface):
    # pylint: disable=abstract-method
    def init(self):
        raise NotImplementedError()

    def put(self, value):
        raise NotImplementedError()

    def get(self, object_ids, timeout=None):
        raise NotImplementedError()

    def remote(self, function: FunctionType, remote_params: dict):
        raise NotImplementedError()

    def register(self, name: str, func: callable, remote_params: dict = None):
        raise NotImplementedError("Implements a way to register new remote functions.")

    def call(self, name: str, *args, **kwargs):
        raise NotImplementedError()

    def call_with_options(self, name, args, kwargs, options):
        raise NotImplementedError()

    def nodes(self):
        raise NotImplementedError()


class TaskScheduler(RayScheduler):
    # pylint: disable=abstract-method
    """
    Basic task-based scheduler. This scheduler relies on underlying
    system's scheduler in distributed memory configurations.
    Simply takes as input a compute module and StoreConfiguration.
    """
    def __init__(self,
                 compute_module,
                 use_head=False):
        self.compute_imp: ComputeImp = compute_module.ComputeCls
        check_implementation(ComputeInterface, self.compute_imp)
        self.remote_functions = {}
        self.available_nodes = []
        self.use_head = use_head
        self.head_node = None

    def init(self):
        # Compute available nodes, based on CPU resource.
        local_ip = get_private_ip()
        total_cpus = 0
        for node in ray.nodes():
            node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
            assert len(node_key) == 1
            node_ip = node_key[0].split(":")[1]
            has_cpu_resources = "CPU" in node["Resources"] and node["Resources"]["CPU"] >= 1.0
            if local_ip == node_ip:
                logging.getLogger().info("head node %s", node_ip)
                self.head_node = node
                if self.use_head and has_cpu_resources:
                    total_cpus += node["Resources"]["CPU"]
                    self.available_nodes.append(node)
            elif has_cpu_resources:
                logging.getLogger().info("worker node %s", node_ip)
                total_cpus += node["Resources"]["CPU"]
                self.available_nodes.append(node)
        logging.getLogger().info("total cpus %s", total_cpus)
        # Collect compute functions.
        module_functions = extract_functions(self.compute_imp)
        function_signatures: dict = {}
        required_methods = inspect.getmembers(ComputeInterface(), predicate=inspect.ismethod)
        for name, func in required_methods:
            function_signatures[name] = func
        for name, func in module_functions.items():
            func_sig = function_signatures[name]
            try:
                remote_params = func_sig.remote_params
            except Exception as _:
                remote_params = {}
            self.remote_functions[name] = self.remote(func, remote_params)

    def put(self, value):
        return ray.put(value)

    def get(self, object_ids, timeout=None):
        return ray.get(object_ids, timeout=timeout)

    def remote(self, function: FunctionType, remote_params: dict):
        r = ray.remote(num_cpus=1, **remote_params)
        return r(function)

    def register(self, name: str, func: callable, remote_params: dict = None):
        if name in self.remote_functions:
            return
        self.remote_functions[name] = self.remote(func, remote_params)

    def call_with_options(self, name, args, kwargs, options):
        return self.remote_functions[name].options(**options).remote(*args, **kwargs)

    def call(self, name: str, *args, **kwargs):
        if "syskwargs" in kwargs:
            kwargs = kwargs.copy()
            syskwargs = kwargs["syskwargs"]
            del kwargs["syskwargs"]
            if "options" in syskwargs:
                options = syskwargs["options"]
                return self.call_with_options(name, args, kwargs, options)
        return self.remote_functions[name].remote(*args, **kwargs)

    def nodes(self):
        return self.available_nodes


class BlockCyclicScheduler(TaskScheduler):
    # pylint: disable=abstract-method
    """
    Operations with dimensions larger than the grid axis
    to which they are mapped wrap along that axis.
    Replication not implemented, but would include the following property:
    Operations with 1 dim along any axis are replicated for each dimension along that axis.
    """

    def __init__(self, compute_module, cluster_shape: Tuple, use_head=False, verbose=False):
        super(BlockCyclicScheduler, self).__init__(compute_module, use_head)
        self.verbose = verbose
        self.cluster_shape: Tuple = cluster_shape
        self.cluster_grid: np.ndarray = np.empty(shape=self.cluster_shape, dtype=np.object)

    def init(self):
        super().init()
        err_str = "Not enough nodes %d for cluster shape %s." % (len(self.available_nodes),
                                                                 str(self.cluster_shape))
        assert len(self.available_nodes) >= np.prod(self.cluster_shape), err_str
        for i, cluster_entry in enumerate(self.get_cluster_entry_iterator()):
            self.cluster_grid[cluster_entry] = self.available_nodes[i]
            logging.getLogger().info("cluster_grid %s %s",
                                     cluster_entry,
                                     self.get_node_key(cluster_entry))
        logging.getLogger().info("cluster_shape %s", str(self.cluster_shape))

    def get_cluster_entry_iterator(self):
        return itertools.product(*map(range, self.cluster_shape))

    def get_cluster_entry(self, grid_entry):
        cluster_entry = []
        num_grid_entry_axes = len(grid_entry)
        num_cluster_axes = len(self.cluster_shape)
        if num_grid_entry_axes <= num_cluster_axes:
            # When array has fewer or equal # of axes than cluster.
            for cluster_axis in range(num_cluster_axes):
                if cluster_axis < num_grid_entry_axes:
                    cluster_dim = self.cluster_shape[cluster_axis]
                    grid_entry_dim = grid_entry[cluster_axis]
                    cluster_entry.append(grid_entry_dim % cluster_dim)
                else:
                    cluster_entry.append(0)
        elif num_grid_entry_axes > num_cluster_axes:
            # When array has more axes then cluster.
            for cluster_axis in range(num_cluster_axes):
                cluster_dim = self.cluster_shape[cluster_axis]
                grid_entry_dim = grid_entry[cluster_axis]
                cluster_entry.append(grid_entry_dim % cluster_dim)
            # Ignore trailing axes, as these are "cycled" to 0 by assuming
            # the dimension of those cluster axes is 1.
        return tuple(cluster_entry)

    def get_node_key(self, cluster_entry):
        node = self.cluster_grid[cluster_entry]
        node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
        assert len(node_key) == 1
        node_key = node_key[0]
        return node_key

    def call(self, name: str, *args, **kwargs):
        assert "syskwargs" in kwargs
        syskwargs = kwargs["syskwargs"]
        grid_entry = syskwargs["grid_entry"]
        grid_shape = syskwargs["grid_shape"]

        if "options" in syskwargs:
            options = syskwargs["options"].copy()
            if "resources" in options:
                resources = options["resources"].copy()
            else:
                resources = {}
        else:
            options = {}
            resources = {}

        # Make sure no node ip addresses are already in resources.
        for key, _ in resources.items():
            assert "node" not in key

        cluster_entry: tuple = self.get_cluster_entry(grid_entry)
        node_key = self.get_node_key(cluster_entry)
        # TODO (hme): This will be problematic. Only able to assign 10k tasks-per-node.
        resources[node_key] = 1.0/10**4
        options["resources"] = resources

        kwargs = kwargs.copy()
        del kwargs["syskwargs"]

        if self.verbose:
            if name == "bop":
                fname = args[0]
                log_str = "BCS: bop_name=%s, " \
                          "grid_entry=%s, grid_shape=%s " \
                          "on cluster_grid[%s] == %s"
                logging.getLogger().info(log_str, fname, str(grid_entry),
                                         str(grid_shape), str(cluster_entry),
                                         node_key)
            else:
                log_str = "BCS: remote_name=%s, " \
                          "grid_entry=%s, grid_shape=%s " \
                          "on cluster_grid[%s] == %s"
                logging.getLogger().info(log_str, name, str(grid_entry),
                                         str(grid_shape), str(cluster_entry),
                                         node_key)

        return self.call_with_options(name, args, kwargs, options)
