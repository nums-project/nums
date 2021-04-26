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
from types import FunctionType
from typing import Any, Union, List

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.compute.compute_interface import ComputeInterface, RNGInterface
from nums.core.systems import utils as systems_utils
from nums.core.systems.schedulers import TaskScheduler, BlockCyclicScheduler
from nums.core.systems.system_interface import SystemInterface
from nums.core.systems.systems import SerialSystem, RaySystem


class ComputeManager(ComputeInterface):
    """
    Abstraction to support multiple systems;
    namely simultaneous support for CPU and GPU system implementations.
    """

    # pylint: disable=abstract-method,useless-super-delegation

    instance = None

    @classmethod
    def create(cls, system: SystemInterface, compute_module):
        if cls.instance is not None:
            raise Exception()
        cls.instance: ComputeManager = ComputeManager(system, compute_module)
        return cls.instance

    @classmethod
    def destroy(cls):
        cls.instance = None

    def __init__(self, system: SystemInterface, compute_module):
        self.system: SystemInterface = system
        self.rng_cls = None
        self.methods: dict = {}
        self._block_shape_map = {}
        self.init_compute(compute_module)

    def init_compute(self, compute_module):
        compute_imp = compute_module.ComputeCls

        # Check that all of kernel interface is implemented.
        systems_utils.check_implementation(ComputeInterface, compute_imp)
        if getattr(compute_module, "RNG", None) is None:
            raise Exception("No random number generator implemented "
                            "for compute module %s" % str(compute_module))
        self.rng_cls = compute_module.RNG

        # Collect implemented module functions.
        module_functions = systems_utils.extract_functions(compute_imp)
        # Collect function signatures.
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
            self.register(name, func, remote_params)

        # Add functions as methods of this class.
        for name, _ in module_functions.items():
            self.methods[name] = self.get_callable(name)

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

    def num_cores_total(self):
        if isinstance(self.system, RaySystem):
            system: RaySystem = self.system
            if isinstance(system.scheduler, BlockCyclicScheduler):
                scheduler: BlockCyclicScheduler = system.scheduler
                nodes = scheduler.cluster_grid.flatten().tolist()
                num_cores = sum(map(lambda n: n["Resources"]["CPU"], nodes))
            else:
                assert isinstance(system.scheduler, TaskScheduler)
                scheduler: TaskScheduler = system.scheduler
                nodes = scheduler.available_nodes
                num_cores = sum(map(lambda n: n["Resources"]["CPU"], nodes))
        else:
            assert isinstance(self.system, SerialSystem)
            num_cores = systems_utils.get_num_cores()
        return int(num_cores)

    def compute_block_shape(self,
                            shape: tuple,
                            dtype: Union[type, np.dtype],
                            cluster_shape=None,
                            num_cores=None):
        # TODO (hme): This should also compute parameters for DeviceGrid.
        if array_utils.is_float(dtype, type_test=True):
            dtype = np.finfo(dtype).dtype
        elif array_utils.is_int(dtype, type_test=True) \
                or array_utils.is_uint(dtype, type_test=True):
            dtype = np.iinfo(dtype).dtype
        elif array_utils.is_complex(dtype, type_test=True):
            dtype = np.dtype(dtype)
        elif dtype in (bool, np.bool_):
            dtype = np.dtype(np.bool_)
        else:
            raise ValueError("dtype %s not supported" % str(dtype))

        nbytes = dtype.alignment
        size = np.product(shape) * nbytes
        # If the object is less than 100 megabytes, there's not much value in constructing
        # a block tensor.
        if size < 10 ** 8:
            block_shape = shape
            return block_shape

        if num_cores is not None:
            pass
        else:
            num_cores = self.num_cores_total()

        if cluster_shape is not None:
            pass
        elif isinstance(self.system, RaySystem) \
                and isinstance(self.system.scheduler, BlockCyclicScheduler):
            # This configuration is the default.
            cluster_shape = self.system.scheduler.cluster_shape
        else:
            assert isinstance(self.system, SerialSystem)
            cluster_shape = (1, 1)

        if len(shape) < len(cluster_shape):
            cluster_shape = cluster_shape[:len(shape)]
        elif len(shape) > len(cluster_shape):
            cluster_shape = list(cluster_shape)
            for axis in range(len(shape)):
                if axis >= len(cluster_shape):
                    cluster_shape.append(1)
            cluster_shape = tuple(cluster_shape)

        shape_np = np.array(shape, dtype=int)
        # Softmax on cluster shape gives strong preference to larger dimensions.
        cluster_weights = np.exp(np.array(cluster_shape)) / np.sum(np.exp(cluster_shape))
        shape_fracs = np.array(shape) / np.sum(shape)
        # cluster_weights weight the proportion of cores available along each axis,
        # and shape_fracs is the proportion of data along each axis.
        weighted_shape_fracs = cluster_weights * shape_fracs
        weighted_shape_fracs = weighted_shape_fracs / np.sum(weighted_shape_fracs)

        # Compute dimensions of grid shape
        # so that the number of blocks are close to the number of cores.
        grid_shape_frac = num_cores ** weighted_shape_fracs
        grid_shape = np.floor(grid_shape_frac)
        # Put remainder on largest axis.
        remaining = np.sum(grid_shape_frac - grid_shape)
        grid_shape[np.argmax(shape)] += remaining
        grid_shape = np.ceil(grid_shape).astype(int)

        # We use ceiling of floating block shape
        # so that resulting grid shape is <= to what we compute above.
        block_shape = tuple((shape_np + grid_shape - 1) // grid_shape)
        return block_shape

    def get_block_shape(self, shape, dtype):
        # Simple way to ensure shape compatibility for basic linear algebra operations.
        block_shape = self.compute_block_shape(shape, dtype)
        final_block_shape = []

        for axis in range(len(shape)):
            shape_dim = shape[axis]
            block_shape_dim = block_shape[axis]
            if shape_dim not in self._block_shape_map:
                self._block_shape_map[shape_dim] = block_shape_dim
            final_block_shape.append(self._block_shape_map[shape_dim])
        return tuple(final_block_shape)

    ####################
    # System Interface
    ####################

    def put(self, value: Any):
        return self.system.put(value)

    def get(self, object_ids: Union[Any, List]):
        return self.system.get(object_ids)

    def remote(self, function: FunctionType, remote_params: dict):
        return self.system.remote(function, remote_params)

    def nodes(self):
        return self.system.nodes()

    def register(self, name: str, func: callable, remote_params: dict = None):
        self.system.register(name, func, remote_params)

    def call(self, name: str, *args, **kwargs):
        return self.system.call(name, *args, **kwargs)

    def call_with_options(self, name: str, args, kwargs, options):
        return self.system.call_with_options(name, args, kwargs, options)

    def get_options(self, cluster_entry, cluster_shape):
        return self.system.get_options(cluster_entry, cluster_shape)

    def get_block_addresses(self, grid):
        return self.system.get_block_addresses(grid)


def instance() -> ComputeManager:
    return ComputeManager.instance
