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
import warnings

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.kernel.kernel_interface import Kernel, RNGInterface
from nums.core.grid.grid import DeviceGrid, Device
from nums.core.backends import utils as backend_utils
from nums.core.backends import Backend


class KernelManager(Kernel):
    """
    Abstraction to support multiple backends;
    namely simultaneous support for CPU and GPU backend implementations.
    """

    # pylint: disable=abstract-method,useless-super-delegation

    instance = None

    @classmethod
    def create(cls, backend: Backend, kernel_module, device_grid: DeviceGrid):
        if cls.instance is not None:
            raise Exception()
        cls.instance: KernelManager = KernelManager(backend, kernel_module, device_grid)
        return cls.instance

    @classmethod
    def destroy(cls):
        cls.instance = None

    def __init__(self, backend: Backend, kernel_module, device_grid: DeviceGrid):
        self.backend: Backend = backend
        self.device_grid: DeviceGrid = device_grid
        self.rng_cls = None
        self.methods: dict = {}
        self._block_shape_map = {}
        self.fuseable_functions = {}
        self.init_kernel(kernel_module)

    def init_kernel(self, kernel_module):
        kernel_imp = kernel_module.KernelCls

        # Check that all of kernel interface is implemented.
        backend_utils.check_implementation(Kernel, kernel_imp)
        if getattr(kernel_module, "RNG", None) is None:
            raise Exception(
                "No random number generator implemented "
                "for compute module %s" % str(kernel_module)
            )
        self.rng_cls = kernel_module.RNG

        # Collect implemented module functions.
        module_functions = backend_utils.extract_functions(kernel_imp)
        # Collect function signatures.
        function_signatures: dict = {}
        required_methods = inspect.getmembers(Kernel(), predicate=inspect.ismethod)
        for name, func in required_methods:
            function_signatures[name] = func
        for name, func in module_functions.items():
            self.fuseable_functions[name] = func
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

    ####################
    # Backend Interface
    ####################

    def put(self, value: Any, **kwargs):
        assert "syskwargs" in kwargs
        kwargs = kwargs.copy()
        syskwargs = kwargs["syskwargs"]
        del kwargs["syskwargs"]
        assert "options" not in syskwargs
        device, options = self._process_syskwargs(syskwargs)
        assert len(options) == 0
        return self.backend.put(value, device)

    def get(self, object_ids: Union[Any, List]):
        return self.backend.get(object_ids)

    def remote(self, function: FunctionType, remote_params: dict):
        return self.backend.remote(function, remote_params)

    def devices(self):
        return self.backend.devices()

    def register(self, name: str, func: callable, remote_params: dict = None):
        self.backend.register(name, func, remote_params)

    def _process_syskwargs(self, syskwargs):
        if "grid_entry" in syskwargs:
            assert "grid_shape" in syskwargs
            assert "device" not in syskwargs
            grid_entry = syskwargs["grid_entry"]
            grid_shape = syskwargs["grid_shape"]
            device: Device = self.device_grid.get_device(grid_entry, grid_shape)
        elif "device" in syskwargs:
            assert "grid_entry" not in syskwargs and "grid_shape" not in syskwargs
            device: Device = syskwargs["device"]
        else:
            raise Exception("All calls require device or grid_entry and grid_shape.")
        if "options" in syskwargs:
            options = syskwargs["options"]
        else:
            options = {}
        return device, options

    def call(self, name: str, *args, **kwargs):
        assert "syskwargs" in kwargs
        kwargs = kwargs.copy()
        syskwargs = kwargs["syskwargs"]
        del kwargs["syskwargs"]
        device, options = self._process_syskwargs(syskwargs)
        return self.backend.call(name, args, kwargs, device, options)

    def num_cores_total(self):
        return self.backend.num_cores_total()

    def register_actor(self, name: str, cls: type):
        return self.backend.register_actor(name, cls)

    def make_actor(self, name: str, *args, device: Device = None, **kwargs):
        return self.backend.make_actor(name, *args, device=device, **kwargs)

    def call_actor_method(self, actor, method: str, *args, **kwargs):
        return self.backend.call_actor_method(actor, method, *args, **kwargs)

    #########################
    # Block Shape Management
    #########################

    @staticmethod
    def compute_block_shape_static(
        shape: tuple, dtype: Union[type, np.dtype], cluster_shape: tuple, num_cores: int
    ):
        # TODO (hme): This should also compute parameters for DeviceGrid.
        if array_utils.is_float(dtype, type_test=True):
            dtype = np.finfo(dtype).dtype
        elif array_utils.is_int(dtype, type_test=True) or array_utils.is_uint(
            dtype, type_test=True
        ):
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
        if size < 10**8:
            block_shape = shape
            return block_shape

        if len(shape) < len(cluster_shape):
            cluster_shape = cluster_shape[: len(shape)]
        elif len(shape) > len(cluster_shape):
            cluster_shape = list(cluster_shape)
            for axis in range(len(shape)):
                if axis >= len(cluster_shape):
                    cluster_shape.append(1)
            cluster_shape = tuple(cluster_shape)

        shape_np = np.array(shape, dtype=int)
        # Softmax on cluster shape gives strong preference to larger dimensions.
        cluster_weights = np.exp(np.array(cluster_shape)) / np.sum(
            np.exp(cluster_shape)
        )
        shape_fracs = np.array(shape) / np.sum(shape)
        # cluster_weights weight the proportion of cores available along each axis,
        # and shape_fracs is the proportion of data along each axis.
        weighted_shape_fracs = cluster_weights * shape_fracs
        weighted_shape_fracs = weighted_shape_fracs / np.sum(weighted_shape_fracs)

        # Compute dimensions of grid shape
        # so that the number of blocks are close to the number of cores.
        grid_shape_frac = num_cores**weighted_shape_fracs
        grid_shape = np.floor(grid_shape_frac)
        # Put remainder on largest axis.
        remaining = np.sum(grid_shape_frac - grid_shape)
        grid_shape[np.argmax(shape)] += remaining
        grid_shape = np.ceil(grid_shape).astype(int)

        # We use ceiling of floating block shape
        # so that resulting grid shape is <= to what we compute above.
        block_shape = tuple((shape_np + grid_shape - 1) // grid_shape)
        return block_shape

    def compute_block_shape(
        self,
        shape: tuple,
        dtype: Union[type, np.dtype],
        cluster_shape=None,
        num_cores=None,
    ):

        if num_cores is None:
            num_cores = self.num_cores_total()

        if cluster_shape is None:
            cluster_shape = self.device_grid.grid_shape

        return KernelManager.compute_block_shape_static(
            shape, dtype, cluster_shape, num_cores
        )

    def update_block_shape_map(self, shape_dim, block_shape_dim):
        if shape_dim in self._block_shape_map:
            if self._block_shape_map[shape_dim] != block_shape_dim:
                warnings.warn(
                    "Block size differs for dimensions of size %s, "
                    "this may cause some operations to be slower." % shape_dim
                )
        self._block_shape_map[shape_dim] = block_shape_dim

    def get_block_shape(self, shape, dtype):
        # Simple way to ensure shape compatibility for basic linear algebra operations.
        block_shape = self.compute_block_shape(shape, dtype)
        final_block_shape = []

        for axis in range(len(shape)):
            shape_dim = shape[axis]
            block_shape_dim = block_shape[axis]
            if shape_dim not in self._block_shape_map:
                self.update_block_shape_map(shape_dim, block_shape_dim)
            final_block_shape.append(self._block_shape_map[shape_dim])
        return tuple(final_block_shape)

    def get_fuseable(self, name):
        return self.fuseable_functions[name]


def instance() -> KernelManager:
    return KernelManager.instance
