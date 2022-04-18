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


import logging
import sys

import numpy as np

from nums.core import settings
from nums.core.array.application import ArrayApplication
from nums.core.kernel import numpy_kernel
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.grid.grid import DeviceGrid, CyclicDeviceGrid, PackedDeviceGrid
from nums.core.filesystem import FileSystem
from nums.core.backends import Backend
from nums.core.backends import (
    SerialBackend,
    RayBackend,
    RayBackendStockScheduler,
    GPUSerialBackend,
    GPUParallelBackend,
    GPURayActorBackend,
    GPUIntraBackend,
    MPIBackend,
)
from nums.core.backends import utils as backend_utils

# pylint: disable=global-statement


_instance: ArrayApplication = None
_call_on_create: list = []


def call_on_create(func):
    global _call_on_create
    # Always include funcs in _call_on_create.
    # If the app is destroyed, the hooks need to be invoked again on creation.
    _call_on_create.append(func)
    if is_initialized():
        func(_instance)


def is_initialized():
    return _instance is not None


def instance():
    # Lazy-initialize to initialize on use instead of initializing on import.
    global _instance
    if _instance is None:
        _instance = create()
        for func in _call_on_create:
            func(_instance)
    return _instance


def create():
    configure_logging()

    global _instance

    if _instance is not None:
        raise Exception("create() called more than once.")

    num_cpus = (
        int(backend_utils.get_num_cores())
        if settings.num_cpus is None
        else settings.num_cpus
    )
    cluster_shape = (1, 1) if settings.cluster_shape is None else settings.cluster_shape

    # Initialize kernel interface and backend.
    backend_name = settings.backend_name
    if backend_name == "serial":
        backend: Backend = SerialBackend(num_cpus)
    elif backend_name == "ray":
        use_head = settings.use_head
        num_devices = int(np.product(cluster_shape))
        backend: Backend = RayBackend(
            address=settings.address,
            use_head=use_head,
            num_nodes=num_devices,
            num_cpus=num_cpus,
        )
    elif backend_name == "mpi":
        backend: Backend = MPIBackend()




    # TODO: These are temporary names, change once finalized

    #TODO: Don't really need a head node for GPU support in general.
    elif backend_name == "gpu":
        backend: Backend = GPUSerialBackend()
    elif backend_name == "gpu-ray":
        num_nodes = int(np.product(cluster_shape))
        backend: Backend = GPUParallelBackend(
            address=settings.address,
            use_head=True,  # TODO: temporary
            num_nodes=num_nodes,
            num_cpus=settings.num_cpus,
            num_gpus=settings.num_gpus,
        )
    elif backend_name == "gpu-ray-actor":
        num_nodes = int(np.product(cluster_shape))
        backend: Backend = GPURayActorBackend(
            address=settings.address,
            use_head=True,  # TODO: temporary
            num_nodes=num_nodes,
            num_cpus=settings.num_cpus,
            num_gpus=settings.num_gpus,
        )
    elif backend_name == "gpu-intra":
        backend: Backend = GPUIntraBackend()
        #TODO: figure out how to explicitly handle the cluster shape
        # Maybe handle a intra cluster shape?
        cluster_shape = (8, 1) # make this parameterizable based on configuration

    elif backend_name == "ray-scheduler":
        use_head = settings.use_head
        num_devices = int(np.product(cluster_shape))
        backend: Backend = RayBackendStockScheduler(
            address=settings.address,
            use_head=use_head,
            num_nodes=num_devices,
            num_cpus=num_cpus,
        )
    elif backend_name == "dask":
        # pylint: disable=import-outside-toplevel
        from nums.experimental.nums_dask.dask_backend import DaskBackend

        cluster_shape = (
            (num_cpus,) if settings.cluster_shape is None else settings.cluster_shape
        )
        num_devices = int(np.product(cluster_shape))
        backend: Backend = DaskBackend(
            address=settings.address, num_devices=num_devices, num_cpus=num_cpus
        )
    elif backend_name == "dask-scheduler":
        # pylint: disable=import-outside-toplevel
        from nums.experimental.nums_dask.dask_backend import DaskBackendStockScheduler

        cluster_shape = (
            (num_cpus,) if settings.cluster_shape is None else settings.cluster_shape
        )
        num_devices = int(np.product(cluster_shape))
        backend: Backend = DaskBackendStockScheduler(
            address=settings.address, num_devices=num_devices, num_cpus=num_cpus
        )
    else:
        raise Exception("Unexpected backend name %s" % settings.backend_name)
    backend.init()

    device_type = "cpu"
    # TODO: this is temporary
    if backend_name == "gpu" or backend_name == "gpu-ray" or backend_name == "gpu-ray-actor":
        from nums.core.kernel import cupy_kernel

        kernel_module = {"cupy": cupy_kernel}[
            "cupy"
        ]
        device_type = "gpu"
    elif backend_name == "gpu-intra":
        from nums.core.kernel import cupy_parallel_kernel

        kernel_module = {"cupy":  cupy_parallel_kernel}[
            "cupy"
        ]
        device_type = "gpu"
    else:
        kernel_module = {"numpy": numpy_kernel}[settings.kernel_name]

    if settings.device_grid_name == "cyclic":
        device_grid: DeviceGrid = CyclicDeviceGrid(
            cluster_shape, device_type, backend.devices()
        )
    elif settings.device_grid_name == "packed":
        device_grid: DeviceGrid = PackedDeviceGrid(
            cluster_shape, device_type, backend.devices()  # TODO: GPU?
        )
    else:
        raise Exception("Unexpected device grid name %s" % settings.device_grid_name)

    km = KernelManager.create(backend, kernel_module, device_grid)
    fs = FileSystem(km)
    return ArrayApplication(km, fs)


def destroy():
    global _instance
    if _instance is None:
        return
    # This will shutdown ray if ray was started by NumS.
    _instance.km.backend.shutdown()
    KernelManager.destroy()
    del _instance
    _instance = None


def configure_logging():
    # TODO (hme): Fix this to avoid debug messages for all packages.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
