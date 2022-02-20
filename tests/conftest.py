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

import time

import pytest
import ray

from nums.core.array.application import ArrayApplication
from nums.core.compute import numpy_compute
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import DeviceGrid, CyclicDeviceGrid, PackedDeviceGrid
from nums.core.backends import utils as backend_utils
from nums.core.backends.filesystem import FileSystem
from nums.core.backends.backends import Backend, SerialBackend, RayBackend, MPIBackend


# pylint: disable=protected-access, import-outside-toplevel


def pytest_collection_modifyitems(config, items):
    keywordexpr = config.option.keyword
    markexpr = config.option.markexpr
    if keywordexpr or markexpr:
        return  # let pytest handle this

    skip_slow = pytest.mark.skip(reason="slow tests not selected.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    parser.addoption("--backend-name")
    parser.addoption("--device-grid-name")


@pytest.fixture(scope="module")
def nps_app_inst(request):
    # This triggers initialization; it's not to be mixed with the app_inst fixture.
    # Observed (core dumped) after updating this fixture to run functions with "serial" backend.
    # Last time this happened, it was due to poor control over the
    # scope and duration of ray resources.
    from nums.core import settings
    from nums.core import application_manager
    import nums.numpy as nps
    import numpy as np

    settings.backend_name = request.config.getoption("--backend-name") or "serial"
    settings.device_grid_name = (
        request.config.getoption("--device-grid-name") or "cyclic"
    )

    # Need to reset numpy random state.
    # It's the only stateful numpy API object.
    nps.random.reset()
    np.random.seed(1331)
    yield application_manager.instance()
    if settings.backend_name == "ray":
        assert application_manager.instance().cm.backend._manage_ray
    application_manager.destroy()
    time.sleep(2)


@pytest.fixture(scope="module")
def app_inst(request):
    backend_name = request.config.getoption("--backend-name") or "serial"
    device_grid_name = request.config.getoption("--device-grid-name") or "cyclic"

    _app_inst = get_app(backend_name, device_grid_name)
    yield _app_inst
    if backend_name == "ray":
        assert _app_inst.cm.backend._manage_ray
    _app_inst.cm.backend.shutdown()
    _app_inst.cm.destroy()
    time.sleep(2)


@pytest.fixture(scope="module", params=[("serial", "cyclic")])
def app_inst_s3(request):
    _app_inst = get_app(*request.param)
    assert isinstance(_app_inst.cm.backend, SerialBackend)
    yield _app_inst
    _app_inst.cm.backend.shutdown()
    _app_inst.cm.destroy()
    time.sleep(2)


@pytest.fixture(
    scope="module",
    params=[
        ("serial", "cyclic"),
        ("dask", "cyclic"),
        ("ray", "cyclic"),
        ("ray", "packed"),
    ],
)
def app_inst_all(request):
    _app_inst = get_app(*request.param)
    yield _app_inst
    if request.param[0] == "ray":
        assert _app_inst.cm.backend._manage_ray
    _app_inst.cm.backend.shutdown()
    _app_inst.cm.destroy()
    time.sleep(2)


def get_app(backend_name, device_grid_name="cyclic"):
    if backend_name == "serial":
        backend: Backend = SerialBackend()
    elif backend_name == "ray":
        assert not ray.is_initialized()
        backend: Backend = RayBackend(
            use_head=True, num_cpus=backend_utils.get_num_cores()
        )
    elif backend_name == "dask":
        from nums.experimental.nums_dask.dask_backend import DaskBackend

        backend: Backend = DaskBackend(
            num_cpus=backend_utils.get_num_cores(), num_nodes=1
        )
    elif system_name == "mpi":
        system: SystemInterface = MPISystem()
    else:
        raise Exception("Unexpected backend name %s" % backend_name)
    backend.init()

    cluster_shape = (1, 1)
    if device_grid_name == "cyclic":
        device_grid: DeviceGrid = CyclicDeviceGrid(
            cluster_shape, "cpu", backend.devices()
        )
    elif device_grid_name == "packed":
        device_grid: DeviceGrid = PackedDeviceGrid(
            cluster_shape, "cpu", backend.devices()
        )
    else:
        raise Exception("Unexpected device grid name %s" % device_grid_name)

    cm = ComputeManager.create(backend, numpy_compute, device_grid)
    fs = FileSystem(cm)
    return ArrayApplication(cm, fs)
