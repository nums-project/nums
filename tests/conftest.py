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

import pytest
import ray

from nums.core.array.application import ArrayApplication
from nums.core.compute import numpy_compute
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import DeviceGrid, CyclicDeviceGrid, PackedDeviceGrid
from nums.core.systems import utils as systems_utils
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.systems import SystemInterface, SerialSystem, RaySystem


@pytest.fixture(scope="module", params=[("serial", "cyclic"), ("ray", "packed")])
def nps_app_inst(request):
    # This triggers initialization; it's not to be mixed with the app_inst fixture.
    # Observed (core dumped) after updating this fixture to run functions with "serial" backend.
    # Last time this happened, it was due poor control over the
    # scope and duration of ray resources.
    # pylint: disable = import-outside-toplevel
    from nums.core import settings
    from nums.core import application_manager
    import nums.numpy as nps
    settings.system_name, settings.device_grid_name = request.param

    # Need to reset numpy random state.
    # It's the only stateful numpy API object.
    nps.random.reset()
    yield application_manager.instance()
    application_manager.destroy()


@pytest.fixture(scope="module", params=[("serial", "cyclic"), ("ray", "packed")])
def app_inst(request):
    # pylint: disable=protected-access
    app_inst = get_app(*request.param)
    yield app_inst
    app_inst.cm.destroy()
    ray.shutdown()


@pytest.fixture(scope="module", params=[("serial", "cyclic")])
def app_inst_s3(request):
    # pylint: disable=protected-access
    app_inst = get_app(*request.param)
    yield app_inst
    app_inst.cm.destroy()
    ray.shutdown()


@pytest.fixture(scope="module", params=[("serial", "cyclic"), ("ray", "cyclic"), ("ray", "packed")])
def app_inst_all(request):
    # pylint: disable=protected-access
    app_inst = get_app(*request.param)
    yield app_inst
    app_inst.cm.destroy()
    ray.shutdown()


def get_app(system_name, device_grid_name="cyclic"):
    if system_name == "serial":
        system: SystemInterface = SerialSystem()
    elif system_name == "ray":
        assert not ray.is_initialized()
        ray.init(num_cpus=systems_utils.get_num_cores())
        system: SystemInterface = RaySystem(use_head=True)
    else:
        raise Exception("Unexpected system name %s" % system_name)
    system.init()

    cluster_shape = (1, 1)
    if device_grid_name == "cyclic":
        device_grid: DeviceGrid = CyclicDeviceGrid(cluster_shape, "cpu", system.devices())
    elif device_grid_name == "packed":
        device_grid: DeviceGrid = PackedDeviceGrid(cluster_shape, "cpu", system.devices())
    else:
        raise Exception("Unexpected device grid name %s" % device_grid_name)

    cm = ComputeManager.create(system, numpy_compute, device_grid)
    fs = FileSystem(cm)
    return ArrayApplication(cm, fs)
