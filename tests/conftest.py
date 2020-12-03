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

from nums.core.systems import numpy_compute
from nums.core.systems.systems import System, SerialSystem, RaySystem
from nums.core.systems import utils as systems_utils
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.schedulers import RayScheduler, TaskScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication


@pytest.fixture(scope="module", params=["serial", "ray-task", "ray-cyclic"])
def app_inst(request):
    # pylint: disable=protected-access
    app_inst = get_app(request.param)
    yield app_inst
    app_inst.system.shutdown()
    ray.shutdown()


@pytest.fixture(scope="module", params=["serial", "ray-cyclic"])
def nps_app_inst(request):
    # This triggers initialization; it's not to be mixed with the app_inst fixture.
    # Observed (core dumped) after updating this fixture to run functions with "serial" backend.
    # Last time this happened, it was due poor control over the
    # scope and duration of ray resources.
    # pylint: disable = import-outside-toplevel
    from nums.core import settings
    from nums.core import application_manager
    settings.system_name = request.param
    yield application_manager.instance()
    application_manager.destroy()


def get_app(mode):
    if mode == "serial":
        system: System = SerialSystem(compute_module=numpy_compute)
    elif mode.startswith("ray"):
        ray.init(num_cpus=systems_utils.get_num_cores())
        if mode == "ray-task":
            scheduler: RayScheduler = TaskScheduler(compute_module=numpy_compute,
                                                    use_head=True)
        elif mode == "ray-cyclic":
            cluster_shape = (1, 1)
            scheduler: RayScheduler = BlockCyclicScheduler(compute_module=numpy_compute,
                                                           cluster_shape=cluster_shape,
                                                           use_head=True,
                                                           verbose=True)
        else:
            raise Exception()
        system: System = RaySystem(compute_module=numpy_compute,
                                   scheduler=scheduler)
    else:
        raise Exception()
    system.init()
    return ArrayApplication(system=system, filesystem=FileSystem(system))
