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


from nums.core import settings
from nums.core.systems.filesystem import FileSystem
from nums.core.systems import numpy_compute
from nums.core.systems.systems import System, SerialSystem, RaySystem
from nums.core.systems.schedulers import RayScheduler, TaskScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication


_instance: ArrayApplication = None


def instance():
    # pylint: disable=global-statement
    # Lazy-initialize to initialize on use instead of initializing on import.
    global _instance
    if _instance is None:
        _instance = create()
    return _instance


def create():
    # pylint: disable=global-statement
    global _instance

    if _instance is not None:
        raise Exception("create() called more than once.")

    system_name = settings.system_name

    compute_module = {
        "numpy": numpy_compute
    }[settings.compute_name]

    if system_name == "serial":
        system: System = SerialSystem(compute_module=compute_module)
    elif system_name == "ray-task":
        scheduler: RayScheduler = TaskScheduler(compute_module=compute_module,
                                                use_head=settings.use_head)
        system: System = RaySystem(compute_module=compute_module,
                                   scheduler=scheduler)
    elif system_name == "ray-cyclic":
        cluster_shape = settings.cluster_shape
        scheduler: RayScheduler = BlockCyclicScheduler(compute_module=compute_module,
                                                       cluster_shape=cluster_shape,
                                                       use_head=settings.use_head)
        system: System = RaySystem(compute_module=compute_module,
                                   scheduler=scheduler)
    else:
        raise Exception()
    system.init()
    return ArrayApplication(system=system, filesystem=FileSystem(system))
