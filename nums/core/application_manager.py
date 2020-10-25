# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from nums.core import settings
from nums.core.systems.filesystem import FileSystem
from nums.core.systems import numpy_compute
from nums.core.systems.systems import System, SerialSystem, RaySystem
from nums.core.systems.schedulers import RayScheduler, TaskScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication


_instance: ArrayApplication = None


def instance():
    # pylint: disable=global-statement
    global _instance
    if _instance is None:
        _instance = create()
    return _instance


def create():
    # pylint: disable=global-statement
    global _instance

    if _instance is not None:
        raise Exception("init called more than once.")

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
