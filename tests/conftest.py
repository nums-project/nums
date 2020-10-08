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


import pytest
import ray

from nums.core.systems import numpy_compute
from nums.core.systems.systems import System, SerialSystem, RaySystem
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.schedulers import RayScheduler, TaskScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication


@pytest.fixture(scope="module", params=["serial", "ray-task", "ray-cyclic"])
def app_inst(request):
    # pylint: disable=protected-access
    app_inst = get_app(request.param)
    yield app_inst
    app_inst._system.shutdown()
    ray.shutdown()


def get_app(mode):
    if mode == "serial":
        system: System = SerialSystem(compute_module=numpy_compute)
    elif mode.startswith("ray"):
        ray.init(num_cpus=4)
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
