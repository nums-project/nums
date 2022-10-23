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
import ray
import os
import nums
from nums import numpy as nps
from nums.core import settings
from nums.core.application_manager import instance

# pylint: disable=import-outside-toplevel

def test_serial_gpu():
    settings.backend_name = "gpu"
    nums.init()
    a = nps.array([[1, 2], [3, 4]])
    b = nps.array([[1, 2], [3, 4]])
    # a = nps.random.rand(1000, 1000)
    # b = nps.random.rand(1000, 1000)
    a.get()
    print(a.shape)
    print(a.block_shape)
    print(a.grid_shape)
    c = nps.add(a, b)
    # print(c.shape)
    # print(c.block_shape)
    # print(c.grid_shape)
    # # c.get()
    # print(c.get())
    assert c.shape == a.shape


def test_serial_gpu_benchmark():
    settings.backend_name = "gpu"
    nums.init()
    import time

    a = nps.random.rand(1000, 1000)
    b = nps.random.rand(1000, 1000)
    begin = time.time()
    c = a @ b
    c.touch()
    c.get()
    end = time.time()
    print(end - begin)


def test_serial_cpu_benchmark():
    settings.backend_name = "ray"
    nums.init()
    import time

    a = nps.random.rand(1000, 1000)
    b = nps.random.rand(1000, 1000)
    begin = time.time()
    c = a @ b
    c.touch()
    end = time.time()
    print(end - begin)


def test_ray_gpu():
    settings.backend_name = "gpu-intra"
    settings.device_grid_name = "packed"
    # settings.backend_name = "gpu"
    print(settings.backend_name)
    # num_gpus = 4
    # ray.init(num_gpus=num_gpus, num_cpus=42)
    # nums.init(cluster_shape=(4, 4), num_gpus=num_gpus, num_cpus=42)
    nums.init()
    os.environ["RAY_PROFILING"] = "1"
    import time
    # time.sleep(5)
    n = 10 ** 4
    a = nps.random.rand(n, n)
    # a.get()
    b = nps.random.rand(n, n)

    # if settings.backend_name == "gpu-intra":
    #     a = a.reshape(block_shape=(a.block_shape[0] // 8, a.block_shape[1] // 8))
    #     b = b.reshape(block_shape=(b.block_shape[0] // 8, b.block_shape[1] // 8))
    a.touch()
    b.touch()
    # time.sleep(5)
    print(a.grid_shape, a.block_shape)
    begin = time.time()
    c = a @ b
    # c += a * b
    # c += a - b
    # c += a * b
    # c += a - b
    # c += a * b
    # c += a - b
    # c += a * b
    # c += a - b

    # d = a + b
    # e = a + b
    # f = a + b

    c.touch()

    # c.get()
    end = time.time()
    print(end - begin)
    # time.sleep(5)
    # ray.timeline(filename="/tmp/timeline.json")


if __name__ == "__main__":
    # test_serial_gpu()
    test_ray_gpu()
