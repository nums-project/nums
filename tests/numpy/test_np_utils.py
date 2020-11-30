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


import itertools

import numpy as np

from nums.core.storage.storage import ArrayGrid
from nums.core.systems import utils as systems_utils


# pylint: disable=import-outside-toplevel


def _inspect_block_shape(nps_app_inst):
    app = nps_app_inst
    dtypes = [np.float32, np.float64]
    shapes = [(10**9, 250), (10**4, 10**4), (10**7, 10), (10, 10**7)]
    cluster_shapes = [(1, 1), (2, 1), (4, 1), (16, 1)]
    cores_per_node = 64
    combos = itertools.product(dtypes, shapes, cluster_shapes)
    for dtype, shape, cluster_shape in combos:
        num_cores = np.product(cluster_shape) * cores_per_node
        block_shape = app.compute_block_shape(shape=shape,
                                              dtype=dtype,
                                              cluster_shape=cluster_shape,
                                              num_cores=num_cores)
        grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
        print()
        print("dtype=%s" % dtype.__name__,
              "cluster_shape=%s" % str(cluster_shape),
              "shape=%s" % str(shape))
        print("grid_shape", grid.grid_shape,
              "block_shape", block_shape)
        print("array size (GB)", np.product(shape)*dtype().nbytes / 10**9,
              "block size (GB)", np.product(block_shape)*dtype().nbytes / 10**9)


def test_block_shape(nps_app_inst):
    app = nps_app_inst
    dtype = np.float64
    shape = (10**9, 250)
    cluster_shape = (1, 1)
    num_cores = 64
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          cluster_shape=cluster_shape,
                                          num_cores=num_cores)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape == (num_cores, 1)

    cluster_shape = (16, 1)
    num_cores = 64 * np.product(cluster_shape)
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          cluster_shape=cluster_shape,
                                          num_cores=num_cores)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape == (num_cores, 1)

    shape = (250, 10**9)
    cluster_shape = (1, 16)
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          cluster_shape=cluster_shape,
                                          num_cores=num_cores)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape == (1, num_cores)

    shape = (10**4, 10**4)
    cluster_shape = (1, 1)
    num_cores = 64
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          cluster_shape=cluster_shape,
                                          num_cores=num_cores)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape == (int(num_cores**.5), int(num_cores**.5))

    shape = (10**4, 10**4//dtype(0).nbytes)
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          cluster_shape=cluster_shape,
                                          num_cores=num_cores)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape != (1, 1)

    shape = (10**4, 10**4//dtype(0).nbytes-1)
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          cluster_shape=cluster_shape,
                                          num_cores=num_cores)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape == (1, 1)

    shape = (10**4, 10**4)
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          num_cores=num_cores)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape == (int(num_cores**.5), int(num_cores**.5))

    cluster_shape = (12, 1)
    num_cores = systems_utils.get_num_cores()
    block_shape = app.compute_block_shape(shape=shape,
                                          dtype=dtype,
                                          cluster_shape=cluster_shape)
    grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
    assert grid.grid_shape == (num_cores, 1)


if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    _inspect_block_shape(nps_app_inst)
    test_block_shape(nps_app_inst)
