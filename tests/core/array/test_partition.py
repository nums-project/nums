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

from nums.core.array.application import ArrayApplication


def test_quickselect(app_inst: ArrayApplication):
    # Simple tests
    np_x = np.array([3, 7, 2, 4, 5, 1, 5, 6])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    ba_oids = ba_x.flattened_oids()
    correct = [1, 2, 3, 4, 5, 5, 6, 7]
    for i in range(-8, 8):
        value = app_inst.quickselect(ba_oids, i)
        if i < 0:
            assert value == correct[i + 8]
        else:
            assert value == correct[i]

    # Randomized tests
    shapes = [(50,), (437,), (1000,)]
    block_shapes = [(10,), (23,), (50,)]
    kth = [-50, -42, -25, -13, 0, 8, 25, 36, 49]
    for shape, block_shape, k in itertools.product(shapes, block_shapes, kth):
        ba_x = app_inst.random.random(shape=shape, block_shape=block_shape)
        ba_oids = ba_x.flattened_oids()
        value = app_inst.quickselect(ba_oids, k)
        if k < 0:
            assert value == np.partition(ba_x.get(), k + shape[0])[k + shape[0]]
        else:
            assert value == np.partition(ba_x.get(), k)[k]


def test_median(app_inst: ArrayApplication):
    # Simple tests
    np_x = np.array([7, 2, 4, 5, 1, 5, 6])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    assert app_inst.median(ba_x) == np.median(np_x)

    np_x = np.array([3, 7, 2, 4, 5, 1, 5, 6])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    assert app_inst.median(ba_x) == np.median(np_x)

    # Randomized tests
    shapes = [(50,), (437,), (1000,)]
    block_shapes = [(10,), (23,), (50,)]
    for shape, block_shape in itertools.product(shapes, block_shapes):
        ba_x = app_inst.random.random(shape=shape, block_shape=block_shape)
        assert app_inst.median(ba_x) == np.median(ba_x.get())


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst: ArrayApplication = conftest.get_app("serial")
    test_quickselect(app_inst)
    test_median(app_inst)
