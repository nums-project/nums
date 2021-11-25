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


def test_quantile_percentile(app_inst: ArrayApplication):
    # see https://github.com/dask/dask/blob/main/dask/array/tests/test_percentiles.py
    qs = [0, 50, 100]
    methods = ["tdigest"]
    interpolations = ["linear"]

    np_x = np.ones((10,))
    ba_x = app_inst.ones(shape=(10,), block_shape=(2,))
    for q, method, interpolation in itertools.product(qs, methods, interpolations):
        assert app_inst.quantile(
            ba_x, q / 100, method=method, interpolation=interpolation
        ).get() == np.quantile(np_x, q / 100)
        assert app_inst.percentile(
            ba_x, q, method=method, interpolation=interpolation
        ).get() == np.percentile(np_x, q)

    np_x = np.array([0, 0, 5, 5, 5, 5, 20, 20])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    for q, method, interpolation in itertools.product(qs, methods, interpolations):
        assert app_inst.quantile(
            ba_x, q / 100, method=method, interpolation=interpolation
        ).get() == np.quantile(np_x, q / 100)
        assert app_inst.percentile(
            ba_x, q, method=method, interpolation=interpolation
        ).get() == np.percentile(np_x, q)


def test_quickselect(app_inst: ArrayApplication):
    # pylint: disable=protected-access
    # Simple tests
    np_x = np.array([3, 7, 2, 4, 5, 1, 5, 6])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    ba_oids = ba_x.flattened_oids()
    correct = [7, 6, 5, 5, 4, 3, 2, 1]
    for i in range(-8, 8):
        value_oid = app_inst._quickselect(ba_oids, i)
        value = app_inst.cm.get(value_oid)
        assert value == correct[i]

    # Randomized tests
    shapes = [(50,), (437,), (1000,)]
    block_shapes = [(10,), (23,), (50,)]
    kth = [-50, -42, -25, -13, 0, 8, 25, 36, 49]
    for shape, block_shape, k in itertools.product(shapes, block_shapes, kth):
        ba_x = app_inst.random.random(shape=shape, block_shape=block_shape)
        ba_oids = ba_x.flattened_oids()
        value_oid = app_inst._quickselect(ba_oids, k)
        value = app_inst.cm.get(value_oid)
        assert value == np.partition(ba_x.get(), -k - 1)[-k - 1]


def test_median(app_inst: ArrayApplication):
    # Simple tests
    np_x = np.array([7, 2, 4, 5, 1, 5, 6])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    assert app_inst.median(ba_x).get() == np.median(np_x)

    np_x = np.array([3, 7, 2, 4, 5, 1, 5, 6])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    assert app_inst.median(ba_x).get() == np.median(np_x)

    # Randomized tests
    shapes = [(50,), (437,), (1000,)]
    block_shapes = [(10,), (23,), (50,)]
    for shape, block_shape in itertools.product(shapes, block_shapes):
        ba_x = app_inst.random.random(shape=shape, block_shape=block_shape)
        assert app_inst.median(ba_x).get() == np.median(ba_x.get())


def test_top_k(app_inst: ArrayApplication):
    # Simple tests
    np_x = np.array([3, 7, 2, 4, 5, 1, 5, 6])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    for k in range(1, len(np_x) + 1):
        # Largest
        ba_v, ba_i = app_inst.top_k(ba_x, k)
        np_v = np.partition(np_x, -k)[-k:]
        assert len(ba_v.get()) == k and len(ba_i.get()) == k
        for v, i in zip(ba_v.get(), ba_i.get()):
            assert v in np_v
            assert np_x[i] == v
        # Smallest
        ba_v, ba_i = app_inst.top_k(ba_x, k, largest=False)
        np_v = np.partition(np_x, k - 1)[:k]
        assert len(ba_v.get()) == k and len(ba_i.get()) == k
        for v, i in zip(ba_v.get(), ba_i.get()):
            assert v in np_v
            assert np_x[i] == v

    # Randomized tests
    shapes = [(50,), (437,), (1000,)]
    block_shapes = [(10,), (23,), (50,)]
    ks = range(1, 51, 15)
    for shape, block_shape, k in itertools.product(shapes, block_shapes, ks):
        ba_x = app_inst.random.random(shape=shape, block_shape=block_shape)
        np_x = ba_x.get()
        # Largest
        ba_v, ba_i = app_inst.top_k(ba_x, k)
        np_v = np.partition(np_x, -k)[-k:]
        assert len(ba_v.get()) == k and len(ba_i.get()) == k
        for v, i in zip(ba_v.get(), ba_i.get()):
            assert v in np_v
            assert np_x[i] == v
        # Smallest
        ba_v, ba_i = app_inst.top_k(ba_x, k, largest=False)
        np_v = np.partition(np_x, k - 1)[:k]
        assert len(ba_v.get()) == k and len(ba_i.get()) == k
        for v, i in zip(ba_v.get(), ba_i.get()):
            assert v in np_v
            assert np_x[i] == v


def test_cov(app_inst):

    np_x = np.arange(30).reshape(10, 3)
    ba_x = app_inst.array(np_x, block_shape=(3, 2))
    assert np.allclose(
        np.cov(np_x, rowvar=False, bias=False),
        app_inst.cov(ba_x, rowvar=False, bias=False).get(),
    )
    assert np.allclose(
        np.cov(np_x, rowvar=False, bias=True),
        app_inst.cov(ba_x, rowvar=False, bias=True).get(),
    )
    assert np.allclose(
        np.cov(np_x, rowvar=True, bias=False),
        app_inst.cov(ba_x, rowvar=True, bias=False).get(),
    )
    assert np.allclose(
        np.cov(np_x, rowvar=True, bias=True),
        app_inst.cov(ba_x, rowvar=True, bias=True).get(),
    )
    assert (
        np.cov(np_x, dtype=np.float16).dtype
        == app_inst.cov(ba_x, dtype=np.float16).get().dtype
    )


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "serial"
    app_inst = application_manager.instance()

    test_quantile_percentile(app_inst)
    test_cov(app_inst)
    # test_quickselect(app_inst)
    # test_median(app_inst)
    # test_top_k(app_inst)
