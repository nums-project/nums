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

import os

import numpy as np

from nums.numpy import BlockArray


# pylint: disable=import-outside-toplevel


def test_explicit_init():
    import nums
    import nums.core.application_manager as am

    nums.init()
    assert am.is_initialized()
    am.destroy()


def test_array_copy(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    ba = nps.arange(10)
    ba2 = nps.array(ba, copy=True)
    assert ba is not ba2


def test_loadtxt(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    seed = 1337
    rs = np.random.RandomState(seed)

    fname = "test_text.out"
    data = rs.random_sample(99).reshape(33, 3)

    np.savetxt(fname=fname, X=data)
    da: BlockArray = nps.loadtxt(fname)
    assert np.allclose(da.get(), data)

    os.remove(fname)
    assert not os.path.exists(fname)


def test_where(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    shapes = [
        (),
        (10 ** 6,),
        (10 ** 6, 1),
        (10 ** 5, 10)
    ]
    for shape in shapes:
        arr: BlockArray = nps.random.rand(*shape)
        x: BlockArray = nps.random.rand(*shape)
        y: BlockArray = nps.random.rand(*shape)
        if len(shape) == 1:
            bs = (shape[0] // 12,)
            arr = arr.reshape(block_shape=bs)
            x = x.reshape(block_shape=bs)
            y = y.reshape(block_shape=bs)
        elif len(shape) == 2:
            bs = (shape[0] // 12, shape[1])
            arr = arr.reshape(block_shape=bs)
            x = x.reshape(block_shape=bs)
            y = y.reshape(block_shape=bs)
        results: tuple = nps.where(arr < 0.5)
        np_results = np.where(arr.get() < 0.5)
        for i in range(len(np_results)):
            assert np.allclose(np_results[i], results[i].get())
        results: tuple = nps.where(arr >= 0.5)
        np_results = np.where(arr.get() >= 0.5)
        for i in range(len(np_results)):
            assert np.allclose(np_results[i], results[i].get())

        # Do an xy test.
        np_results = np.where(arr.get() < 0.5, x.get(), y.get())
        result = nps.where(arr < 0.5, x, y)
        assert np.allclose(np_results, result.get())

        np_results = np.where(arr.get() >= 0.5, x.get(), y.get())
        result = nps.where(arr >= 0.5, x, y)
        assert np.allclose(np_results, result.get())


def test_reshape(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None
    ba = nps.arange(2 * 3 * 4).reshape((2, 3, 4), block_shape=(2, 3, 4))
    assert nps.allclose(ba.reshape((6, 4), block_shape=(6, 4)),
                        nps.reshape(ba, shape=(6, 4)))


def test_all_alltrue_any(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    true_int = np.array([[1, 2, 3], [1, 2, 3]])
    false_int = np.array([[1, 2, 0], [1, 2, 3]])
    true_bool = np.array([[True, True, True], [True, True, True]])
    false_bool = np.array([[True, False, False], [False, True, True]])
    true_float = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    false_float = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 3.0]])

    checks = [true_int, false_int, true_bool, false_bool, true_float, false_float]

    for array in checks:
        nps_array = nps.array(array).reshape(block_shape=(2, 2))
        assert nps.all(nps_array).get() == np.all(array)
        assert nps.alltrue(nps_array).get() == np.alltrue(array)

        assert nps.all(nps_array).dtype is bool
        assert nps.alltrue(nps_array).dtype is bool

    false_int = np.array([[0, 0, 0], [0, 0, 0]])
    false_bool = np.array([[False, False, False], [False, False, False]])
    false_float = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    checks = [true_int, false_int, true_bool, false_bool, true_float, false_float]

    for array in checks:
        nps_array = nps.array(array).reshape(block_shape=(2, 2))
        assert nps.any(nps_array).get() == np.any(array)

        assert nps.any(nps_array).dtype is bool


def test_array_eq(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    int_array_1 = np.array([[1, 2, 3], [4, 5, 6]])
    int_array_2 = np.array([[3, 9, 1], [8, 4, 2]])
    bool_array_1 = np.array([[True, False, True], [True, False, True]])
    bool_array_2 = np.array([[False, False, True], [False, False, True]])
    float_array_1 = np.array([[1e10, 1e-8, 1e-8], [1e10, 1e-8, 1e-8]])
    float_array_2 = np.array([[1.00001e10, 1e-9, 1e-9], [1.00001e10, 1e-9, 1e-9]])

    checks = [(int_array_1, int_array_2),
              (bool_array_1, bool_array_2),
              (float_array_1, float_array_2)]

    for check in checks:
        nps_array_1 = nps.array(check[0]).reshape(block_shape=(2, 2))
        nps_array_2 = nps.array(check[1]).reshape(block_shape=(2, 2))
        assert nps.array_equal(nps_array_1, nps_array_1).get() == np.array_equal(check[0], check[0])
        assert nps.array_equal(nps_array_1, nps_array_2).get() == np.array_equal(check[0], check[1])
        assert nps.array_equiv(nps_array_1, nps_array_1).get() == np.array_equiv(check[0], check[0])
        assert nps.array_equiv(nps_array_1, nps_array_2).get() == np.array_equiv(check[0], check[1])
        assert nps.allclose(nps_array_1, nps_array_1).get() == np.allclose(check[0], check[0])
        assert nps.allclose(nps_array_1, nps_array_2).get() == np.allclose(check[0], check[1])

        assert nps.array_equal(nps_array_1, nps_array_2).dtype is bool
        assert nps.array_equiv(nps_array_1, nps_array_2).dtype is bool
        assert nps.allclose(nps_array_1, nps_array_2).dtype is bool

    # False interaction test
    checks_1 = [np.array([False]), np.array([False]),
                np.array([0]), np.array([0]),
                np.array([0.0]), np.array([0.0])]
    checks_2 = [np.array([0]), np.array([0.0]),
                np.array([False]), np.array([0.0]),
                np.array([False]), np.array([0])]
    for check_1, check_2 in zip(checks_1, checks_2):
        nps_check_1 = nps.array(check_1)
        nps_check_2 = nps.array(check_2)
        assert nps.array_equal(nps_check_1, nps_check_2) == \
               np.array_equal(check_1, check_2)
        assert nps.array_equiv(nps_check_1, nps_check_2) == \
               np.array_equiv(check_1, check_2)

    # Infinity interaction test
    assert nps.array_equal(nps.array([nps.inf, nps.NINF]), nps.array([nps.NINF, nps.inf])) == \
           np.array_equal(np.array([np.inf, np.NINF]), np.array([np.NINF, np.inf]))


def test_properties(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None
    A: BlockArray = nps.random.randn(10, 20, 1)
    assert A.shape == nps.shape(A)
    assert A.size == nps.size(A)
    assert A.ndim == nps.ndim(A)
    assert A.squeeze().shape == nps.squeeze(A).shape
    assert nps.allclose(A.T, nps.transpose(A))
    A_copy = nps.copy(A)
    assert A_copy is not A


if __name__ == "__main__":
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    # test_where(nps_app_inst)
    # test_loadtxt(nps_app_inst)
    # test_reshape(nps_app_inst)
    # test_all_alltrue_any(nps_app_inst)
    # test_array_eq(nps_app_inst)
    test_properties(nps_app_inst)
