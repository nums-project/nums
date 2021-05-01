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


# pylint: disable=import-outside-toplevel

def try_multiple_nd(np_atleast_nd, nps_atleast_nd):
    import nums.numpy as nps

    python_types = [
        1,
        [1, 2],
        [[1, 2]],
        [[[1, 2]]]
    ]
    all_types = python_types + list(map(np.array, python_types))
    test_cases = list(itertools.product(all_types, repeat=3))
    nps_types = list(map(nps.array, python_types))
    nps_cases = list(itertools.product(nps_types, repeat=3))

    for arguments in test_cases:
        np_res = np_atleast_nd(*arguments)
        nps_res = nps_atleast_nd(*arguments)

        for i in range(len(np_res)):
            assert np.allclose(np.array(nps_res[i].get()), np.array(np_res[i]))

    for arguments in nps_cases:
        np_args = list(map(lambda obj: np.array(obj.get()), arguments))
        np_res = np_atleast_nd(*np_args)
        nps_res = nps_atleast_nd(*arguments)
        for i in range(len(np_res)):
            assert np.allclose(np.array(nps_res[i].get()), np.array(np_res[i]))


def test_atleast_1d(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    x = 1.0
    assert np.allclose(nps.atleast_1d(x).get(), np.atleast_1d(x))

    x_np = np.arange(9).reshape(3, 3)
    x_nps = nps.arange(9).reshape(3, 3)
    assert np.allclose(nps.atleast_1d(x_np).get(), np.atleast_1d(x_np))
    assert np.allclose(nps.atleast_1d(x_nps).get(), np.atleast_1d(x_np))
    assert np.atleast_1d(x_np) is x_np
    assert nps.atleast_1d(x_nps) is x_nps

    try_multiple_nd(np.atleast_1d, nps.atleast_1d)


def test_atleast_2d(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    x = 1.0
    assert np.allclose(nps.atleast_2d(x).get(), np.atleast_2d(x))

    x = np.arange(3.0)
    assert np.allclose(nps.atleast_2d(x).get(), np.atleast_2d(x))

    try_multiple_nd(np.atleast_2d, nps.atleast_2d)


def test_atleast_3d(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    x = 1.0
    assert np.allclose(nps.atleast_1d(x).get(), np.atleast_1d(x))

    x = np.arange(12.0).reshape(4, 3)
    assert np.allclose(nps.atleast_1d(x).get(), np.atleast_1d(x))

    try_multiple_nd(np.atleast_3d, nps.atleast_3d)


def test_hstack(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    a1 = nps.array((1, 2, 3))
    a2 = np.array((1, 2, 3))
    b = np.array((2, 3, 4))
    assert np.allclose(nps.hstack((a1, b)).get(), np.hstack((a2, b)))
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    assert np.allclose(nps.hstack((a, b)).get(), np.hstack((a, b)))


def test_vstack(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    a1 = nps.array([1, 2, 3])
    a2 = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    assert np.allclose(nps.vstack((a1, b)).get(), np.vstack((a2, b)))
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    assert np.allclose(nps.vstack((a, b)).get(), np.vstack((a, b)))


def test_dstack(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    a1 = nps.array((1, 2, 3))
    a2 = np.array((1, 2, 3))
    b = np.array((2, 3, 4))
    assert np.allclose(nps.dstack((a1, b)).get(), np.dstack((a2, b)))
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    assert np.allclose(nps.dstack((a, b)).get(), np.dstack((a, b)))


def test_row_stack(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    a1 = nps.array([1, 2, 3])
    a2 = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    assert np.allclose(nps.row_stack((a1, b)).get(), np.row_stack((a2, b)))
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    assert np.allclose(nps.row_stack((a, b)).get(), np.row_stack((a, b)))


def test_column_stack(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    a1 = nps.array((1, 2, 3))
    a2 = np.array((1, 2, 3))
    b = np.array((2, 3, 4))
    assert np.allclose(nps.column_stack((a1, b)).get(), np.column_stack((a2, b)))


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_atleast_1d(nps_app_inst)
    test_atleast_2d(nps_app_inst)
    test_atleast_3d(nps_app_inst)
    test_hstack(nps_app_inst)
    test_vstack(nps_app_inst)
    test_dstack(nps_app_inst)
    test_row_stack(nps_app_inst)
    test_column_stack(nps_app_inst)
