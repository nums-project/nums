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


import numpy as np
import scipy.special

from nums.core.array.application import ArrayApplication


def test_stats(app_inst: ArrayApplication):
    np_x = np.arange(100)
    ba_x = app_inst.array(np_x, block_shape=np_x.shape)
    assert np.allclose(np.mean(np_x), app_inst.mean(ba_x).get())
    assert np.allclose(np.std(np_x), app_inst.std(ba_x).get())


def test_uops(app_inst: ArrayApplication):
    np_x = np.arange(100)
    ba_x = app_inst.array(np_x, block_shape=np_x.shape)
    assert np.allclose(np.abs(np_x), app_inst.abs(ba_x).get())
    assert np.allclose(np.linalg.norm(np_x), app_inst.norm(ba_x).get())


def test_bops(app_inst: ArrayApplication):
    # pylint: disable=no-member
    pairs = [(1, 2),
             (2.0, 3.0),
             (2, 3.0),
             (2.0, 3)]
    for a, b in pairs:
        np_a, np_b = np.array(a), np.array(b)
        ba_a, ba_b = app_inst.scalar(a), app_inst.scalar(b)
        assert np.allclose(np_a + np_b, (ba_a + ba_b).get())
        assert np.allclose(np_a - np_b, (ba_a - ba_b).get())
        assert np.allclose(np_a * np_b, (ba_a * ba_b).get())
        assert np.allclose(np_a / np_b, (ba_a / ba_b).get())
        assert np.allclose(np_a ** np_b, (ba_a ** ba_b).get())
        assert np.allclose(scipy.special.xlogy(np_a, np_b),
                           app_inst.xlogy(ba_a, ba_b).get())


def test_bools(app_inst):
    np_one, np_two = np.array(1), np.array(2)
    ba_one, ba_two = app_inst.scalar(1), app_inst.scalar(2)
    assert (ba_one < ba_two) == (np_one < np_two)
    assert (ba_one <= ba_two) == (np_one <= np_two)
    assert (ba_one > ba_two) == (np_one > np_two)
    assert (ba_one >= ba_two) == (np_one >= np_two)
    assert (ba_one == ba_two) == (np_one == np_two)
    assert (ba_one != ba_two) == (np_one != np_two)


def test_bool_reduction(app_inst):
    np_arr = np.array([True, False, True, True, False, False], dtype=np.bool_)
    ba = app_inst.array(np_arr, block_shape=(2,))
    result_sum = app_inst.sum(ba, axis=0).get()
    np_sum = np.sum(np_arr)
    assert result_sum.dtype == np_sum.dtype
    assert result_sum == np_sum


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_stats(app_inst)
    test_uops(app_inst)
    test_bops(app_inst)
    test_bools(app_inst)
    test_bool_reduction(app_inst)
