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


import pytest
import numpy as np

from nums.numpy import BlockArray


# pylint: disable=import-outside-toplevel, protected-access


def test_basic(nps_app_inst):
    import nums.numpy as nps

    app = nps_app_inst

    x_api = nps.random.RandomState(1337).random_sample((500, 1))
    x_app = app.random_state(1337).random(
        shape=x_api.shape, block_shape=x_api.block_shape
    )
    assert nps.allclose(x_api, x_app)

    x_api = nps.random.RandomState(1337).rand(500, 1)
    x_app = app.random_state(1337).random(
        shape=x_api.shape, block_shape=x_api.block_shape
    )
    assert nps.allclose(x_api, x_app)

    x_api = nps.random.RandomState(1337).randn(500, 1) + 5.0
    x_app = app.random_state(1337).normal(
        loc=5.0, shape=x_api.shape, block_shape=x_api.block_shape
    )
    assert nps.allclose(x_api, x_app)

    x_api = nps.random.RandomState(1337).randint(0, 10, size=(100, 1))
    x_app = app.random_state(1337).integers(
        0, 10, shape=x_api.shape, block_shape=x_api.block_shape
    )
    assert nps.allclose(x_api, x_app)


def test_shuffle(nps_app_inst):
    def select(idx, np_idx, arr, np_arr):
        if axis == 0:
            arr_shuffle = arr[idx]
            np_arr_shuffle = np_arr[np_idx]
        else:
            ss = [slice(None, None) for _ in range(3)]
            ss[axis] = idx
            np_ss = [slice(None, None) for _ in range(3)]
            np_ss[axis] = np_idx
            arr_shuffle = arr[tuple(ss)]
            np_arr_shuffle = np_arr[tuple(np_ss)]
        assert np.all(np_arr_shuffle == arr_shuffle.get())

    def assign(idx, np_idx, arr, np_arr, axis, mode, idx_axes=None, idx_vals=None):
        arr = arr.copy()
        np_arr = np_arr.copy()
        if mode == "scalar":
            np_value = np.random.randint(-np.product(shape), -1, size=1).item()
        elif mode == "single-dim":
            np_value = np.random.randint(-np.product(shape), -1, size=len(np_idx))
        elif mode == "multi-dim":
            value_shape = tuple(
                list(np_arr.shape[:axis])
                + [len(np_idx)]
                + list(np_arr.shape[axis + 1 :])
            )
            np_value = np.random.randint(-np.product(shape), -1, size=value_shape)
        else:
            raise Exception()
        value = nps.array(np_value)

        ss = [slice(None, None) for _ in range(3)]
        ss[axis] = idx
        np_ss = [slice(None, None) for _ in range(3)]
        np_ss[axis] = np_idx
        if mode == "single-dim" and axis != 2:
            if idx_axes:
                # If idx_axes is set, then we should not expect and exception.
                ss[idx_axes[0]], ss[idx_axes[1]] = idx_vals[0], idx_vals[1]
                np_ss[idx_axes[0]], np_ss[idx_axes[1]] = idx_vals[0], idx_vals[1]
                arr[tuple(ss)] = value
                np_arr[tuple(np_ss)] = np_value
                assert np.all(np_arr == arr.get())
            else:
                with pytest.raises(ValueError):
                    np_arr[tuple(np_ss)] = np_value
                with pytest.raises(ValueError):
                    arr[tuple(ss)] = value
        else:
            if mode == "scalar":
                # Run indexed subscripts on scalar values.
                if idx_axes:
                    ss[idx_axes[0]], ss[idx_axes[1]] = idx_vals[0], idx_vals[1]
                    np_ss[idx_axes[0]], np_ss[idx_axes[1]] = idx_vals[0], idx_vals[1]
            arr[tuple(ss)] = value
            np_arr[tuple(np_ss)] = np_value
            assert np.all(np_arr == arr.get())

    import nums.numpy as nps

    assert nps_app_inst is not None

    shape = (12, 34, 56)
    block_shape = (2, 5, 7)
    arr: BlockArray = nps.arange(np.product(shape)).reshape(
        shape, block_shape=block_shape
    )
    np_arr = arr.get()

    for axis in range(3):
        idx_axes = [(1, 2), (0, 2), None][axis]
        idx_values = [(13, 40), (3, 55), None][axis]
        for axis_frac in (1.0, 0.5):
            rs = nps.random.RandomState(1337)
            idx = rs.permutation(int(shape[axis] * axis_frac))
            np_idx = idx.get()
            select(idx, np_idx, arr, np_arr)
            for mode in ["scalar", "single-dim", "multi-dim"]:
                assign(idx, np_idx, arr, np_arr, axis, mode)
                assign(
                    idx,
                    np_idx,
                    arr,
                    np_arr,
                    axis,
                    mode,
                    idx_axes=idx_axes,
                    idx_vals=idx_values,
                )
            # Also test boolean mask.
            np_mask = np.zeros(shape[axis], dtype=bool)
            np_mask[np_idx] = True
            mask = nps.array(np_mask)
            assert np.allclose(mask.get(), np_mask)
            select(mask, np_mask, arr, np_arr)
            for mode in ["scalar", "single-dim", "multi-dim"]:
                assign(idx, np_idx, arr, np_arr, axis, mode)
                assign(
                    idx,
                    np_idx,
                    arr,
                    np_arr,
                    axis,
                    mode,
                    idx_axes=idx_axes,
                    idx_vals=idx_values,
                )


def test_shuffle_subscript_ops(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    shape = (123, 45)
    block_shape = (10, 20)
    arr: BlockArray = nps.arange(np.product(shape)).reshape(
        shape, block_shape=block_shape
    )
    np_arr = arr.get()
    rs = nps.random.RandomState(1337)
    idx: BlockArray = rs.permutation(shape[1])
    np_idx = idx.get()
    arr_shuffle = arr[:, idx]
    np_arr_shuffle = np_arr[:, np_idx]
    assert np.all(np_arr_shuffle == arr_shuffle.get())


def test_blockarray_perm(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    shape = (12, 34)
    block_shape = (5, 10)
    arr: BlockArray = nps.arange(np.product(shape)).reshape(
        shape, block_shape=block_shape
    )
    np_arr = arr.get()
    rs = nps.random.RandomState(1337)
    np_arr_shuffle: BlockArray = rs.permutation(arr).get()
    for i in range(shape[0]):
        num_found = 0
        for j in range(shape[0]):
            if np.allclose(np_arr[i], np_arr_shuffle[j]):
                num_found += 1
        assert num_found == 1


def test_default_random(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    num1 = nps.random.random_sample()
    num2 = nps.random.random_sample()
    assert not nps.allclose(num1, num2)
    # Test default random seed.
    nps.random.seed(1337)
    num1 = nps.random.random_sample()
    nps.random.seed(1337)
    num2 = nps.random.random_sample()
    assert nps.allclose(num1, num2)


if __name__ == "__main__":
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    from nums.core import application_manager

    nps_app_inst = application_manager.instance()

    # test_basic(nps_app_inst)
    test_shuffle(nps_app_inst)
    # test_shuffle_subscript_ops(nps_app_inst)
    # test_default_random(nps_app_inst)
    # test_blockarray_perm(nps_app_inst)
