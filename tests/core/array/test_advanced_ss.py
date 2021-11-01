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


def test_select_assign(nps_app_inst):
    def select(idx, np_idx, arr, np_arr, idx_axes=None, idx_vals=None):
        ss = [slice(None, None) for _ in range(3)]
        ss[axis] = idx
        np_ss = [slice(None, None) for _ in range(3)]
        np_ss[axis] = np_idx
        if idx_axes:
            # If idx_axes is set, then we should not expect and exception.
            ss[idx_axes[0]], ss[idx_axes[1]] = idx_vals[0], idx_vals[1]
            np_ss[idx_axes[0]], np_ss[idx_axes[1]] = idx_vals[0], idx_vals[1]
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
            select(idx, np_idx, arr, np_arr, idx_axes=idx_axes, idx_vals=idx_values)
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
            select(mask, np_mask, arr, np_arr, idx_axes=idx_axes, idx_vals=idx_values)
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


def test_subscript_permutation(nps_app_inst):
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


def test_subscript_edge_cases(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None
    rs = nps.random.RandomState(1337)

    testset = [
        [(123,), (10,), rs.randint(123, size=7)],
        [(123, 45), (10, 20), rs.randint(123, size=13)],
    ]
    for shape, block_shape, idx in testset:
        arr: BlockArray = nps.arange(np.product(shape)).reshape(
            shape, block_shape=block_shape
        )
        np_arr = arr.get()
        np_idx = idx.get()
        result = arr[idx]
        np_result = np_arr[np_idx]
        assert np.all(np_result == result.get())

    # NumPy integer idx.
    arr: BlockArray = nps.arange(1)
    np_arr = arr.get()
    idx = nps.array(0)
    np_idx = idx.get()
    result = arr[idx]
    np_result = np_arr[np_idx]
    assert np.all(np_result == result.get())

    # NumPy array of length 1.
    arr: BlockArray = nps.arange(10)
    np_arr = arr.get()
    idx = nps.array([2])
    np_idx = idx.get()
    result = arr[idx]
    np_result = np_arr[np_idx]
    assert np.all(np_result == result.get())


if __name__ == "__main__":
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    from nums.core import application_manager

    nps_app_inst = application_manager.instance()

    # test_select_assign(nps_app_inst)
    # test_subscript_permutation(nps_app_inst)
    # test_subscript_edge_cases(nps_app_inst)
