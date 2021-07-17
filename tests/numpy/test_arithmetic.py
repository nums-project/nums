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


# pylint: disable=import-outside-toplevel, no-member
import numpy as np
import pytest

from nums.numpy import BlockArray
from nums.numpy import numpy_utils
from nums.core.array.utils import is_array_like
from nums import numpy as nps


def test_ufunc(nps_app_inst):
    assert nps_app_inst is not None

    uops, bops = numpy_utils.ufunc_op_signatures()
    for name, _ in sorted(uops):
        if name in ("arccosh", "arcsinh"):
            np_val = np.array([np.e])
        elif (
            name == "invert" or name.startswith("bitwise") or name.startswith("logical")
        ):
            np_val = np.array([True, False], dtype=np.bool_)
        else:
            np_val = np.array([0.1, 0.2, 0.3])
        ns_val = nps.array(np_val)
        ns_ufunc = nps.__getattribute__(name)
        np_ufunc = np.__getattribute__(name)
        np_result = np_ufunc(np_val)
        ns_result: BlockArray = ns_ufunc(ns_val)
        assert np.allclose(np_result, ns_result.get())

    def check_bop(_name, _np_a, _np_b):
        np_ufunc = np.__getattribute__(_name)
        ns_ufunc = nps.__getattribute__(_name)
        if _name in ("ldexp",) and str(_np_b.dtype) not in ("int", "int32", "int64"):
            return
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)
        _np_result = np_ufunc(_np_a, _np_b)
        _ns_result = ns_ufunc(_ns_a, _ns_b)
        assert np.allclose(_np_result, _ns_result.get())

    for name, _ in bops:
        if name.startswith("bitwise") or name.startswith("logical"):
            np_a = np.array([True, False, True, False], dtype=np.bool_)
            np_b = np.array([True, True, False, False], dtype=np.bool_)
            check_bop(name, np_a, np_b)
        elif name in ("gcd", "lcm"):
            np_a = np.array([8, 3, 7], dtype=int)
            np_b = np.array([4, 12, 13], dtype=int)
            check_bop(name, np_a, np_b)
        elif name.endswith("shift"):
            np_a = np.array([7 * 10 ** 3, 8 * 10 ** 3, 9 * 10 ** 3], dtype=int)
            np_b = np.array([1, 2, 3], dtype=int)
            check_bop(name, np_a, np_b)
        else:
            pairs = [
                (np.array([0.1, 5.0, 0.3]), np.array([0.2, 6.0, 0.3])),
                (np.array([0.1, 5.0, 0.3]), np.array([4, 2, 6], dtype=int)),
                (np.array([3, 7, 3], dtype=int), np.array([4, 2, 6], dtype=int)),
            ]
            for np_a, np_b in pairs:
                check_bop(name, np_a, np_b)


def test_matmul_tensordot(nps_app_inst):
    assert nps_app_inst is not None

    def check_matmul_op(_np_a, _np_b):
        _name = "matmul"
        np_ufunc = np.__getattribute__(_name)
        ns_ufunc = nps.__getattribute__(_name)
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)
        _np_result = np_ufunc(_np_a, _np_b)
        _ns_result = ns_ufunc(_ns_a, _ns_b)
        assert np.allclose(_np_result, _ns_result.get())

    def check_tensordot_op(_np_a, _np_b, axes):
        _name = "tensordot"
        np_ufunc = np.__getattribute__(_name)
        ns_ufunc = nps.__getattribute__(_name)
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)
        _np_result = np_ufunc(_np_a, _np_b, axes=axes)
        _ns_result = ns_ufunc(_ns_a, _ns_b, axes=axes)
        assert np.allclose(_np_result, _ns_result.get())

    np_v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    np_w = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    check_tensordot_op(np_v, np_w, axes=1)
    check_tensordot_op(np_v, np_v, axes=1)

    np_A = np.stack([np_v + i for i in range(20)])
    np_B = np.transpose(np.stack([np_w + i for i in range(20)]))
    check_matmul_op(np_A, np_B)


def test_matmul_tensor_error(nps_app_inst):
    assert nps_app_inst is not None

    # TODO (bcp): Replace with matmul tests for rank > 2 once implemented.
    def check_matmul_tensor_error(_ns_a, _ns_b):
        with pytest.raises(NotImplementedError):
            nps.matmul(_ns_a, _ns_b)

    ns_a = nps.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    ns_b = nps.array([[[7, 6], [5, 4]], [[3, 2], [1, 0]]])
    check_matmul_tensor_error(ns_a, ns_b)


def test_inner_outer(nps_app_inst):
    assert nps_app_inst is not None
    A = np.random.randn(10)
    B = np.random.randn(10)
    nps_A = nps.array(A)
    nps_B = nps.array(B)
    assert np.allclose(np.inner(A, B), nps.inner(nps_A, nps_B).get())
    assert np.allclose(np.outer(A, B), nps.outer(nps_A, nps_B).get())


def test_broadcast(nps_app_inst):
    assert nps_app_inst is not None

    _ops = ["add", "subtract", "divide"]

    def check_basic_broadcast_correctness(
        _np_a, _np_b, _a_blockshape=None, _b_blockshape=None
    ):
        ns_a = nps.array(_np_a)
        ns_b = nps.array(_np_b)

        if _a_blockshape:
            ns_a = ns_a.reshape(block_shape=_a_blockshape)
        if _b_blockshape:
            ns_b = ns_b.reshape(block_shape=_b_blockshape)

        for _op in _ops:
            np_op = np.__getattribute__(_op)
            ns_op = nps.__getattribute__(_op)

            _np_result = np_op(_np_a, _np_b)
            _ns_result = ns_op(ns_a, ns_b)

            assert np.allclose(_np_result, _ns_result.get())

    np_a = np.random.randn(10)
    np_b = np.random.randn()
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10)
    np_b = np.random.randn(1)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 1)
    np_b = np.random.randn()
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(1, 10)
    np_b = np.random.randn(1)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(1, 10)
    np_b = np.random.randn()
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10)
    np_b = np.random.randn()
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10)
    np_b = np.random.randn(1)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10)
    np_b = np.random.randn(10)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10)
    np_b = np.random.randn(10, 1)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10)
    np_b = np.random.randn(1, 10)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10, 10)
    np_b = np.random.randn()
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10, 10)
    np_b = np.random.randn(1)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10, 10)
    np_b = np.random.randn(10)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10, 10)
    np_b = np.random.randn(1, 10)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10, 10)
    np_b = np.random.randn(10, 1)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10, 10)
    np_b = np.random.randn(10, 10)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(10, 10, 10)
    np_b = np.random.randn(10, 10, 1)
    check_basic_broadcast_correctness(np_a, np_b)

    np_a = np.random.randn(3, 20, 20)
    np_b = np.random.randn(20, 20)
    check_basic_broadcast_correctness(
        np_a, np_b, _a_blockshape=(3, 10, 10), _b_blockshape=(10, 10)
    )


def test_broadcast_block(nps_app_inst):
    assert nps_app_inst is not None

    def check_broadcast_block_correctness(block, grid_shape):
        _np_result = np.tile(block, grid_shape)

        ns_a = nps.zeros(_np_result.shape).reshape(block_shape=block.shape)
        ns_b = nps.array(block)

        _ns_result = ns_a + ns_b

        assert np.allclose(_np_result, _ns_result.get())

    check_broadcast_block_correctness(np.random.randn(10, 10), (2, 2))
    check_broadcast_block_correctness(np.random.randn(10, 10), (2, 1))
    check_broadcast_block_correctness(np.random.randn(10, 10), (1, 2))
    check_broadcast_block_correctness(np.random.randn(10, 10, 10), (2, 2, 2))
    check_broadcast_block_correctness(np.random.randn(10, 10, 10), (2, 2))


def test_broadcast_block_shape_error(nps_app_inst):
    assert nps_app_inst is not None

    _ops = ["add", "subtract", "divide"]

    def check_value_error(_ns_a, _ns_b, _a_blockshape=None, _b_blockshape=None):
        for _op in _ops:
            ns_op = nps.__getattribute__(_op)

            if _a_blockshape:
                _ns_a = _ns_a.reshape(block_shape=_a_blockshape)
            if _b_blockshape:
                _ns_b = _ns_b.reshape(block_shape=_b_blockshape)

            with pytest.raises(ValueError):
                ns_op(_ns_a, _ns_b)

    nps_A = nps.random.randn(20, 20)
    nps_B = nps.random.randn(20, 20)
    check_value_error(nps_A, nps_B, _a_blockshape=(10, 10), _b_blockshape=(2, 2))

    nps_A = nps.random.randn(20, 20)
    nps_B = nps.random.randn(20)
    check_value_error(nps_A, nps_B, _a_blockshape=(10, 10))

    nps_A = nps.random.randn(20, 20)
    nps_B = nps.random.randn(20, 1)
    check_value_error(nps_A, nps_B, _a_blockshape=(10, 10))

    nps_A = nps.random.randn(20, 20, 20)
    nps_B = nps.random.randn(20, 20, 20)
    check_value_error(nps_A, nps_B, _a_blockshape=(10, 10, 10), _b_blockshape=(2, 2, 2))

    nps_A = nps.random.randn(1, 1, 3, 100, 100)
    nps_B = nps.random.randn(10, 10)
    check_value_error(nps_A, nps_B)


def test_dot(nps_app_inst):
    assert nps_app_inst is not None
    npsa = nps.array
    A = np.random.randn(10)
    B = np.random.randn(10)
    assert np.allclose(np.dot(A, B), nps.dot(npsa(A), npsa(B)).get())
    A = np.random.randn(10, 2)
    B = np.random.randn(2, 10)
    assert np.allclose(np.dot(A, B), nps.dot(npsa(A), npsa(B)).get())
    A = np.random.randn(10, 2)
    B = np.random.randn()
    assert np.allclose(np.dot(A, B), nps.dot(npsa(A), npsa(B)).get())


def test_tensordot_shape_error(nps_app_inst):
    assert nps_app_inst is not None

    def check_tensordot_mismatch_simple_error(_np_a, _np_b, axes):
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)

        with pytest.raises(ValueError):
            np.tensordot(_np_a, _np_b, axes=axes)
        with pytest.raises(ValueError):
            nps.tensordot(_ns_a, _ns_b, axes=axes)

    def check_tensordot_axes_type_error(_np_a, _np_b, axes):
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)

        # TODO (bcp): Remove test once tensordot over multiple axes is implemented.
        if is_array_like(axes):
            with pytest.raises(NotImplementedError):
                nps.tensordot(_ns_a, _ns_b, axes=axes)
        else:
            with pytest.raises(TypeError):
                np.tensordot(_np_a, _np_b, axes=axes)
            with pytest.raises(TypeError):
                nps.tensordot(_ns_a, _ns_b, axes=axes)

    np_A = np.random.randn(2, 1)
    np_B = np.random.randn(2, 1)
    check_tensordot_mismatch_simple_error(np_A, np_B, 1)

    np_A = np.random.randn(2, 2)
    np_B = np.random.randn(2, 1)
    check_tensordot_mismatch_simple_error(np_A, np_B, 2)

    np_A = np.random.randn(2, 2, 3)
    np_B = np.random.randn(2, 2, 2)
    check_tensordot_mismatch_simple_error(np_A, np_B, 3)

    np_A = np.random.randn(2, 2, 3)
    np_B = np.random.randn(2, 2, 2)
    check_tensordot_mismatch_simple_error(np_A, np_B, 2)

    np_A = np.random.randn(2, 2, 2)
    np_B = np.random.randn(2, 2, 2)
    check_tensordot_axes_type_error(np_A, np_B, 2.1)
    check_tensordot_axes_type_error(np_A, np_B, [0, 1])


if __name__ == "__main__":
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_inner_outer(nps_app_inst)
    test_dot(nps_app_inst)
