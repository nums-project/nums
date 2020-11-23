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


def test_ufunc(nps_app_inst):
    import numpy as np

    from nums.numpy import BlockArray
    from nums.numpy import numpy_utils
    from nums import numpy as nps

    assert nps_app_inst is not None

    uops, bops = numpy_utils.ufunc_op_signatures()
    for name, _ in sorted(uops):
        if name in ("arccosh", "arcsinh"):
            np_val = np.array([np.e])
        elif name == "invert" or name.startswith("bitwise") or name.startswith("logical"):
            np_val = np.array([True, False], dtype=np.bool_)
        else:
            np_val = np.array([.1, .2, .3])
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
            np_a = np.array([8, 3, 7], dtype=np.int)
            np_b = np.array([4, 12, 13], dtype=np.int)
            check_bop(name, np_a, np_b)
        elif name.endswith("shift"):
            np_a = np.array([7*10**3, 8*10**3, 9*10**3], dtype=np.int)
            np_b = np.array([1, 2, 3], dtype=np.int)
            check_bop(name, np_a, np_b)
        else:
            pairs = [
                (np.array([.1, 5.0, .3]),
                 np.array([.2, 6.0, .3])),
                (np.array([.1, 5.0, .3]),
                 np.array([4, 2, 6], dtype=np.int)),
                (np.array([3, 7, 3], dtype=np.int),
                 np.array([4, 2, 6], dtype=np.int)),
            ]
            for np_a, np_b in pairs:
                check_bop(name, np_a, np_b)


def test_matmul_tensordot(nps_app_inst):
    import numpy as np

    from nums import numpy as nps

    assert nps_app_inst is not None

    def check_matmul_op(_np_a, _np_b):
        _name = 'matmul'
        np_ufunc = np.__getattribute__(_name)
        ns_ufunc = nps.__getattribute__(_name)
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)
        _np_result = np_ufunc(_np_a, _np_b)
        _ns_result = ns_ufunc(_ns_a, _ns_b)
        assert np.allclose(_np_result, _ns_result.get())

    def check_tensordot_op(_np_a, _np_b, axes):
        _name = 'tensordot'
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


# TODO(hme): Add broadcast tests.


if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_ufunc(nps_app_inst)
