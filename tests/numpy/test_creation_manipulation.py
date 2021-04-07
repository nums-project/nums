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

from nums.numpy import BlockArray


# pylint: disable = import-outside-toplevel, no-member


def test_basic_creation(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    ops = "empty", "zeros", "ones"
    shape = (2, 3, 4)
    for op in ops:
        ba: BlockArray = nps.__getattribute__(op)(shape=shape)
        if "zeros" in op:
            assert nps.allclose(nps.zeros(shape), ba)
        if "ones" in op:
            assert nps.allclose(nps.ones(shape), ba)

        ba2: BlockArray = nps.__getattribute__(op + "_like")(ba)
        assert ba.shape == ba2.shape
        assert ba.dtype == ba2.dtype
        assert ba.block_shape == ba2.block_shape


def test_eye(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    eyes = [
        (10, 10),
        (7, 10),
        (10, 13),
    ]
    for N, M in eyes:
        ba: BlockArray = nps.eye(N, M)
        np_arr = np.eye(N, M)
        assert np.allclose(ba.get(), np_arr)
        # Also test identity.
        ba: BlockArray = nps.identity(N)
        np_arr = np.identity(N)
        assert np.allclose(ba.get(), np_arr)


def test_diag(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    ba: BlockArray = nps.array([1.0, 2.0, 3.0])
    np_arr = ba.get()
    # Make a diag matrix.
    ba = nps.diag(ba)
    np_arr = np.diag(np_arr)
    assert np.allclose(ba.get(), np_arr)
    # Take diag of diag matrix.
    ba = nps.diag(ba)
    np_arr = np.diag(np_arr)
    assert np.allclose(ba.get(), np_arr)


def test_arange(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    ba: BlockArray = nps.arange(5)
    np_arr = np.arange(5)
    assert np.allclose(ba.get(), np_arr)


def test_concatenate(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    ba1: BlockArray = nps.arange(5)
    ba2: BlockArray = nps.arange(6)
    ba = nps.concatenate((ba1, ba2))
    np_arr = np.concatenate((np.arange(5), np.arange(6)))
    assert np.allclose(ba.get(), np_arr)


def test_split(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    ba: BlockArray = nps.arange(10)
    np_arr = np.arange(10)
    ba_list = nps.split(ba, 2)
    np_arr_list = np.split(np_arr, 2)
    for i in range(len(np_arr_list)):
        assert np.allclose(ba_list[i].get(), np_arr_list[i])


def test_func_space(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    ba: BlockArray = nps.linspace(12.3, 45.6, 23).reshape(block_shape=(10,))
    np_arr = np.linspace(12.3, 45.6, 23)
    assert np.allclose(ba.get(), np_arr)
    ba: BlockArray = nps.logspace(12.3, 45.6, 23).reshape(block_shape=(10,))
    np_arr = np.logspace(12.3, 45.6, 23)
    assert np.allclose(ba.get(), np_arr)

def test_shape(nps_app_inst):
    from nums import numpy as nps
    from nums.core import application_manager

    assert nps_app_inst is not None

    shape = (10, 20, 30, 40)
    block_shape = (10, 10, 10, 10)
    ns_ins = application_manager.instance()

    def check_expand_and_squeeze(_np_a, axes):
        np_expand_dims = np.__getattribute__('expand_dims')
        ns_expand_dims = nps.__getattribute__('expand_dims')
        np_squeeze = np.__getattribute__('squeeze')
        ns_squeeze = nps.__getattribute__('squeeze')

        _ns_a = nps.array(_np_a)
        _ns_ins_a = ns_ins.array(_np_a, block_shape=block_shape)

        _np_result = np_expand_dims(_np_a, axes)
        _ns_result = ns_expand_dims(_ns_a, axes)
        _ns_ins_result = ns_expand_dims(_ns_ins_a, axes)
        assert np.allclose(_np_result, _ns_result.get())
        assert np.allclose(_np_result, _ns_ins_result.get())
        check_dim(_np_result, _ns_result)
        check_dim(_np_result, _ns_ins_result)

        _np_result = np_squeeze(_np_a)
        _ns_result = ns_squeeze(_ns_a)
        _ns_ins_result = ns_squeeze(_ns_ins_a)
        assert np.allclose(_np_result, _ns_result.get())
        assert np.allclose(_np_result, _ns_ins_result.get())
        check_dim(_np_result, _ns_result)
        check_dim(_np_result, _ns_ins_result)

    def check_dim(_np_a, _ns_a):
        np_ndim = np.__getattribute__('ndim')
        assert np_ndim(_np_a) == np_ndim(_ns_a)

    def check_swapaxes(_np_a, axis1, axis2):
        ns_ins = application_manager.instance()
        np_swapaxes = np.__getattribute__('swapaxes')
        ns_swapaxes = nps.__getattribute__('swapaxes')

        _ns_a = nps.array(_np_a)
        _ns_ins_a = ns_ins.array(_np_a, block_shape=block_shape)

        _np_result = np_swapaxes(_np_a, axis1, axis2)
        _ns_result = ns_swapaxes(_ns_a, axis1, axis2)
        _ns_ins_result = ns_swapaxes(_ns_ins_a, axis1, axis2)
        assert np.allclose(_np_result, _ns_result.get())
        assert np.allclose(_np_result, _ns_ins_result.get())

    np_A = np.ones(shape)
    check_expand_and_squeeze(np_A, axes=0)
    check_expand_and_squeeze(np_A, axes=2)
    check_expand_and_squeeze(np_A, axes=4)
    check_expand_and_squeeze(np_A, axes=(2, 3))
    check_expand_and_squeeze(np_A, axes=(0, 5))
    check_expand_and_squeeze(np_A, axes=(0, 5, 6))
    check_expand_and_squeeze(np_A, axes=(2, 3, 5, 6, 7))

    for a1 in range(4):
        for a2 in range(4):
            check_swapaxes(np_A, axis1=a1, axis2=a2)

if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_basic_creation(nps_app_inst)
    test_eye(nps_app_inst)
    test_diag(nps_app_inst)
    test_arange(nps_app_inst)
    test_concatenate(nps_app_inst)
    test_split(nps_app_inst)
    test_func_space(nps_app_inst)
    test_shape(nps_app_inst)
