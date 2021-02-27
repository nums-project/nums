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


# pylint: disable=import-outside-toplevel


def test_reshape_int(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    shape = (3, 5, 10)
    arr = nps.arange(np.product(shape))
    np_arr = arr.get()
    assert np.allclose(np_arr.reshape(shape), arr.reshape(shape).get())
    assert np.allclose(np_arr.reshape(shape).reshape(-1),
                       arr.reshape(shape).reshape(-1).get())
    assert np.allclose(np_arr.reshape(shape).reshape(np.product(shape)),
                       arr.reshape(shape).reshape(np.product(shape)).get())


def test_reshape_noops(nps_app_inst):
    shape, block_shape = (3, 5, 10), (3, 2, 5)
    arr = nps_app_inst.random_state(1337).random(shape, block_shape)
    new_arr = arr.reshape()
    assert arr is new_arr
    new_arr = arr.reshape(shape)
    assert arr is new_arr
    new_arr = arr.reshape(block_shape=block_shape)
    assert arr is new_arr
    new_arr = arr.reshape(shape, block_shape=block_shape)
    assert arr is new_arr


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    import nums.core.settings
    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_reshape_int(nps_app_inst)
    test_reshape_noops(nps_app_inst)
