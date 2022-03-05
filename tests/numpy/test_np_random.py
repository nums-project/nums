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

    nums.core.settings.backend_name = "serial"
    from nums.core import application_manager

    nps_app_inst = application_manager.instance()

    test_basic(nps_app_inst)
    # test_default_random(nps_app_inst)
    # test_blockarray_perm(nps_app_inst)
