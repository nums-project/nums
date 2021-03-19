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

from nums.core.storage.storage import BimodalGaussian, ArrayGrid
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray

# pylint: disable=wrong-import-order
import common


def test_array_integrity(app_inst: ArrayApplication):
    shape = 12, 21
    npX = np.arange(np.product(shape)).reshape(*shape)
    X = app_inst.array(npX, block_shape=(6, 7))
    common.check_block_integrity(X)


def test_transpose(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(100, 9)
    X = app_inst.array(real_X, block_shape=(100, 1))
    assert np.allclose(X.T.get(), real_X.T)
    # Identity.
    assert np.allclose(X.T.T.get(), X.get())
    assert np.allclose(X.T.T.get(), real_X)


def test_reshape(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(1000, 9)
    X = app_inst.array(real_X, block_shape=(100, 9))
    X = X.reshape((1000, 9), block_shape=(1000, 1))
    assert np.allclose(X.get(), real_X)


def test_concatenate(app_inst: ArrayApplication):
    axis = 1
    real_X, _ = BimodalGaussian.get_dataset(1000, 9)
    real_ones = np.ones(shape=(1000, 1))
    X = app_inst.array(real_X, block_shape=(100, 9))
    ones = app_inst.ones((1000, 1), (100, 1), dtype=X.dtype)
    X_concated = app_inst.concatenate([X, ones], axis=axis, axis_block_size=X.block_shape[axis])
    real_X_concated = np.concatenate([real_X, real_ones], axis=axis)
    assert np.allclose(X_concated.get(), real_X_concated)

    real_X2 = np.random.random_sample(1000*17).reshape(1000, 17)
    X2 = app_inst.array(real_X2, block_shape=(X.block_shape[0], 3))
    X_concated = app_inst.concatenate([X, ones, X2], axis=axis, axis_block_size=X.block_shape[axis])
    real_X_concated = np.concatenate([real_X, real_ones, real_X2], axis=axis)
    assert np.allclose(X_concated.get(), real_X_concated)


def test_split(app_inst: ArrayApplication):
    # TODO (hme): Implement a split leveraging block_shape param in reshape op.
    x = app_inst.array(np.array([1.0, 2.0, 3.0, 4.0]), block_shape=(4,))
    syskwargs = x.blocks[0].syskwargs()
    syskwargs["options"] = {"num_returns": 2}
    res1, res2 = x.system.split(x.blocks[0].oid,
                                2,
                                axis=0,
                                transposed=False,
                                syskwargs=syskwargs)
    ba = BlockArray(ArrayGrid((4,), (2,), x.dtype.__name__), x.system)
    ba.blocks[0].oid = res1
    ba.blocks[1].oid = res2
    assert np.allclose([1.0, 2.0, 3.0, 4.0], ba.get())


def test_touch(app_inst: ArrayApplication):
    ones = app_inst.ones((123, 456), (12, 34))
    assert ones.touch() is ones


if __name__ == "__main__":
    # pylint: disable=import-error, no-member
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_array_integrity(app_inst)
    test_concatenate(app_inst)
    test_touch(app_inst)
    test_split(app_inst)
