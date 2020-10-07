# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import numpy as np
import pytest

from nums.core.storage.storage import BimodalGaussian
from nums.core.array.application import ArrayApplication

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
    X = X.reshape(shape=(1000, 9), block_shape=(1000, 1))
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


@pytest.mark.skip
def test_split(app_inst: ArrayApplication):
    raise NotImplementedError("Requires app-level imp. and tests.")


def test_touch(app_inst: ArrayApplication):
    ones = app_inst.ones((123, 456), (12, 34))
    assert ones.touch() is None


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_array_integrity(app_inst)
    test_concatenate(app_inst)
    test_touch(app_inst)
