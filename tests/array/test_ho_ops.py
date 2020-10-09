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

from nums.core.storage.storage import BimodalGaussian
from nums.core.array.application import ArrayApplication


def test_log(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(100, 9)
    X = app_inst.array(real_X, block_shape=(10, 2))
    assert np.allclose(app_inst.log(X).get(), np.log(real_X))


def test_stats(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(3, 2)
    X = app_inst.array(real_X, block_shape=(2, 1))
    assert np.allclose(app_inst.mean(X, axis=0).get(), np.mean(real_X, axis=0))
    assert np.allclose(app_inst.std(X, axis=1).get(), np.std(real_X, axis=1))

    real_X, _ = BimodalGaussian.get_dataset(100, 9)
    X = app_inst.array(real_X, block_shape=(10, 2))
    assert np.allclose(app_inst.mean(X, axis=0).get(), np.mean(real_X, axis=0))
    assert np.allclose(app_inst.std(X, axis=1).get(), np.std(real_X, axis=1))


def test_sum(app_inst: ArrayApplication):
    shape = (5, 6, 7)
    real_X = np.random.random_sample(np.product(shape)).reshape(shape)
    X = app_inst.array(real_X, block_shape=(2, 1, 4))
    assert np.allclose(app_inst.sum(X, axis=1).get(), np.sum(real_X, axis=1))


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_log(app_inst)
    test_stats(app_inst)
    test_sum(app_inst)
