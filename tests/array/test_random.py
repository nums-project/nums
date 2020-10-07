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
from scipy import stats

from nums.core.array.random import NumsRandomState
from nums.core.array.blockarray import BlockArray
from nums.core.array.application import ArrayApplication


# pylint: disable=unused-variable


def test_np_random(app_inst: ArrayApplication):

    # Sample a single value.
    sample = app_inst.random_state(1337).random().get()
    assert sample.shape == ()
    assert isinstance(sample.item(), np.float)

    shape, block_shape = (15, 10), (5, 5)
    # Probably not equal if pvalue falls below this threshold.
    epsilon = 1e-2
    rs1: NumsRandomState = app_inst.random_state(1337)
    ba1: BlockArray = rs1.random(shape, block_shape)
    # The Kolmogorov–Smirnov test for arbitrary distributions.
    # Under the null hypothesis, the distributions are equal,
    # so we say distributions are neq if pvalue < epsilon.
    stat, pvalue = stats.kstest(ba1.get().flatten(), stats.uniform.cdf)
    assert pvalue > epsilon

    rs2: NumsRandomState = app_inst.random_state(1337)
    ba2: BlockArray = rs2.random(shape, block_shape)
    assert app_inst.allclose(ba1, ba2)

    rs3: NumsRandomState = app_inst.random_state(1338)
    ba3: BlockArray = rs3.random(shape, block_shape)
    assert not app_inst.allclose(ba2, ba3)

    # If block shape differs, so does generated arrays.
    # This is a non-issue since we don't expose block shape as a param.
    rs4: NumsRandomState = app_inst.random_state(1337)
    ba4: BlockArray = rs4.random(shape, block_shape=(6, 7)).reshape(block_shape=block_shape)
    assert not app_inst.allclose(ba2, ba4)

    # dtype tests.
    rs: NumsRandomState = app_inst.random_state(1337)
    ba4: BlockArray = rs.random(shape, block_shape, dtype=np.float32)
    assert ba4.dtype is np.float32
    assert str(ba4.get().dtype) == "float32"


def test_np_distributions(app_inst: ArrayApplication):
    shape, block_shape = (15, 10), (5, 5)
    epsilon = 1e-2
    rs: NumsRandomState = app_inst.random_state(1337)

    # Type test.
    low, high = -3.2, 5.7
    ba: BlockArray = rs.uniform(low, high, shape, block_shape, dtype=np.float32)
    assert ba.dtype is np.float32
    assert str(ba.get().dtype) == "float32"

    # Distribution test.
    cdf = lambda x: stats.uniform.cdf(x, loc=low, scale=high-low)
    stat, pvalue = stats.kstest(ba.get().flatten(), cdf)
    assert pvalue > epsilon
    # Also just confirm standard uniform distribution fails test.
    assert stats.kstest(ba.get().flatten(), stats.uniform.cdf)[1] < epsilon

    loc, scale = -123, 42
    ba: BlockArray = rs.normal(loc, scale, shape, block_shape)
    cdf = lambda x: stats.norm.cdf(x, loc=loc, scale=scale)
    stat, pvalue = stats.kstest(ba.get().flatten(), cdf)
    assert pvalue > epsilon
    assert stats.kstest(ba.get().flatten(), stats.norm.cdf)[1] < epsilon


def test_np_integer(app_inst: ArrayApplication):
    shape, block_shape = (15, 10), (5, 5)
    sample = app_inst.random_state(1337).integers(100).get()
    assert sample.shape == ()
    assert isinstance(sample.item(), np.int)

    rs: NumsRandomState = app_inst.random_state(1337)
    ba: BlockArray = rs.integers(-10, 20, shape, block_shape, np.int32)
    assert ba.get().dtype == "int32"
    arr: np.array = ba.get()
    assert -10 <= np.min(arr) <= np.max(arr) < 20


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_np_random(app_inst)
    test_np_distributions(app_inst)
    test_np_integer(app_inst)
