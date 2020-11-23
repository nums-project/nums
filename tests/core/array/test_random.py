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


import warnings

import numpy as np
from scipy import stats

from nums.core.array.random import NumsRandomState
from nums.core.array.blockarray import BlockArray
from nums.core.array.application import ArrayApplication


# pylint: disable=unused-variable


def test_basic(app_inst: ArrayApplication):
    # TODO (hme): Add more comprehensive tests for these distributions.

    dists = [("beta", (1, 2, (3,), (3,))),
             ("binomial", (3, .5, (3,), (3,))),
             ("chisquare", (2, (3,), (3,))),
             ("exponential", (1.0, (3,), (3,))),
             ("f", (2, 1.0, (3,), (3,))),
             ("gamma", (2, .7, (3,), (3,))),
             ("geometric", (.5, (3,), (3,))),
             ("gumbel", (0.0, 1.0, (3,), (3,))),
             ("hypergeometric", (3, 2, 5, (3,), (3,))),
             ("laplace", (0.0, 1.0, (3,), (3,))),
             ("logistic", (0.0, 1.0, (3,), (3,))),
             ("lognormal", (0.0, 1.0, (3,), (3,))),
             ("logseries", (.5, (3,), (3,))),
             ("negative_binomial", (5, .5, (3,), (3,))),
             ("noncentral_chisquare", (2, 1.0, (3,), (3,))),
             ("noncentral_f", (2, 3.0, 1.0, (3,), (3,))),
             ("pareto", (2.0, (3,), (3,))),
             ("poisson", (1.0, (3,), (3,))),
             ("power", (2.0, (3,), (3,))),
             ("rayleigh", (1.0, (3,), (3,))),
             ("standard_cauchy", ((3,), (3,))),
             ("standard_t", (2, (3,), (3,))),
             ("triangular", (1, 3, 4, (3,), (3,))),
             ("vonmises", (1.0, 3.0, (3,), (3,))),
             ("wald", (4.0, 2.0, (3,), (3,))),
             ("weibull", (2.0, (3,), (3,))),
             ("zipf", (2.0, (3,), (3,))),
             ]
    rs = app_inst.random_state()
    for dist_name, dist_params in dists:
        assert len(rs.__getattribute__(dist_name)(*dist_params).get()) == 3


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


def test_default_random(app_inst: ArrayApplication):
    num1 = app_inst.random_state().random()
    num2 = app_inst.random_state().random()
    num_iters = 0
    max_iters = 10
    while app_inst.allclose(num1, num2) and num_iters < max_iters:
        num_iters += 1
        num2 = app_inst.random_state().random()
    if num_iters > 0:
        warnings.warn("More than one iteration required to generate unequal random numbers.")
    assert not app_inst.allclose(num1, num2)

    # Test default random seed.
    app_inst.random.seed(1337)
    num1 = app_inst.random.random()
    app_inst.random.seed(1337)
    num2 = app_inst.random.random()
    assert app_inst.allclose(num1, num2)


def test_numpy_random(app_inst: ArrayApplication):
    # Make sure the numpy version of our RNG works properly.
    # This uses the underlying RNG to generate a numpy array instead of a block array.
    # This ensures deterministic behavior when we need random numbers on the driver node.
    app_inst.random.seed(1337)
    nps_num = app_inst.random.random()
    app_inst.random.seed(1337)
    np_num = app_inst.random.numpy().random()
    assert np.allclose(nps_num.get(), np_num)


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    # test_basic(app_inst)
    # test_np_random(app_inst)
    # test_np_distributions(app_inst)
    # test_np_integer(app_inst)
    # test_default_random(app_inst)
    # test_numpy_random(app_inst)
