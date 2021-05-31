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

from nums.core.array.blockarray import BlockArray, Block
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid


class NumsRandomState(object):
    def __init__(self, cm: ComputeManager, seed):
        self._cm = cm
        self._rng = self._cm.get_rng(seed)

    def seed(self, seed=None):
        # New RNG based on given seed.
        self._rng = self._cm.get_rng(seed)

    def numpy(self):
        # pylint: disable = import-outside-toplevel
        from nums.core.compute.numpy_compute import block_rng

        return block_rng(*self._rng.new_block_rng_params())

    def random(self, shape=None, block_shape=None, dtype=None):
        if dtype is None:
            dtype = np.float64
        assert isinstance(dtype, type)
        return self._sample_basic("random", shape, block_shape, dtype, (dtype,))

    def integers(
        self, low, high=None, shape=None, block_shape=None, dtype=None, endpoint=False
    ):
        if dtype is None:
            dtype = np.int64
        assert isinstance(dtype, type)
        return self._sample_basic(
            "integers", shape, block_shape, dtype, (low, high, dtype, endpoint)
        )

    def uniform(self, low=0.0, high=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("uniform", shape, block_shape, dtype, (low, high))

    def normal(self, loc=0.0, scale=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("normal", shape, block_shape, dtype, (loc, scale))

    def beta(self, a, b, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("beta", shape, block_shape, dtype, (a, b))

    def binomial(self, n, p, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("binomial", shape, block_shape, dtype, (n, p))

    def chisquare(self, df, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("chisquare", shape, block_shape, dtype, (df,))

    def exponential(self, scale=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("exponential", shape, block_shape, dtype, (scale,))

    def f(self, dfnum, dfden, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("f", shape, block_shape, dtype, (dfnum, dfden))

    def gamma(self, k, theta, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("gamma", shape, block_shape, dtype, (k, theta))

    def geometric(self, p, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("geometric", shape, block_shape, dtype, (p,))

    def gumbel(self, loc=0.0, scale=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("gumbel", shape, block_shape, dtype, (loc, scale))

    def hypergeometric(
        self, ngood, nbad, nsample, shape=None, block_shape=None, dtype=None
    ):
        return self._sample_basic(
            "hypergeometric", shape, block_shape, dtype, (ngood, nbad, nsample)
        )

    def laplace(self, loc=0.0, scale=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("laplace", shape, block_shape, dtype, (loc, scale))

    def logistic(self, loc=0.0, scale=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("logistic", shape, block_shape, dtype, (loc, scale))

    def lognormal(self, mean=0.0, sigma=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("lognormal", shape, block_shape, dtype, (mean, sigma))

    def logseries(self, p, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("logseries", shape, block_shape, dtype, (p,))

    def negative_binomial(self, n, p, shape=None, block_shape=None, dtype=None):
        return self._sample_basic(
            "negative_binomial", shape, block_shape, dtype, (n, p)
        )

    def noncentral_chisquare(self, df, nonc, shape=None, block_shape=None, dtype=None):
        return self._sample_basic(
            "noncentral_chisquare", shape, block_shape, dtype, (df, nonc)
        )

    def noncentral_f(
        self, dfnum, dfden, nonc, shape=None, block_shape=None, dtype=None
    ):
        return self._sample_basic(
            "noncentral_f", shape, block_shape, dtype, (dfnum, dfden, nonc)
        )

    def pareto(self, a, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("pareto", shape, block_shape, dtype, (a,))

    def poisson(self, lam=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("poisson", shape, block_shape, dtype, (lam,))

    def power(self, a, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("power", shape, block_shape, dtype, (a,))

    def rayleigh(self, scale=1.0, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("rayleigh", shape, block_shape, dtype, (scale,))

    def standard_cauchy(self, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("standard_cauchy", shape, block_shape, dtype, ())

    def standard_t(self, df, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("standard_t", shape, block_shape, dtype, (df,))

    # TODO Restrict params to scalars.
    def triangular(self, left, mode, right, shape=None, block_shape=None, dtype=None):
        return self._sample_basic(
            "triangular", shape, block_shape, dtype, (left, mode, right)
        )

    def vonmises(self, mu, kappa, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("vonmises", shape, block_shape, dtype, (mu, kappa))

    def wald(self, mean, scale, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("wald", shape, block_shape, dtype, (mean, scale))

    def weibull(self, a, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("weibull", shape, block_shape, dtype, (a,))

    def zipf(self, a, shape=None, block_shape=None, dtype=None):
        return self._sample_basic("zipf", shape, block_shape, dtype, (a,))

    # TODO (hme): Add multivariate samplers.

    def _sample_basic(
        self, rfunc_name, shape, block_shape, dtype, rfunc_args
    ) -> BlockArray:
        if shape is None:
            assert block_shape is None
            shape = ()
            block_shape = ()
        else:
            assert block_shape is not None
        if dtype is None:
            dtype = np.float64
        assert isinstance(dtype, type)
        grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype=dtype.__name__)
        ba: BlockArray = BlockArray(grid, self._cm)
        for grid_entry in ba.grid.get_entry_iterator():
            rng_params = list(self._rng.new_block_rng_params())
            # Size and dtype to begin with.
            this_block_shape = grid.get_block_shape(grid_entry)
            size = int(np.product(this_block_shape))
            # Inconsistent param orderings.
            if rfunc_name == "random":
                rfunc_args_final = tuple([size] + list(rfunc_args))
            elif rfunc_name == "integers":
                # rfunc_args == (low, high, dtype, endpoint)
                rfunc_args_final = tuple(
                    list(rfunc_args[:2]) + [size] + list(rfunc_args[2:])
                )
            else:
                rfunc_args_final = tuple(list(rfunc_args) + [size])
            block: Block = ba.blocks[grid_entry]
            block.oid = self._cm.random_block(
                rng_params,
                rfunc_name,
                rfunc_args_final,
                this_block_shape,
                dtype,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
        return ba

    def permutation(self, size, block_size):
        shape = (size,)
        block_shape = (block_size,)
        grid: ArrayGrid = ArrayGrid(
            shape=shape, block_shape=shape, dtype=np.int64.__name__
        )
        ba = BlockArray(grid, self._cm)
        for grid_entry in ba.grid.get_entry_iterator():
            rng_params = list(self._rng.new_block_rng_params())
            block: Block = ba.blocks[grid_entry]
            block.oid = self._cm.permutation(
                rng_params,
                size,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
        return ba.reshape(block_shape=block_shape)
