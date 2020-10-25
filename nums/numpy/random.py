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

from nums.core.application_manager import instance
from nums.core.array import utils as array_utils
import numpy as np


class RandomState(object):

    def __init__(self, seed=None):
        self.rs = instance().random_state(seed)

    def _get_shapes(self, size=None, dtype=None):
        if dtype is None:
            dtype = np.float64
        if size is None:
            size = ()
        if not isinstance(size, tuple):
            assert array_utils.is_int(size)
            shape = (size,)
        else:
            shape = size
        block_shape = instance().get_block_shape(shape, dtype)
        return shape, block_shape

    def random_sample(self, size=None):
        shape, block_shape = self._get_shapes(size, np.float64)
        return self.rs.random(shape=shape, block_shape=block_shape)

    def rand(self, *shape):
        shape, block_shape = self._get_shapes(shape, np.float64)
        return self.rs.random(shape=shape, block_shape=block_shape)

    def randn(self, *shape):
        shape, block_shape = self._get_shapes(shape, np.float64)
        return self.rs.normal(shape=shape, block_shape=block_shape)

    def randint(self, low, high=None, size=None, dtype=None):
        if high is None:
            high = low
            low = 0
        shape, block_shape = self._get_shapes(size, dtype)
        return self.rs.integers(low, high, shape=shape, block_shape=block_shape)

    def random_integers(self):
        # This requires endpoint to be implemented by integers.
        raise NotImplementedError()
