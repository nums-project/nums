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


from nums.core.application_manager import instance
from nums.core.array import utils as array_utils
from nums.core.array.blockarray import BlockArray
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

    def permutation(self, x):
        app = instance()
        if array_utils.is_int(x):
            shape = (x, )
            block_shape = app.compute_block_shape(shape=shape, dtype=np.int64)
            return self.rs.permutation(shape[0], block_shape[0])
        else:
            assert isinstance(x, BlockArray)
            shape = x.shape
            block_shape = x.shape
            arr_perm = self.rs.permutation(shape[0], block_shape[0]).get()
            return x[arr_perm]
