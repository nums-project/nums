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


import itertools

import tqdm
import numpy as np
import pytest

from nums.core.array import utils as array_utils


def test_assign_broadcasting():
    # https://numpy.org/doc/stable/user/basics.indexing.html#assigning-values-to-indexed-arrays
    # Note that the above documentation does not fully capture the broadcasting behavior of NumPy.
    # We therefore test our tools for broadcasting with array shapes, instead of arrays.
    # Consider the permutations of a tuple of 10 integers ranging from 0 to 3.
    # We partition this tuple into 2 tuples of 5 integers, and
    # use the tuples to define the shapes of two arrays,
    # to define the LHS and RHS of an assignment.
    # A value of 0 means the axis is not specified in the resulting array shape.
    # Because we're interested in valid assignment operations,
    # we define empty arrays of size equal to the product of
    # axis dims, excluding any axes which are set to 0.
    # If all axes are set to 0,
    # then create a dimensionless array with value 0.
    # There is no formal proof that the proof for arrays with 5 axes
    # is without loss of generality.
    def get_array(shape):
        shape = tuple(filter(lambda x: x > 0, shape))
        if len(shape) == 0:
            return np.array(0)
        else:
            return np.empty(np.product(shape)).reshape(shape)

    perms = list(itertools.product([0, 1, 2, 3], repeat=10))
    pbar = tqdm.tqdm(total=len(perms))
    for shapes in perms:
        A: np.ndarray = get_array(shapes[:5])
        B: np.ndarray = get_array(shapes[5:])
        try:
            if A.shape == ():
                continue
            if B.shape == ():
                A[:] = B
            else:
                A[:] = B[:]
            # This should execute without error.
            assert np.broadcast_to(B, A.shape).shape == array_utils.broadcast_shape_to_alt(B.shape,
                                                                                           A.shape)
            assert array_utils.can_broadcast_shape_to(B.shape, A.shape), \
                "%s can be broadcast to %s" % (B.shape, A.shape)
        except ValueError as _:
            with pytest.raises(ValueError):
                np.broadcast_to(B, A.shape)
            with pytest.raises(ValueError):
                array_utils.broadcast_shape_to_alt(B.shape, A.shape)
            assert not array_utils.can_broadcast_shape_to(B.shape, A.shape), \
                "%s cannot be broadcast to %s" % (B.shape, A.shape)
        pbar.update(1)


def test_bop_broadcasting():
    def get_array(shape):
        shape = tuple(filter(lambda x: x > 0, shape))
        if len(shape) == 0:
            return np.array(0)
        else:
            return np.empty(np.product(shape)).reshape(shape)

    perms = list(itertools.product([0, 1, 2, 3], repeat=10))
    pbar = tqdm.tqdm(total=len(perms))
    for shapes in perms:
        A: np.ndarray = get_array(shapes[:5])
        B: np.ndarray = get_array(shapes[5:])
        try:
            assert (A * B).shape == array_utils.broadcast_shape(A.shape, B.shape)
        except ValueError as _:
            assert not array_utils.can_broadcast_shapes(B.shape, A.shape)
            assert not array_utils.can_broadcast_shapes(A.shape, B.shape)
            with pytest.raises(ValueError):
                array_utils.broadcast_shape(A.shape, B.shape)
        pbar.update(1)


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_assign_broadcasting()
    test_bop_broadcasting()
