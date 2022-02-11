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

# pylint: disable = redefined-builtin, too-many-lines, anomalous-backslash-in-string, unused-wildcard-import, wildcard-import

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray

from nums.numpy.api.logic import *

############################################
# Reduction Ops
############################################


def sum(
    a: BlockArray,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=None,
) -> BlockArray:
    """Sum of array elements over a given axis.

    This docstring was copied from numpy.sum.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : BlockArray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum.
    where : BlockArray of bool, optional
        Elements to include in the sum.

    Returns
    -------
    sum_along_axis : BlockArray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    mean, average

    Notes
    -----
    'initial' is currently not supported.

    'where' is currently not supported.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.sum(nps.array([0.5, 1.5])).get()  # doctest: +SKIP
    array(2.)
    >>> nps.sum(nps.array([[0, 1], [0, 5]])).get()  # doctest: +SKIP
    array(6)
    >>> nps.sum(nps.array([[0, 1], [0, 5]]), axis=0).get()  # doctest: +SKIP
    array([0, 6])
    >>> nps.sum(nps.array([[0, 1], [0, 5]]), axis=1).get()  # doctest: +SKIP
    array([1, 5])
    """
    if initial is not None:
        raise NotImplementedError("'initial' is currently not supported.")
    if where is not None:
        raise NotImplementedError("'where' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().sum(a, axis=axis, keepdims=keepdims, dtype=dtype)
