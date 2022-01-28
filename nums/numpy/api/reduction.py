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

from nums.numpy.api.utility import *

############################################
# Reduction Ops
############################################


def where(condition: BlockArray, x: BlockArray = None, y: BlockArray = None):
    """Return elements chosen from `x` or `y` depending on `condition`.

    This docstring was copied from numpy.where.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    condition : BlockArray, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : BlockArray
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : BlockArray
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    Notes
    -----
    If all the arrays are 1-D, `where` is equivalent to::

        [xv if c else yv
         for c, xv, yv in zip(condition, x, y)]

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.arange(10)  # doctest: +SKIP
    >>> a.get()  # doctest: +SKIP
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> nps.where(a < 5, a, 10*a).get()  # doctest: +SKIP
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
    """
    return _instance().where(condition, x, y)


def all(a: BlockArray, axis=None, out=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to True.

    This docstring was copied from numpy.all.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (``axis=None``) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : BlockArray, optional
        Alternate output array in which to place the result.
        It must have the same shape as the expected output and its
        type is preserved (e.g., if ``dtype(out)`` is float, the result
        will consist of 0.0's and 1.0's). See `ufuncs-output-type` for more
        details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `all` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    all : BlockArray, bool
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    all : equivalent method
    any : Test whether any element along a given axis evaluates to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to `True` because these are not equal to zero.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.all(nps.array([[True,False],[True,True]])).get()  # doctest: +SKIP
    array(False)

    >>> nps.all(nps.array([[True,False],[True,True]]), axis=0).get()  # doctest: +SKIP
    array([ True, False])

    >>> nps.all(nps.array([-1, 4, 5])).get()  # doctest: +SKIP
    array(True)

    >>> nps.all(nps.array([1.0, nps.nan])).get()  # doctest: +SKIP
    array(True)
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("all", a, axis=axis, keepdims=keepdims)


def alltrue(a: BlockArray, axis=None, out=None, dtype=None, keepdims=False):
    """Check if all elements of input array are true.

    This docstring was copied from numpy.alltrue.

    Some inconsistencies with the NumS version may exist.

    See Also
    --------
    all : Equivalent function; see for details.
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("alltrue", a, axis=axis, keepdims=keepdims, dtype=dtype)


def any(a: BlockArray, axis=None, out=None, keepdims=False):
    """Test whether any array element along a given axis evaluates to True.

    This docstring was copied from numpy.any.

    Some inconsistencies with the NumS version may exist.

    Returns single boolean unless `axis` is not ``None``

    Parameters
    ----------
    a : BlockArray
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : BlockArray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output and its type is preserved
        (e.g., if it is of type float, then it will remain so, returning
        1.0 for True and 0.0 for False, regardless of the type of `a`).
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `any` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    any : BlockArray
        A new `BlockArray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    any : equivalent method
    all : Test whether all elements along a given axis evaluate to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to `True` because these are not equal to zero.

    'out' is currently not supported

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.any(nps.array([[True, False], [True, True]])).get()  # doctest: +SKIP
    array(True)

    >>> nps.any(nps.array([[True, False], [False, False]]), axis=0).get()  # doctest: +SKIP
    array([ True, False])

    >>> nps.any(nps.array([-1, 0, 5])).get()  # doctest: +SKIP
    array(True)

    >>> nps.any(nps.array(nps.nan)).get()  # doctest: +SKIP
    array(True)
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("any", a, axis=axis, keepdims=keepdims)
