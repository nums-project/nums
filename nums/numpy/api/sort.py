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

from typing import Tuple

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray

from nums.numpy.api.arithmetic import *
from nums.numpy.api.generated import *

############################################
# Sorting, Searching, and Counting Ops
############################################


def argmax(a: BlockArray, axis=None, out=None):
    """Returns the indices of the maximum values along an axis.

    This docstring was copied from numpy.argmax.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : BlockArray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    argmax, argmin
    amax : The maximum value along a given axis.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    argmax currently only supports one-dimensional arrays.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.


    Indexes of the maximal elements of a N-dimensional array:

    >>> b = nps.arange(6)  # doctest: +SKIP
    >>> b[1] = 5  # doctest: +SKIP
    >>> b.get()  # doctest: +SKIP
    array([0, 5, 2, 3, 4, 5])
    >>> nps.argmax(b).get()  # Only the first occurrence is returned.  # doctest: +SKIP
    array(1)
    """
    if len(a.shape) > 1:
        raise NotImplementedError(
            "argmax currently only supports one-dimensional arrays."
        )
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().argop("argmax", a, axis=axis)


def argmin(a: BlockArray, axis=None, out=None):
    """Returns the indices of the minimum values along an axis.

    This docstring was copied from numpy.argmin.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : BlockArray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    argmin, argmax
    amin : The minimum value along a given axis.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> b = nps.arange(6) + 10  # doctest: +SKIP
    >>> b[4] = 10  # doctest: +SKIP
    >>> b.get()  # doctest: +SKIP
    array([10, 11, 12, 13, 10, 15])
    >>> nps.argmin(b).get()  # Only the first occurrence is returned.  # doctest: +SKIP
    array(0)
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().argop("argmin", a, axis=axis)


def max(
    a: BlockArray, axis=None, out=None, keepdims=False, initial=None, where=None
) -> BlockArray:
    """Return the maximum of an array or maximum along an axis.

    This docstring was copied from numpy.max.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.
        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : BlockArray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `amax` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.
    where : BlockArray of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.
    Returns
    -------
    amax : BlockArray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    argmax :
        Return the indices of the maximum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding max value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmax.

    Don't use `amax` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``amax(a, axis=0)``.

    'initial' is currently not supported.

    'where' is currently not supported.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.arange(4).reshape((2,2))  # doctest: +SKIP
    >>> a.get()  # doctest: +SKIP
    array([[0, 1],
           [2, 3]])
    >>> nps.amax(a).get()           # Maximum of the flattened array  # doctest: +SKIP
    array(3)
    >>> nps.amax(a, axis=0).get()   # Maxima along the first axis  # doctest: +SKIP
    array([2, 3])
    >>> nps.amax(a, axis=1).get()   # Maxima along the second axis  # doctest: +SKIP
    >>> b = nps.arange(5, dtype=float)  # doctest: +SKIP
    >>> b[2] = nps.NaN  # doctest: +SKIP
    >>> nps.amax(b).get()  # doctest: +SKIP
    array(nan)
    >>> nps.nanmax(b).get()  # doctest: +SKIP
    array(4.)
    """
    if initial is not None:
        raise NotImplementedError("'initial' is currently not supported.")
    if where is not None:
        raise NotImplementedError("'where' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().max(a, axis=axis, keepdims=keepdims)


amax = max


def min(
    a: BlockArray, axis=None, out=None, keepdims=False, initial=None, where=None
) -> BlockArray:
    """Return the minimum of an array or minimum along an axis.

    This docstring was copied from numpy.min.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.
        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : BlockArray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `amin` method of sub-classes of
        `BlockArray`, however any non-default value will be. If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.
    where : BlockArray of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    amin : BlockArray
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    argmin :
        Return the indices of the minimum values.

    nanmax, maximum, fmax

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding min value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmin.

    Don't use `amin` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``amin(a, axis=0)``.

    'initial' is currently not supported.

    'where' is currently not supported.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.arange(4).reshape((2,2))  # doctest: +SKIP
    >>> a.get()  # doctest: +SKIP
    array([[0, 1],
           [2, 3]])
    >>> nps.amin(a).get()           # Minimum of the flattened array  # doctest: +SKIP
    array(0)
    >>> nps.amin(a, axis=0).get()   # Minima along the first axis  # doctest: +SKIP
    array([0, 1])
    >>> nps.amin(a, axis=1).get()   # Minima along the second axis  # doctest: +SKIP
    array([0, 2])

    >>> nps.nanmin(b).get()  # doctest: +SKIP
    0.0
    """
    if initial is not None:
        raise NotImplementedError("'initial' is currently not supported.")
    if where is not None:
        raise NotImplementedError("'where' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().min(a, axis=axis, keepdims=keepdims)


amin = min


def top_k(
    a: BlockArray, k: int, largest=True, sorted=False
) -> Tuple[BlockArray, BlockArray]:
    """Find the `k` largest or smallest elements of a BlockArray.

    If there are multiple kth elements that are equal in value, then no guarantees are made as
    to which ones are included in the top k.

    Args:
        a: A BlockArray.
        k: Number of top elements to return.
        largest: Whether to return largest or smallest elements.

    Returns:
        A tuple containing two BlockArrays, (`values`, `indices`).
        values: Values of the top k elements, unsorted.
        indices: Indices of the top k elements, ordered by their corresponding values.
    """
    if sorted:
        # The result can be sorted when sorting is implemented.
        raise NotImplementedError("'sorted' is currently not supported.")
    return _instance().top_k(a, k, largest=largest)


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
