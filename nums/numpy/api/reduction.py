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

from nums.numpy.api.equality import *

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
