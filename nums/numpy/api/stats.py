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

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray
from nums.numpy import numpy_utils

from nums.numpy.api.reduction import *
from nums.numpy.api.utility import *

from nums.numpy.api.generated import *


############################################
# Stats
############################################


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


def mean(a: BlockArray, axis=None, dtype=None, out=None, keepdims=False):
    """Compute the arithmetic mean along the specified axis.

    This docstring was copied from numpy.mean.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : BlockArray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    m : BlockArray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average
    std, var, nanmean, nanstd, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.

    Note that for floating-point input, the mean is computed using the
    same precision the input has.  Depending on the input data, this can
    cause the results to be inaccurate, especially for `float32` (see
    example below).  Specifying a higher-precision accumulator using the
    `dtype` keyword can alleviate this issue.

    By default, `float16` results are computed using `float32` intermediates
    for extra precision.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, 2], [3, 4]])  # doctest: +SKIP
    >>> nps.mean(a).get()  # doctest: +SKIP
    array(2.5)
    >>> nps.mean(a, axis=0).get()  # doctest: +SKIP
    array([2., 3.])
    >>> nps.mean(a, axis=1).get()  # doctest: +SKIP
    array([1.5, 3.5])
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().mean(a, axis=axis, keepdims=keepdims, dtype=dtype)


def var(a: BlockArray, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Compute the variance along the specified axis.

    This docstring was copied from numpy.var.

    Some inconsistencies with the NumS version may exist.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : BlockArray
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.
        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float`; for arrays of float types it is the same as
        the array type.
    out : BlockArray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    variance : BlockArray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance;
        otherwise, a reference to the output array is returned.

    See Also
    --------
    std, mean, nanmean, nanstd, nanvar

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, 2], [3, 4]]) # doctest: +SKIP
    >>> nps.var(a).get()  # doctest: +SKIP
    array(1.25)
    >>> nps.var(a, axis=0).get()  # doctest: +SKIP
    array([1.,  1.])
    >>> nps.var(a, axis=1).get()  # doctest: +SKIP
    array([0.25,  0.25])
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().var(a, axis=axis, ddof=ddof, keepdims=keepdims, dtype=dtype)


def std(a: BlockArray, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Compute the standard deviation along the specified axis.

    This docstring was copied from numpy.std.

    Some inconsistencies with the NumS version may exist.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : BlockArray
        Calculate the standard deviation of these values.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is None.
    out : BlockArray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    standard_deviation : BlockArray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

    See Also
    --------
    var, mean, nanmean, nanstd, nanvar

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
    the divisor ``N - ddof`` is used instead. In standard statistical
    practice, ``ddof=1`` provides an unbiased estimator of the variance
    of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The
    standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.

    Note that, for complex numbers, `std` takes the absolute
    value before squaring, so that the result is always real and nonnegative.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, 2], [3, 4]])  # doctest: +SKIP
    >>> nps.std(a).get()  # doctest: +SKIP
    array(1.1180339887498949) # may vary
    >>> nps.std(a, axis=0).get()  # doctest: +SKIP
    array([1.,  1.])
    >>> nps.std(a, axis=1).get()  # doctest: +SKIP
    array([0.5,  0.5])
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().std(a, axis=axis, ddof=ddof, keepdims=keepdims, dtype=dtype)


def cov(
    m: BlockArray,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    dtype=None,
):
    """Estimate a covariance matrix, given data and weights.

    This docstring was copied from numpy.cov.

    Some inconsistencies with the NumS version may exist.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    See the notes for an outline of the algorithm.

    Parameters
    ----------
    m : BlockArray
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : BlockArray, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average. See the notes for the details. The default value
        is ``None``.
    fweights : BlockArray, int, optional
        1-D array of integer frequency weights; the number of times each
        observation vector should be repeated.
    aweights : BlockArray, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

    Returns
    -------
    out : BlockArray
        The covariance matrix of the variables.

    See Also
    --------
    corrcoef : Normalized covariance matrix

    Notes
    -----
    Assume that the observations are in the columns of the observation
    array `m` and let ``f = fweights`` and ``a = aweights`` for brevity. The
    steps to compute the weighted covariance are as follows::

        >>> m = nps.arange(10, dtype=nps.float64)  # doctest: +SKIP
        >>> f = nps.arange(10) * 2  # doctest: +SKIP
        >>> a = nps.arange(10) ** 2.  # doctest: +SKIP
        >>> ddof = 1  # doctest: +SKIP
        >>> w = f * a  # doctest: +SKIP
        >>> v1 = nps.sum(w)  # doctest: +SKIP
        >>> v2 = nps.sum(w * a)  # doctest: +SKIP
        >>> m -= nps.sum(m * w, axis=None, keepdims=True) / v1  # doctest: +SKIP
        >>> cov = nps.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)  # doctest: +SKIP

    Note that when ``a == 1``, the normalization factor
    ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (nps.sum(f) - ddof).get()``
    as it should.

    y, ddof, fweights, and aweights are not supported.

    Only 2-dimensional arrays are supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = nps.array([[0, 2], [1, 1], [2, 0]]).T  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> nps.cov(x).get()  # doctest: +SKIP
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.
    """
    if not (y is None and ddof is None and fweights is None and aweights is None):
        raise NotImplementedError("y, ddof, fweights, and aweights are not supported.")
    if len(m.shape) != 2:
        raise NotImplementedError("Only 2-dimensional arrays are supported.")
    return _instance().cov(m, rowvar, bias, dtype)


def average(
    a: BlockArray,
    axis: Optional[int] = None,
    weights: Optional[BlockArray] = None,
    returned: bool = False,
) -> Union[BlockArray, Tuple[BlockArray, BlockArray]]:
    """Compute the weighted average along the specified axis.

    This docstring was copied from numpy.average.

    Some inconsistencies with the NumS version may exist.

    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : BlockArray
        Array containing data to be averaged. If `a` is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        axis=None, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : BlockArray, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.  The 1-D calculation is::

            avg = sum(a * weights) / sum(weights)

        The only constraint on `weights` is that `sum(weights)` must not be 0.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.

    Returns
    -------
    retval, [sum_of_weights] : array_type or double
        Return the average along the specified axis. When `returned` is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. `sum_of_weights` is of the
        same type as `retval`. The result dtype follows a genereal pattern.
        If `weights` is None, the result dtype will be that of `a` , or ``float64``
        if `a` is integral. Otherwise, if `weights` is not None and `a` is non-
        integral, the result type will be the type of lowest precision capable of
        representing values of both `a` and `weights`. If `a` happens to be
        integral, the previous rules still applies but the result dtype will
        at least be ``float``.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When the length of 1D `weights` is not the same as the shape of `a`
        along axis.

    See Also
    --------
    mean

    Notes
    -----
    Only single 'axis' is currently supported.

    1D weights broadcasting is currently not supported.

    Weights along one or more axes sum to zero.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> data = nps.arange(1, 5)  # doctest: +SKIP
    >>> data.get()  # doctest: +SKIP
    array([1, 2, 3, 4])
    >>> nps.average(data).get()  # doctest: +SKIP
    array(2.5)

    >>> data = nps.arange(6).reshape((3,2))  # doctest: +SKIP
    >>> data.get()  # doctest: +SKIP
    array([[0, 1],
           [2, 3],
           [4, 5]])
    """
    if axis and not isinstance(axis, int):
        raise NotImplementedError("Only single 'axis' is currently supported.")

    if weights is None:
        avg = mean(a, axis=axis)
        if not returned:
            return avg
        weights_sum = BlockArray.from_scalar(a.size / avg.size, a.cm)
        return avg, weights_sum

    if a.shape != weights.shape:
        raise NotImplementedError(
            "1D weights broadcasting is currently not supported; "
            "dimensions of 'a' and 'weights' must match."
        )
    weights_sum = sum(weights, axis=axis)
    if not all(weights_sum):
        raise ZeroDivisionError("Weights along one or more axes sum to zero.")
    avg = divide(sum(multiply(a, weights), axis=axis), weights_sum)

    if not returned:
        return avg
    if avg.shape != weights_sum.shape:
        weights_sum = weights_sum.broadcast_to(avg.shape)
    return avg, weights_sum


def quantile(
    a: BlockArray,
    q: float,
    axis: Optional[int] = None,
    out: BlockArray = None,
    overwrite_input: bool = False,
    interpolation: {"linear"} = "linear",
    keepdims: bool = False,
) -> BlockArray:
    """Compute the q-th quantile of the data along the specified axis.

    This docstring was copied from numpy.quantile.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array or object that can be converted to an array.
    q : BlockArray of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    out : BlockArray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:

            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    quantile : BlockArray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float``, the output
        data-type is ``float``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    percentile : equivalent to quantile, but with q in the range [0, 100].
    median : equivalent to ``quantile(..., 0.5)``
    nanquantile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th quantile of
    ``V`` is the value ``q`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the quantile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=0.5``, the same as the minimum if ``q=0.0`` and the
    same as the maximum if ``q=1.0``.

    'axis' is currently not supported.

    'out' is currently not supported.

    'overwrite_input' is currently not supported.

    only 'linear' 'interpolation' is currently supported.

    'keepdims' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[10, 7, 4], [3, 2, 1]])  # doctest: +SKIP
    >>> a.get()  # doctest: +SKIP
    array([[10,  7,  4],
           [ 3,  2,  1]])
    """
    if axis is not None:
        raise NotImplementedError("'axis' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    if overwrite_input:
        raise NotImplementedError("'overwrite_input' is currently not supported.")
    if interpolation != "linear":
        raise NotImplementedError(
            "only 'linear' 'interpolation' is currently supported."
        )
    if keepdims:
        raise NotImplementedError("'keepdims' is currently not supported.")
    return _instance().quantile(a, q, interpolation=interpolation)


def percentile(
    a: BlockArray,
    q: float,
    axis: Optional[int] = None,
    out: BlockArray = None,
    overwrite_input: bool = False,
    interpolation: {"linear"} = "linear",
    keepdims: bool = False,
) -> BlockArray:
    """Compute the q-th percentile of the data along the specified axis.

    This docstring was copied from numpy.percentile.

    Some inconsistencies with the NumS version may exist.

    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : BlockArray
        Input array or object that can be converted to an array.
    q : float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the array.
    out : BlockArray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired percentile lies between two data points
        ``i < j``:

        * 'linear': ``i + (j - i) * fraction``, where ``fraction``
          is the fractional part of the index surrounded by ``i``
          and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    percentile : BlockArray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    median : equivalent to ``percentile(..., 50)``
    nanpercentile
    quantile : equivalent to percentile, except with q in the range [0, 1].

    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th percentile of
    ``V`` is the value ``q/100`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the percentile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=50``, the same as the minimum if ``q=0`` and the
    same as the maximum if ``q=100``.

    'axis' is currently not supported.

    'out' is currently not supported.

    'overwrite_input' is currently not supported.

    only 'linear' 'interpolation' is currently supported.

    'keepdims' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[10, 7, 4], [3, 2, 1]])  # doctest: +SKIP
    >>> a.get()  # doctest: +SKIP
    array([[10,  7,  4],
           [ 3,  2,  1]])
    """
    if axis is not None:
        raise NotImplementedError("'axis' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    if overwrite_input:
        raise NotImplementedError("'overwrite_input' is currently not supported.")
    if interpolation != "linear":
        raise NotImplementedError(
            "only 'linear' 'interpolation' is currently supported."
        )
    if keepdims:
        raise NotImplementedError("'keepdims' is currently not supported.")
    return _instance().percentile(a, q, interpolation=interpolation)


def median(a: BlockArray, axis=None, out=None, keepdims=False) -> BlockArray:
    """Compute the median along the specified axis.

    This docstring was copied from numpy.median.

    Some inconsistencies with the NumS version may exist.

    Returns the median of the array elements.

    Parameters
    ----------
    a : BlockArray
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the array.
    out : BlockArray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : BlockArray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``nps.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean, percentile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i
    e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
    two middle values of ``V_sorted`` when ``N`` is even.

    'axis' is currently not supported.

    'out' is currently not supported.

    'keepdims' is currently not supported.
    """
    if axis is not None:
        raise NotImplementedError("'axis' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    if keepdims:
        raise NotImplementedError("'keepdims' is currently not supported.")
    return _instance().median(a)


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
