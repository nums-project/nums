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

############################################
# NaN Ops
############################################


def nanmax(a: BlockArray, axis=None, out=None, keepdims=False):
    """Return the maximum of an array or maximum along an axis, ignoring any
    NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
    raised and NaN is returned for that slice.

    This docstring was copied from numpy.nanmax.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Array containing numbers whose maximum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the maximum is computed. The default is to compute
        the maximum of the flattened array.
    out : BlockArray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        If the value is anything but the default, then
        `keepdims` will be passed through to the `max` method
        of sub-classes of `BlockArray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

    Returns
    -------
    nanmax : BlockArray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, an BlockArray scalar is
        returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amin, fmin, minimum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to nps.max.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, 2], [3, nps.nan]])  # doctest: +SKIP
    >>> nps.nanmax(a).get()  # doctest: +SKIP
    array(3.)
    >>> nps.nanmax(a, axis=0).get()  # doctest: +SKIP
    array([3.,  2.])
    >>> nps.nanmax(a, axis=1).get()  # doctest: +SKIP
    array([2.,  3.])
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("nanmax", a, axis=axis, keepdims=keepdims)


def nanmin(a: BlockArray, axis=None, out=None, keepdims=False):
    """Return minimum of an array or minimum along an axis, ignoring any NaNs.
    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
    Nan is returned for that slice.

    This docstring was copied from numpy.nanmin.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Array containing numbers whose minimum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the minimum is computed. The default is to compute
        the minimum of the flattened array.
    out : BlockArray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `min` method
        of sub-classes of `BlockArray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

    Returns
    -------
    nanmin : BlockArray
        An array with the same shape as `a`, with the specified axis
        removed.  If `a` is a 0-d array, or if axis is None, an BlockArray
        scalar is returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amax, fmax, maximum

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to nps.min.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, 2], [3, nps.nan]])  # doctest: +SKIP
    >>> nps.nanmin(a).get()  # doctest: +SKIP
    arary(1.)
    >>> nps.nanmin(a, axis=0).get()  # doctest: +SKIP
    array([1.,  2.])
    >>> nps.nanmin(a, axis=1).get()  # doctest: +SKIP
    array([1.,  3.])
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("nanmin", a, axis=axis, keepdims=keepdims)


def nansum(a: BlockArray, axis=None, dtype=None, out=None, keepdims=False):
    """Return the sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.

    This docstring was copied from numpy.nansum.

    Some inconsistencies with the NumS version may exist.

    In NumPy versions <= 1.9.0 Nan is returned for slices that are all-NaN or
    empty. In later versions zero is returned.

    Parameters
    ----------
    a : BlockArray
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the sum is computed. The default is to compute the
        sum of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : BlockArray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        `ufuncs-output-type` for more details. The casting of NaN to integer
        can yield unexpected results.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `BlockArray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

    Returns
    -------
    nansum : BlockArray.
        A new array holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.

    See Also
    --------
    numpy.sum : Sum across array propagating NaNs.
    isnan : Show which elements are NaN.
    isfinite: Show which elements are not NaN or +/-inf.

    Notes
    -----
    If both positive and negative infinity are present, the sum will be Not
    A Number (NaN).

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.nansum(nps.array([1])).get()  # doctest: +SKIP
    array(1)
    >>> nps.nansum(nps.array([1, nps.nan])).get()  # doctest: +SKIP
    array(1.)
    >>> a = nps.array([[1, 1], [1, nps.nan]])  # doctest: +SKIP
    >>> nps.nansum(a).get()  # doctest: +SKIP
    array(3.)
    >>> nps.nansum(a, axis=0).get()  # doctest: +SKIP
    array([2.,  1.])
    >>> nps.nansum(nps.array([1, nps.nan, nps.inf])).get()  # doctest: +SKIP
    array(inf)
    >>> nps.nansum(nps.array([1, nps.nan, nps.NINF])).get()  # doctest: +SKIP
    array(-inf)
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("nansum", a, axis=axis, dtype=dtype, keepdims=keepdims)


def nanmean(a: BlockArray, axis=None, dtype=None, out=None, keepdims=False):
    """Compute the arithmetic mean along the specified axis, ignoring NaNs.

    This docstring was copied from numpy.nanmean.

    Some inconsistencies with the NumS version may exist.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float` intermediate and return values are used for integer inputs.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : BlockArray
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for inexact inputs, it is the same as the input
        dtype.
    out : BlockArray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `BlockArray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

    Returns
    -------
    m : BlockArray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned. Nan is
        returned for slices that contain only NaNs.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-NaN elements along the axis
    divided by the number of non-NaN elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32`.  Specifying a
    higher-precision accumulator using the `dtype` keyword can alleviate
    this issue.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, nps.nan], [3, 4]])  # doctest: +SKIP
    >>> nps.nanmean(a).get()  # doctest: +SKIP
    array(2.66666667)
    >>> nps.nanmean(a, axis=0).get()  # doctest: +SKIP
    array([2.,  4.])
    >>> nps.nanmean(a, axis=1).get()  # doctest: +SKIP
    array([1.,  3.5]) # may vary
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().nanmean(a, axis=axis, dtype=dtype, keepdims=keepdims)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Compute the variance along the specified axis, while ignoring NaNs.

    This docstring was copied from numpy.nanvar.

    Some inconsistencies with the NumS version may exist.

    Returns the variance of the array elements, a measure of the spread of
    a distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : BlockArray
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the variance is computed.  The default is to compute
        the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float64`; for arrays of float types it is the same as
        the array type.
    out : BlockArray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN
        elements. By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.


    Returns
    -------
    variance : BlockArray, see dtype parameter above
        If `out` is None, return a new array containing the variance,
        otherwise return a reference to the output array. If ddof is >= the
        number of non-NaN elements in a slice or the slice contains only
        NaNs, then the result for that slice is NaN.

    See Also
    --------
    std : Standard deviation
    mean : Average
    var : Variance while not ignoring NaNs
    nanstd, nanmean

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite
    population.  ``ddof=0`` provides a maximum likelihood estimate of the
    variance for normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    For this function to work on sub-classes of BlockArray, they must define
    `sum` with the kwarg `keepdims`

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, nps.nan], [3, 4]])  # doctest: +SKIP
    >>> nps.nanvar(a).get()  # doctest: +SKIP
    array(1.55555556)
    >>> nps.nanvar(a, axis=0).get()  # doctest: +SKIP
    array([1.,  0.])
    >>> nps.nanvar(a, axis=1).get()  # doctest: +SKIP
    array([0.,  0.25])  # may vary
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Compute the standard deviation along the specified axis, while
    ignoring NaNs.

    This docstring was copied from numpy.nanstd.

    Some inconsistencies with the NumS version may exist.

    Returns the standard deviation, a measure of the spread of a
    distribution, of the non-NaN array elements. The standard deviation is
    computed for the flattened array by default, otherwise over the
    specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : BlockArray
        Calculate the standard deviation of the non-NaN values.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it
        is the same as the array type.
    out : BlockArray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the
        calculated values) will be cast if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of non-NaN
        elements.  By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        If this value is anything but the default it is passed through
        as-is to the relevant functions of the sub-classes.  If these
        functions do not have a `keepdims` kwarg, a RuntimeError will
        be raised.

    Returns
    -------
    standard_deviation : BlockArray, see dtype parameter above.
        If `out` is None, return a new array containing the standard
        deviation, otherwise return a reference to the output array. If
        ddof is >= the number of non-NaN elements in a slice or the slice
        contains only NaNs, then the result for that slice is NaN.

    See Also
    --------
    var, mean, std
    nanvar, nanmean

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is
    specified, the divisor ``N - ddof`` is used instead. In standard
    statistical practice, ``ddof=1`` provides an unbiased estimator of the
    variance of the infinite population. ``ddof=0`` provides a maximum
    likelihood estimate of the variance for normally distributed variables.
    The standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.

    Note that, for complex numbers, `std` takes the absolute value before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the *std* is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example
    below).  Specifying a higher-accuracy accumulator using the `dtype`
    keyword can alleviate this issue.

    'out' is currently not supported'

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, nps.nan], [3, 4]])  # doctest: +SKIP
    >>> nps.nanstd(a).get()  # doctest: +SKIP
    array(1.24721913)
    >>> nps.nanstd(a, axis=0).get()  # doctest: +SKIP
    array([1., 0.])
    >>> nps.nanstd(a, axis=1).get()  # doctest: +SKIP
    array([0.,  0.5]) # may vary
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().nanstd(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
