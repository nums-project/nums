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

import numpy as np

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray

from nums.numpy.api.creation import *
from nums.numpy.api.algebra import *
from nums.numpy.api.utility import *

############################################
# Manipulation Ops
############################################


def arange(start=None, stop=None, step=1, dtype=None) -> BlockArray:
    """Return evenly spaced values within a given interval.

    This docstring was copied from numpy.arange.

    Some inconsistencies with the NumS version may exist.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an BlockArray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `nums.linspace` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    arange : BlockArray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.

    Notes
    -----
    Only step size of 1 is currently supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.arange(3).get()  # doctest: +SKIP
    array([0, 1, 2])
    >>> nps.arange(3.0).get()  # doctest: +SKIP
    array([ 0.,  1.,  2.])
    >>> nps.arange(3,7).get()  # doctest: +SKIP
    array([3, 4, 5, 6])
    """
    if start is None:
        raise TypeError("Missing required argument start")
    if stop is None:
        stop = start
        start = 0
    if step != 1:
        raise NotImplementedError("Only step size of 1 is currently supported.")
    if dtype is None:
        dtype = np.__getattribute__(str(np.result_type(start, stop)))
    shape = (int(np.ceil(stop - start)),)
    app = _instance()
    block_shape = app.get_block_shape(shape, dtype)
    return app.arange(start, shape, block_shape, step, dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """Return evenly spaced numbers over a specified interval.

    This docstring was copied from numpy.linspace.

    Some inconsistencies with the NumS version may exist.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : BlockArray
        The starting value of the sequence.
    stop : BlockArray
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : BlockArray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).
    geomspace : Similar to `linspace`, but with numbers spaced evenly on a log
                scale (a geometric progression).
    logspace : Similar to `geomspace`, but with the end points specified as
               logarithms.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.linspace(2.0, 3.0, num=5).get()  # doctest: +SKIP
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    """
    shape = (num,)
    dtype = np.float64 if dtype is None else dtype
    app = _instance()
    block_shape = app.get_block_shape(shape, dtype)
    return app.linspace(start, stop, shape, block_shape, endpoint, retstep, dtype, axis)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """Return numbers spaced evenly on a log scale.

    This docstring was copied from numpy.logspace.

    Some inconsistencies with the NumS version may exist.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    Parameters
    ----------
    start : BlockArray
        ``base ** start`` is the starting value of the sequence.
    stop : BlockArray
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

        .. versionadded:: 1.16.0


    Returns
    -------
    samples : BlockArray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.

    Notes
    -----
    Logspace is equivalent to the code

    >>> y = nps.linspace(start, stop, num=num, endpoint=endpoint)  # doctest: +SKIP
    ... # doctest: +SKIP
    >>> power(base, y).astype(dtype)  # doctest: +SKIP
    ... # doctest: +SKIP

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.logspace(2.0, 3.0, num=4).get()  # doctest: +SKIP
    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])
    >>> nps.logspace(2.0, 3.0, num=4, base=2.0).get()  # doctest: +SKIP
    array([4.        ,  5.0396842 ,  6.34960421,  8.        ])
    """
    app = _instance()
    ba: BlockArray = linspace(start, stop, num, endpoint, dtype=None, axis=axis)
    ba = power(app.scalar(base), ba)
    if dtype is not None and dtype != ba.dtype:
        ba = ba.astype(dtype)
    return ba


def concatenate(arrays, axis=0, out=None):
    """Join a sequence of arrays along an existing axis.

    This docstring was copied from numpy.concatenate.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : BlockArray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.

    Returns
    -------
    res : BlockArray
        The concatenated array.

    See Also
    --------
    ma.concatenate : Concatenate function that preserves input masks.
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.
    split : Split array into a list of multiple sub-arrays of equal size.
    hsplit : Split array into multiple sub-arrays horizontally (column wise).
    vsplit : Split array into multiple sub-arrays vertically (row wise).
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    stack : Stack a sequence of arrays along a new axis.
    block : Assemble arrays from blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    column_stack : Stack 1-D arrays as columns into a 2-D array.

    Notes
    -----
    When one or more of the arrays to be concatenated is a MaskedArray,
    this function will return a MaskedArray object instead of an BlockArray,
    but the input masks are *not* preserved. In cases where a MaskedArray
    is expected as input, use the ma.concatenate function from the masked
    array module instead.

    out is currently not supported for concatenate.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1, 2], [3, 4]])  # doctest: +SKIP
    >>> b = nps.array([[5, 6]])  # doctest: +SKIP
    >>> nps.concatenate((a, b), axis=0).get()  # doctest: +SKIP
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> nps.concatenate((a, b.T), axis=1).get()  # doctest: +SKIP
    array([[1, 2, 5],
           [3, 4, 6]])
    """
    if out is not None:
        raise NotImplementedError("out is currently not supported for concatenate.")
    # Pick the mode along specified axis.
    axis_block_size = scipy.stats.mode(
        list(map(lambda arr: arr.block_shape[axis], arrays))
    ).mode.item()
    return _instance().concatenate(arrays, axis=axis, axis_block_size=axis_block_size)


def split(ary: BlockArray, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays as views into `ary`.

    This docstring was copied from numpy.split.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    ary : BlockArray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in

          - ary[:2]
          - ary[2:3]
          - ary[3:]

        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of BlockArrays
        A list of sub-arrays as views into `ary`.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.

    See Also
    --------
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.  Does not raise an exception if
                  an equal division cannot be made.
    hsplit : Split array into multiple sub-arrays horizontally (column-wise).
    vsplit : Split array into multiple sub-arrays vertically (row wise).
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).

    Notes
    -----
    Split currently supports integers only.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(9.0)  # doctest: +SKIP
    >>> [a.get() for a in  nps.split(x, 3)]  # doctest: +SKIP
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.,  8.])]
    """
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError("Split currently supports integers only.")
    dim_total = ary.shape[axis]
    # Splits into N equal arrays, and raise if this is not possible.
    if dim_total % indices_or_sections != 0:
        raise ValueError(
            "ary axis %s cannot be split into %s equal arrays."
            % (axis, indices_or_sections)
        )
    dim_partial = dim_total // indices_or_sections
    results = []
    ss_op = [slice(None, None, 1) for _ in ary.shape]
    for i in range(0, dim_total, dim_partial):
        start = i
        stop = i + dim_partial
        ss_op[axis] = slice(start, stop, 1)
        ary_part = ary[tuple(ss_op)]
        results.append(ary_part)
    return tuple(results)


def diag(v: BlockArray, k=0) -> BlockArray:
    """Extract a diagonal or construct a diagonal array.

    This docstring was copied from numpy.diag.

    Some inconsistencies with the NumS version may exist.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Parameters
    ----------
    v : BlockArray
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : BlockArray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.
    triu : Upper triangle of an array.

    Notes
    -----
    offset != 0 is currently not supported.

    out is currently not supported.

    axis1 != 0 or axis2 != 1 is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(9).reshape((3,3))  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> nps.diag(x).get()  # doctest: +SKIP
    array([0, 4, 8])

    >>> nps.diag(nps.diag(x)).get()  # doctest: +SKIP
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """
    app = _instance()
    if k != 0:
        raise NotImplementedError("Only k==0 is currently supported.")
    return app.diag(v)


def atleast_1d(*arys):
    """Convert inputs to arrays with at least one dimension.

    This docstring was copied from numpy.atleast_1d.

    Some inconsistencies with the NumS version may exist.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys1, arys2, ... : BlockArray
        One or more input arrays.

    Returns
    -------
    ret : BlockArray
        An array, or list of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.atleast_1d(1.0).get()  # doctest: +SKIP
    array([1.])

    >>> x = nps.arange(9.0).reshape(3,3)  # doctest: +SKIP
    >>> nps.atleast_1d(x).get()  # doctest: +SKIP
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])

    >>> [a.get() for a in nps.atleast_1d(1, [3, 4])]  # doctest: +SKIP
    [array([1]), array([3, 4])]
    """
    return _instance().atleast_1d(*arys)


def atleast_2d(*arys):
    """View inputs as arrays with at least two dimensions.

    This docstring was copied from numpy.atleast_2d.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    arys1, arys2, ... : BlockArray
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : BlockArray
        An array, or list of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.atleast_2d(3.0).get()  # doctest: +SKIP
    array([[3.]])

    >>> x = nps.arange(3.0)  # doctest: +SKIP
    >>> nps.atleast_2d(x).get()  # doctest: +SKIP
    array([[0., 1., 2.]])

    >>> [a.get() for a in nps.atleast_2d(1, [1, 2], [[1, 3]])]  # doctest: +SKIP
    [array([[1]]), array([[1, 2]]), array([[1, 2]])]
    """
    return _instance().atleast_2d(*arys)


def atleast_3d(*arys):
    """View inputs as arrays with at least three dimensions.

    This docstring was copied from numpy.atleast_3d.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    arys1, arys2, ... : BlockArray
        One or more array-like sequences.  Non-array inputs are converted to
        arrays.  Arrays that already have three or more dimensions are
        preserved.

    Returns
    -------
    res1, res2, ... : BlockArray
        An array, or list of arrays, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
        of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.

    See Also
    --------
    atleast_1d, atleast_2d

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.atleast_3d(3.0).get()  # doctest: +SKIP
    array([[[3.]]])

    >>> x = nps.arange(3.0)  # doctest: +SKIP
    >>> nps.atleast_3d(x).shape  # doctest: +SKIP
    (1, 3, 1)

    >>> x = nps.arange(12.0).reshape(4,3)  # doctest: +SKIP
    >>> nps.atleast_3d(x).shape  # doctest: +SKIP
    (4, 3, 1)

    >>> for arr in nps.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):  # doctest: +SKIP
    ...     print(arr.get(), arr.shape) # doctest: +SKIP
    ...
    [[[1]
      [2]]] (1, 2, 1)
    [[[1]
      [2]]] (1, 2, 1)
    [[[1 2]]] (1, 1, 2)
    """
    return _instance().atleast_3d(*arys)


def hstack(tup):
    """Stack arrays in sequence horizontally (column wise).

    This docstring was copied from numpy.hstack.

    Some inconsistencies with the NumS version may exist.

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of BlockArray
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    Returns
    -------
    stacked : BlockArray
        The array formed by stacking the given arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    hsplit : Split an array into multiple sub-arrays horizontally (column-wise).

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array((1,2,3))  # doctest: +SKIP
    >>> b = nps.array((2,3,4))  # doctest: +SKIP
    >>> nps.hstack((a,b)).get()  # doctest: +SKIP
    array([1, 2, 3, 2, 3, 4])
    >>> a = nps.array([[1],[2],[3]])  # doctest: +SKIP
    >>> b = nps.array([[2],[3],[4]])  # doctest: +SKIP
    >>> nps.hstack((a,b)).get()  # doctest: +SKIP
    array([[1, 2],
           [2, 3],
           [3, 4]])
    """
    return _instance().hstack(tup)


def vstack(tup):
    """Stack arrays in sequence vertically (row wise).

    This docstring was copied from numpy.vstack.

    Some inconsistencies with the NumS version may exist.

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of BlockArrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : BlockArray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([1, 2, 3])  # doctest: +SKIP
    >>> b = nps.array([2, 3, 4])  # doctest: +SKIP
    >>> nps.vstack((a,b)).get()  # doctest: +SKIP
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> a = nps.array([[1], [2], [3]])  # doctest: +SKIP
    >>> b = nps.array([[2], [3], [4]])  # doctest: +SKIP
    >>> nps.vstack((a,b)).get()  # doctest: +SKIP
    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])
    """
    return _instance().vstack(tup)


def dstack(tup):
    """Stack arrays in sequence depth wise (along third axis).

    This docstring was copied from numpy.dstack.

    Some inconsistencies with the NumS version may exist.

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of arrays
        The arrays must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : BlockArray
        The array formed by stacking the given arrays, will be at least 3-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    hstack : Stack arrays in sequence horizontally (column wise).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    dsplit : Split array along third axis.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array((1,2,3))  # doctest: +SKIP
    >>> b = nps.array((2,3,4))  # doctest: +SKIP
    >>> nps.dstack((a,b)).get()  # doctest: +SKIP
    array([[[1, 2],
            [2, 3],
            [3, 4]]])

    >>> a = nps.array([[1],[2],[3]])  # doctest: +SKIP
    >>> b = nps.array([[2],[3],[4]])  # doctest: +SKIP
    >>> nps.dstack((a,b)).get()  # doctest: +SKIP
    array([[[1, 2]],
           [[2, 3]],
           [[3, 4]]])
    """
    return _instance().dstack(tup)


def row_stack(tup):
    """Stack arrays in sequence vertically (row wise).

    This docstring was copied from numpy.row_stack.

    Some inconsistencies with the NumS version may exist.

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of BlockArrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : BlockArray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([1, 2, 3])  # doctest: +SKIP
    >>> b = nps.array([2, 3, 4])  # doctest: +SKIP
    >>> nps.vstack((a,b)).get()  # doctest: +SKIP
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> a = nps.array([[1], [2], [3]])  # doctest: +SKIP
    >>> b = nps.array([[2], [3], [4]])  # doctest: +SKIP
    >>> nps.vstack((a,b)).get()  # doctest: +SKIP
    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])
    """
    return _instance().row_stack(tup)


def column_stack(tup):
    """Stack 1-D arrays as columns into a 2-D array.

    This docstring was copied from numpy.column_stack.

    Some inconsistencies with the NumS version may exist.

    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
    stacked : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array((1,2,3))  # doctest: +SKIP
    >>> b = nps.array((2,3,4))  # doctest: +SKIP
    >>> nps.column_stack((a,b)).get()  # doctest: +SKIP
    array([[1, 2],
           [2, 3],
           [3, 4]])
    """
    return _instance().column_stack(tup)
