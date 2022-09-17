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

import warnings
import numpy as np
import sparse

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray

from nums.numpy.api.stats import *
from nums.numpy.api.logic import *

############################################
# Creation Ops
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


def array(object, dtype=None, copy=True, order="K", ndmin=0, subok=False) -> BlockArray:
    """Creates a BlockArray.

    This docstring was copied from numpy.array.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy will
        only be made if __array__ returns a copy, if obj is a nested sequence,
        or if a copy is needed to satisfy any of the other requirements
        (`dtype`, `order`, etc.).
    order : {'K'}, optional
        Specify the memory layout of the array. If object is not an array, the
        newly created array will be in C order (row major) unless 'F' is
        specified, in which case it will be in Fortran order (column major).
        If object is an array the following holds.

        ===== ========= ===================================================
        order  no copy                     copy=True
        ===== ========= ===================================================
        'K'   unchanged F & C order preserved, otherwise most similar order
        'A'   unchanged F order if input is F and not C, otherwise C order
        'C'   C order   C order
        'F'   F order   F order
        ===== ========= ===================================================

        When ``copy=False`` and a copy is made for other reasons, the result is
        the same as if ``copy=True``, with some exceptions for `A`, see the
        Notes section. The default order is 'K'.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.

    Returns
    -------
    out : BlockArray
        An array object satisfying the specified requirements.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.


    Notes
    -----
    Only order='K' is supported.

    Only ndmin=0 is currently supported.

    subok must be False

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.array([1, 2, 3]).get()  # doctest: +SKIP
    array([1, 2, 3])

    Upcasting:

    >>> nps.array([1, 2, 3.0]).get()  # doctest: +SKIP
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> nps.array([[1, 2], [3, 4]]).get()  # doctest: +SKIP
    array([[1, 2],
           [3, 4]])

    Type provided:

    >>> nps.array([1, 2, 3], dtype=complex).get()  # doctest: +SKIP
    array([ 1.+0.j,  2.+0.j,  3.+0.j])
    """
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    if ndmin != 0:
        raise NotImplementedError("Only ndmin=0 is currently supported.")
    if subok:
        raise ValueError("subok must be False.")
    if isinstance(object, BlockArray):
        if copy:
            object = object.copy()
        if dtype is not None:
            if dtype is not object.dtype:
                object = object.astype(dtype)
        return object
    result = np.array(
        object, dtype=dtype, copy=copy, order=order, ndmin=ndmin, subok=subok
    )
    dtype = np.__getattribute__(str(result.dtype))
    shape = result.shape
    app = _instance()
    block_shape = app.compute_block_shape(shape, dtype)
    return app.array(result, block_shape)


def from_coo(a: sparse.COO):
    assert isinstance(a, sparse.COO)
    dtype = np.__getattribute__(str(a.dtype))
    shape = a.shape
    app = _instance()
    block_shape = app.compute_block_shape(shape, dtype)
    return app.array(a, block_shape)


def copy(a: BlockArray, order="K", subok=False):
    """Return an array copy of the given object.

    This docstring was copied from numpy.copy.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :meth:`BlockArray.copy` are very
        similar, but have different default values for their order=
        arguments.)
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the
        returned array will be forced to be a base-class array (defaults to False).

    Returns
    -------
    arr : BlockArray
        Array interpretation of `a`.

    See Also
    --------
    copy : Preferred method for creating an array copy

    Notes
    -----
    This is equivalent to:

    >>> nps.array(a, copy=True).get()  #doctest: +SKIP

    Only default args supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    Create an array x, with a reference y and a copy z:

    >>> x = nps.array([1, 2, 3])  # doctest: +SKIP
    >>> y = x  # doctest: +SKIP
    >>> z = nps.copy(x)  # doctest: +SKIP

    Note that, when we modify x, y changes, but not z:

    >>> x[0] = 10  # doctest: +SKIP
    >>> (x[0] == y[0]).get()  # doctest: +SKIP
    array(True)
    >>> (x[0] == z[0]).get()  # doctest: +SKIP
    False
    """
    assert order == "K" and not subok, "Only default args supported."
    return a.copy()


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


def empty(shape, dtype=float):
    """Return a new array of given shape and type, without initializing entries.

    This docstring was copied from numpy.empty.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `int`. Default is
        `float`.

    Returns
    -------
    out : BlockArray
        Array of uninitialized (arbitrary) data of the given shape and dtype.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.


    Notes
    -----
    `empty`, unlike `zeros`, does not set the array values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.empty([2, 2]).get()  # doctest: +SKIP
    array([[ -9.74499359e+001,   6.69583040e-309],
           [  2.13182611e-314,   3.06959433e-309]])         #uninitialized

    >>> nps.empty([2, 2], dtype=int).get()  # doctest: +SKIP
    array([[-1073741821, -1067949133],
           [  496041986,    19249760]])                     #uninitialized

    """
    app = _instance()
    if isinstance(shape, int):
        shape = (shape,)
    block_shape = app.compute_block_shape(shape, dtype)
    return app.empty(shape=shape, block_shape=block_shape, dtype=dtype)


def empty_like(prototype: BlockArray, dtype=None, order="K", shape=None):
    """Return a new array with the same shape and type as a given array.

    This docstring was copied from numpy.empty_like.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    prototype : BlockArray
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if ``prototype`` is Fortran
        contiguous, 'C' otherwise. 'K' means match the layout of ``prototype``
        as closely as possible.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.

    Returns
    -------
    out : BlockArray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `prototype`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.

    Only order='K' is supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = ([1,2,3], [4,5,6])                         # a is array-like  # doctest: +SKIP
    >>> nps.empty_like(a, shape=(2, 3), dtype=float).get()  # doctest: +SKIP
    array([[-1073741821, -1073741821,           3],    # uninitialized
           [          0,           0, -1073741821]])
    >>> a = nps.array([[1., 2., 3.],[4.,5.,6.]]).get()  # doctest: +SKIP
    >>> nps.empty_like(a, shape=(2, 3), dtype=float).get()  # doctest: +SKIP
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000], # uninitialized
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    """
    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    return empty(shape, dtype)


def eye(N, M=None, k=0, dtype=float):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    This docstring was copied from numpy.eye.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.

    Returns
    -------
    I : BlockArray of shape (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    See Also
    --------
    identity : (almost) equivalent function
    diag : diagonal 2-D array from a 1-D array specified by the user.

    Notes
    -----
    Only k==0 is currently supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.eye(2, dtype=int).get()  # doctest: +SKIP
    array([[1, 0],
           [0, 1]])
    >>> nps.eye(3, k=1).get()  # doctest: +SKIP
    array([[0.,  1.,  0.],
           [0.,  0.,  1.],
           [0.,  0.,  0.]])
    """
    app = _instance()
    if k != 0:
        raise NotImplementedError("Only k==0 is currently supported.")
    if M is None:
        M = N
    shape = (N, M)
    block_shape = app.get_block_shape(shape, dtype)
    return app.eye(shape, block_shape, dtype)


def identity(n: int, dtype=float) -> BlockArray:
    """Return the identity array.

    This docstring was copied from numpy.identity.

    Some inconsistencies with the NumS version may exist.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : BlockArray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.identity(3).get()  # doctest: +SKIP
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])
    """
    return eye(n, n, dtype=dtype)


def loadtxt(
    fname,
    dtype=float,
    comments="# ",
    delimiter=" ",
    converters=None,
    skiprows=0,
    usecols=None,
    unpack=False,
    ndmin=0,
    encoding="bytes",
    max_rows=None,
) -> BlockArray:
    """Load data from a text file.

    This docstring was copied from numpy.loadtxt.

    Some inconsistencies with the NumS version may exist.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file, str, or pathlib.Path
        File, filename, or generator to read.  If the filename extension is
        ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators should return byte strings.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The string used to separate values. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will parse the
        column string into the desired value.  E.g., if column 0 is a date
        string: ``converters = {0: datestr2num}``.  Converters can also be
        used to provide a default value for missing data (but see also
        `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
        Default: None.
    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The default, None, results in all columns being read.

            When a single column has to be read it is possible to use
            an integer instead of a tuple. E.g ``usecols = 3`` reads the
            fourth column the same way as ``usecols = (3,)`` would.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
        data-type, arrays are returned for each field.  Default is False.
    ndmin : int, optional
        The returned array will have at least `ndmin` dimensions.
        Otherwise mono-dimensional axes will be squeezed.
        Legal values: 0 (default), 1 or 2.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply to input streams.
        The special value 'bytes' enables backward compatibility workarounds
        that ensures you receive byte arrays as results if possible and passes
        'latin1' encoded strings to converters. Override this value to receive
        unicode arrays and pass strings as input to converters.  If set to None
        the system default is used. The default value is 'bytes'.
    max_rows : int, optional
        Read `max_rows` lines of content after `skiprows` lines. The default
        is to read all the lines.

    Returns
    -------
    out : BlockArray
        Data read from the text file.

    Notes
    -----
    This function aims to be a fast reader for simply formatted files.  The
    `genfromtxt` function provides more sophisticated handling of, e.g.,
    lines with missing values.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> from io import StringIO  # doctest: +SKIP
    >>> c = StringIO("0 1\\n2 3")  # doctest: +SKIP
    >>> nps.loadtxt(c).get()  # doctest: +SKIP
    array([[0., 1.],
           [2., 3.]])

    >>> c = StringIO("1,0,2\\n3,0,4")  # doctest: +SKIP
    >>> x, y = nps.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array([1., 3.])
    >>> y.get()  # doctest: +SKIP
    array([2., 4.])
    """
    app = _instance()
    num_rows = app.km.num_cores_total()
    try:
        ba: BlockArray = app.loadtxt(
            fname,
            dtype=dtype,
            comments=comments,
            delimiter=delimiter,
            converters=converters,
            skiprows=skiprows,
            usecols=usecols,
            unpack=unpack,
            ndmin=ndmin,
            encoding=encoding,
            max_rows=max_rows,
            num_workers=num_rows,
        )
        shape = ba.shape
        block_shape = app.compute_block_shape(shape, dtype)
        return ba.reshape(block_shape=block_shape)
    except Exception as _:
        warnings.warn("Failed to load text data in parallel; using np.loadtxt locally.")
        np_arr = np.loadtxt(
            fname,
            dtype=dtype,
            comments=comments,
            delimiter=delimiter,
            converters=converters,
            skiprows=skiprows,
            usecols=usecols,
            unpack=unpack,
            ndmin=ndmin,
            encoding=encoding,
            max_rows=max_rows,
        )
        shape = np_arr.shape
        block_shape = app.compute_block_shape(shape, dtype)
        return app.array(np_arr, block_shape=block_shape)


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


def ones(shape, dtype=float):
    """Return a new array of given shape and type, filled with ones.

    This docstring was copied from numpy.ones.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `int`.  Default is
        `float`.

    Returns
    -------
    out : BlockArray
        Array of ones with the given shape and dtype.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.ones(5).get()  # doctest: +SKIP
    array([1., 1., 1., 1., 1.])

    >>> nps.ones((5,), dtype=int).get()  # doctest: +SKIP
    array([1, 1, 1, 1, 1])

    >>> nps.ones((2, 1)).get()  # doctest: +SKIP
    array([[1.],
           [1.]])

    >>> s = (2,2)  # doctest: +SKIP
    >>> nps.ones(s).get()  # doctest: +SKIP
    array([[1.,  1.],
           [1.,  1.]])
    """
    app = _instance()
    if isinstance(shape, int):
        shape = (shape,)
    block_shape = app.get_block_shape(shape, dtype)
    return app.ones(shape=shape, block_shape=block_shape, dtype=dtype)


def ones_like(prototype, dtype=None, order="K", shape=None):
    """Return an array of ones with the same shape and type as a given array.

    This docstring was copied from numpy.ones_like.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.

    Returns
    -------
    out : BlockArray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    ones : Return a new array setting values to one.

    Notes
    -----
    Only order='K' is supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(6)  # doctest: +SKIP
    >>> x = x.reshape((2, 3))  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> nps.ones_like(x).get()  # doctest: +SKIP
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = nps.arange(3, dtype=float)  # doctest: +SKIP
    >>> y.get()  # doctest: +SKIP
    array([0., 1., 2.])
    >>> nps.ones_like(y).get()  # doctest: +SKIP
    array([1.,  1.,  1.])
    """
    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    return ones(shape, dtype)


def zeros(shape, dtype=float):
    """Return a new array of given shape and type, without initializing entries.

    This docstring was copied from numpy.zeros.

    Some inconsistencies with the NumS version may exist.

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `int`.  Default is
        `float`.

    Returns
    -------
    out : BlockArray
        Array of zeros with the given shape and dtype.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.zeros(5).get()  # doctest: +SKIP
    array([ 0.,  0.,  0.,  0.,  0.])

    >>> nps.zeros((5,), dtype=int).get()  # doctest: +SKIP
    array([0, 0, 0, 0, 0])

    >>> nps.zeros((2, 1)).get()  # doctest: +SKIP
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)  # doctest: +SKIP
    >>> nps.zeros(s).get()  # doctest: +SKIP
    array([[ 0.,  0.],
           [ 0.,  0.]])
    """
    app = _instance()
    if isinstance(shape, int):
        shape = (shape,)
    block_shape = app.get_block_shape(shape, dtype)
    return app.zeros(shape=shape, block_shape=block_shape, dtype=dtype)


def zeros_like(prototype, dtype=None, order="K", shape=None):
    """Return an array of zeros with the same shape and type as a given array.

    This docstring was copied from numpy.zeros_like.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `prototype` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.

    Returns
    -------
    out : BlockArray
        Array of zeros with the same shape and type as `prototype`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    Only order='K' is supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(6)  # doctest: +SKIP
    >>> x = x.reshape((2, 3))  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> nps.zeros_like(x).get()  # doctest: +SKIP
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = nps.arange(3, dtype=float)  # doctest: +SKIP
    >>> y.get()  # doctest: +SKIP
    array([0., 1., 2.])
    >>> nps.zeros_like(y).get()  # doctest: +SKIP
    array([0.,  0.,  0.])
    """
    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    return zeros(shape, dtype)


def asarray(a, dtype=None, order=None, *, like=None):
    """Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : BlockArray
        Array interpretation of `a`.  No copy is performed if the input
        is already a BlockArray with matching dtype and order.

    Raises
    ------
    NotImplementedError
        If you pass a non-`None` value to `order` or `like`.

    See Also
    --------
    asanyarray : Similar function which passes through subclasses.
    asfarray : Convert input to a floating point ndarray.
    asarray_chkfinite : Similar function which checks input for NaNs and Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> nps.asarray(a).get()
    array([1, 2])

    Existing arrays are not copied:

    >>> a = nps.array([1, 2])
    >>> nps.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = nps.array([1, 2], dtype=np.float32)
    >>> nps.asarray(a, dtype=np.float32) is a
    True
    >>> nps.asarray(a, dtype=np.float64) is a
    False
    """
    if order is not None:
        raise NotImplementedError("The `order` parameter isn't supported.")
    if like is not None:
        raise NotImplementedError("The `like` parameter isn't supported.")

    if type(a) is BlockArray:  # pylint: disable=unidiomatic-typecheck
        is_matching_dtype = not dtype or a.dtype == dtype
        if is_matching_dtype:
            return a
        else:
            return a.astype(dtype)

    a = np.asarray(a, dtype=dtype)
    return array(a, dtype=a.dtype, copy=True)
