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

# pylint: disable = redefined-builtin, too-many-lines, W1401, W0401, W0614

import warnings
import scipy.stats

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray

from nums.numpy.api.stats import *
from nums.numpy.api.utility import *

############################################
# Creation and I/O Ops
############################################


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
    num_rows = app.cm.num_cores_total()
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


def trace(a: BlockArray, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Return the sum along diagonals of the array.

    This docstring was copied from numpy.trace.

    Some inconsistencies with the NumS version may exist.

    If `a` is 2-D, the sum along its diagonal with the given offset
    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

    If `a` has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of `a` with `axis1`
    and `axis2` removed.

    Parameters
    ----------
    a : BlockArray
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the first two
        axes of `a`.
    dtype : dtype, optional
        Determines the data-type of the returned array and of the accumulator
        where the elements are summed. If dtype has the value None and `a` is
        of integer type of precision less than the default integer
        precision, then the default integer precision is used. Otherwise,
        the precision is the same as that of `a`.
    out : BlockArray, optional
        Array into which the output is placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : BlockArray
        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
        larger dimensions, then an array of sums along diagonals is returned.

    See Also
    --------
    diag, diagonal

    Notes
    -----
    offset != 0 is currently not supported.

    out is currently not supported.

    axis1 != 0 or axis2 != 1 is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.trace(nps.eye(3)).get()  # doctest: +SKIP
    array(3.)
    >>> a = nps.arange(8).reshape((2,2,2))  # doctest: +SKIP
    """
    if offset != 0:
        raise NotImplementedError("offset != 0 is currently not supported.")
    if out is not None:
        raise NotImplementedError("out is currently not supported.")
    if axis1 != 0 or axis2 != 1:
        raise NotImplementedError(
            " axis1 != 0 or axis2 != 1 is currently not supported."
        )
    return sum(diag(a, offset), dtype=dtype, out=out)


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
