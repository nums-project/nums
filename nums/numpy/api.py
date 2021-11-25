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


import warnings

from typing import Tuple, Optional, Union

import numpy as np
import scipy.stats

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray
from nums.numpy import numpy_utils
from nums.core.utils import derived_from


# pylint: disable = redefined-builtin, too-many-lines


def _not_implemented(func):
    # From project JAX: https://github.com/google/jax/blob/master/jax/numpy/lax_numpy.py
    def wrapped(*args, **kwargs):
        # pylint: disable=unused-argument
        msg = "NumPy function {} not yet implemented."
        raise NotImplementedError(msg.format(func))

    return wrapped


# TODO (mwe): Convert this to invoke the NumPy op on a worker instead of the driver.
def _default_to_numpy(func):
    def wrapped(*args, **kwargs):
        warnings.warn(
            "Operation "
            + func.__name__
            + " not implemented, falling back to NumPy. "
            + "If this is too slow or failing, please open an issue on GitHub.",
            RuntimeWarning,
        )
        new_args = [arg.get() if isinstance(arg, BlockArray) else arg for arg in args]
        new_kwargs = {
            k: v.get() if isinstance(v, BlockArray) else v
            for k, v in zip(kwargs.keys(), kwargs.values())
        }
        res = np.__getattribute__(func.__name__)(*new_args, **new_kwargs)
        if isinstance(res, tuple):
            nps_res = tuple(array(x) for x in res)
        else:
            nps_res = array(res)
        return nps_res

    return wrapped


############################################
# Constants
############################################

# Distributed memory access of these values will be optimized downstream.
pi = np.pi
e = np.e
euler_gamma = np.euler_gamma
inf = infty = Inf = Infinity = PINF = np.inf
NINF = np.NINF
PZERO = np.PZERO
NZERO = np.NZERO
nan = NAN = NaN = np.nan

############################################
# Data Types
############################################


bool_ = np.bool_

uint = np.uint
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

float16 = np.float16
float32 = np.float32
float64 = np.float64

complex64 = np.complex64
complex128 = np.complex128


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
    this function will return a MaskedArray object instead of an ndarray,
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
    sub-arrays : list of ndarrays
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
    tup : sequence of ndarrays
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
    tup : sequence of ndarrays
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
    `range` function, but returns an ndarray rather than a list.

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


############################################
# Linear Algebra Ops
############################################


def tensordot(x1: BlockArray, x2: BlockArray, axes=2) -> BlockArray:
    """Compute tensor dot product along specified axes.

    This docstring was copied from numpy.tensordot.

    Some inconsistencies with the NumS version may exist.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : BlockArray
        Tensors to "dot".

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    output : BlockArray
        The tensor dot product of the input.

    See Also
    --------
    dot

    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.

    Non-integer axes is currently not supported.
    """
    return _instance().tensordot(arr_1=x1, arr_2=x2, axes=axes)


def matmul(x1: BlockArray, x2: BlockArray) -> BlockArray:
    """Matrix product of two arrays.

    This docstring was copied from numpy.matmul.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays, scalars not allowed.

    Returns
    -------
    y : BlockArray
        The matrix product of the inputs.
        This is a scalar only when both x1, x2 are 1-d vectors.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

        If a scalar value is passed in.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : alternative matrix product with different broadcasting rules.
    """
    return _instance().matmul(arr_1=x1, arr_2=x2)


def inner(a: BlockArray, b: BlockArray):
    """Inner product of two arrays.

    This docstring was copied from numpy.inner.

    Some inconsistencies with the NumS version may exist.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : BlockArray
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : BlockArray
        `out.shape = a.shape[:-1] + b.shape[:-1]`

    Raises
    ------
    ValueError
        If the last dimension of `a` and `b` has different size.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.

    Notes
    -----
    Only single-axis inputs supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    Ordinary inner product for vectors:

    >>> a = nps.array([1,2,3])  # doctest: +SKIP
    >>> b = nps.array([0,1,0])  # doctest: +SKIP
    >>> nps.inner(a, b).get() # doctest: +SKIP
    array(2)
    """
    assert len(a.shape) == len(b.shape) == 1, "Only single-axis inputs supported."
    return a.T @ b


def outer(a: BlockArray, b: BlockArray):
    """Compute the outer product of two vectors.

    This docstring was copied from numpy.outer.

    Some inconsistencies with the NumS version may exist.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::

      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) BlockArray
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) BlockArray
        Second input vector.  Input is flattened if
        not already 1-dimensional.

    Returns
    -------
    out : (M, N) BlockArray
        ``out[i, j] = a[i] * b[j]``

    See also
    --------
    inner
    outer
    tensordot

    Notes
    -----
    Only single-axis inputs supported.
    """
    assert len(a.shape) == len(b.shape) == 1, "Only single-axis inputs supported."
    return a.reshape((a.shape[0], 1)) @ b.reshape((1, b.shape[0]))


def dot(a: BlockArray, b: BlockArray, out=None) -> BlockArray:
    """Dot product of two arrays.

    This docstring was copied from numpy.dot.

    Some inconsistencies with the NumS version may exist.

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
      and using ``numpy.multiply(a, b)`` or ``a * b`` is preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : BlockArray
        First argument.
    b : BlockArray
        Second argument.
    out : BlockArray, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : BlockArray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    matmul : '@' operator as method with out parameter.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    For 2-D arrays it is the matrix product:

    >>> a = nps.array([[1, 0], [0, 1]])  # doctest: +SKIP
    >>> b = nps.array([[4, 1], [2, 2]])  # doctest: +SKIP
    >>> nps.dot(a, b).get()  # doctest: +SKIP
    array([[4, 1],
           [2, 2]])
    """
    assert out is None, "Specifying an output array is not supported."
    a_len, b_len = len(a.shape), len(b.shape)
    if a_len == b_len == 1:
        return inner(a, b)
    elif a_len == b_len == 2:
        return matmul(a, b)
    elif a_len == 0 or b_len == 0:
        return multiply(a, b)
    else:
        raise NotImplementedError(
            "The dot operation on arbitrary arrays is not yet supported."
        )


############################################
# Shape Ops
############################################


def shape(a: BlockArray):
    """Return the shape of an array.

    This docstring was copied from numpy.shape.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.shape(nps.eye(3))  # doctest: +SKIP
    (3, 3)
    """
    return a.shape


def size(a: BlockArray):
    """Return the number of elements along a given axis.

    This docstring was copied from numpy.size.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input data.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    See Also
    --------
    shape : dimensions of array

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([[1,2,3],[4,5,6]])  # doctest: +SKIP
    >>> nps.size(a)  # doctest: +SKIP
    6
    """
    return a.size


def ndim(a: BlockArray):
    """Return the number of dimensions of an array.

    This docstring was copied from numpy.ndim.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array.  If it is not already an ndarray, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    shape : dimensions of array

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.ndim(nps.array([[1,2,3],[4,5,6]])).get()  # doctest: +SKIP
    2
    """
    return a.ndim


def reshape(a: BlockArray, shape):
    """Gives a new shape to an array without changing its data.

    This docstring was copied from numpy.reshape.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Array to be reshaped.

    Returns
    -------
    reshaped_array : BlockArray
        This will be a new view object if possible; otherwise, it will
        be a copy.
    """
    block_shape = _instance().compute_block_shape(shape, a.dtype)
    return a.reshape(shape, block_shape=block_shape)


def expand_dims(a: BlockArray, axis):
    """Expand the shape of an array.

    This docstring was copied from numpy.expand_dims.

    Some inconsistencies with the NumS version may exist.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Parameters
    ----------
    a : BlockArray
        Input array.
    axis : int or tuple of ints
        Position in the expanded axes where the new axis (or axes) is placed.

    Returns
    -------
    result : BlockArray
        View of `a` with the number of dimensions increased.

    See Also
    --------
    squeeze : The inverse operation, removing singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones
    atleast_1d, atleast_2d, atleast_3d

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.array([1, 2])  # doctest: +SKIP
    >>> x.shape  # doctest: +SKIP
    (2,)

    The following is equivalent to ``x[nps.newaxis, :]`` or ``x[nps.newaxis]``:

    >>> y = nps.expand_dims(x, axis=0)  # doctest: +SKIP
    >>> y.get()  # doctest: +SKIP
    array([[1, 2]])
    >>> y.shape  # doctest: +SKIP
    (1, 2)

    The following is equivalent to ``x[:, nps.newaxis]``:

    >>> y = nps.expand_dims(x, axis=1)  # doctest: +SKIP
    >>> y.get()  # doctest: +SKIP
    array([[1],
           [2]])
    >>> y.shape  # doctest: +SKIP
    (2, 1)

    ``axis`` may also be a tuple:

    >>> y = nps.expand_dims(x, axis=(0, 1))  # doctest: +SKIP
    >>> y.get()  # doctest: +SKIP
    array([[[1, 2]]])

    >>> y = nps.expand_dims(x, axis=(2, 0))  # doctest: +SKIP
    >>> y.get()  # doctest: +SKIP
    array([[[1],
            [2]]])
    """
    return a.expand_dims(axis)


def squeeze(a: BlockArray, axis=None):
    """Remove single-dimensional entries from the shape of an array.

    This docstring was copied from numpy.squeeze.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input data.

    Returns
    -------
    squeezed : BlockArray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`.

    See Also
    --------
    expand_dims : The inverse operation, adding singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Notes
    -----
    axis not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.array([[[0], [1], [2]]])  # doctest: +SKIP
    >>> x.shape  # doctest: +SKIP
    (1, 3, 1)
    >>> nps.squeeze(x).shape  # doctest: +SKIP
    (3,)
    >>> x = nps.array([[1234]])  # doctest: +SKIP
    >>> x.shape  # doctest: +SKIP
    (1, 1)
    >>> nps.squeeze(x).get()  # doctest: +SKIP
    array(1234)  # 0d array
    >>> nps.squeeze(x).shape  # doctest: +SKIP
    ()
    >>> nps.squeeze(x)[()].get()  # doctest: +SKIP
    array(1234)
    """
    assert axis is None, "axis not supported."
    return a.squeeze()


def swapaxes(a: BlockArray, axis1: int, axis2: int):
    """Interchange two axes of an array.

    This docstring was copied from numpy.swapaxes.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : BlockArray
        For NumPy >= 1.10.0, if `a` is an ndarray, then a view of `a` is
        returned; otherwise a new array is created. For earlier NumPy
        versions a view of `a` is returned only if the order of the
        axes is changed, otherwise the input array is returned.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.array([[1,2,3]])  # doctest: +SKIP
    >>> nps.swapaxes(x,0,1).get()  # doctest: +SKIP
    array([[1],
           [2],
           [3]])

    >>> x = nps.array([[[0,1],[2,3]],[[4,5],[6,7]]])  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])

    >>> nps.swapaxes(x,0,2).get()  # doctest: +SKIP
    array([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])
    """
    return a.swapaxes(axis1, axis2)


def transpose(a: BlockArray, axes=None):
    """Reverse or permute the axes of an array; returns the modified array.

    This docstring was copied from numpy.transpose.

    Some inconsistencies with the NumS version may exist.

    For an array a with two axes, transpose(a) gives the matrix transpose.

    Parameters
    ----------
    a : BlockArray
        Input array.

    Returns
    -------
    p : BlockArray
        `a` with its axes permuted.  A view is returned whenever
        possible.

    Notes
    -----
    Transposing a 1-D array returns an unchanged view of the original array.

    axes not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(4).reshape((2,2))  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array([[0, 1],
           [2, 3]])

    >>> nps.transpose(x).get()  # doctest: +SKIP
    array([[0, 2],
           [1, 3]])
    """
    assert axes is None, "axes not supported."
    return a.T


############################################
# Misc
############################################


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
        as possible. (Note that this function and :meth:`ndarray.copy` are very
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


############################################
# Reduction Ops
############################################


@derived_from(np)
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
        `ndarray`, however any non-default value will be. If the
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
        `ndarray`, however any non-default value will be.  If the
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
        `ndarray`, however any non-default value will be.  If the
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
        `ndarray`, however any non-default value will be.  If the
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


@derived_from(np)
def where(condition: BlockArray, x: BlockArray = None, y: BlockArray = None):
    return _instance().where(condition, x, y)


@derived_from(np)
def all(a: BlockArray, axis=None, out=None, keepdims=False):
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("all", a, axis=axis, keepdims=keepdims)


@derived_from(np)
def alltrue(a: BlockArray, axis=None, out=None, dtype=None, keepdims=False):
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("alltrue", a, axis=axis, keepdims=keepdims, dtype=dtype)


@derived_from(np)
def any(a: BlockArray, axis=None, out=None, keepdims=False):
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("any", a, axis=axis, keepdims=keepdims)


def average(
    a: BlockArray,
    axis: Union[None, int] = None,
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


@derived_from(np)
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
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

    Returns
    -------
    nanmax : BlockArray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, an ndarray scalar is
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
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.

    Returns
    -------
    nanmin : BlockArray
        An array with the same shape as `a`, with the specified axis
        removed.  If `a` is a 0-d array, or if axis is None, an ndarray
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
        of sub-classes of `ndarray`.  If the sub-classes methods
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

    For this function to work on sub-classes of ndarray, they must define
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


############################################
# Utility Ops
############################################


@derived_from(np)
def array_equal(a: BlockArray, b: BlockArray, equal_nan=False) -> BlockArray:
    """True if two arrays have the same shape and elements, False otherwise.

    This docstring was copied from numpy.array_equal.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a1, a2 : BlockArray
        Input arrays.
    equal_nan : bool
        Whether to compare NaN's as equal. If the dtype of a1 and a2 is
        complex, values will be considered equal if either the real or the
        imaginary component of a given value is ``nan``.

    Returns
    -------
    b : bool
        Returns True if the arrays are equal.

    See Also
    --------
    allclose: Returns True if two arrays are element-wise equal within a
              tolerance.
    array_equiv: Returns True if input arrays are shape consistent and all
                 elements equal.

    Notes
    -----
    equal_nan=True not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.array_equal(nps.array([1, 2]), nps.array([1, 2])).get()  # doctest: +SKIP
    array(True)
    >>> a = nps.array([1, nps.nan])  # doctest: +SKIP
    >>> nps.array_equal(a, a).get()  # doctest: +SKIP
    array(False)
    """
    if equal_nan is not False:
        raise NotImplementedError("equal_nan=True not supported.")
    return _instance().array_equal(a, b)


def array_equiv(a: BlockArray, b: BlockArray) -> BlockArray:
    """Returns True if input arrays are shape consistent and all elements equal.

    This docstring was copied from numpy.array_equiv.

    Some inconsistencies with the NumS version may exist.

    Shape consistent means they are either the same shape, or one input array
    can be broadcasted to create the same shape as the other one.

    Parameters
    ----------
    a1, a2 : BlockArray
        Input arrays.

    Returns
    -------
    out : BlockArray
        True if equivalent, False otherwise.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.array_equiv(nps.array([1, 2]), nps.array([1, 2])).get()  # doctest: +SKIP
    array(True)
    >>> nps.array_equiv(nps.array([1, 2]), nps.array([1, 3])).get()  # doctest: +SKIP
    array(False)
    """
    return _instance().array_equiv(a, b)


def allclose(
    a: BlockArray, b: BlockArray, rtol=1.0e-5, atol=1.0e-8, equal_nan=False
) -> BlockArray:
    """Returns True if two arrays are element-wise equal within a tolerance.

    This docstring was copied from numpy.allclose.

    Some inconsistencies with the NumS version may exist.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    Parameters
    ----------
    a, b : BlockArray
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    isclose, all, any, equal

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    ``allclose(a, b)`` might be different from ``allclose(b, a)`` in
    some rare cases.

    The comparison of `a` and `b` uses standard broadcasting, which
    means that `a` and `b` need not have the same shape in order for
    ``allclose(a, b)`` to evaluate to True.  The same is true for
    `equal` but not `array_equal`.

    equal_nan=True not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.allclose(nps.array([1e10,1e-7]), nps.array([1.00001e10,1e-8])).get()  # doctest: +SKIP
    array(False)
    >>> nps.allclose(nps.array([1e10,1e-8]), nps.array([1.00001e10,1e-9])).get()  # doctest: +SKIP
    array(True)
    >>> nps.allclose(nps.array([1e10,1e-8]), nps.array([1.0001e10,1e-9])).get()  # doctest: +SKIP
    array(False)
    >>> nps.allclose(nps.array([1.0, nps.nan]), nps.array([1.0, nps.nan])).get()  # doctest: +SKIP
    array(False)
    """
    if equal_nan is not False:
        raise NotImplementedError("equal_nan=True not supported.")
    return _instance().allclose(a, b, rtol, atol)


############################################
# Generated Ops (Unary, Binary)
############################################


@derived_from(np)
def abs(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="abs",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def absolute(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="absolute",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def arccos(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="arccos",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def arccosh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="arccosh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def arcsin(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="arcsin",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def arcsinh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="arcsinh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def arctan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="arctan",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def arctanh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="arctanh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def bitwise_not(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_uop(
        op_name="bitwise_not",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def cbrt(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="cbrt",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def ceil(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="ceil",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def conj(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="conj",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def conjugate(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_uop(
        op_name="conjugate",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def cos(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="cos",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def cosh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="cosh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def deg2rad(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="deg2rad",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def degrees(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="degrees",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def exp(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="exp",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def exp2(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="exp2",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def expm1(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="expm1",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def fabs(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="fabs",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def floor(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="floor",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def invert(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="invert",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def isfinite(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="isfinite",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def isinf(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="isinf",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def isnan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="isnan",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def log(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="log",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def log10(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="log10",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def log1p(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="log1p",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def log2(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="log2",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def logical_not(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_uop(
        op_name="logical_not",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def negative(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="negative",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def positive(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="positive",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def rad2deg(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="rad2deg",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def radians(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="radians",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def reciprocal(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_uop(
        op_name="reciprocal",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def rint(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="rint",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def sign(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="sign",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def signbit(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="signbit",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def sin(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="sin",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def sinh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="sinh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def spacing(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="spacing",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def sqrt(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="sqrt",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def square(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="square",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def tan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="tan",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def tanh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="tanh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def trunc(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(
        op_name="trunc",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def add(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="add",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def arctan2(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="arctan2",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def bitwise_and(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="bitwise_and",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def bitwise_or(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="bitwise_or",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def bitwise_xor(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="bitwise_xor",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def copysign(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="copysign",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def divide(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="divide",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def equal(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="equal",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def float_power(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="float_power",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def floor_divide(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="floor_divide",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def fmax(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="fmax",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def fmin(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="fmin",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def fmod(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="fmod",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def gcd(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="gcd",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def greater(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="greater",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def greater_equal(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="greater_equal",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def heaviside(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="heaviside",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def hypot(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="hypot",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def lcm(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="lcm",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def ldexp(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="ldexp",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def left_shift(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="left_shift",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def less(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="less",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def less_equal(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="less_equal",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def logaddexp(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="logaddexp",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def logaddexp2(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="logaddexp2",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def logical_and(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="logical_and",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def logical_or(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="logical_or",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def logical_xor(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="logical_xor",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def maximum(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="maximum",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def minimum(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="minimum",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def mod(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="mod",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def multiply(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="multiply",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def nextafter(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="nextafter",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def not_equal(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="not_equal",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def power(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="power",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def remainder(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="remainder",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def right_shift(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="right_shift",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def subtract(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="subtract",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def true_divide(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    return _instance().map_bop(
        op_name="true_divide",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )
