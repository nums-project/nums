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
        Input array.  If it is not already an BlockArray, a conversion is
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
        For NumPy >= 1.10.0, if `a` is an BlockArray, then a view of `a` is
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


def abs(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Calculate the absolute value element-wise.

    ``nps.abs`` is a shorthand for this function.

    This docstring was copied from numpy.abs.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    absolute : BlockArray
        A BlockArray containing the absolute value of
        each element in `x`.  For complex input, ``a + ib``, the
        absolute value is :math:`\sqrt{ a^2 + b^2 }`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.array([-1.2, 1.2])  # doctest: +SKIP
    >>> nps.absolute(x).get()  # doctest: +SKIP
    array([ 1.2,  1.2])
    >>> nps.absolute(nps.array([1.2 + 1j])).get()  # doctest: +SKIP
    array([1.56204994])
    """
    return _instance().map_uop(
        op_name="abs",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def absolute(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Calculate the absolute value element-wise.

    ``nps.abs`` is a shorthand for this function.

    This docstring was copied from numpy.absolute.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    absolute : BlockArray
        A BlockArray containing the absolute value of
        each element in `x`.  For complex input, ``a + ib``, the
        absolute value is :math:`\sqrt{ a^2 + b^2 }`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.array([-1.2, 1.2])  # doctest: +SKIP
    >>> nps.absolute(x).get()  # doctest: +SKIP
    array([ 1.2,  1.2])
    >>> nps.absolute(nps.array(1.2 + 1j)).get()  # doctest: +SKIP
    array(1.56204994)
    """
    return _instance().map_uop(
        op_name="absolute",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def arccos(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if ``y = cos(x)``, then ``x = arccos(y)``.

    This docstring was copied from numpy.arccos.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        `x`-coordinate on the unit circle.
        For real arguments, the domain is [-1, 1].
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    angle : BlockArray
        The angle of the ray intersecting the unit circle at the given
        `x`-coordinate in radians [0, pi].

    See Also
    --------
    cos, arctan, arcsin, emath.arccos

    Notes
    -----
    `arccos` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cos(z) = x`. The convention is to return
    the angle `z` whose real part lies in `[0, pi]`.

    For real-valued input data types, `arccos` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccos` is a complex analytic function that
    has branch cuts `[-inf, -1]` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse `cos` is also known as `acos` or cos^-1.

    References
    ----------
    M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
    10th printing, 1964, pp. 79. http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    We expect the arccos of 1 to be 0, and of -1 to be pi:

    >>> nps.arccos(nps.array([1, -1])).get()  # doctest: +SKIP
    array([ 0.        ,  3.14159265])
    """
    return _instance().map_uop(
        op_name="arccos",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def arccosh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Inverse hyperbolic cosine, element-wise.

    This docstring was copied from numpy.arccosh.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    arccosh : BlockArray
        Array of the same shape as `x`.

    See Also
    --------

    cosh, arcsinh, sinh, arctanh, tanh

    Notes
    -----
    `arccosh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cosh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi, pi]` and the real part in
    ``[0, inf]``.

    For real-valued input data types, `arccosh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccosh` is a complex analytical function that
    has a branch cut `[-inf, 1]` and is continuous from above on it.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arccosh

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.arccosh(nps.array([nps.e, 10.0])).get()  # doctest: +SKIP
    array([ 1.65745445,  2.99322285])
    >>> nps.arccosh(nps.array(1)).get()  # doctest: +SKIP
    array(0.0)
    """
    return _instance().map_uop(
        op_name="arccosh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def arcsin(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Inverse sine, element-wise.

    This docstring was copied from numpy.arcsin.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        `y`-coordinate on the unit circle.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    angle : BlockArray
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.

    See Also
    --------
    sin, cos, arccos, tan, arctan, arctan2, emath.arcsin

    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arcsin` is a complex analytic function that
    has, by convention, the branch cuts [-inf, -1] and [1, inf]  and is
    continuous from above on the former and from below on the latter.

    The inverse sine is also known as `asin` or sin^{-1}.

    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.arcsin(nps.array(1)).get()     # pi/2  # doctest: +SKIP
    array(1.57079633)
    >>> nps.arcsin(nps.array(-1)).get()    # -pi/2  # doctest: +SKIP
    array(-1.57079633)
    >>> nps.arcsin(nps.array(0)).get()  # doctest: +SKIP
    array(0.)
    """
    return _instance().map_uop(
        op_name="arcsin",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def arcsinh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Inverse hyperbolic sine element-wise.

    This docstring was copied from numpy.arcsinh.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Array of the same shape as `x`.

    Notes
    -----
    `arcsinh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sinh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arcsinh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    returns ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccos` is a complex analytical function that
    has branch cuts `[1j, infj]` and `[-1j, -infj]` and is continuous from
    the right on the former and from the left on the latter.

    The inverse hyperbolic sine is also known as `asinh` or ``sinh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arcsinh

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.arcsinh(nps.array([nps.e, 10.0])).get()  # doctest: +SKIP
    array([ 1.72538256,  2.99822295])
    """
    return _instance().map_uop(
        op_name="arcsinh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def arctan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Trigonometric inverse tangent, element-wise.

    The inverse of tan, so that if ``y = tan(x)`` then ``x = arctan(y)``.

    This docstring was copied from numpy.arctan.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Out has the same shape as `x`.  Its real part is in
        ``[-pi/2, pi/2]`` (``arctan(+/-inf)`` returns ``+/-pi/2``).

    See Also
    --------
    arctan2 : The "four quadrant" arctan of the angle formed by (`x`, `y`)
        and the positive `x`-axis.
    angle : Argument of complex values.

    Notes
    -----
    `arctan` is a multi-valued function: for each `x` there are infinitely
    many numbers `z` such that tan(`z`) = `x`.  The convention is to return
    the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, `arctan` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctan` is a complex analytic function that
    has [`1j, infj`] and [`-1j, -infj`] as branch cuts, and is continuous
    from the left on the former and from the right on the latter.

    The inverse tangent is also known as `atan` or tan^{-1}.

    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79.
    http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    We expect the arctan of 0 to be 0, and of 1 to be pi/4:

    >>> nps.arctan(nps.array([0, 1])).get()  # doctest: +SKIP
    array([ 0.        ,  0.78539816])

    >>> nps.pi/4  # doctest: +SKIP
    0.78539816339744828
    """
    return _instance().map_uop(
        op_name="arctan",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def arctanh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Inverse hyperbolic tangent element-wise.

    This docstring was copied from numpy.arctanh.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Array of the same shape as `x`.

    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tanh(z) = x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctanh` is a complex analytical function
    that has branch cuts `[-1, -inf]` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse hyperbolic tangent is also known as `atanh` or ``tanh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arctanh

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.arctanh(nps.array([0, -0.5])).get()  # doctest: +SKIP
    array([ 0.        , -0.54930614])
    """
    return _instance().map_uop(
        op_name="arctanh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def bitwise_not(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    This docstring was copied from numpy.bitwise_not.

    Some inconsistencies with the NumS version may exist.

    Computes the bit-wise NOT of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``~``.

    For signed integer inputs, the two's complement is returned.  In a
    two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit
    two's-complement system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    x : BlockArray
        Only integer and boolean types are handled.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Result.

    See Also
    --------
    bitwise_and, bitwise_or, bitwise_xor
    logical_not
    binary_repr :
        Return the binary representation of the input number as a string.

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        https://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    We've seen that 13 is represented by ``00001101``.
    The invert or bit-wise NOT of 13 is then:

    >>> x = nps.invert(nps.array(13, dtype=nps.uint8))  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array(242, dtype=uint8)

    The result depends on the bit-width:

    >>> x = nps.invert(nps.array(13, dtype=nps.uint16)).get()  # doctest: +SKIP
    >>> x  # doctest: +SKIP
    65522
    >>> nps.binary_repr(x, width=16).get()  # doctest: +SKIP
    '1111111111110010'

    When using signed integer types the result is the two's complement of
    the result for the unsigned type:

    >>> nps.invert(nps.array([13], dtype=nps.int8)).get()  # doctest: +SKIP
    array([-14], dtype=int8)
    >>> nps.binary_repr(-14, width=8).get()  # doctest: +SKIP
    '11110010'

    Booleans are accepted as well:

    >>> nps.invert(nps.array([True, False])).get()  # doctest: +SKIP
    array([False,  True])
    """
    return _instance().map_uop(
        op_name="bitwise_not",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def cbrt(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the cube-root of an array, element-wise.

    This docstring was copied from numpy.cbrt.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        The values whose cube-roots are required.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        An array of the same shape as `x`, containing the cube
        cube-root of each element in `x`.
        If `out` was provided, `y` is a reference to it.


    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.cbrt(nps.array([1,8,27])).get()  # doctest: +SKIP
    array([ 1.,  2.,  3.])
    """
    return _instance().map_uop(
        op_name="cbrt",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def ceil(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the ceiling of the input, element-wise.

    This docstring was copied from numpy.ceil.

    Some inconsistencies with the NumS version may exist.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\lceil x \rceil`.

    Parameters
    ----------
    x : BlockArray
        Input data.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray or scalar
        The ceiling of each element in `x`, with `float` dtype.

    See Also
    --------
    floor, trunc, rint

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])  # doctest: +SKIP
    >>> nps.ceil(a).get()  # doctest: +SKIP
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])
    """
    return _instance().map_uop(
        op_name="ceil",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def conj(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the complex conjugate, element-wise.

    This docstring was copied from numpy.conj.

    Some inconsistencies with the NumS version may exist.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    Parameters
    ----------
    x : BlockArray
        Input value.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The complex conjugate of `x`, with same dtype as `y`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.conjugate(nps.array(1+2j)).get()  # doctest: +SKIP
    array(1.-2.j)
    """
    return _instance().map_uop(
        op_name="conj",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def conjugate(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return the complex conjugate, element-wise.

    This docstring was copied from numpy.conjugate.

    Some inconsistencies with the NumS version may exist.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    Parameters
    ----------
    x : BlockArray
        Input value.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The complex conjugate of `x`, with same dtype as `y`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.conjugate(nps.array(1+2j)).get()  # doctest: +SKIP
    array(1.-2.j)
    """
    return _instance().map_uop(
        op_name="conjugate",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def cos(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Cosine element-wise.

    This docstring was copied from numpy.cos.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array in radians.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The corresponding cosine values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.cos(nps.array([0, nps.pi/2, nps.pi])).get()  # doctest: +SKIP
    array([  1.00000000e+00,   6.12303177e-17,  -1.00000000e+00])
    """
    return _instance().map_uop(
        op_name="cos",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def cosh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Hyperbolic cosine, element-wise.

    This docstring was copied from numpy.cosh.

    Some inconsistencies with the NumS version may exist.

    Equivalent to ``1/2 * (nps.exp(x) + nps.exp(-x))`` and ``nps.cos(1j*x).get()``.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Output array of same shape as `x`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.cosh(nps.array(0)).get()  # doctest: +SKIP
    array(1.)
    """
    return _instance().map_uop(
        op_name="cosh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def deg2rad(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Convert angles from degrees to radians.

    This docstring was copied from numpy.deg2rad.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Angles in degrees.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The corresponding angle in radians.

    See Also
    --------
    rad2deg : Convert angles from radians to degrees.
    unwrap : Remove large jumps in angle by wrapping.

    Notes
    -----
    ``deg2rad(x)`` is ``x * pi / 180``.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.deg2rad(nps.array(180)).get()  # doctest: +SKIP
    array(3.14159265)
    """
    return _instance().map_uop(
        op_name="deg2rad",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def degrees(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Convert angles from radians to degrees.

    This docstring was copied from numpy.degrees.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array in radians.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray of floats
        The corresponding degree values; if `out` was supplied this is a
        reference to it.

    See Also
    --------
    rad2deg : equivalent function

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    Convert a radian array to degrees

    >>> rad = nps.arange(12.) *nps.pi/6  # doctest: +SKIP
    >>> nps.degrees(rad).get()  # doctest: +SKIP
    array([   0.,   30.,   60.,   90.,  120.,  150.,  180.,  210.,  240.,
            270.,  300.,  330.])

    >>> out = nps.zeros((rad.shape))  # doctest: +SKIP
    >>> r = nps.degrees(rad, out)  # doctest: +SKIP
    >>> nps.all(r == out).get()  # doctest: +SKIP
    array(True)
    """
    return _instance().map_uop(
        op_name="degrees",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def exp(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Calculate the exponential of all elements in the input array.

    This docstring was copied from numpy.exp.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Output array, element-wise exponential of `x`.

    See Also
    --------
    expm1 : Calculate ``exp(x) - 1`` for all elements in the array.
    exp2  : Calculate ``2**x`` for all elements in the array.

    Notes
    -----
    The irrational number ``e`` is also known as Euler's number.  It is
    approximately 2.718281, and is the base of the natural logarithm,
    ``ln`` (this means that, if :math:`x = \ln y = \log_e y`,
    then :math:`e^x = y`. For real input, ``exp(x)`` is always positive.

    For complex arguments, ``x = a + ib``, we can write
    :math:`e^x = e^a e^{ib}`.  The first term, :math:`e^a`, is already
    known (it is the real argument, described above).  The second term,
    :math:`e^{ib}`, is :math:`\cos b + i \sin b`, a function with
    magnitude 1 and a periodic phase.

    References
    ----------
    .. [1] Wikipedia, "Exponential function",
           https://en.wikipedia.org/wiki/Exponential_function
    .. [2] M. Abramovitz and I. A. Stegun, "Handbook of Mathematical Functions
           with Formulas, Graphs, and Mathematical Tables," Dover, 1964, p. 69,
           http://www.math.sfu.ca/~cbm/aands/page_69.htm
    """
    return _instance().map_uop(
        op_name="exp",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def exp2(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Calculate `2**p` for all `p` in the input array.

    This docstring was copied from numpy.exp2.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Element-wise 2 to the power `x`.

    See Also
    --------
    power

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.exp2(nps.array([2, 3])).get()  # doctest: +SKIP
    array([ 4.,  8.])
    """
    return _instance().map_uop(
        op_name="exp2",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def expm1(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Calculate ``exp(x) - 1`` for all elements in the array.

    This docstring was copied from numpy.expm1.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Element-wise exponential minus one: ``out = exp(x) - 1``.

    See Also
    --------
    log1p : ``log(1 + x)``, the inverse of expm1.

    Notes
    -----
    This function provides greater precision than ``exp(x) - 1``
    for small values of ``x``.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    The true value of ``exp(1e-10) - 1`` is ``1.00000000005e-10`` to
    about 32 significant digits. This example shows the superiority of
    expm1 in this case.

    >>> nps.expm1(nps.array(1e-10)).get()  # doctest: +SKIP
    array(1.e-10)
    >>> nps.exp(nps.array(1e-10)).get() - 1  # doctest: +SKIP
    1.000000082740371e-10
    """
    return _instance().map_uop(
        op_name="expm1",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def fabs(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Compute the absolute values element-wise.

    This docstring was copied from numpy.fabs.

    Some inconsistencies with the NumS version may exist.

    This function returns the absolute values (positive magnitude) of the
    data in `x`. Complex values are not handled, use `absolute` to find the
    absolute values of complex data.

    Parameters
    ----------
    x : BlockArray
        The array of numbers for which the absolute values are required. If
        `x` is a scalar, the result `y` will also be a scalar.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray or scalar
        The absolute values of `x`, the returned values are always floats.

    See Also
    --------
    absolute : Absolute values including `complex` types.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.fabs(nps.array(-1)).get()  # doctest: +SKIP
    array(1.)
    >>> nps.fabs(nps.array([-1.2, 1.2])).get()  # doctest: +SKIP
    array([ 1.2,  1.2])
    """
    return _instance().map_uop(
        op_name="fabs",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def floor(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the floor of the input, element-wise.

    This docstring was copied from numpy.floor.

    Some inconsistencies with the NumS version may exist.

    The floor of the scalar `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    x : BlockArray
        Input data.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray or scalar
        The floor of each element in `x`.

    See Also
    --------
    ceil, trunc, rint

    Notes
    -----
    Some spreadsheet programs calculate the "floor-towards-zero", in other
    words ``floor(-2.5) == -2``.  NumPy instead uses the definition of
    `floor` where `floor(-2.5) == -3`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])  # doctest: +SKIP
    >>> nps.floor(a).get()  # doctest: +SKIP
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])
    """
    return _instance().map_uop(
        op_name="floor",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def invert(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    This docstring was copied from numpy.invert.

    Some inconsistencies with the NumS version may exist.

    Computes the bit-wise NOT of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``~``.

    For signed integer inputs, the two's complement is returned.  In a
    two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit
    two's-complement system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    x : BlockArray
        Only integer and boolean types are handled.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Result.

    See Also
    --------
    bitwise_and, bitwise_or, bitwise_xor
    logical_not
    binary_repr :
        Return the binary representation of the input number as a string.

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        https://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    We'vThe doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    e seen that 13 is represented by ``00001101``.
    The invert or bit-wise NOT of 13 is then:

    >>> x = nps.invert(nps.array(13, dtype=nps.uint8))  # doctest: +SKIP
    >>> x.get()  # doctest: +SKIP
    array(242, dtype=uint8)

    When using signed integer types the result is the two's complement of
    the result for the unsigned type:

    >>> nps.invert(nps.array([13], dtype=nps.int8)).get()  # doctest: +SKIP
    array([-14], dtype=int8)

    Booleans are accepted as well:

    >>> nps.invert(nps.array([True, False])).get()  # doctest: +SKIP
    array([False,  True])
    """
    return _instance().map_uop(
        op_name="invert",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def isfinite(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Test element-wise for finiteness (not infinity or not Not a Number).

    This docstring was copied from numpy.isfinite.

    Some inconsistencies with the NumS version may exist.

    The result is returned as a boolean array.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray, bool
        True where ``x`` is not positive infinity, negative infinity,
        or NaN; false otherwise.

    See Also
    --------
    isinf, isneginf, isposinf, isnan

    Notes
    -----
    Not a Number, positive infinity and negative infinity are considered
    to be non-finite.

    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.  Errors result if the
    second argument is also supplied when `x` is a scalar input, or if
    first and second arguments have different shapes.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.isfinite(nps.array(1)).get()  # doctest: +SKIP
    array(True)
    >>> nps.isfinite(nps.array(0)).get()  # doctest: +SKIP
    array(True)
    >>> nps.isfinite(nps.array(nps.nan)).get()  # doctest: +SKIP
    array(False)
    >>> nps.isfinite(nps.array(nps.inf)).get()  # doctest: +SKIP
    array(False)
    >>> nps.isfinite(nps.array(nps.NINF)).get()  # doctest: +SKIP
    array(False)

    >>> x = nps.array([-nps.inf, 0., nps.inf])  # doctest: +SKIP
    >>> y = nps.array([2, 2, 2])  # doctest: +SKIP
    >>> nps.isfinite(x, y).get()  # doctest: +SKIP
    array([0, 1, 0])
    >>> y.get()  # doctest: +SKIP
    array([0, 1, 0])
    """
    return _instance().map_uop(
        op_name="isfinite",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def isinf(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Test element-wise for positive or negative infinity.

    This docstring was copied from numpy.isinf.

    Some inconsistencies with the NumS version may exist.

    Returns a boolean array of the same shape as `x`, True where ``x ==
    +/-inf``, otherwise False.

    Parameters
    ----------
    x : BlockArray
        Input values
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : boolean BlockArray
        True where ``x`` is positive or negative infinity, false otherwise.

    See Also
    --------
    isneginf, isposinf, isnan, isfinite

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is supplied when the first
    argument is a scalar, or if the first and second arguments have
    different shapes.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.isinf(nps.array(nps.inf)).get()  # doctest: +SKIP
    array(True)
    >>> nps.isinf(nps.array(nps.nan)).get()  # doctest: +SKIP
    array(False)
    >>> nps.isinf(nps.array(nps.NINF)).get()  # doctest: +SKIP
    array(True)
    >>> nps.isinf(nps.array([nps.inf, -nps.inf, 1.0, nps.nan])).get()  # doctest: +SKIP
    array([ True,  True, False, False])

    >>> x = nps.array([-nps.inf, 0., nps.inf])  # doctest: +SKIP
    >>> y = nps.array([2, 2, 2])  # doctest: +SKIP
    >>> nps.isinf(x, y).get()  # doctest: +SKIP
    array([1, 0, 1])
    >>> y.get()  # doctest: +SKIP
    array([1, 0, 1])
    """
    return _instance().map_uop(
        op_name="isinf",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def isnan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Test element-wise for NaN and return result as a boolean array.

    This docstring was copied from numpy.isnan.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray or bool
        True where ``x`` is NaN, false otherwise.

    See Also
    --------
    isinf, isneginf, isposinf, isfinite, isnat

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.  # doctest: +SKIP

    >>> nps.isnan(nps.array(nps.nan)).get()
    array(True)
    >>> nps.isnan(nps.array(nps.inf)).get()  # doctest: +SKIP
    array(False)
    """
    return _instance().map_uop(
        op_name="isnan",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def log(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Natural logarithm, element-wise.

    This docstring was copied from numpy.log.

    Some inconsistencies with the NumS version may exist.

    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base
    `e`.

    Parameters
    ----------
    x : BlockArray
        Input value.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The natural logarithm of `x`, element-wise.

    See Also
    --------
    log10, log2, log1p

    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log` always returns real output. For
    each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it. `log`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.log(nps.array([1, nps.e, nps.e**2, 0])).get()  # doctest: +SKIP
    array([  0.,   1.,   2., -Inf])
    """
    return _instance().map_uop(
        op_name="log",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


@derived_from(np)
def log10(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the base 10 logarithm of the input array, element-wise.

    This docstring was copied from numpy.log10.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The logarithm to the base 10 of `x`, element-wise. NaNs are
        returned where x is negative.

    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `10**z = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log10` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log10` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it.
    `log10` handles the floating-point negative zero as an infinitesimal
    negative number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.log10(nps.array([1e-15, -3.])).get()  # doctest: +SKIP
    array([-15.,  nan])
    """
    return _instance().map_uop(
        op_name="log10",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def log1p(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the natural logarithm of one plus the input array, element-wise.

    This docstring was copied from numpy.log1p.

    Some inconsistencies with the NumS version may exist.

    Calculates ``log(1 + x)``.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        Natural logarithm of `1 + x`, element-wise.

    See Also
    --------
    expm1 : ``exp(x) - 1``, the inverse of `log1p`.

    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.

    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = 1 + x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log1p` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log1p` is a complex analytical function that
    has a branch cut `[-inf, -1]` and is continuous from above on it.
    `log1p` handles the floating-point negative zero as an infinitesimal
    negative number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.log1p(nps.array(1e-99)).get()  # doctest: +SKIP
    array(1.e-99)
    >>> nps.log(nps.array(1 + 1e-99)).get()  # doctest: +SKIP
    array(0.)
    """
    return _instance().map_uop(
        op_name="log1p",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def log2(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Base-2 logarithm of `x`.

    This docstring was copied from numpy.log2.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        Base-2 logarithm of `x`.

    See Also
    --------
    log, log10, log1p

    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `2**z = x`. The convention is to return the `z`
    whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log2` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log2` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it. `log2`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.array([0, 1, 2, 2**4])  # doctest: +SKIP
    >>> nps.log2(x).get()  # doctest: +SKIP
    array([-Inf,   0.,   1.,   4.])

    >>> xi = nps.array([0+1.j, 1, 2+0.j, 4.j])  # doctest: +SKIP
    >>> nps.log2(xi).get()  # doctest: +SKIP
    array([ 0.+2.26618007j,  0.+0.j        ,  1.+0.j        ,  2.+2.26618007j])
    """
    return _instance().map_uop(
        op_name="log2",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def logical_not(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Compute the truth value of NOT x element-wise.

    This docstring was copied from numpy.logical_not.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Logical NOT is applied to the elements of `x`.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray of bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.

    See Also
    --------
    logical_and, logical_or, logical_xor

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.logical_not(nps.array(3)).get()  # doctest: +SKIP
    array(False)
    >>> nps.logical_not(nps.array([True, False, 0, 1])).get()  # doctest: +SKIP
    array([False,  True,  True, False])

    >>> x = nps.arange(5)  # doctest: +SKIP
    >>> nps.logical_not(x<3).get()  # doctest: +SKIP
    array([False, False, False,  True,  True])
    """
    return _instance().map_uop(
        op_name="logical_not",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def negative(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Numerical negative, element-wise.

    This docstring was copied from numpy.negative.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray or scalar
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray or scalar
        Returned array or scalar: `y = -x`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.negative(nps.array([1.,-1.])).get()  # doctest: +SKIP
    array([-1.,  1.])
    """
    return _instance().map_uop(
        op_name="negative",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def positive(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Numerical positive, element-wise.

    This docstring was copied from numpy.positive.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.

    Returns
    -------
    y : BlockArray
        Returned array: `y = +x`.

    Notes
    -----
    Equivalent to `x.copy()`, but only defined for types that support
    arithmetic.
    """
    return _instance().map_uop(
        op_name="positive",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def rad2deg(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Convert angles from radians to degrees.

    This docstring was copied from numpy.rad2deg.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Angle in radians.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The corresponding angle in degrees.

    See Also
    --------
    deg2rad : Convert angles from degrees to radians.
    unwrap : Remove large jumps in angle by wrapping.

    Notes
    -----
    rad2deg(x) is ``180 * x / pi``.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.rad2deg(nps.array(nps.pi/2)).get()  # doctest: +SKIP
    array(90.)
    """
    return _instance().map_uop(
        op_name="rad2deg",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def radians(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Convert angles from degrees to radians.

    This docstring was copied from numpy.radians.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array in degrees.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The corresponding radian values.

    See Also
    --------
    deg2rad : equivalent function

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    Convert a degree array to radians

    >>> deg = nps.arange(12.) * 30.  # doctest: +SKIP
    >>> nps.radians(deg).get()  # doctest: +SKIP
    array([ 0.        ,  0.52359878,  1.04719755,  1.57079633,  2.0943951 ,
            2.61799388,  3.14159265,  3.66519143,  4.1887902 ,  4.71238898,
            5.23598776,  5.75958653])

    >>> out = nps.zeros((deg.shape))  # doctest: +SKIP
    >>> ret = nps.radians(deg, out)  # doctest: +SKIP
    >>> ret is out  # doctest: +SKIP
    True
    """
    return _instance().map_uop(
        op_name="radians",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def reciprocal(
    x: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return the reciprocal of the argument, element-wise.

    This docstring was copied from numpy.reciprocal.

    Some inconsistencies with the NumS version may exist.

    Calculates ``1/x``.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        Return array.

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.reciprocal(nps.array(2.)).get()  # doctest: +SKIP
    array(0.5)
    >>> nps.reciprocal(nps.array([1, 2., 3.33])).get()  # doctest: +SKIP
    array([ 1.       ,  0.5      ,  0.3003003])
    """
    return _instance().map_uop(
        op_name="reciprocal",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def rint(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Round elements of the array to the nearest integer.

    This docstring was copied from numpy.rint.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Output array is same shape and type as `x`.

    See Also
    --------
    ceil, floor, trunc

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])  # doctest: +SKIP
    >>> nps.rint(a).get()  # doctest: +SKIP
    array([-2., -2., -0.,  0.,  2.,  2.,  2.])
    """
    return _instance().map_uop(
        op_name="rint",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def sign(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Returns an element-wise indication of the sign of a number.

    This docstring was copied from numpy.sign.

    Some inconsistencies with the NumS version may exist.

    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.  nan
    is returned for nan inputs.

    For complex inputs, the `sign` function returns
    ``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``.

    complex(nan, 0) is returned for complex nan inputs.

    Parameters
    ----------
    x : BlockArray
        Input values.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The sign of `x`.

    Notes
    -----
    There is more than one definition of sign in common use for complex
    numbers.  The definition used here is equivalent to :math:`x/\sqrt{x*x}`
    which is different from a common alternative, :math:`x/|x|`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.sign(nps.array([-5., 4.5])).get()  # doctest: +SKIP
    array([-1.,  1.])
    >>> nps.sign(nps.array(0)).get()  # doctest: +SKIP
    array(0)
    >>> nps.sign(nps.array(5-2j)).get()  # doctest: +SKIP
    array(1.+0.j)
    """
    return _instance().map_uop(
        op_name="sign",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def signbit(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Returns element-wise True where signbit is set (less than zero).

    This docstring was copied from numpy.signbit.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        The input value(s).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    result : BlockArray of bool
        Output array, or reference to `out` if that was supplied.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.signbit(nps.array(-1.2)).get()  # doctest: +SKIP
    array(True)
    >>> nps.signbit(nps.array([1, -2.3, 2.1])).get()  # doctest: +SKIP
    array([False,  True, False])
    """
    return _instance().map_uop(
        op_name="signbit",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def sin(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Trigonometric sine, element-wise.

    This docstring was copied from numpy.sin.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The sine of each element of x.

    See Also
    --------
    arcsin, sinh, cos

    Notes
    -----
    The sine is one of the fundamental functions of trigonometry (the
    mathematical study of triangles).  Consider a circle of radius 1
    centered on the origin.  A ray comes in from the :math:`+x` axis, makes
    an angle at the origin (measured counter-clockwise from that axis), and
    departs from the origin.  The :math:`y` coordinate of the outgoing
    ray's intersection with the unit circle is the sine of that angle.  It
    ranges from -1 for :math:`x=3\pi / 2` to +1 for :math:`\pi / 2.`  The
    function has zeroes where the angle is a multiple of :math:`\pi`.
    Sines of angles between :math:`\pi` and :math:`2\pi` are negative.
    The numerous properties of the sine and related functions are included
    in any standard trigonometry text.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    Print sine of one angle:

    >>> nps.sin(nps.array(nps.pi/2.)).get()  # doctest: +SKIP
    array(1.)

    Print sines of an array of angles given in degrees:

    >>> nps.sin(nps.array((0., 30., 45., 60., 90.)) * nps.pi / 180. ).get()  # doctest: +SKIP
    array([ 0.        ,  0.5       ,  0.70710678,  0.8660254 ,  1.        ])
    """
    return _instance().map_uop(
        op_name="sin",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def sinh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Hyperbolic sine, element-wise.

    This docstring was copied from numpy.sinh.

    Some inconsistencies with the NumS version may exist.

    Equivalent to ``1/2 * (nps.exp(x) - nps.exp(-x)).get()`` or
    ``-1j * nps.sin(1j*x).get()``.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The corresponding hyperbolic sine values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972, pg. 83.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.sinh(nps.array(0)).get()  # doctest: +SKIP
    nps.array(0.)
    >>> nps.sinh(nps.array(nps.pi*1j/2)).get()  # doctest: +SKIP
    array(0.+1.j)
    >>> nps.sinh(nps.array(nps.pi*1j)).get() # (exact value is 0)  # doctest: +SKIP
    array(0.+1.2246468e-16j)
    >>> # Discrepancy due to vagaries of floating point arithmetic.

    >>> # Example of providing the optional output parameter
    >>> out1 = nps.array([0], dtype='d')  # doctest: +SKIP
    >>> out2 = nps.sinh(nps.array([0.1]), out1)  # doctest: +SKIP
    >>> out2 is out1  # doctest: +SKIP
    True
    """
    return _instance().map_uop(
        op_name="sinh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def spacing(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the distance between x and the nearest adjacent number.

    This docstring was copied from numpy.spacing.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Values to find the spacing of.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        The spacing of values of `x`.
    """
    return _instance().map_uop(
        op_name="spacing",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def sqrt(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the non-negative square-root of an array, element-wise.

    This docstring was copied from numpy.sqrt.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        The values whose square-roots are required.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`.  If any element in `x` is
        complex, a complex array is returned (and the square-roots of
        negative reals are calculated).  If all of the elements in `x`
        are real, so is `y`, with negative elements returning ``nan``.
        If `out` was provided, `y` is a reference to it.

    Notes
    -----
    *sqrt* has--consistent with common convention--as its branch cut the
    real "interval" [`-inf`, 0), and is continuous from above on it.
    A branch cut is a curve in the complex plane across which a given
    complex function fails to be continuous.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.sqrt(nps.array([1,4,9])).get()  # doctest: +SKIP
    array([ 1.,  2.,  3.])

    >>> nps.sqrt(nps.array([4, -1, -3+4J])).get()  # doctest: +SKIP
    array([ 2.+0.j,  0.+1.j,  1.+2.j])

    >>> nps.sqrt(nps.array([4, -1, nps.inf])).get()  # doctest: +SKIP
    array([ 2., nan, inf])
    """
    return _instance().map_uop(
        op_name="sqrt",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def square(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the element-wise square of the input.

    This docstring was copied from numpy.square.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x : BlockArray
        Input data.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Element-wise `x*x`, of the same shape and dtype as `x`.

    See Also
    --------
    sqrt
    power

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.square(nps.array([-1j, 1])).get()  # doctest: +SKIP
    array([-1.-0.j,  1.+0.j])
    """
    return _instance().map_uop(
        op_name="square",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def tan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Compute tangent element-wise.

    This docstring was copied from numpy.tan.

    Some inconsistencies with the NumS version may exist.

    Equivalent to ``nps.sin(x)/nps.cos(x).get()`` element-wise.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The corresponding tangent values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> from math import pi  # doctest: +SKIP
    >>> nps.tan(nps.array([-pi,pi/2,pi])).get()  # doctest: +SKIP
    array([  1.22460635e-16,   1.63317787e+16,  -1.22460635e-16])
    >>>
    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = nps.array([0], dtype='d')  # doctest: +SKIP
    >>> out2 = nps.cos(nps.array([0.1]), out1)  # doctest: +SKIP
    >>> out2 is out1  # doctest: +SKIP
    True
    """
    return _instance().map_uop(
        op_name="tan",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def tanh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Compute hyperbolic tangent element-wise.

    This docstring was copied from numpy.tanh.

    Some inconsistencies with the NumS version may exist.

    Equivalent to ``nps.sinh(x)/nps.cosh(x)`` or ``-1j * nps.tan(1j*x).get()``.

    Parameters
    ----------
    x : BlockArray
        Input array.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The corresponding hyperbolic tangent values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
           New York, NY: Dover, 1972, pg. 83.
           http://www.math.sfu.ca/~cbm/aands/

    .. [2] Wikipedia, "Hyperbolic function",
           https://en.wikipedia.org/wiki/Hyperbolic_function

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.tanh(nps.array([0, nps.pi*1j, nps.pi*1j/2])).get()  # doctest: +SKIP
    array([ 0. +0.00000000e+00j,  0. -1.22460635e-16j,  0. +1.63317787e+16j])

    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = nps.array([0], dtype='d')  # doctest: +SKIP
    >>> out2 = nps.tanh(nps.array([0.1]), out1)  # doctest: +SKIP
    >>> out2 is out1  # doctest: +SKIP
    True
    """
    return _instance().map_uop(
        op_name="tanh",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def trunc(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    """Return the truncated value of the input, element-wise.

    This docstring was copied from numpy.trunc.

    Some inconsistencies with the NumS version may exist.

    The truncated value of the scalar `x` is the nearest integer `i` which
    is closer to zero than `x` is. In short, the fractional part of the
    signed number `x` is discarded.

    Parameters
    ----------
    x : BlockArray
        Input data.
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The truncated value of each element in `x`.

    See Also
    --------
    ceil, floor, rint

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> a = nps.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])  # doctest: +SKIP
    >>> nps.trunc(a).get()  # doctest: +SKIP
    array([-1., -1., -0.,  0.,  1.,  1.,  2.])
    """
    return _instance().map_uop(
        op_name="trunc",
        arr=x,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def add(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Add arguments element-wise.

    This docstring was copied from numpy.add.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        The arrays to be added.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    add : BlockArray or scalar
        The sum of `x1` and `x2`, element-wise.

    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.add(nps.array(1.0), nps.array(4.0)).get()  # doctest: +SKIP
    array(5.)
    >>> x1 = nps.arange(9.0).reshape((3, 3))  # doctest: +SKIP
    >>> x2 = nps.arange(3.0)  # doctest: +SKIP
    >>> nps.add(x1, x2).get()  # doctest: +SKIP
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])

    """
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
    """Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    This docstring was copied from numpy.arctan2.

    Some inconsistencies with the NumS version may exist.

    The quadrant (i.e., branch) is chosen so that ``arctan2(x1, x2)`` is
    the signed angle in radians between the ray ending at the origin and
    passing through the point (1,0), and the ray ending at the origin and
    passing through the point (`x2`, `x1`).  (Note the role reversal: the
    "`y`-coordinate" is the first function parameter, the "`x`-coordinate"
    is the second.)  By IEEE convention, this function is defined for
    `x2` = +/-0 and for either or both of `x1` and `x2` = +/-inf (see
    Notes for specific values).

    This function is not defined for complex-valued arguments; for the
    so-called argument of complex values, use `angle`.

    Parameters
    ----------
    x1 : BlockArray, real-valued
        `y`-coordinates.
    x2 : BlockArray, real-valued
        `x`-coordinates.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    angle : BlockArray
        Array of angles in radians, in the range ``[-pi, pi]``.

    See Also
    --------
    arctan, tan, angle

    Notes
    -----
    *arctan2* is identical to the `atan2` function of the underlying
    C library.  The following special values are defined in the C
    standard: [1]_

    ====== ====== ================
    `x1`   `x2`   `arctan2(x1,x2)`
    ====== ====== ================
    +/- 0  +0     +/- 0
    +/- 0  -0     +/- pi
     > 0   +/-inf +0 / +pi
     < 0   +/-inf -0 / -pi
    +/-inf +inf   +/- (pi/4)
    +/-inf -inf   +/- (3*pi/4)
    ====== ====== ================

    Note that +0 and -0 are distinct floating point numbers, as are +inf
    and -inf.

    References
    ----------
    .. [1] ISO/IEC standard 9899:1999, "Programming language C."

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    Consider four points in different quadrants:

    >>> x = nps.array([-1, +1, +1, -1])  # doctest: +SKIP
    >>> y = nps.array([-1, -1, +1, +1])  # doctest: +SKIP
    >>> (nps.arctan2(y, x) * 180 / nps.pi).get()  # doctest: +SKIP
    array([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. `arctan2` is defined also when `x2` = 0
    and at several other special points, obtaining values in
    the range ``[-pi, pi]``:

    >>> nps.arctan2(nps.array([1., -1.]), nps.array([0., 0.])).get()  # doctest: +SKIP
    array([ 1.57079633, -1.57079633])
    >>> nps.arctan2(nps.array([0., 0., nps.inf]), nps.array([+0., -0., nps.inf])).get()  # doctest: +SKIP
    array([ 0.        ,  3.14159265,  0.78539816])
    """
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
    """Compute the bit-wise AND of two arrays element-wise.

    This docstring was copied from numpy.bitwise_and.

    Some inconsistencies with the NumS version may exist.

    Computes the bit-wise AND of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``&``.

    Parameters
    ----------
    x1, x2 : BlockArray
        Only integer and boolean types are handled.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar
        Result.

    See Also
    --------
    logical_and
    bitwise_or
    bitwise_xor
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    The The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    number 13 is represented by ``00001101``.  Likewise, 17 is
    represented by ``00010001``.  The bit-wise AND of 13 and 17 is
    therefore ``000000001``, or 1:

    >>> nps.bitwise_and(nps.array(13), nps.array(17)).get()  # doctest: +SKIP
    array(1)
    >>> nps.bitwise_and(nps.array([2,5,255]), nps.array([3,14,16])).get()  # doctest: +SKIP
    array([ 2,  4, 16])
    >>> nps.bitwise_and(nps.array([True, True]), nps.array([False, True])).get()  # doctest: +SKIP
    array([False,  True])
    """
    return _instance().map_bop(
        op_name="bitwise_and",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def bitwise_or(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Compute the bit-wise OR of two arrays element-wise.

    This docstring was copied from numpy.bitwise_or.

    Some inconsistencies with the NumS version may exist.

    Computes the bit-wise OR of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``|``.

    Parameters
    ----------
    x1, x2 : BlockArray
        Only integer and boolean types are handled.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Result.

    See Also
    --------
    logical_or
    bitwise_and
    bitwise_xor
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    The The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    number 13 has the binaray representation ``00001101``. Likewise,
    16 is represented by ``00010000``.  The bit-wise OR of 13 and 16 is
    then ``000111011``, or 29:

    >>> nps.bitwise_or(nps.array(13), nps.array(16)).get()  # doctest: +SKIP
    array(29)
    >>> nps.bitwise_or(nps.array([2, 5, 255]), nps.array([4, 4, 4])).get()  # doctest: +SKIP
    array([  6,   5, 255])
    >>> (nps.array([2, 5, 255]) | nps.array([4, 4, 4])).get()  # doctest: +SKIP
    array([  6,   5, 255])
    >>> nps.bitwise_or(nps.array([True, True]), nps.array([False, True])).get()  # doctest: +SKIP
    array([ True,  True])
    """
    return _instance().map_bop(
        op_name="bitwise_or",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def bitwise_xor(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Compute the bit-wise XOR of two arrays element-wise.

    This docstring was copied from numpy.bitwise_xor.

    Some inconsistencies with the NumS version may exist.

    Computes the bit-wise XOR of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``^``.

    Parameters
    ----------
    x1, x2 : BlockArray
        Only integer and boolean types are handled.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Result.

    See Also
    --------
    logical_xor
    bitwise_and
    bitwise_or
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    The The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    number 13 is represented by ``00001101``. Likewise, 17 is
    represented by ``00010001``.  The bit-wise XOR of 13 and 17 is
    therefore ``00011100``, or 28:

    >>> nps.bitwise_xor(nps.array(13), nps.array(17)).get()  # doctest: +SKIP
    nps.array(28)

    >>> nps.bitwise_xor(nps.array([31,3]), nps.array([5,6])).get()  # doctest: +SKIP
    array([26,  5])
    >>> nps.bitwise_xor(nps.array([True, True]), nps.array([False, True])).get()  # doctest: +SKIP
    array([ True, False])
    """
    return _instance().map_bop(
        op_name="bitwise_xor",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def copysign(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Change the sign of x1 to that of x2, element-wise.

    This docstring was copied from numpy.copysign.

    Some inconsistencies with the NumS version may exist.

    If `x2` is a scalar, its sign will be copied to all elements of `x1`.

    Parameters
    ----------
    x1 : BlockArray
        Values to change the sign of.
    x2 : BlockArray
        The sign of `x2` is copied to `x1`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        The values of `x1` with the sign of `x2`.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.copysign(nps.array(1.3), nps.array(-1)).get()  # doctest: +SKIP
    array(-1.3)
    >>> (1/nps.copysign(nps.array(0), nps.array(1))).get()  # doctest: +SKIP
    array(inf)
    >>> (1/nps.copysign(nps.array(0), nps.array(-1))).get()  # doctest: +SKIP
    array(-inf)
    """
    return _instance().map_bop(
        op_name="copysign",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def divide(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Returns a true division of the inputs, element-wise.

    This docstring was copied from numpy.divide.

    Some inconsistencies with the NumS version may exist.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Parameters
    ----------
    x1 : BlockArray
        Dividend array.
    x2 : BlockArray
        Divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray or scalar

    Notes
    -----
    In Python, ``//`` is the floor division operator and ``/`` the
    true division operator.  The ``true_divide(x1, x2)`` function is
    equivalent to true division in Python.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(5)  # doctest: +SKIP
    >>> nps.true_divide(x, nps.array(4)).get()  # doctest: +SKIP
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    >>> (x/4).get()  # doctest: +SKIP
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
    """
    return _instance().map_bop(
        op_name="divide",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def equal(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return (x1 == x2) element-wise.

    This docstring was copied from numpy.equal.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.equal(nps.array([0, 1, 3]), nps.arange(3)).get()  # doctest: +SKIP
    array([ True,  True, False])

    What is compared are values, not types. So an int (1) and an array of
    length one can evaluate as True:

    >>> nps.equal(nps.array(1), nps.ones(1)).get()  # doctest: +SKIP
    array([ True])
    """
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
    """First array elements raised to powers from second array, element-wise.

    This docstring was copied from numpy.float_power.

    Some inconsistencies with the NumS version may exist.

    Raise each base in `x1` to the positionally-corresponding power in `x2`.
    `x1` and `x2` must be broadcastable to the same shape. This differs from
    the power function in that integers, float16, and float32  are promoted to
    floats with a minimum precision of float64 so that the result is always
    inexact.  The intent is that the function will return a usable result for
    negative powers and seldom overflow for positive powers.

    Parameters
    ----------
    x1 : BlockArray
        The bases.
    x2 : BlockArray
        The exponents.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The bases in `x1` raised to the exponents in `x2`.

    See Also
    --------
    power : power function that preserves type
    """
    return _instance().map_bop(
        op_name="float_power",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def floor_divide(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return the largest integer smaller or equal to the division of the inputs.
    It is equivalent to the Python ``//`` operator and pairs with the
    Python ``%`` (`remainder`), function so that ``a = a % b + b * (a // b)``
    up to roundoff.

    This docstring was copied from numpy.floor_divide.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1 : BlockArray
        Numerator.
    x2 : BlockArray
        Denominator.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        y = floor(`x1`/`x2`)

    See Also
    --------
    remainder : Remainder complementary to floor_divide.
    divmod : Simultaneous floor division and remainder.
    divide : Standard division.
    floor : Round a number to the nearest integer toward minus infinity.
    ceil : Round a number to the nearest integer toward infinity.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.floor_divide(nps.array(7),nps.array(3)).get()  # doctest: +SKIP
    nps.array(2)
    >>> nps.floor_divide(nps.array([1., 2., 3., 4.]), nps.array(2.5)).get()  # doctest: +SKIP
    array([ 0.,  0.,  1.,  1.])
    """
    return _instance().map_bop(
        op_name="floor_divide",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def fmax(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Element-wise maximum of array elements.

    This docstring was copied from numpy.fmax.

    Some inconsistencies with the NumS version may exist.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then the
    non-nan element is returned. If both elements are NaNs then the first
    is returned.  The latter distinction is important for complex NaNs,
    which are defined as at least one of the real or imaginary parts being
    a NaN. The net effect is that NaNs are ignored when possible.

    Parameters
    ----------
    x1, x2 : BlockArray
        The arrays holding the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The maximum of `x1` and `x2`, element-wise.

    See Also
    --------
    fmin :
        Element-wise minimum of two arrays, ignores NaNs.
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.
    amax :
        The maximum value of an array along a given axis, propagates NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignores NaNs.

    minimum, amin, nanmin

    Notes
    -----
    The fmax is equivalent to ``nps.where(x1 >= x2, x1, x2).get()`` when neither
    x1 nor x2 are NaNs, but it is faster and does proper broadcasting.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.fmax(nps.array([2, 3, 4]), nps.array([1, 5, 2])).get()  # doctest: +SKIP
    array([ 2.,  5.,  4.])

    >>> nps.fmax(nps.eye(2), nps.array([0.5, 2])).get()  # doctest: +SKIP
    array([[ 1. ,  2. ],
           [ 0.5,  2. ]])

    >>> nps.fmax(nps.array([nps.nan, 0, nps.nan]),nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
    array([ 0.,  0., nan])
    """
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
    """Element-wise minimum of array elements.

    This docstring was copied from numpy.fmin.

    Some inconsistencies with the NumS version may exist.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then the
    non-nan element is returned. If both elements are NaNs then the first
    is returned.  The latter distinction is important for complex NaNs,
    which are defined as at least one of the real or imaginary parts being
    a NaN. The net effect is that NaNs are ignored when possible.

    Parameters
    ----------
    x1, x2 : BlockArray
        The arrays holding the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The minimum of `x1` and `x2`, element-wise.

    See Also
    --------
    fmax :
        Element-wise maximum of two arrays, ignores NaNs.
    minimum :
        Element-wise minimum of two arrays, propagates NaNs.
    amin :
        The minimum value of an array along a given axis, propagates NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignores NaNs.

    maximum, amax, nanmax

    Notes
    -----
    The fmin is equivalent to ``nps.where(x1 <= x2, x1, x2).get()`` when neither
    x1 nor x2 are NaNs, but it is faster and does proper broadcasting.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.fmin(nps.array([2, 3, 4]), nps.array([1, 5, 2])).get()  # doctest: +SKIP
    array([1, 3, 2])

    >>> nps.fmin(nps.eye(2), nps.array([0.5, 2])).get()  # doctest: +SKIP
    array([[ 0.5,  0. ],
           [ 0. ,  1. ]])

    >>> nps.fmin(nps.array([nps.nan, 0, nps.nan]),nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
    array([ 0.,  0., nan])
    """
    return _instance().map_bop(
        op_name="fmin",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def fmod(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return the element-wise remainder of division.

    This docstring was copied from numpy.fmod.

    Some inconsistencies with the NumS version may exist.

    This is the NumPy implementation of the C library function fmod, the
    remainder has the same sign as the dividend `x1`. It is equivalent to
    the Matlab(TM) ``rem`` function and should not be confused with the
    Python modulus operator ``x1 % x2``.

    Parameters
    ----------
    x1 : BlockArray
        Dividend.
    x2 : BlockArray
        Divisor.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The remainder of the division of `x1` by `x2`.

    See Also
    --------
    remainder : Equivalent to the Python ``%`` operator.
    divide

    Notes
    -----
    The result of the modulo operation for negative dividend and divisors
    is bound by conventions. For `fmod`, the sign of result is the sign of
    the dividend, while for `remainder` the sign of the result is the sign
    of the divisor. The `fmod` function is equivalent to the Matlab(TM)
    ``rem`` function.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.fmod(nps.array([-3, -2, -1, 1, 2, 3]), nps.array(2)).get()  # doctest: +SKIP
    array([-1,  0, -1,  1,  0,  1])
    >>> nps.remainder(nps.array([-3, -2, -1, 1, 2, 3]), nps.array(2)).get()  # doctest: +SKIP
    array([1, 0, 1, 1, 0, 1])

    >>> nps.fmod(nps.array([5, 3]), nps.array([2, 2.])).get()  # doctest: +SKIP
    array([ 1.,  1.])
    >>> a = nps.arange(-3, 3).reshape(3, 2)  # doctest: +SKIP
    >>> a.get()  # doctest: +SKIP
    array([[-3, -2],
           [-1,  0],
           [ 1,  2]])
    >>> nps.fmod(a, nps.array([2,2])).get()  # doctest: +SKIP
    array([[-1,  0],
           [-1,  0],
           [ 1,  0]])
    """
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
    """Returns the greatest common divisor of ``|x1|`` and ``|x2|``

    This docstring was copied from numpy.gcd.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray, int
        Arrays of values.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).

    Returns
    -------
    y : BlockArray
        The greatest common divisor of the absolute value of the inputs

    See Also
    --------
    lcm : The lowest common multiple

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.gcd(nps.array(12), nps.array(20)).get()  # doctest: +SKIP
    array(4)
    >>> nps.gcd(nps.arange(6), nps.array(20)).get()  # doctest: +SKIP
    array([20,  1,  2,  1,  4,  5])
    """
    return _instance().map_bop(
        op_name="gcd",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def greater(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return the truth value of (x1 > x2) element-wise.

    This docstring was copied from numpy.greater.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater_equal, less, less_equal, equal, not_equal

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.greater(nps.array([4,2]),nps.array([2,2])).get()  # doctest: +SKIP
    array([ True, False])

    If the inputs are BlockArray, then nps.greater is equivalent to '>'.

    >>> a = nps.array([4,2])  # doctest: +SKIP
    >>> b = nps.array([2,2])  # doctest: +SKIP
    >>> (a > b).get()  # doctest: +SKIP
    array([ True, False])
    """
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
    """Return the truth value of (x1 >= x2) element-wise.

    This docstring was copied from numpy.greater_equal.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray of bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater, less, less_equal, equal, not_equal

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.greater_equal(nps.array([4, 2, 1]), nps.array([2, 2, 2])).get()  # doctest: +SKIP
    array([ True, True, False])
        """
    return _instance().map_bop(
        op_name="greater_equal",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def heaviside(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Compute the Heaviside step function.

    This docstring was copied from numpy.heaviside.

    Some inconsistencies with the NumS version may exist.

    The Heaviside step function is defined as::

                              0   if x1 < 0
        heaviside(x1, x2) =  x2   if x1 == 0
                              1   if x1 > 0

    where `x2` is often taken to be 0.5, but 0 and 1 are also sometimes used.

    Parameters
    ----------
    x1 : BlockArray
        Input values.
    x2 : BlockArray
        The value of the function when x1 is 0.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        The output array, element-wise Heaviside step function of `x1`.

    References
    ----------
    .. Wikipedia, "Heaviside step function",
       https://en.wikipedia.org/wiki/Heaviside_step_function

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.heaviside(nps.array([-1.5, 0, 2.0]), nps.array(0.5)).get()  # doctest: +SKIP
    array([ 0. ,  0.5,  1. ])
    >>> nps.heaviside(nps.array([-1.5, 0, 2.0]), nps.array(1)).get()  # doctest: +SKIP
    array([ 0.,  1.,  1.])
    """
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
    """Given the "legs" of a right triangle, return its hypotenuse.

    This docstring was copied from numpy.hypot.

    Some inconsistencies with the NumS version may exist.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    it is broadcast for use with each element of the other argument.
    (See Examples)

    Parameters
    ----------
    x1, x2 : BlockArray
        Leg of the triangle(s).
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    z : BlockArray
        The hypotenuse of the triangle(s).

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.hypot(3*nps.ones((3, 3)), 4*nps.ones((3, 3))).get()  # doctest: +SKIP
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    Example showing broadcast of scalar_like argument:

    >>> nps.hypot(3*nps.ones((3, 3)), nps.array([4])).get()  # doctest: +SKIP
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])
    """
    return _instance().map_bop(
        op_name="hypot",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def lcm(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """This docstring was copied from numpy.lcm.

    Some inconsistencies with the NumS version may exist.

    Returns the lowest common multiple of ``|x1|`` and ``|x2|``

    Parameters
    ----------
    x1, x2 : BlockArray, int
        Arrays of values.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).

    Returns
    -------
    y : BlockArray
        The lowest common multiple of the absolute value of the inputs

    See Also
    --------
    gcd : The greatest common divisor

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.lcm(nps.array(12), nps.array(20)).get()  # doctest: +SKIP
    array(60)
    >>> nps.lcm(nps.arange(6), nps.array(20)).get()  # doctest: +SKIP
    array([ 0, 20, 20, 60, 20, 20])
    """
    return _instance().map_bop(
        op_name="lcm",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def ldexp(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Returns x1 * 2**x2, element-wise.

    This docstring was copied from numpy.ldexp.

    Some inconsistencies with the NumS version may exist.

    The mantissas `x1` and twos exponents `x2` are used to construct
    floating point numbers ``x1 * 2**x2``.

    Parameters
    ----------
    x1 : BlockArray
        Array of multipliers.
    x2 : BlockArray, int
        Array of twos exponents.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The result of ``x1 * 2**x2``.

    See Also
    --------
    frexp : Return (y1, y2) from ``x = y1 * 2**y2``, inverse to `ldexp`.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.

    `ldexp` is useful as the inverse of `frexp`, if used by itself it is
    more clear to simply use the expression ``x1 * 2**x2``.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(6)  # doctest: +SKIP
    >>> nps.ldexp(*nps.frexp(x)).get()  # doctest: +SKIP
    array([ 0.,  1.,  2.,  3.,  4.,  5.])
    """
    return _instance().map_bop(
        op_name="ldexp",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def left_shift(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Shift the bits of an integer to the left.

    This docstring was copied from numpy.left_shift.

    Some inconsistencies with the NumS version may exist.

    Bits are shifted to the left by appending `x2` 0s at the right of `x1`.
    Since the internal representation of numbers is in binary format, this
    operation is equivalent to multiplying `x1` by ``2**x2``.

    Parameters
    ----------
    x1 : BlockArray of integer type
        Input values.
    x2 : BlockArray of integer type
        Number of zeros to append to `x1`. Has to be non-negative.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : array of integer type
        Return `x1` with bits shifted `x2` times to the left.

    See Also
    --------
    right_shift : Shift the bits of an integer to the right.
    binary_repr : Return the binary representation of the input number
        as a string.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.left_shift(nps.array(5), nps.array(2)).get()  # doctest: +SKIP
    array(20)

    Note that the dtype of the second argument may change the dtype of the
    result and can lead to unexpected results in some cases (see
    :ref:`Casting Rules <ufuncs.casting>`):

    >>> a = nps.left_shift(nps.array(255, dtype=nps.uint8), nps.array(1)) # Expect 254  # doctest: +SKIP
    >>> print(a, type(a)) # Unexpected result due to upcasting  # doctest: +SKIP
    510 <class 'nums.core.array.blockarray.BlockArray'>
    """
    return _instance().map_bop(
        op_name="left_shift",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def less(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return the truth value of (x1 < x2) element-wise.

    This docstring was copied from numpy.less.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater, less_equal, greater_equal, equal, not_equal

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.less(nps.array([1, 2]), nps.array([2, 2])).get()  # doctest: +SKIP
    array([ True, False])

    """
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
    """Return the truth value of (x1 =< x2) element-wise.

    This docstring was copied from numpy.less_equal.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    greater, less, greater_equal, equal, not_equal

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.less_equal(nps.array([4, 2, 1]), nps.array([2, 2, 2])).get()  # doctest: +SKIP
    array([False,  True,  True])
    """
    return _instance().map_bop(
        op_name="less_equal",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def logaddexp(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Logarithm of the sum of exponentiations of the inputs.

    This docstring was copied from numpy.logaddexp.

    Some inconsistencies with the NumS version may exist.

    Calculates ``log(exp(x1) + exp(x2))``. This function is useful in
    statistics where the calculated probabilities of events may be so small
    as to exceed the range of normal floating point numbers.  In such cases
    the logarithm of the calculated probability is stored. This function
    allows adding probabilities stored in such a fashion.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input values.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    result : BlockArray
        Logarithm of ``exp(x1) + exp(x2)``.

    See Also
    --------
    logaddexp2: Logarithm of the sum of exponentiations of inputs in base 2.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> prob1 = nps.log(nps.array(1e-50))  # doctest: +SKIP
    >>> prob2 = nps.log(nps.array(2.5e-50))  # doctest: +SKIP
    >>> prob12 = nps.logaddexp(prob1, prob2)  # doctest: +SKIP
    >>> prob12.get()  # doctest: +SKIP
    array(-113.87649168)
    >>> nps.exp(prob12).get()  # doctest: +SKIP
    array(3.5e-50)
    """
    return _instance().map_bop(
        op_name="logaddexp",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def logaddexp2(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Logarithm of the sum of exponentiations of the inputs in base-2.

    This docstring was copied from numpy.logaddexp2.

    Some inconsistencies with the NumS version may exist.

    Calculates ``log2(2**x1 + 2**x2)``. This function is useful in machine
    learning when the calculated probabilities of events may be so small as
    to exceed the range of normal floating point numbers.  In such cases
    the base-2 logarithm of the calculated probability can be used instead.
    This function allows adding probabilities stored in such a fashion.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input values.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    result : BlockArray
        Base-2 logarithm of ``2**x1 + 2**x2``.

    See Also
    --------
    logaddexp: Logarithm of the sum of exponentiations of the inputs.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.
    >>> prob1 = nps.log(nps.array(1e-50))  # doctest: +SKIP
    >>> prob2 = nps.log(nps.array(2.5e-50))  # doctest: +SKIP
    >>> prob12 = nps.logaddexp2(prob1, prob2)  # doctest: +SKIP
    >>> prob1.get(), prob2.get(), prob12.get()  # doctest: +SKIP
    (array(-115.12925465), array(-114.21296392), array(-113.59955523))
    >>> 2**prob12  # doctest: +SKIP
    array(6.35515844e-35)
    """
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
    """Compute the truth value of x1 AND x2 element-wise.

    This docstring was copied from numpy.logical_and.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        Boolean result of the logical AND operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    See Also
    --------
    logical_or, logical_not, logical_xor
    bitwise_and

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.logical_and(nps.array(True), nps.array(False)).get()  # doctest: +SKIP
    array(False)
    >>> nps.logical_and(nps.array([True, False]), nps.array([False, False])).get()  # doctest: +SKIP
    array([False, False])

    >>> x = nps.arange(5)  # doctest: +SKIP
    >>> nps.logical_and(x>1, x<4).get()  # doctest: +SKIP
    array([False, False,  True,  True, False])
    """
    return _instance().map_bop(
        op_name="logical_and",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def logical_or(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Compute the truth value of x1 OR x2 element-wise.

    This docstring was copied from numpy.logical_or.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Logical OR is applied to the elements of `x1` and `x2`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        Boolean result of the logical OR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    See Also
    --------
    logical_and, logical_not, logical_xor
    bitwise_or

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.logical_or(nps.array(True), nps.array(False)).get()  # doctest: +SKIP
    array(True)
    >>> nps.logical_or(nps.array([True, False]), nps.array([False, False])).get()  # doctest: +SKIP
    array([ True, False])

    >>> x = nps.arange(5)  # doctest: +SKIP
    >>> nps.logical_or(x < 1, x > 3).get()  # doctest: +SKIP
    array([ True, False, False, False,  True])
    """
    return _instance().map_bop(
        op_name="logical_or",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def logical_xor(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Compute the truth value of x1 XOR x2, element-wise.

    This docstring was copied from numpy.logical_xor.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Logical XOR is applied to the elements of `x1` and `x2`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray of bool
        Boolean result of the logical XOR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    See Also
    --------
    logical_and, logical_or, logical_not, bitwise_xor

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.logical_xor(nps.array(True), nps.array(False)).get()  # doctest: +SKIP
    array(True)
    >>> nps.logical_xor(nps.array([True, True, False, False]), nps.array([True, False, True, False])).get()  # doctest: +SKIP
    array([False,  True,  True, False])

    >>> x = nps.arange(5)  # doctest: +SKIP
    >>> nps.logical_xor(x < 1, x > 3).get()  # doctest: +SKIP
    array([ True, False, False, False,  True])
    """
    return _instance().map_bop(
        op_name="logical_xor",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def maximum(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Element-wise maximum of array elements.

    This docstring was copied from numpy.maximum.

    Some inconsistencies with the NumS version may exist.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : BlockArray
        The arrays holding the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The maximum of `x1` and `x2`, element-wise.

    See Also
    --------
    minimum :
        Element-wise minimum of two arrays, propagates NaNs.
    fmax :
        Element-wise maximum of two arrays, ignores NaNs.
    amax :
        The maximum value of an array along a given axis, propagates NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignores NaNs.

    fmin, amin, nanmin

    Notes
    -----
    The maximum is equivalent to ``nps.where(x1 >= x2, x1, x2).get()`` when
    neither x1 nor x2 are nans, but it is faster and does proper
    broadcasting.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.maximum(nps.array([2, 3, 4]), nps.array([1, 5, 2])).get()  # doctest: +SKIP
    array([2, 5, 4])

    >>> nps.maximum(nps.eye(2), nps.array([0.5, 2])).get() # broadcasting  # doctest: +SKIP
    array([[ 1. ,  2. ],
           [ 0.5,  2. ]])

    >>> nps.maximum(nps.array([nps.nan, 0, nps.nan]), nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
    array([nan, nan, nan])
    >>> nps.maximum(nps.array(nps.Inf), nps.array(1)).get()  # doctest: +SKIP
    arrray(inf)
    """
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
    """Element-wise minimum of array elements.

    This docstring was copied from numpy.minimum.

    Some inconsistencies with the NumS version may exist.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : BlockArray
        The arrays holding the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The minimum of `x1` and `x2`, element-wise.

    See Also
    --------
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.
    fmin :
        Element-wise minimum of two arrays, ignores NaNs.
    amin :
        The minimum value of an array along a given axis, propagates NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignores NaNs.

    fmax, amax, nanmax

    Notes
    -----
    The minimum is equivalent to ``nps.where(x1 <= x2, x1, x2).get()`` when
    neither x1 nor x2 are NaNs, but it is faster and does proper
    broadcasting.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.minimum(nps.array([2, 3, 4]), nps.array([1, 5, 2])).get()  # doctest: +SKIP
    array([1, 3, 2])

    >>> nps.minimum(nps.eye(2), nps.array([0.5, 2])).get() # broadcasting  # doctest: +SKIP
    array([[ 0.5,  0. ],
           [ 0. ,  1. ]])

    >>> nps.minimum(nps.array([nps.nan, 0, nps.nan]),nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
    array([nan, nan, nan])
    >>> nps.minimum(nps.array(-nps.Inf), nps.array(1)).get()  # doctest: +SKIP
    array(-inf)
    """
    return _instance().map_bop(
        op_name="minimum",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def mod(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return element-wise remainder of division.

    This docstring was copied from numpy.mod.

    Some inconsistencies with the NumS version may exist.

    Computes the remainder complementary to the `floor_divide` function.  It is
    equivalent to the Python modulus operator``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to ``nps.remainder``
    is ``mod``.

    Parameters
    ----------
    x1 : BlockArray
        Dividend array.
    x2 : BlockArray
        Divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The element-wise remainder of the quotient ``floor_divide(x1, x2)``.

    See Also
    --------
    floor_divide : Equivalent of Python ``//`` operator.
    divmod : Simultaneous floor division and remainder.
    fmod : Equivalent of the MATLAB ``rem`` function.
    divide, floor

    Notes
    -----
    Returns 0 when `x2` is 0 and both `x1` and `x2` are (arrays of)
    integers.
    ``mod`` is an alias of ``remainder``.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.remainder(nps.array([4, 7]), nps.array([2, 3])).get()  # doctest: +SKIP
    array([0, 1])
    >>> nps.remainder(nps.arange(7), nps.array(5)).get()  # doctest: +SKIP
    array([0, 1, 2, 3, 4, 0, 1])
    """
    return _instance().map_bop(
        op_name="mod",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def multiply(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Multiply arguments element-wise.

    This docstring was copied from numpy.multiply.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays to be multiplied.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The product of `x1` and `x2`, element-wise.

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of array broadcasting.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.multiply(nps.array(2.0), nps.array(4.0)).get()  # doctest: +SKIP
    array(8.)

    >>> x1 = nps.arange(9.0).reshape((3, 3))  # doctest: +SKIP
    >>> x2 = nps.arange(3.0)  # doctest: +SKIP
    >>> nps.multiply(x1, x2).get()  # doctest: +SKIP
    array([[  0.,   1.,   4.],
           [  0.,   4.,  10.],
           [  0.,   7.,  16.]])
    """
    return _instance().map_bop(
        op_name="multiply",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def nextafter(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return the next floating-point value after x1 towards x2, element-wise.

    This docstring was copied from numpy.nextafter.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1 : BlockArray
        Values to find the next representable value of.
    x2 : BlockArray
        The direction where to look for the next representable value of `x1`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        The next representable values of `x1` in the direction of `x2`.
    """
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
    """This docstring was copied from numpy.not_equal.

    Some inconsistencies with the NumS version may exist.

    Return (x1 != x2) element-wise.

    Parameters
    ----------
    x1, x2 : BlockArray
        Input arrays.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.

    See Also
    --------
    equal, greater, greater_equal, less, less_equal

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.not_equal(nps.array([1.,2.]), nps.array([1., 3.])).get()  # doctest: +SKIP
    array([False,  True])
    """
    return _instance().map_bop(
        op_name="not_equal",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def power(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """First array elements raised to powers from second array, element-wise.

    This docstring was copied from numpy.power.

    Some inconsistencies with the NumS version may exist.

    Raise each base in `x1` to the positionally-corresponding power in
    `x2`.  `x1` and `x2` must be broadcastable to the same shape. Note that an
    integer type raised to a negative integer power will raise a ValueError.

    Parameters
    ----------
    x1 : BlockArray
        The bases.
    x2 : BlockArray
        The exponents.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The bases in `x1` raised to the exponents in `x2`.

    See Also
    --------
    float_power : power function that promotes integers to float
    """
    return _instance().map_bop(
        op_name="power",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def remainder(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Return element-wise remainder of division.

    This docstring was copied from numpy.remainder.

    Some inconsistencies with the NumS version may exist.

    Computes the remainder complementary to the `floor_divide` function.  It is
    equivalent to the Python modulus operator``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to ``nps.remainder``
    is ``mod``.

    Parameters
    ----------
    x1 : BlockArray
        Dividend array.
    x2 : BlockArray
        Divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The element-wise remainder of the quotient ``floor_divide(x1, x2)``.

    See Also
    --------
    floor_divide : Equivalent of Python ``//`` operator.
    divmod : Simultaneous floor division and remainder.
    fmod : Equivalent of the MATLAB ``rem`` function.
    divide, floor

    Notes
    -----
    Returns 0 when `x2` is 0 and both `x1` and `x2` are (arrays of)
    integers.
    ``mod`` is an alias of ``remainder``.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.remainder(nps.array([4, 7]), nps.array([2, 3])).get()  # doctest: +SKIP
    array([0, 1])
    >>> nps.remainder(nps.arange(7), nps.array(5)).get()  # doctest: +SKIP
    array([0, 1, 2, 3, 4, 0, 1])
    """
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
    """Shift the bits of an integer to the right.

    This docstring was copied from numpy.right_shift.

    Some inconsistencies with the NumS version may exist.

    Bits are shifted to the right `x2`.  Because the internal
    representation of numbers is in binary format, this operation is
    equivalent to dividing `x1` by ``2**x2``.

    Parameters
    ----------
    x1 : BlockArray, int
        Input values.
    x2 : BlockArray, int
        Number of bits to remove at the right of `x1`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray
        Return `x1` with bits shifted `x2` times to the right.

    See Also
    --------
    left_shift : Shift the bits of an integer to the left.
    binary_repr : Return the binary representation of the input number
        as a string.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.right_shift(nps.array(10), nps.array(1)).get()  # doctest: +SKIP
    5
    """
    return _instance().map_bop(
        op_name="right_shift",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )


def subtract(
    x1: BlockArray, x2: BlockArray, out: BlockArray = None, where=True, **kwargs
) -> BlockArray:
    """Subtract arguments, element-wise.

    This docstring was copied from numpy.subtract.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    x1, x2 : BlockArray
        The arrays to be subtracted from each other.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : BlockArray
        The difference of `x1` and `x2`, element-wise.

    Notes
    -----
    Equivalent to ``x1 - x2`` in terms of array broadcasting.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.subtract(nps.array(1.0), nps.array(4.0)).get()  # doctest: +SKIP
    array(-3.)

    >>> x1 = nps.arange(9.0).reshape((3, 3))  # doctest: +SKIP
    >>> x2 = nps.arange(3.0)  # doctest: +SKIP
    >>> nps.subtract(x1, x2).get()  # doctest: +SKIP
    array([[ 0.,  0.,  0.],
           [ 3.,  3.,  3.],
           [ 6.,  6.,  6.]])
    """
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
    """Returns a true division of the inputs, element-wise.

    This docstring was copied from numpy.true_divide.

    Some inconsistencies with the NumS version may exist.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Parameters
    ----------
    x1 : BlockArray
        Dividend array.
    x2 : BlockArray
        Divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : BlockArray, None, or optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : BlockArray, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : BlockArray

    Notes
    -----
    In Python, ``//`` is the floor division operator and ``/`` the
    true division operator.  The ``true_divide(x1, x2)`` function is
    equivalent to true division in Python.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> x = nps.arange(5)  # doctest: +SKIP
    >>> nps.true_divide(x, nps.array(4)).get()  # doctest: +SKIP
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    >>> (x/4).get()  # doctest: +SKIP
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
    """
    return _instance().map_bop(
        op_name="true_divide",
        arr_1=x1,
        arr_2=x2,
        out=out,
        where=where,
        kwargs=numpy_utils.ufunc_kwargs(kwargs),
    )
