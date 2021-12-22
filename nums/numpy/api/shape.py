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

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray

from nums.numpy.api.utility import *

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
