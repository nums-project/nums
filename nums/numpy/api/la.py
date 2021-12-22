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


from nums.numpy.api.creation import *
from nums.numpy.api.generated import *
from nums.numpy.api.utility import *

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
    return a.transpose(defer=True) @ b


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