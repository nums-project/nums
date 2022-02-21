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

from nums.core.array.blockarray import BlockArray

from nums.numpy.api.logic import *

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
