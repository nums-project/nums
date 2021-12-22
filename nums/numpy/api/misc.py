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

from nums.numpy.api.utility import *


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
