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
# Logic Ops
############################################


def all(a: BlockArray, axis=None, out=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to True.

    This docstring was copied from numpy.all.

    Some inconsistencies with the NumS version may exist.

    Parameters
    ----------
    a : BlockArray
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (``axis=None``) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : BlockArray, optional
        Alternate output array in which to place the result.
        It must have the same shape as the expected output and its
        type is preserved (e.g., if ``dtype(out)`` is float, the result
        will consist of 0.0's and 1.0's). See `ufuncs-output-type` for more
        details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `all` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    all : BlockArray, bool
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    all : equivalent method
    any : Test whether any element along a given axis evaluates to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to `True` because these are not equal to zero.

    'out' is currently not supported.

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.all(nps.array([[True,False],[True,True]])).get()  # doctest: +SKIP
    array(False)

    >>> nps.all(nps.array([[True,False],[True,True]]), axis=0).get()  # doctest: +SKIP
    array([ True, False])

    >>> nps.all(nps.array([-1, 4, 5])).get()  # doctest: +SKIP
    array(True)

    >>> nps.all(nps.array([1.0, nps.nan])).get()  # doctest: +SKIP
    array(True)
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("all", a, axis=axis, keepdims=keepdims)


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


def alltrue(a: BlockArray, axis=None, out=None, dtype=None, keepdims=False):
    """Check if all elements of input array are true.

    This docstring was copied from numpy.alltrue.

    Some inconsistencies with the NumS version may exist.

    See Also
    --------
    all : Equivalent function; see for details.
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("alltrue", a, axis=axis, keepdims=keepdims, dtype=dtype)


def any(a: BlockArray, axis=None, out=None, keepdims=False):
    """Test whether any array element along a given axis evaluates to True.

    This docstring was copied from numpy.any.

    Some inconsistencies with the NumS version may exist.

    Returns single boolean unless `axis` is not ``None``

    Parameters
    ----------
    a : BlockArray
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : BlockArray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output and its type is preserved
        (e.g., if it is of type float, then it will remain so, returning
        1.0 for True and 0.0 for False, regardless of the type of `a`).
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `any` method of sub-classes of
        `BlockArray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    any : BlockArray
        A new `BlockArray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    any : equivalent method
    all : Test whether all elements along a given axis evaluate to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to `True` because these are not equal to zero.

    'out' is currently not supported

    Examples
    --------
    The doctests shown below are copied from NumPy.
    They won’t show the correct result until you operate ``get()``.

    >>> nps.any(nps.array([[True, False], [True, True]])).get()  # doctest: +SKIP
    array(True)

    >>> nps.any(nps.array([[True, False], [False, False]]), axis=0).get()  # doctest: +SKIP
    array([ True, False])

    >>> nps.any(nps.array([-1, 0, 5])).get()  # doctest: +SKIP
    array(True)

    >>> nps.any(nps.array(nps.nan)).get()  # doctest: +SKIP
    array(True)
    """
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return _instance().reduce("any", a, axis=axis, keepdims=keepdims)


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
