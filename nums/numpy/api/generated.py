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

# pylint: disable = redefined-builtin, too-many-lines, anomalous-backslash-in-string, unused-wildcard-import, wildcard-import, unused-import

from typing import Tuple, Optional, Union

import numpy as np

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray
from nums.numpy import numpy_utils

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
    >>> nps.arctan2(nps.array([0., 0., nps.inf]),
    ...     nps.array([+0., -0., nps.inf])).get()  # doctest: +SKIP
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

    >>> nps.fmax(nps.array([nps.nan, 0, nps.nan]),
    ...     nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
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

    >>> nps.fmin(nps.array([nps.nan, 0, nps.nan]),
    ...     nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
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

    >>> a = nps.left_shift(nps.array(255, dtype=nps.uint8),
    ...     nps.array(1)) # Expect 254  # doctest: +SKIP
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
    >>> nps.logical_xor(nps.array([True, True, False, False]),
    ...     nps.array([True, False, True, False])).get()  # doctest: +SKIP
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

    >>> nps.maximum(nps.array([nps.nan, 0, nps.nan]),
    ...     nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
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

    >>> nps.minimum(nps.array([nps.nan, 0, nps.nan]),
    ...     nps.array([0, nps.nan, nps.nan])).get()  # doctest: +SKIP
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
