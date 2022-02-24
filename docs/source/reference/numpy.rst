nums.numpy
==========

Functions in the ``nums.numpy`` module.

.. currentmodule:: nums.numpy

NumS Supported API
~~~~~~~~~~~~~~~~~~


Linear Algebra
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    dot
    inner
    matmul
    tensordot
    outer
    trace

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    sum
    sin
    cos
    tan
    arcsin
    arccos
    arctan
    hypot
    arctan2
    degrees
    radians
    deg2rad
    rad2deg
    sinh
    cosh
    tanh
    arcsinh
    arccosh
    arctanh
    rint
    floor
    ceil
    trunc
    sum
    nansum
    exp
    expm1
    exp2
    log
    log10
    log2
    log1p
    logaddexp
    logaddexp2
    signbit
    copysign
    ldexp
    nextafter
    spacing
    lcm
    gcd
    add
    reciprocal
    positive
    negative
    multiply
    divide
    power
    subtract
    true_divide
    floor_divide
    float_power
    fmod
    mod
    remainder
    conj
    conjugate
    maximum
    fmax
    amax
    nanmax
    minimum
    fmin
    amin
    nanmin
    sqrt
    cbrt
    square
    absolute
    fabs
    sign
    heaviside


Array Creation Routines
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    empty
    empty_like
    eye
    identity
    ones
    ones_like
    zeros
    zeros_like
    array
    copy
    loadtxt
    arange
    linspace
    logspace
    mgrid
    ogrid
    diag
    diagflat


Logic Functions
~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    all
    any
    isfinite
    isinf
    isnan
    logical_and
    logical_or
    logical_not
    logical_xor
    allclose
    array_equal
    array_equiv
    greater
    greater_equal
    less
    less_equal
    equal
    not_equal

Array Manipulation Routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    shape
    reshape
    swapaxes
    transpose
    atleast_1d
    atleast_2d
    atleast_3d
    expand_dims
    squeeze
    concatenate
    vstack
    hstack
    dstack
    column_stack
    row_stack
    split
    reshape

Sorting, Searching, and Counting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    argmax
    argmin
    where
    max
    min
    top_k

Statistics
~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    percentile
    quantile
    median
    average
    mean
    std
    var
    nanmean
    nanstd
    nanvar
    cov