"""
NuMojo Top-Level Package (`numojo`)
==================================

This is the main public entry point for NuMojo.

Exports
-------
Core container types:
- `Matrix` (2D array type)
- `NDArray` (N-dimensional array type)
- `Shape` / `NDArrayShape`, `Strides` / `NDArrayStrides`

Core utilities:
- dtype aliases (`f32`, `f64`, `i32`, `i64`, etc.) and complex dtypes
- common error types (`IndexError`, `ShapeError`, ...)

Routines
--------
For convenience, this module also re-exports a curated set of NumPy-like routines
from `numojo.routines` (creation, manipulation, math, linalg, statistics, IO, etc.).

Notes
-----
- This file is intended to provide a stable import surface for users.
- Internal code should prefer importing from the canonical submodules/packages
  (e.g. `numojo.core.matrix`, `numojo.core.layout`, `numojo.routines.math`) rather
  than relying on extensive top-level re-exports.
NuMojo Top-Level Package (`numojo`)
==================================

This is the main public entry point for NuMojo.

Exports
-------
Core container types:
- `Matrix` (2D array type)
- `NDArray` (N-dimensional array type)
- `Shape` / `NDArrayShape`, `Strides` / `NDArrayStrides`

Core utilities:
- dtype aliases (`f32`, `f64`, `i32`, `i64`, etc.) and complex dtypes
- common error types (`IndexError`, `ShapeError`, ...)

Routines
--------
For convenience, this module also re-exports a curated set of NumPy-like routines
from `numojo.routines` (creation, manipulation, math, linalg, statistics, IO, etc.).

Notes
-----
- This file is intended to provide a stable import surface for users.
- Internal code should prefer importing from the canonical submodules/packages
  (e.g. `numojo.core.matrix`, `numojo.core.layout`, `numojo.routines.math`) rather
  than relying on extensive top-level re-exports.
"""

comptime __version__: String = "V0.8.0"

# ===----------------------------------------------------------------------=== #
# Import core types
# ===----------------------------------------------------------------------=== #

from numojo.core.ndarray import NDArray
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.layout.ndstrides import NDArrayStrides
from numojo.core.indexing.item import Item
from numojo.core.indexing import IndexMethods
from numojo.core.matrix import Matrix
from numojo.core.complex.complex_simd import ComplexSIMD

from numojo.core.complex.complex_ndarray import ComplexNDArray
from numojo.core.dtype.complex_dtype import (
    ComplexDType,
    ci8,
    ci16,
    ci32,
    ci64,
    ci128,
    ci256,
    cint,
    cu8,
    cu16,
    cu32,
    cu64,
    cu128,
    cu256,
    cuint,
    cbf16,
    cf16,
    cf32,
    cf64,
    cboolean,
    cinvalid,
)
from numojo.core.dtype.default_dtype import (
    i8,
    i16,
    i32,
    i64,
    i128,
    i256,
    int,
    u8,
    u16,
    u32,
    u64,
    u128,
    u256,
    uint,
    bf16,
    f16,
    f32,
    f64,
    boolean,
)
from numojo.core.error import NumojoError
from numojo.core.type_aliases import (
    Shape,
    Strides,
    ComplexScalar,
    CScalar,
    `1j`,
)

# ===----------------------------------------------------------------------=== #
# Import routines and objects
# ===----------------------------------------------------------------------=== #

# Objects
from numojo.routines.constants import Constants

comptime pi = numojo.routines.constants.Constants.pi
comptime e = numojo.routines.constants.Constants.e
comptime c = numojo.routines.constants.Constants.c

# Functions
# TODO Make explicit imports of each individual function in future
# to avoid polluting the root namespace.
from numojo.routines import io
from numojo.routines.io import (
    loadtxt,
    savetxt,
    load,
    save,
)
from numojo.routines.io import set_printoptions

from numojo.routines import linalg
from numojo.routines.linalg.misc import diagonal

from numojo.routines import logic
from numojo.routines.logic import (
    greater,
    greater_equal,
    less,
    less_equal,
    equal,
    not_equal,
    isinf,
    isfinite,
    isnan,
    any,
    all,
)

from numojo.routines import math
from numojo.routines.math import (
    add,
    sub,
    diff,
    mod,
    mul,
    div,
    floor_div,
    fma,
    remainder,
)
from numojo.routines.math import gradient, trapz
from numojo.routines.math import exp, exp2, expm1, log, ln, log2, log10, log1p
from numojo.routines.math import (
    max,
    min,
    mimimum,
    maximum,
)
from numojo.routines.math import copysign
from numojo.routines.math import (
    arccosh,
    acosh,
    arcsinh,
    asinh,
    arctanh,
    atanh,
    cosh,
    sinh,
    tanh,
)
from numojo.routines.math import cbrt, clip, rsqrt, sqrt, scalb
from numojo.routines.math import prod, cumprod
from numojo.routines.math import (
    tabs,
    tfloor,
    tceil,
    ttrunc,
    tround,
    roundeven,
    nextafter,
)
from numojo.routines.math import sum, cumsum
from numojo.routines.math import (
    arccos,
    acos,
    arcsin,
    asin,
    arctan,
    atan,
    atan2,
    cos,
    sin,
    tan,
    hypot,
    hypot_fma,
)

from numojo.routines import statistics
from numojo.routines.statistics import (
    mean,
    mode,
    median,
    variance,
    std,
)

from numojo.routines import bitwise
from numojo.routines.bitwise import invert

from numojo.routines import creation
from numojo.routines.creation import (
    arange,
    linspace,
    logspace,
    geomspace,
    empty,
    empty_like,
    eye,
    identity,
    ones,
    ones_like,
    zeros,
    zeros_like,
    full,
    full_like,
    diag,
    diagflat,
    tri,
    tril,
    triu,
    vander,
    fromstring,
    array,
)

from numojo.routines import indexing
from numojo.routines.indexing import `where`, compress, take_along_axis


from numojo.routines import manipulation
from numojo.routines.manipulation import (
    ndim,
    shape,
    size,
    reshape,
    ravel,
    transpose,
    broadcast_to,
    flip,
)

from numojo.routines import random

from numojo.routines import sorting
from numojo.routines.sorting import sort, argsort

from numojo.routines import searching
from numojo.routines.searching import argmax, argmin

# ===----------------------------------------------------------------------=== #
# Alias for users
# For ease of use, the name of the types may not follow the Mojo convention,
# e.g., lower case can also be used for comptime of structs.
# ===----------------------------------------------------------------------=== #
