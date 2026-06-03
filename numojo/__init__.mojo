# ===----------------------------------------------------------------------=== #
# NuMojo: A numerical computation library for Mojo.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
NuMojo Top-Level Package (`numojo`)
==================================

Central public surface for NuMojo that exposes the primary containers, dtype helpers, common errors,
and a curated set of NumPy-inspired routines.

Exports
-------
Core container types:
- `Matrix` and `NDArray`
- `Shape` / `NDArrayShape`, `Strides` / `NDArrayStrides`

Core utilities:
- dtype aliases (`f32`, `f64`, `i32`, `i64`, ...) along with their complex counterparts and SIMD helpers
- shared error types such as `NumojoError`, `IndexError`, and `ShapeError`

Routines
--------
Re-exports a carefully selected subset of functionality from `numojo.routines` covering creation,
manipulation, math, logic, statistics, I/O, and related domains so users have a stable convenience import.

Notes
-----
- This module is intended to provide a stable import surface for users.
- Internal code should prefer importing directly from the canonical submodules/packages
  (`numojo.core.matrix`, `numojo.core.layout`, `numojo.routines.math`, etc.) rather than relying on
  extensive top-level re-exports.
- Public APIs in this module adhere to the Mojo docstring style guide to keep documentation precise
  and predictable for users.

FORMAT FOR DOCSTRING (See "Mojo docstring style guide" for more information)
1. Description *
2. Parameters *
3. Args *
4. Raises *
5. Constraints *
6. Returns *
7. Notes
9. References
10. Examples *
(Items marked with * are defined by the Mojo docstring style guide.)
"""

comptime __version__: String = "V0.9.0"

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
from numojo.core.accelerator.device import Device, cpu, cuda, mps, rocm

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

comptime pi = Constants.pi
comptime e = Constants.e
comptime c = Constants.c

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
from numojo.routines.math import gradient, diff
from numojo.routines.math import exp, exp2, expm1, log, log2, log10, log1p
from numojo.routines.math import (
    max,
    min,
    minimum,
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
