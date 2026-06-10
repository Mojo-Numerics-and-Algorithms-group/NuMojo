# ===----------------------------------------------------------------------=== #
# NuMojo: Routines module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Routines module (numojo.routines).

This modules groups NumPy-like functionality by topic (math, linalg, statistics,
creation, manipulation, etc.).

What this `__init__` exports:
- Topic namespaces (e.g. `numojo.routines.math`, `numojo.routines.linalg`, ...)
- A curated set of convenience functions at `numojo.routines.*` for ergonomic
  internal use and power users.

Notes / conventions:
- Public user-facing imports should generally come from the top-level `numojo`
  module (or `numojo.prelude`) rather than importing deeply from this package.
- Keep this initializer predictable: add new re-exports only when they are
  stable and widely used.
"""

import numojo.routines.linalg
import numojo.routines.logic
import numojo.routines.math
import numojo.routines.statistics
import numojo.routines.bitwise
import numojo.routines.creation
import numojo.routines.indexing
import numojo.routines.manipulation
import numojo.routines.random
import numojo.routines.sorting
import numojo.routines.searching
import numojo.routines.functional
import numojo.routines.operations

from numojo.routines.io import (
    loadtxt,
    savetxt,
    load,
    save,
    set_printoptions,
)

from numojo.routines.linalg.misc import diagonal

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

from numojo.routines.math import (
    add,
    sub,
    mod,
    mul,
    div,
    floor_div,
    fma,
    remainder,
    gradient,
    diff,
    exp,
    exp2,
    expm1,
    log,
    log2,
    log10,
    log1p,
    max,
    min,
    minimum,
    maximum,
    copysign,
    arccosh,
    acosh,
    arcsinh,
    asinh,
    arctanh,
    atanh,
    cosh,
    sinh,
    tanh,
    cbrt,
    clip,
    rsqrt,
    sqrt,
    scalb,
    prod,
    cumprod,
    tabs,
    tfloor,
    tceil,
    ttrunc,
    tround,
    roundeven,
    nextafter,
    sum,
    cumsum,
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

from numojo.routines.statistics import mean, mode, median, variance, std_dev

from numojo.routines.bitwise import invert

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

from numojo.routines.indexing import `where`, compress, take_along_axis

from numojo.routines.functional import (
    apply_along_axis_reduce,
    apply_along_axis_reduce_to_int,
    apply_along_axis_reduce_with_dtype,
    apply_along_axis_preserve,
    apply_along_axis_inplace,
    apply_along_axis_indices,
)

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

from numojo.routines.sorting import sort, argsort
from numojo.routines.searching import argmax, argmin

from numojo.routines.operations import (
    HostExecutor,
    UnaryKernel,
    BinaryKernel,
    UnaryPredicate,
    BinaryPredicate,
    BinaryIntKernel,
    TernaryKernel,
)
