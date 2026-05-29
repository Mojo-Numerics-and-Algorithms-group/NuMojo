# ===----------------------------------------------------------------------=== #
# NuMojo: Routines module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Routines module (numojo.routines)

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

import .linalg
import .logic
import .math
import .statistics
import .bitwise
import .creation
import .indexing
import .manipulation
import .random
import .sorting
import .searching
import .functional
import .operations

from .io import (
    loadtxt,
    savetxt,
    load,
    save,
    set_printoptions,
)

from .linalg.misc import diagonal

from .logic import (
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

from .math import (
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

from .statistics import mean, mode, median, variance

from .bitwise import invert

from .creation import (
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

from .indexing import `where`, compress, take_along_axis

from .functional import (
    apply_along_axis_reduce,
    apply_along_axis_reduce_to_int,
    apply_along_axis_reduce_with_dtype,
    apply_along_axis_preserve,
    apply_along_axis_inplace,
    apply_along_axis_indices,
)

from .manipulation import (
    ndim,
    shape,
    size,
    reshape,
    ravel,
    transpose,
    broadcast_to,
    flip,
)

from .sorting import sort, argsort
from .searching import argmax, argmin

from .operations import HostExecutor
