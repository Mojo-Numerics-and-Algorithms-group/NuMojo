# ===----------------------------------------------------------------------=== #
# NuMojo: Routines module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Routines (numojo.routines).
===========================
NumPy-like functionality grouped by topic (math, linalg, statistics, creation,
manipulation, etc.).

Exports
-------
- Topic namespaces (e.g. `numojo.routines.math`, `numojo.routines.linalg`, ...).
- A curated set of convenience functions at `numojo.routines.*` for ergonomic
  internal use and power users.

Notes
-----
- Public user-facing imports should generally come from the top-level `numojo`
  module (or `numojo.prelude`) rather than importing deeply from this package.
- Keep this initializer predictable: add new re-exports only when they are
  stable and widely used.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from . import (
    bitwise,
    creation,
    functional,
    indexing,
    linalg,
    logic,
    manipulation,
    math,
    operations,
    random,
    searching,
    sorting,
    statistics,
)
from .bitwise import invert
from .creation import (
    arange,
    array,
    diag,
    diagflat,
    empty,
    empty_like,
    eye,
    fromstring,
    full,
    full_like,
    geomspace,
    identity,
    linspace,
    logspace,
    ones,
    ones_like,
    tri,
    tril,
    triu,
    vander,
    zeros,
    zeros_like,
)
from .functional import (
    apply_along_axis_indices,
    apply_along_axis_inplace,
    apply_along_axis_preserve,
    apply_along_axis_reduce,
    apply_along_axis_reduce_to_int,
    apply_along_axis_reduce_with_dtype,
)
from .indexing import (
    compress,
    fancy_index,
    flatnonzero,
    nonzero,
    ravel_multi_index,
    take,
    take_along_axis,
    unravel_index,
    `where`,
)
from .io import (
    load,
    loadtxt,
    save,
    savetxt,
    set_printoptions,
)
from .linalg.misc import diagonal
from .logic import (
    all,
    any,
    equal,
    greater,
    greater_equal,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    not_equal,
)
from .manipulation import (
    broadcast_to,
    column_stack,
    concatenate,
    flip,
    hstack,
    ndim,
    ravel,
    reshape,
    row_stack,
    shape,
    size,
    transpose,
    vstack,
)
from .math import (
    acos,
    acosh,
    add,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    cbrt,
    clip,
    copysign,
    cos,
    cosh,
    cumprod,
    cumsum,
    diff,
    div,
    exp,
    exp2,
    expm1,
    floor_div,
    fma,
    gradient,
    hypot,
    hypot_fma,
    log,
    log10,
    log1p,
    log2,
    max,
    maximum,
    min,
    minimum,
    mod,
    mul,
    nextafter,
    prod,
    remainder,
    roundeven,
    rsqrt,
    scalb,
    sin,
    sinh,
    sqrt,
    sub,
    sum,
    tabs,
    tan,
    tanh,
    tceil,
    tfloor,
    tround,
    ttrunc,
)
from .operations import HostExecutor
from .searching import (
    argmax,
    argmin,
)
from .sorting import (
    argsort,
    sort,
)
from .statistics import (
    mean,
    median,
    mode,
    stddev,
    variance,
)
