"""
NuMojo Routines Package (`numojo.routines`)
==========================================

This package groups NumPy-like functionality by topic (math, linalg, statistics,
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

from . import io
from . import linalg
from . import logic
from . import math
from . import statistics
from . import bitwise
from . import creation
from . import indexing
from . import manipulation
from . import random
from . import sorting
from . import searching
from . import functional

from .io import loadtxt, savetxt, load, save, set_printoptions

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
    diff,
    mod,
    mul,
    div,
    floor_div,
    fma,
    remainder,
    gradient,
    trapz,
    exp,
    exp2,
    expm1,
    log,
    ln,
    log2,
    log10,
    log1p,
    max,
    min,
    mimimum,
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

from .statistics import mean, mode, median, variance, std

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

from .functional import apply_along_axis

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
