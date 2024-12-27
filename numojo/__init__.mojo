"""
NuMojo is a library for numerical computing in Mojo 🔥 similar to NumPy, SciPy in Python.
"""

alias __version__ = "V0.2"

# ===----------------------------------------------------------------------=== #
# Alias for users
# For ease of use, the name of the types may not follow the Mojo convention,
# e.g., lower case can also be used for alias of structs.
# ===----------------------------------------------------------------------=== #

alias idx = numojo.core.index.Idx

# ===----------------------------------------------------------------------=== #
# Import core types
# ===----------------------------------------------------------------------=== #

from numojo.core.ndarray import NDArray
from numojo.core.complex_ndarray import ComplexNDArray
from numojo.core._complex_dtype import CDType, ComplexSIMD
from numojo.core.ndshape import NDArrayShape, Shape
from numojo.core.index import Idx
from numojo.core.datatypes import (
    i8,
    i16,
    i32,
    i64,
    isize,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    ci8,
    ci16,
    ci32,
    ci64,
    cu8,
    cu16,
    cu32,
    cu64,
    cf16,
    cf32,
    cf64,
)

# ===----------------------------------------------------------------------=== #
# Import routines and objects
# ===----------------------------------------------------------------------=== #

# Objects
from numojo.routines.constants import Constants

alias pi = numojo.routines.constants.Constants.pi
alias e = numojo.routines.constants.Constants.e
alias c = numojo.routines.constants.Constants.c

# Functions
# TODO Make explicit imports of each individual function in future
# to avoid polluting the root namespace.
from numojo.routines import io
from numojo.routines.io import loadtxt, savetxt, format_float_scientific

from numojo.routines import linalg

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
from numojo.routines.math import maxT, minT, amin, amax, mimimum, maximum
from numojo.routines.math import copysign
from numojo.routines.math import acosh, asinh, atanh, cosh, sinh, tanh
from numojo.routines.math import cbrt, rsqrt, sqrt, scalb
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
    acos,
    asin,
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
    meanall,
    max,
    min,
    cummean,
    mode,
    median,
    cumpvariance,
    cumvariance,
    cumpstdev,
    cumstdev,
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
from numojo.routines.indexing import where

from numojo.routines import manipulation
from numojo.routines.manipulation import (
    ndim,
    shape,
    size,
    reshape,
    ravel,
    transpose,
    flip,
)

from numojo.routines import random

from numojo.routines import sorting
from numojo.routines.sorting import sort, argsort

from numojo.routines import searching
from numojo.routines.searching import argmax, argmin
