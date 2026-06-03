# ===----------------------------------------------------------------------=== #
# NuMojo: Math routines submodule
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Math routines for NuMojo (numojo.routines.math).

Aggregates arithmetic, trigonometric, hyperbolic, and utility routines for NDArrays and Matrices.
"""

from numojo.routines.math.arithmetic import add, sub, mod, mul, div, floor_div, fma, remainder
from numojo.routines.math.differences import gradient, diff
from numojo.routines.math.exponents import exp, exp2, expm1, log, log2, log10, log1p
from numojo.routines.math.extrema import max, min, minimum, maximum
from numojo.routines.math.floating import copysign
from numojo.routines.math.hyper import (
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
from numojo.routines.math.misc import cbrt, clip, rsqrt, sqrt, scalb
from numojo.routines.math.products import prod, cumprod
from numojo.routines.math.rounding import (
    round,
    tabs,
    tfloor,
    tceil,
    ttrunc,
    tround,
    roundeven,
    nextafter,
)
from numojo.routines.math.sums import sum, cumsum
from numojo.routines.math.trig import (
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
