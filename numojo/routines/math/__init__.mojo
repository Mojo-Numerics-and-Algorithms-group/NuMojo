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

from .arithmetic import add, sub, mod, mul, div, floor_div, fma, remainder
from .differences import gradient, diff
from .exponents import exp, exp2, expm1, log, log2, log10, log1p
from .extrema import max, min, minimum, maximum
from .floating import copysign
from .hyper import (
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
from .misc import cbrt, clip, rsqrt, sqrt, scalb
from .products import prod, cumprod
from .rounding import (
    round,
    tabs,
    tfloor,
    tceil,
    ttrunc,
    tround,
    roundeven,
    nextafter,
)
from .sums import sum, cumsum
from .trig import (
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
