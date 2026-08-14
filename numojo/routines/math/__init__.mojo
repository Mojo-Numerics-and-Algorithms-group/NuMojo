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
# ===----------------------------------------------------------------------===#
# Local
# ===----------------------------------------------------------------------===#
from .arithmetic import (
    add,
    div,
    floor_div,
    fma,
    mod,
    mul,
    remainder,
    sub,
)
from .differences import (
    diff,
    gradient,
)
from .exponents import (
    exp,
    exp2,
    expm1,
    log,
    log10,
    log1p,
    log2,
)
from .extrema import (
    max,
    maximum,
    min,
    minimum,
)
from .floating import copysign
from .hyper import (
    acosh,
    arccosh,
    arcsinh,
    arctanh,
    asinh,
    atanh,
    cosh,
    sinh,
    tanh,
)
from .misc import (
    cbrt,
    clip,
    rsqrt,
    scalb,
    sqrt,
)
from .products import (
    cumprod,
    prod,
)
from .rounding import (
    nextafter,
    round,
    roundeven,
    tabs,
    tceil,
    tfloor,
    tround,
    ttrunc,
)
from .sums import (
    cumsum,
    sum,
)
from .trig import (
    acos,
    arccos,
    arcsin,
    arctan,
    asin,
    atan,
    atan2,
    cos,
    hypot,
    hypot_fma,
    sin,
    tan,
)
