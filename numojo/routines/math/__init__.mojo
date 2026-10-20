# ===----------------------------------------------------------------------=== #
# NuMojo: Math routines and operations
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Math routines (numojo.routines.math).
=====================================
Arithmetic, trigonometric, hyperbolic, exponential, and utility mathematical operations for arrays.

Exports
-------
Arithmetic: `add`, `sub`, `mul`, `div`, `floor_div`, `mod`, `remainder`, `fma`.

Trigonometric: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `atan2`, `hypot`, `hypot_fma`.

Hyperbolic: `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`.

Exponential: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`.

Extrema: `max`, `min`, `maximum`, `minimum`.

Floating point: `copysign`, `nextafter`, `scalb`, `cbrt`, `sqrt`, `rsqrt`, `clip`.

Rounding: `round`, `roundeven`, `trunc`, `ceil`, `floor`.

Absolute value: `abs`.

Differences and aggregation: `diff`, `gradient`, `ediff1d`, `sum`, `cumsum`, `prod`, `cumprod`.

Integration: `trapezoid`, `cumtrapz`.

Signal processing: `convolve`, `interp`.

Element-wise: `sign`, `square`, `power`/`pow`, `negative`, `positive`, `fmax`,
`fmin`, `fmod`, `rint`.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
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
    ediff1d,
    gradient,
)
from .elementwise import (
    fmax,
    fmin,
    fmod,
    negative,
    positive,
    power,
    power as pow,
    rint,
    sign,
    square,
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
    ptp,
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
from .integrate import (
    cumtrapz,
    trapezoid,
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
from .signal import (
    convolve,
    interp,
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
