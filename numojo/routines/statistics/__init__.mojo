# ===----------------------------------------------------------------------=== #
# NuMojo: Statistical functions
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Statistics routines (numojo.routines.statistics).
=================================================
Statistical functions including averages, dispersion measures, and order statistics.

Exports
-------
- `mean`: Arithmetic mean of array elements.
- `median`: Middle value of array elements.
- `mode`: Most frequent element.
- `min`, `max`: Minimum and maximum values.
- `variance`: Variance of array elements.
- `stddev`: Standard deviation of array elements.
- `nansum`, `nanmean`, `nanmax`, `nanmin`, `nanvar`, `nanstd`, `nanmedian`: NaN-aware reductions.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .averages import (
    max,
    mean,
    median,
    min,
    mode,
    stddev,
    variance,
)
from .nanfunctions import (
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanstd,
    nansum,
    nanvar,
)
