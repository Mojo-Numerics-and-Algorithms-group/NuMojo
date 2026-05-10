# ===----------------------------------------------------------------------=== #
# NuMojo: Statistics routines submodule
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Statistics routines for NuMojo (numojo.routines.statistics).

Aggregates averages, modes, and dispersion helpers for NDArrays and Matrices.
"""

from .averages import (
    mean,
    max,
    min,
    mode,
    median,
    variance,
    stddev,
)
