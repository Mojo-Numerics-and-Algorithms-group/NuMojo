# ===----------------------------------------------------------------------=== #
# NuMojo: Logic routines submodule
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Logic routines for NuMojo (numojo.routines.logic).

This module provides a collection of logic routines for numerical computations, including comparison operations, content checks, and truth evaluations.
"""
from .comparison import (
    greater,
    greater_equal,
    less,
    less_equal,
    equal,
    not_equal,
)
from .contents import isinf, isfinite, isnan
from .truth import any, all
