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
# ===----------------------------------------------------------------------===#
# Local
# ===----------------------------------------------------------------------===#
from .comparison import (
    allclose,
    array_equal,
    equal,
    greater,
    greater_equal,
    isclose,
    less,
    less_equal,
    not_equal,
)
from .contents import (
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
)
from .logical_ops import (
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)
from .truth import (
    all,
    any,
)
