# ===----------------------------------------------------------------------=== #
# NuMojo: Logic and comparison operations
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Logic routines (numojo.routines.logic).
======================================
Comparison operations, logical operators, and truth value evaluations for arrays.

Exports
-------
- Comparison: `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, `allclose`, `array_equal`, `isclose`.
- Contents: `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`.
- Logical: `logical_and`, `logical_or`, `logical_xor`, `logical_not`.
- Truth: `any`, `all`.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
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
