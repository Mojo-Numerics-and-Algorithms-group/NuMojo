# ===----------------------------------------------------------------------=== #
# NuMojo: Array layout metadata
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Layout (numojo.core.layout).
============================

Layout metadata types used by NuMojo arrays and matrices (shape, strides, and flags).

Exports
-------
- `NDArrayShape`: Represents the shape (dimensions) of an array.
- `NDArrayStrides`: Represents the strides (memory layout) of an array.
- `Flags`: Layout flags controlling array behavior.
- `newaxis`: Marker for expanding dimensions.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .array_methods import newaxis
from .flags import Flags
from .ndshape import NDArrayShape
from .ndstrides import NDArrayStrides
