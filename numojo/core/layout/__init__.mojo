# ===----------------------------------------------------------------------=== #
# NuMojo: Layout Module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Layout (numojo.core.layout)

Layout metadata types used by NuMojo arrays and matrices (shape, strides, and flags).
"""

from .ndshape import NDArrayShape
from .ndstrides import NDArrayStrides
from .flags import Flags
