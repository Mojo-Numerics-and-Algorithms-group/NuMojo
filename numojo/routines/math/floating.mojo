# ===----------------------------------------------------------------------=== #
# NuMojo: Floating-point routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Floating-point routines for NuMojo (numojo.routines.math.floating)
---------------------------------------------------------------------
Implements floating-point specific helpers on NDArrays, such as `copysign`.
"""

import std.math as math

from numojo.routines import HostExecutor
from numojo.core.ndarray import NDArray

# ===------------------------------------------------------------------------===#
# Sign Copy
# ===------------------------------------------------------------------------===#


def copysign[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Copy the sign of one array onto another.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Raises:
        Error if shape of `array1` and `array2` do not match.

    Returns:
        A NDArray with the magnitude of `array2` and the sign of `array1`.
    """
    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return math.copysign(simd1, simd2)
    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)
