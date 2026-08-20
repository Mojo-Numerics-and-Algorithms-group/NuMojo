# ===----------------------------------------------------------------------=== #
# NuMojo: Floating-point routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Floating (numojo.routines.math.floating).
=========================================
Floating-point specific operations for NDArrays.

Implements floating-point helper functions such as `copysign` for element-wise
sign manipulation.

Exports
-------
- `copysign`: Copy sign from one array to another.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as math

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor

# ===----------------------------------------------------------------------=== #
# Sign Copy
# ===----------------------------------------------------------------------=== #


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
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return math.copysign(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)
