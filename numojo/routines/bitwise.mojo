# ===----------------------------------------------------------------------=== #
# NuMojo: Routines module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Bit-wise operations module (`numojo.routines.bitwise`)

This module implements bit-wise operations on NDArrays, such as bitwise AND, OR, XOR, and NOT (invert).
"""

from numojo.routines import HostExecutor
from numojo.core.ndarray import NDArray

# ===------------------------------------------------------------------------===#
# Bitwise operations
# ===------------------------------------------------------------------------===#


def invert[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[dtype] where (
    dtype.is_integral() or dtype == DType.bool
):
    """
    Element-wise invert of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Constraints:
        The array must be either a boolean or integral array.

    Returns:
        A NDArray equal to the bitwise inversion of array.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.bitwise import invert

        var arr1 = nm.array[nm.i8]([1, 2, 3], shape=[3])
        var result1 = invert(arr1) # result1 is [-2, -3, -4]

        var arr2 = nm.array[nm.boolean]([True, False, True], shape=[3])
        var result2 = invert(arr2) # result2 is [false, true, false
        ```
    """
    return HostExecutor.apply_unary[dtype, SIMD.__invert__](array)
