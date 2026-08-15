# ===----------------------------------------------------------------------=== #
# NuMojo: Truth testing
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Truth value testing (numojo.routines.logic.truth)
----------------------------------------------------
This module implements the truth value testing functions, such as `all` and `any`, for `NDArray`.
"""

from std.algorithm import vectorize
from max.algorithm import parallelize
from std.sys import simd_width_of

from numojo.core.ndarray import NDArray

# TODO: Add all and any algorithm to backend.

# ===----------------------------------------------------------------------=== #
# Truth operations for NDArray
# ===----------------------------------------------------------------------=== #


def all(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    Checks whether all elements of the array evaluate to True.

    Args:
        array: Input NDArray (DType.bool).

    Returns:
        True if all elements of the array evaluate to True, False if not.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.truth import all

        var a = arange[i32](24).reshape(Shape(2, 3, 4))
        var result = all(a > 5) # outputs False
        ```
    """
    var result = Scalar[DType.bool](True)
    comptime opt_nelts: Int = simd_width_of[DType.bool]()

    def closure[simd_width: Int](idx: Int) {mut result, imm array} -> None:
        var simd_data = array.unsafe_load[width=simd_width](idx)
        result = (result & simd_data).reduce_and()

    vectorize[opt_nelts](array.size, closure)
    return result


def any(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    Checks whether any element of the array evaluate to True.

    Args:
        array: Input NDArray (DType.bool).

    Returns:
        True if any element of the array evaluate to True, False if not.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.truth import any

        var a = arange[i32](24).reshape(Shape(2, 3, 4))
        var result = any(a > 5) # outputs True
        ```
    """
    var result = Scalar[DType.bool](False)
    comptime opt_nelts: Int = simd_width_of[DType.bool]()

    def closure[simd_width: Int](idx: Int) {mut result, imm array} -> None:
        var simd_data = array.unsafe_load[width=simd_width](idx)
        result = (result | simd_data).reduce_or()

    vectorize[opt_nelts](array.size, closure)
    return result


# ===----------------------------------------------------------------------=== #
# Truth operations for NDArray
# ===----------------------------------------------------------------------=== #
