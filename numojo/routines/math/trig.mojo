# ===----------------------------------------------------------------------=== #
# NuMojo: Trigonometric routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Trigonometric routines for NuMojo (numojo.routines.math.trig).

Implements trigonometric and inverse trigonometric functions for NDArrays and Matrices.
"""

import std.math as math

from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
from numojo.core.matrix.base import _arithmetic_func_matrix_to_matrix
from numojo.routines import HostExecutor
from numojo.routines.math.misc import sqrt
from numojo.routines.math.arithmetic import fma

# ===------------------------------------------------------------------------===#
# Inverse Trig (NDArray)
# ===------------------------------------------------------------------------===#


def acos[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise acos of `array`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.acos(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


def asin[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise asin of `array`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.asin(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


def atan[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise atan of `array`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.atan(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


def atan2[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse tangent with two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise atan2 of `array1` and `array2`.

    References:
        https://en.wikipedia.org/wiki/Atan2.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.atan2(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)


# ===------------------------------------------------------------------------===#
# Inverse Trig (Matrix)
# ===------------------------------------------------------------------------===#


def arccos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise acos of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.acos(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def acos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise acos of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.acos(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def arcsin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse sine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise asin of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.asin(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def asin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse sine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise asin of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.asin(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def arctan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse tangent.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise atan of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.atan(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def atan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse tangent.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise atan of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.atan(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


# ===------------------------------------------------------------------------===#
# Trig (NDArray)
# ===------------------------------------------------------------------------===#


def cos[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise cos of `array`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.cos(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


def sin[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise sin of `array`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.sin(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


def tan[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise tan of `array`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.tan(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


# ===------------------------------------------------------------------------===#
# Trig (Matrix)
# ===------------------------------------------------------------------------===#


def cos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise cos of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.cos(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def sin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply sine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise sin of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.sin(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def tan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply tangent.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise tan of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.tan(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


# ===------------------------------------------------------------------------===#
# Hypotenuse
# ===------------------------------------------------------------------------===#


def hypot[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hypotenuse calculation to two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise hypotenuse of `array1` and `array2`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.hypot(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)


def hypot_fma[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hypotenuse calculation using fused multiply-add.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise hypotenuse of `array1` and `array2`.
    """
    var array2_squared = fma[dtype](array2, array2, SIMD[dtype, 1](0))
    return sqrt[dtype](fma[dtype](array1, array1, array2_squared))
