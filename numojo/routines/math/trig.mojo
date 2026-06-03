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
from numojo.routines import HostExecutor, UnaryKernel, BinaryKernel
from numojo.routines.math.misc import sqrt
from numojo.routines.math.arithmetic import fma

# ===------------------------------------------------------------------------===#
# Kernel functors
# ===------------------------------------------------------------------------===#


struct _Sin(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.sin(x)
        else:
            return x


struct _Cos(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.cos(x)
        else:
            return x


struct _Tan(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.tan(x)
        else:
            return x


struct _Asin(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.asin(x)
        else:
            return x


struct _Acos(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.acos(x)
        else:
            return x


struct _Atan(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.atan(x)
        else:
            return x


struct _Atan2(BinaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        a: SIMD[type, simd_w], b: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.atan2(a, b)
        else:
            return a


struct _Hypot(BinaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        a: SIMD[type, simd_w], b: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.hypot(a, b)
        else:
            return a

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
    return HostExecutor.apply_unary[dtype, _Acos](array)


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
    return HostExecutor.apply_unary[dtype, _Asin](array)


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
    return HostExecutor.apply_unary[dtype, _Atan](array)


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
    return HostExecutor.apply_binary[dtype, _Atan2](array1, array2)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Acos](A)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Acos](A)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Asin](A)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Asin](A)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Atan](A)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Atan](A)


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
    return HostExecutor.apply_unary[dtype, _Cos](array)


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
    return HostExecutor.apply_unary[dtype, _Sin](array)


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
    return HostExecutor.apply_unary[dtype, _Tan](array)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Cos](A)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Sin](A)


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
    return _arithmetic_func_matrix_to_matrix[dtype, _Tan](A)


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
    return HostExecutor.apply_binary[dtype, _Hypot](array1, array2)


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
