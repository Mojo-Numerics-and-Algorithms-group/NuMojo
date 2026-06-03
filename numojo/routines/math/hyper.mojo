# ===----------------------------------------------------------------------=== #
# NuMojo: Hyperbolic routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Hyperbolic routines for NuMojo (numojo.routines.math.hyper).

Implements hyperbolic and inverse hyperbolic functions for NDArrays and Matrices.
"""

import std.math as math

from numojo.routines import HostExecutor, UnaryKernel
from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
from numojo.core.matrix.base import _arithmetic_func_matrix_to_matrix

# TODO: add dtype in backends and pass it here.

# ===------------------------------------------------------------------------===#
# Unary kernel functors
# ===------------------------------------------------------------------------===#


struct _Acosh(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.acosh(x)
        else:
            return x


struct _Asinh(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.asinh(x)
        else:
            return x


struct _Atanh(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.atanh(x)
        else:
            return x


struct _Cosh(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.cosh(x)
        else:
            return x


struct _Sinh(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.sinh(x)
        else:
            return x


struct _Tanh(UnaryKernel):
    @staticmethod
    def apply[type: DType, simd_w: Int](
        x: SIMD[type, simd_w]
    ) -> SIMD[type, simd_w]:
        comptime if type.is_floating_point():
            return math.tanh(x)
        else:
            return x

# ===------------------------------------------------------------------------===#
# Inverse Hyperbolic Trig (NDArray)
# ===------------------------------------------------------------------------===#


def acosh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse hyperbolic cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise acosh of `array`.
    """
    return HostExecutor.apply_unary[dtype, _Acosh](array)


def asinh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse hyperbolic sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise asinh of `array`.
    """
    return HostExecutor.apply_unary[dtype, _Asinh](array)


def atanh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse hyperbolic tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise atanh of `array`.
    """
    return HostExecutor.apply_unary[dtype, _Atanh](array)


# ===------------------------------------------------------------------------===#
# Inverse Hyperbolic Trig (Matrix)
# ===------------------------------------------------------------------------===#


def arccosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic cosine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic cosine (arccosh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Acosh](A)


def acosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic cosine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic cosine (acosh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Acosh](A)


def arcsinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic sine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic sine (arcsinh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Asinh](A)


def asinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic sine element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic sine (asinh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Asinh](A)


def arctanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic tangent element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic tangent (arctanh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Atanh](A)


def atanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse hyperbolic tangent element-wise to a Matrix.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise inverse hyperbolic tangent (atanh) of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Atanh](A)


# ===------------------------------------------------------------------------===#
# Hyperbolic Trig (NDArray)
# ===------------------------------------------------------------------------===#


def cosh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hyperbolic cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise cosh of `array`.
    """
    return HostExecutor.apply_unary[dtype, _Cosh](array)


def sinh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hyperbolic sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise sinh of `array`.
    """
    return HostExecutor.apply_unary[dtype, _Sinh](array)


def tanh[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hyperbolic tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise tanh of `array`.
    """
    return HostExecutor.apply_unary[dtype, _Tanh](array)


# ===------------------------------------------------------------------------===#
# Hyperbolic Trig (Matrix)
# ===------------------------------------------------------------------------===#


def cosh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise cosh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Cosh](A)


def sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic sin.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise sinh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Sinh](A)


def tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic tan.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise tanh of `A`.
    """
    return _arithmetic_func_matrix_to_matrix[dtype, _Tanh](A)
