# ===----------------------------------------------------------------------=== #
# NuMojo: Hyperbolic routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Hyperbolic routines for NuMojo (numojo.routines.math.hyper)
--------------------------------------------------------------
Implements hyperbolic and inverse hyperbolic functions for NDArrays and Matrices.
"""

import std.math as math

from numojo.routines import HostExecutor
from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
from numojo.core.matrix.base import _arithmetic_func_matrix_to_matrix

# TODO: add dtype in backends and pass it here.

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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.acosh(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.asinh(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.atanh(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.acosh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.acosh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.asinh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.asinh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.atanh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.atanh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.cosh(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.sinh(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.tanh(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.cosh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def sinh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic sin.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise sinh of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.sinh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)


def tanh[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Apply hyperbolic tan.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise tanh of `A`.
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.tanh(simd)

    return _arithmetic_func_matrix_to_matrix[dtype, _kernel](A)
