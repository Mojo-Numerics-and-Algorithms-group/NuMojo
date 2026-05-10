# ===----------------------------------------------------------------------=== #
# NuMojo: Miscellaneous math routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Miscellaneous math routines for NuMojo (numojo.routines.math.misc)
---------------------------------------------------------------------
Implements miscellaneous math helpers on NDArrays, including cube root, clipping, reciprocal square root, square root, and scalb.
"""

from std.algorithm import vectorize
import std.math as builtin_math
import std.math.math as stdlib_math
from std.sys import simd_width_of

from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor


# TODO: Implement same routines for Matrix.
def cbrt[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise cube root of a NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to array**(1/3).
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return stdlib_math.cbrt(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


# ===------------------------------------------------------------------------===#
# Clipping
# ===------------------------------------------------------------------------===#


def clip[
    dtype: DType, //
](
    a: NDArray[dtype], a_min: Scalar[dtype], a_max: Scalar[dtype]
) raises -> NDArray[dtype]:
    """
    Limit values in an array to the range [a_min, a_max].
    If a_min is greater than a_max, values are set to a_max.

    Parameters:
        dtype: The element type.

    Args:
        a: A NDArray.
        a_min: The minimum value.
        a_max: The maximum value.

    Returns:
        A NDArray with the clipped values.
    """

    var result = a.contiguous()  # Owned, C-contiguous copy

    for i in range(result.size):
        if result._buf.ptr[i] < a_min:
            result._buf.ptr[i] = a_min
        if result._buf.ptr[i] > a_max:
            result._buf.ptr[i] = a_max

    return result^


# ===------------------------------------------------------------------------===#
# Reciprocal Square Root
# ===------------------------------------------------------------------------===#


def _mt_rsqrt[
    dtype: DType, simd_width: Int
](value: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """
    Element-wise reciprocal square root of SIMD.

    Parameters:
        dtype: The element type.
        simd_width: The SIMD width.

    Args:
        value: A SIMD vector.

    Returns:
        A SIMD equal to 1/SIMD**(1/2).
    """
    return stdlib_math.sqrt(SIMD.__truediv__(SIMD[dtype, simd_width](1), value))


def rsqrt[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise reciprocal square root of NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to 1/NDArray**(1/2).
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return _mt_rsqrt(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


def sqrt[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise square root of a NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to NDArray**(1/2).
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return stdlib_math.sqrt(simd)

    return HostExecutor.apply_unary[dtype, _kernel](array)


# ===------------------------------------------------------------------------===#
# Scaling
# ===------------------------------------------------------------------------===#


def scalb[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply scalb element-wise to two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray with values equal to scalb(array1, array2).
    """

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return stdlib_math.scalb(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)
