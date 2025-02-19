# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

# ===------------------------------------------------------------------------===#
# Miscellaneous mathematical functions
# ===------------------------------------------------------------------------===#

import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from utils import Variant

import numojo.core._math_funcs as _mf
from numojo.core.ndarray import NDArray


fn cbrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise cuberoot of NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to NDArray**(1/3).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.cbrt](array)


fn _mt_rsqrt[
    dtype: DType, simd_width: Int
](value: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """
    Element-wise reciprocal squareroot of SIMD.
    Parameters:
        dtype: The element type.
        simd_width: The SIMD width.
    Args:
        value: A SIMD vector.
    Returns:
        A SIMD equal to 1/SIMD**(1/2).
    """
    return math.sqrt(SIMD.__truediv__(1, value))


fn rsqrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise reciprocal squareroot of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to 1/NDArray**(1/2).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, _mt_rsqrt](array)


fn sqrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise squareroot of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to NDArray**(1/2).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, math.sqrt](array)


# this is a temporary doc, write a more explanatory one
fn scalb[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Calculate the scalb of array1 and array2.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray with the shape of `NDArray` with values equal to the negative one plus
        e to the power of the value in the original NDArray at each position.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, math.scalb](
        array1, array2
    )
