# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

# ===------------------------------------------------------------------------===#
# Miscellaneous mathematical functions
# ===------------------------------------------------------------------------===#

from algorithm import parallelize, vectorize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
import builtin.math as builtin_math
import stdlib.math.math as stdlib_math
from sys import simdwidthof
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
    return backend().math_func_1_array_in_one_array_out[
        dtype, stdlib_math.cbrt
    ](array)


fn clip[
    dtype: DType, //
](a: NDArray[dtype], a_min: Scalar[dtype], a_max: Scalar[dtype]) -> NDArray[
    dtype
]:
    """
    Limit the values in an array between [a_min, a_max].
    If a_min is greater than a_max, the value is equal to a_max.

    Parameters:
        dtype: The data type.

    Args:
        a: A array.
        a_min: The minimum value.
        a_max: The maximum value.

    Returns:
        An array with the clipped values.
    """

    var res = a  # Deep copy of the array

    for i in range(res.size):
        if res._buf.ptr[i] < a_min:
            res._buf.ptr[i] = a_min
        if res._buf.ptr[i] > a_max:
            res._buf.ptr[i] = a_max

    return res


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
    return stdlib_math.sqrt(SIMD.__truediv__(1, value))


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
    Element-wise square root of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to NDArray**(1/2).
    """
    return backend().math_func_1_array_in_one_array_out[
        dtype, stdlib_math.sqrt
    ](array)


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
    return backend().math_func_2_array_in_one_array_out[
        dtype, stdlib_math.scalb
    ](array1, array2)
