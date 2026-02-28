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

import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from utils import Variant

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray
from numojo.core.layout import NDArrayShape
from numojo.core.dtype.utility import is_inttype, is_floattype, is_booltype


fn invert[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype] where (
    dtype.is_integral() or dtype == DType.bool
):
    """
    Element-wise invert of an array.

    Constraints:
        The array must be either a boolean or integral array.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to the bitwise inversion of array.
    """
    comptime assert (
        is_inttype[dtype]() or is_booltype[dtype]()
    ), "Only Bools and integral types can be inverted."

    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__invert__](
        array
    )
