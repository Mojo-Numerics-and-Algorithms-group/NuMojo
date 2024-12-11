# ===------------------------------------------------------------------------===#

# Extrema finding
# ===------------------------------------------------------------------------===#

"""
TODO: 
1) Add support for axis parameter.  
2) Currently, constrained is crashing mojo, so commented it out and added raise Error. Check later.
3) Relax constrained[] to let user get whatever output they want, but make a warning instead.
"""

import math
from algorithm import vectorize
from sys import simdwidthof
from collections.optional import Optional

from numojo.core.ndarray import NDArray
from numojo.routines.sorting import binary_sort


# for max and min, I can later change to the latest reduce.max, reduce.min()
fn maxT[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """
    Maximum value of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.
    Returns:
        The maximum of all of the member values of array as a SIMD Value of `dtype`.
    """
    # TODO: Test this
    alias width = simdwidthof[dtype]()
    var max_value = NDArray[dtype](NDArrayShape(width))
    for i in range(width):
        max_value.__setitem__(i, array[0])
    # var max_value: SIMD[ dtype, width] = SIMD[ dtype, width](array[0])

    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        max_value.store[width=simd_width](
            0,
            max(
                max_value.load[width=simd_width](0),
                array.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_max, width](array.num_elements())

    var result: Scalar[dtype] = Scalar[dtype](max_value.get(0))
    for i in range(max_value.__len__()):
        if max_value.get(i) > result:
            result = max_value.get(i)
    return result


fn minT[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """
    Minimum value of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The minimum of all of the member values of array as a SIMD Value of `dtype`.
    """
    alias width = simdwidthof[dtype]()
    var min_value = NDArray[dtype](NDArrayShape(width))
    for i in range(width):
        min_value.__setitem__(i, array[0])

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        min_value.store[width=simd_width](
            0,
            min(
                min_value.load[width=simd_width](0),
                array.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_min, width](array.num_elements())

    var result: Scalar[dtype] = Scalar[dtype](min_value.get(0))
    for i in range(min_value.__len__()):
        if min_value.get(i) < result:
            result = min_value.get(i)

    return result


# this roughly seems to be just an alias for min in numpy
fn amin[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """
    Minimum value of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An array.
    Returns:
        The minimum of all of the member values of array as a SIMD Value of `dtype`.
    """
    return minT[dtype](array)


# this roughly seems to be just an alias for max in numpy
fn amax[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """
    Maximum value of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A array.
    Returns:
        The maximum of all of the member values of array as a SIMD Value of `dtype`.
    """
    return maxT[dtype](array)


fn mimimum[
    dtype: DType = DType.float64
](s1: SIMD[dtype, 1], s2: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    """
    Minimum value of two SIMD values.

    Parameters:
         dtype: The element type.

    Args:
        s1: A SIMD Value.
        s2: A SIMD Value.
    Returns:
        The minimum of the two SIMD Values as a SIMD Value of `dtype`.
    """
    return min(s1, s2)


fn maximum[
    dtype: DType = DType.float64
](s1: SIMD[dtype, 1], s2: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    """
    Maximum value of two SIMD values.

    Parameters:
         dtype: The element type.

    Args:
        s1: A SIMD Value.
        s2: A SIMD Value.
    Returns:
        The maximum of the two SIMD Values as a SIMD Value of `dtype`.
    """
    return max(s1, s2)


fn minimum[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element wise minimum of two arrays.

    Parameters:
         dtype: The element type.

    Args:
        array1: An array.
        array2: An array.
    Returns:
        The element wise minimum of the two arrays as a array of `dtype`.
    """
    var result: NDArray[dtype] = NDArray[dtype](array1.shape)

    alias width = simdwidthof[dtype]()
    if array1.shape != array2.shape:
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            min(
                array1.load[width=simd_width](idx),
                array2.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_min, width](array1.num_elements())
    return result


fn maximum[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element wise maximum of two arrays.

    Parameters:
         dtype: The element type.

    Args:
        array1: A array.
        array2: A array.
    Returns:
        The element wise maximum of the two arrays as a array of `dtype`.
    """

    var result: NDArray[dtype] = NDArray[dtype](array1.shape)
    alias width = simdwidthof[dtype]()
    if array1.shape != array2.shape:
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            max(
                array1.load[width=simd_width](idx),
                array2.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_max, width](array1.num_elements())
    return result
