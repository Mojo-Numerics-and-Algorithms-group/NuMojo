"""
Cumulative reduction statistics functions for NDArrays
"""
# ===----------------------------------------------------------------------=== #
# Statistics Module - Implements cumulative reduce functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #

from sys.info import simdwidthof
from math import max, min
from collections.optional import Optional

import math
from algorithm import vectorize
from ...core.ndarray import NDArray, NDArrayShape
from ...core.utility_funcs import is_inttype, is_floattype
from ...core.sort import binary_sort

"""
TODO: 
1) Add support for axis parameter.  
2) Currently, constrained is crashing mojo, so commented it out and added raise Error. Check later.
3) Relax constrained[] to let user get whatever output they want, but make a warning instead.
"""


# ===------------------------------------------------------------------------===#
# Reduce Cumulative Operations
# ===------------------------------------------------------------------------===#


fn cumsum[
    dtype: DType = DType.float64
](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """Sum of all items of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The sum of all items in the array as a SIMD Value of `dtype`.
    """
    var result = Scalar[dtype]()
    alias opt_nelts: Int = simdwidthof[dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result += simd_data.reduce_add()

    vectorize[vectorize_sum, opt_nelts](array.num_elements())
    return result


fn cumprod[
    dtype: DType = DType.float64
](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """Product of all items in an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The product of all items in the array as a SIMD Value of `dtype`.
    """

    var result: SIMD[dtype, 1] = SIMD[dtype, 1](1.0)
    alias opt_nelts = simdwidthof[dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result *= simd_data.reduce_mul()

    vectorize[vectorize_sum, opt_nelts](array.num_elements())
    return result


# ===------------------------------------------------------------------------===#

# Statistics Cumulative Operations
# ===------------------------------------------------------------------------===#


fn cummean[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """Arithmatic mean of all items of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The mean of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[ dtype]() and is_inttype[dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[dtype]() and is_inttype[dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    return cumsum[dtype](array) / (array.num_elements())


fn mode[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """Mode of all items of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The mode of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_array: NDArray[dtype] = binary_sort[dtype](array)
    var max_count = 0
    var mode_value = sorted_array.item(0)
    var current_count = 1

    for i in range(1, array.num_elements()):
        if sorted_array[i] == sorted_array[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                mode_value = sorted_array.item(i - 1)
            current_count = 1

    if current_count > max_count:
        mode_value = sorted_array.item(array.num_elements() - 1)

    return mode_value


# * IMPLEMENT median high and low
fn median[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    """Median value of all items of an array.

    Parameters:
         dtype: The element type.

    Args:
        array: An NDArray.

    Returns:
        The median of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_array = binary_sort[dtype](array)
    var n = array.num_elements()
    if n % 2 == 1:
        return sorted_array.item(n // 2)
    else:
        return (sorted_array.item(n // 2 - 1) + sorted_array.item(n // 2)) / 2


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
    alias opt_nelts = simdwidthof[dtype]()
    var max_value = NDArray[dtype](NDArrayShape(opt_nelts))
    for i in range(opt_nelts):
        max_value[i] = array[0]
    # var max_value: SIMD[ dtype, opt_nelts] = SIMD[ dtype, opt_nelts](array[0])

    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        max_value.store[width=simd_width](
            0,
            SIMD.max(
                max_value.load[width=simd_width](0),
                array.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_max, opt_nelts](array.num_elements())

    var result: Scalar[dtype] = Scalar[dtype](max_value.get_scalar(0))
    for i in range(max_value.__len__()):
        if max_value.get_scalar(i) > result:
            result = max_value.get_scalar(i)
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
    alias opt_nelts = simdwidthof[dtype]()
    var min_value = NDArray[dtype](NDArrayShape(opt_nelts))
    for i in range(opt_nelts):
        min_value[i] = array[0]

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        min_value.store[width=simd_width](
            0,
            min(
                min_value.load[width=simd_width](0),
                array.load[width=simd_width](idx),
            ),
            # SIMD.min(
                # min_value.load[width=simd_width](0),
                # array.load[width=simd_width](idx),
            # ),
        )

    vectorize[vectorized_min, opt_nelts](array.num_elements())

    var result: Scalar[dtype] = Scalar[dtype](min_value.get_scalar(0))
    for i in range(min_value.__len__()):
        if min_value.get_scalar(i) < result:
            result = min_value.get_scalar(i)

    return result


fn cumpvariance[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Population variance of a array.

    Parameters:
         dtype: The element type..

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.
    Returns:
        The variance of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[ dtype]() and is_inttype[dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[dtype]() and is_inttype[dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )

    var mean_value: Scalar[dtype]
    if not mu:
        mean_value = cummean[dtype](array)
    else:
        mean_value = mu.value()[]

    var result = Scalar[dtype]()

    for i in range(array.num_elements()):
        result += (array.get_scalar(i) - mean_value) ** 2

    return result / (array.num_elements())


fn cumvariance[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Variance of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.

    Returns:
        The variance of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[ dtype]() and is_inttype[dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[dtype]() and is_inttype[dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    var mean_value: Scalar[dtype]

    if not mu:
        mean_value = cummean[dtype](array)
    else:
        mean_value = mu.value()[]

    var result = Scalar[dtype]()
    for i in range(array.num_elements()):
        result += (array.get_scalar(i) - mean_value) ** 2

    return result / (array.num_elements() - 1)


fn cumpstdev[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Population standard deviation of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.

    Returns:
        The standard deviation of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[ dtype]() and is_inttype[dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[dtype]() and is_inttype[dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    return math.sqrt(cumpvariance[dtype](array, mu))


fn cumstdev[
    dtype: DType = DType.float64
](array: NDArray[dtype], mu: Optional[Scalar[dtype]] = None) raises -> SIMD[
    dtype, 1
]:
    """
    Standard deviation of a array.

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.
    Returns:
        The standard deviation of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[ dtype]() and is_inttype[dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[dtype]() and is_inttype[dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    return math.sqrt(cumvariance[dtype](array, mu))


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
    var result: NDArray[dtype] = NDArray[dtype](array1.shape())

    alias nelts = simdwidthof[dtype]()
    if array1.shape() != array2.shape():
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            min(
                array1.load[width=simd_width](idx),
                array2.load[width=simd_width](idx),
            ),
            # SIMD.min(
                # array1.load[width=simd_width](idx),
                # array2.load[width=simd_width](idx),
            # ),
        )

    vectorize[vectorized_min, nelts](array1.num_elements())
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

    var result: NDArray[dtype] = NDArray[dtype](array1.shape())
    alias nelts = simdwidthof[dtype]()
    if array1.shape() != array2.shape():
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            max(
                array1.load[width=simd_width](idx),
                array2.load[width=simd_width](idx),
            ),
            # SIMD.max(
            #     array1.load[width=simd_width](idx),
            #     array2.load[width=simd_width](idx),
            # ),
        )

    vectorize[vectorized_max, nelts](array1.num_elements())
    return result


# * for loop version works fine for argmax and argmin, need to vectorize it
fn argmax[dtype: DType](array: NDArray[dtype]) raises -> Int:
    """
    Argmax of a array.

    Parameters:
        dtype: The element type.

    Args:
        array: A array.
    Returns:
        The index of the maximum value of the array.
    """
    if array.num_elements() == 0:
        raise Error("array is empty")

    var idx: Int = 0
    var max_val: Scalar[dtype] = array.get_scalar(0)
    for i in range(1, array.num_elements()):
        if array.get_scalar(i) > max_val:
            max_val = array.get_scalar(i)
            idx = i
    return idx


fn argmin[dtype: DType](array: NDArray[dtype]) raises -> Int:
    """
    Argmin of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A array.
    Returns:
        The index of the minimum value of the array.
    """
    if array.num_elements() == 0:
        raise Error("array is empty")

    var idx: Int = 0
    var min_val: Scalar[dtype] = array.get_scalar(0)

    for i in range(1, array.num_elements()):
        if array.get_scalar(i) < min_val:
            min_val = array.get_scalar(i)
            idx = i
    return idx
