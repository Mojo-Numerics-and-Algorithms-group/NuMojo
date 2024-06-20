"""
# ===----------------------------------------------------------------------=== #
# Statistics Module - Implements cumulative reduce functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

import math
from algorithm import vectorize
from ...core.ndarray import NDArray, NDArrayShape
from ...core.utility_funcs import is_inttype, is_floattype

"""
TODO: 
1) Add support for axis parameter.  
2) Currently, constrained is crashing mojo, so commented it out and added raise Error. Check later.
3) Relax constrained[] to let user get whatever output they want, but make a warning instead.
"""

# ===------------------------------------------------------------------------===#
# Sort
# ===------------------------------------------------------------------------===#


fn binary_sort[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> NDArray[out_dtype]:
    """
    Binary sorting of NDArray.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.

    Returns:
        The sorted NDArray of type `out_dtype`.
    """
    var result: NDArray[out_dtype] = NDArray[out_dtype](array.shape())
    for i in range(array.ndshape._size):
        result[i] = array[i].cast[out_dtype]()

    var n = array.num_elements()
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i - 1] > result[i]:
                var temp: Scalar[out_dtype] = result[i - 1]
                result[i - 1] = result[i]
                result[i] = temp
    return result


# ===------------------------------------------------------------------------===#
# Reduce Cumulative Operations
# ===------------------------------------------------------------------------===#


fn sum[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) -> SIMD[out_dtype, 1]:
    """
    Cumulative Sum of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.
    Returns:
        The cumulative sum of the array as a SIMD Value of `dtype`.
    """
    var result = Scalar[out_dtype]()
    alias opt_nelts: Int = simdwidthof[in_dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result += (simd_data.reduce_add()).cast[out_dtype]()

    vectorize[vectorize_sum, opt_nelts](array.num_elements())
    return result


fn prod[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) -> SIMD[out_dtype, 1]:
    """
    Cumulative Product of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.
    Args:
        array: A NDArray.
    Returns:
        The cumulative product of the array as a SIMD Value of `dtype`.
    """

    var result: SIMD[out_dtype, 1] = SIMD[out_dtype, 1](1)
    alias opt_nelts = simdwidthof[in_dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result *= (simd_data.reduce_mul()).cast[out_dtype]()

    vectorize[vectorize_sum, opt_nelts](array.num_elements())
    return result


# ===------------------------------------------------------------------------===#

# Statistics Cumulative Operations
# ===------------------------------------------------------------------------===#


fn mean[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Cumulative Arithmatic Mean of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.
    Args:
        array: A NDArray.
    Returns:
        The mean of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[in_dtype]() and is_inttype[out_dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[in_dtype]() and is_inttype[out_dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    return sum[in_dtype, out_dtype](array) / (array.num_elements())


fn mode[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Cumulative Mode of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.
    Returns:
        The mode of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_tensor: NDArray[out_dtype] = binary_sort[in_dtype, out_dtype](
        array
    )
    var max_count = 0
    var mode_value = sorted_tensor[0]
    var current_count = 1

    for i in range(1, array.num_elements()):
        if sorted_tensor[i] == sorted_tensor[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                mode_value = sorted_tensor[i - 1]
            current_count = 1

    if current_count > max_count:
        mode_value = sorted_tensor[array.num_elements() - 1]

    return mode_value


# * IMPLEMENT median high and low
fn median[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Median value of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.
    Returns:
        The median of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_tensor = binary_sort[in_dtype, out_dtype](array)
    var n = array.num_elements()
    if n % 2 == 1:
        return sorted_tensor[n // 2]
    else:
        return (sorted_tensor[n // 2 - 1] + sorted_tensor[n // 2]) / 2


# for max and min, I can later change to the latest reduce.max, reduce.min()
fn maxT[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Maximum value of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.
    Returns:
        The maximum of all of the member values of array as a SIMD Value of `dtype`.
    """
    # TODO: Test this
    alias opt_nelts = simdwidthof[in_dtype]()
    var max_value = NDArray[in_dtype](NDArrayShape(opt_nelts))
    for i in range(opt_nelts):
        max_value[i] = array[0]
    # var max_value: SIMD[in_dtype, opt_nelts] = SIMD[in_dtype, opt_nelts](array[0])

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

    var result: Scalar[in_dtype] = Scalar[out_dtype](max_value[0])
    for i in range(max_value.__len__()):
        if max_value[i] > result:
            result = max_value[i]

    return result.cast[out_dtype]()


fn minT[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Minimum value of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.

    Returns:
        The minimum of all of the member values of array as a SIMD Value of `dtype`.
    """
    alias opt_nelts = simdwidthof[in_dtype]()
    var min_value = NDArray[in_dtype](NDArrayShape(opt_nelts))
    for i in range(opt_nelts):
        min_value[i] = array[0]

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        min_value.store[width=simd_width](
            0,
            SIMD.min(
                min_value.load[width=simd_width](0),
                array.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_min, opt_nelts](array.num_elements())

    var result: Scalar[in_dtype] = Scalar[out_dtype](min_value[0])
    for i in range(min_value.__len__()):
        if min_value[i] < result:
            result = min_value[i]

    return result.cast[out_dtype]()


fn pvariance[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    array: NDArray[in_dtype], mu: Optional[Scalar[in_dtype]] = None
) raises -> SIMD[out_dtype, 1]:
    """
    Population variance of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type..

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.
    Returns:
        The variance of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[in_dtype]() and is_inttype[out_dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[in_dtype]() and is_inttype[out_dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )

    var mean_value: Scalar[out_dtype]
    if not mu:
        mean_value = mean[in_dtype, out_dtype](array)
    else:
        mean_value = mu.value()[].cast[out_dtype]()

    var result = Scalar[out_dtype]()

    for i in range(array.num_elements()):
        result += (array[i].cast[out_dtype]() - mean_value) ** 2

    return result / (array.num_elements())


fn variance[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    array: NDArray[in_dtype], mu: Optional[Scalar[in_dtype]] = None
) raises -> SIMD[out_dtype, 1]:
    """
    Variance of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.

    Returns:
        The variance of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[in_dtype]() and is_inttype[out_dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[in_dtype]() and is_inttype[out_dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    var mean_value: Scalar[out_dtype]

    if not mu:
        mean_value = mean[in_dtype, out_dtype](array)
    else:
        mean_value = mu.value()[].cast[out_dtype]()

    var result = Scalar[out_dtype]()
    for i in range(array.num_elements()):
        result += (array[i].cast[out_dtype]() - mean_value) ** 2

    return result / (array.num_elements() - 1)


fn pstdev[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    array: NDArray[in_dtype], mu: Optional[Scalar[in_dtype]] = None
) raises -> SIMD[out_dtype, 1]:
    """
    Population standard deviation of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.

    Returns:
        The standard deviation of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[in_dtype]() and is_inttype[out_dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[in_dtype]() and is_inttype[out_dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    return math.sqrt(pvariance[in_dtype, out_dtype](array, mu))


fn stdev[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    array: NDArray[in_dtype], mu: Optional[Scalar[in_dtype]] = None
) raises -> SIMD[out_dtype, 1]:
    """
    Standard deviation of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.
    Returns:
        The standard deviation of all of the member values of array as a SIMD Value of `dtype`.
    """
    # constrained[is_inttype[in_dtype]() and is_inttype[out_dtype](), "Input and output both cannot be `Integer` datatype as it may lead to precision errors"]()
    if is_inttype[in_dtype]() and is_inttype[out_dtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    return math.sqrt(variance[in_dtype, out_dtype](array, mu))


# this roughly seems to be just an alias for min in numpy
fn amin[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Minimum value of an array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: An array.
    Returns:
        The minimum of all of the member values of array as a SIMD Value of `dtype`.
    """
    return minT[in_dtype, out_dtype](array)


# this roughly seems to be just an alias for max in numpy
fn amax[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> SIMD[out_dtype, 1]:
    """
    Maximum value of a array.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A array.
    Returns:
        The maximum of all of the member values of array as a SIMD Value of `dtype`.
    """
    return maxT[in_dtype, out_dtype](array)


fn mimimum[
    in_dtype: DType, out_dtype: DType = DType.float64
](s1: SIMD[in_dtype, 1], s2: SIMD[in_dtype, 1]) -> SIMD[out_dtype, 1]:
    """
    Minimum value of two SIMD values.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        s1: A SIMD Value.
        s2: A SIMD Value.
    Returns:
        The minimum of the two SIMD Values as a SIMD Value of `dtype`.
    """
    return SIMD.min(s1, s2).cast[out_dtype]()


fn maximum[
    in_dtype: DType, out_dtype: DType = DType.float64
](s1: SIMD[in_dtype, 1], s2: SIMD[in_dtype, 1]) -> SIMD[out_dtype, 1]:
    """
    Maximum value of two SIMD values.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        s1: A SIMD Value.
        s2: A SIMD Value.
    Returns:
        The maximum of the two SIMD Values as a SIMD Value of `dtype`.
    """
    return SIMD.max(s1, s2).cast[out_dtype]()


fn minimum[
    in_dtype: DType, out_dtype: DType = DType.float64
](array1: NDArray[in_dtype], array2: NDArray[in_dtype]) raises -> NDArray[
    out_dtype
]:
    """
    Element wise minimum of two tensors.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array1: An array.
        array2: An array.
    Returns:
        The element wise minimum of the two tensors as a array of `dtype`.
    """
    var result: NDArray[out_dtype] = NDArray[out_dtype](array1.shape())

    alias nelts = simdwidthof[in_dtype]()
    if array1.shape() != array2.shape():
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            SIMD.min(
                array1.load[width=simd_width](idx),
                array2.load[width=simd_width](idx),
            ).cast[out_dtype](),
        )

    vectorize[vectorized_min, nelts](array1.num_elements())
    return result


fn maximum[
    in_dtype: DType, out_dtype: DType = DType.float64
](array1: NDArray[in_dtype], array2: NDArray[in_dtype]) raises -> NDArray[
    out_dtype
]:
    """
    Element wise maximum of two tensors.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array1: A array.
        array2: A array.
    Returns:
        The element wise maximum of the two tensors as a array of `dtype`.
    """

    var result: NDArray[out_dtype] = NDArray[out_dtype](array1.shape())
    alias nelts = simdwidthof[in_dtype]()
    if array1.shape() != array2.shape():
        raise Error("array shapes are not the same")

    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            SIMD.max(
                array1.load[width=simd_width](idx),
                array2.load[width=simd_width](idx),
            ).cast[out_dtype](),
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
    var max_val: Scalar[dtype] = array[0]
    for i in range(1, array.num_elements()):
        if array[i] > max_val:
            max_val = array[i]
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
    var min_val: Scalar[dtype] = array[0]

    for i in range(1, array.num_elements()):
        if array[i] < min_val:
            min_val = array[i]
            idx = i
    return idx
