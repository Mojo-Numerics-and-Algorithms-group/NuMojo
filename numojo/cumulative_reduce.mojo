"""
# ===----------------------------------------------------------------------=== #
# Statistics Module - Implements cumulative reduce functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

import math
from algorithm import vectorize
from .ndarray import NDArray, NDArrayShape

"""
# TODO:
Right now, all stats functions have problems when the output datatype is different than input dataype, 
such as when input tensors have integer, but it's mean is float. In such cases, function returns 0 currently. 

1) Add support for type conversion
2) Add support for axis parameter
"""

# ===------------------------------------------------------------------------===#
# Sort
# ===------------------------------------------------------------------------===#


fn binary_sort[dtype: DType](array: NDArray[dtype]) -> NDArray[dtype]:
    var result: NDArray[dtype] = array
    var n = array.num_elements()
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i - 1] > result[i]:
                var temp = result[i - 1]
                result[i - 1] = result[i]
                result[i] = temp
    return result


# ===------------------------------------------------------------------------===#
# Reduce Cumulative Operations
# ===------------------------------------------------------------------------===#


fn sum[dtype: DType](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Sum of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
    Returns:
        The cumulative sum of the array as a SIMD Value of `dtype`.
    """
    var result = Scalar[dtype]()
    alias simd_width: Int = simdwidthof[dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result += simd_data.reduce_add()

    vectorize[vectorize_sum, simd_width](array.num_elements())
    return result


fn prod[dtype: DType](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Product of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
    Returns:
        The cumulative product of the array as a SIMD Value of `dtype`.
    """

    var result: SIMD[dtype, 1] = SIMD[dtype, 1](1)
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(
        0, opt_nelts * (array.num_elements() // opt_nelts), opt_nelts
    ):
        var simd_data = array.load[width=opt_nelts](i)
        result *= simd_data.reduce_mul()

    if array.num_elements() % opt_nelts != 0:
        for i in range(
            opt_nelts * (array.num_elements() // opt_nelts),
            array.num_elements(),
        ):
            var simd_data = array.load[width=1](i)
            result *= simd_data.reduce_mul()
    return result


# ===------------------------------------------------------------------------===#
# Statistics Cumulative Operations
# ===------------------------------------------------------------------------===#


fn mean[dtype: DType](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Arithmatic Mean of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
    Returns:
        The mean of all of the member values of array as a SIMD Value of `dtype`.
    """
    return sum[dtype](array) / array.num_elements()


fn mode[dtype: DType](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Mode of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
    Returns:
        The mode of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_tensor = binary_sort[dtype](array)
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
fn median[dtype: DType](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Median value of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
    Returns:
        The median of all of the member values of array as a SIMD Value of `dtype`.
    """
    var sorted_tensor = binary_sort[dtype](array)
    var n = array.num_elements()
    if n % 2 == 1:
        return sorted_tensor[n // 2]
    else:
        return (sorted_tensor[n // 2 - 1] + sorted_tensor[n // 2]) / 2


# for max and min, I can later change to the latest reduce.max, reduce.min()
fn maxT[dtype: DType](array: NDArray[dtype]) -> SIMD[dtype, 1]:
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
    alias nelts = simdwidthof[dtype]()
    var max_value = NDArray[dtype](NDArrayShape(nelts))
    for i in range(nelts):
        max_value[i] = array[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        max_value.store[width=simd_width](
            0,
            SIMD.max(
                max_value.load[width=simd_width](0),
                array.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized, nelts](array.num_elements())
    return SIMD.max(max_value[0], max_value[1])


fn minT[dtype: DType](array: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Minimum value of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The minimum of all of the member values of array as a SIMD Value of `dtype`.
    """
    alias nelts = simdwidthof[dtype]()
    var min_value = NDArray[dtype](NDArrayShape(nelts))
    for i in range(nelts):
        min_value[i] = array[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        min_value.store[width=simd_width](
            0,
            SIMD.min(
                min_value.load[width=simd_width](0),
                array.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized, nelts](array.num_elements())
    return SIMD.min(min_value[0], min_value[1])


fn pvariance[
    dtype: DType
](array: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[
    dtype, 1
]:
    """
    Population variance of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        mu: The mean of the array, if provided.
    Returns:
        The variance of all of the member values of array as a SIMD Value of `dtype`.
    """
    var mean_value: Scalar[dtype]

    if mu == Scalar[dtype]():
        mean_value = mean[dtype](array)
    else:
        mean_value = mu

    var result = Scalar[dtype]()
    for i in range(array.num_elements()):
        result += (array[i] - mean_value) ** 2
    return result / array.num_elements()


fn variance[
    dtype: DType
](array: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[
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
    var mean_value: Scalar[dtype]

    if mu == Scalar[dtype]():
        mean_value = mean[dtype](array)
    else:
        mean_value = mu

    var result = Scalar[dtype]()
    for i in range(array.num_elements()):
        result += (array[i] - mean_value) ** 2
    return result / (array.num_elements() - 1)


fn pstdev[
    dtype: DType
](array: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[
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
    return math.sqrt(pvariance(array, mu))


fn stdev[
    dtype: DType
](array: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[
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
    return math.sqrt(variance(array, mu))


# this roughly seems to be just an alias for min in numpy
fn amin[dtype: DType](arr: Tensor[dtype]) -> SIMD[dtype, 1]:
    """
    Minimum value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        arr: A Tensor.
    Returns:
        The minimum of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return minT[dtype](arr)

# this roughly seems to be just an alias for max in numpy
fn amax[dtype: DType](arr: Tensor[dtype]) -> SIMD[dtype, 1]:
    """
    Maximum value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        arr: A Tensor.
    Returns:
        The maximum of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return maxT[dtype](arr)

fn mimimum[
    dtype: DType
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
    return SIMD.min(s1, s2)


fn maximum[
    dtype: DType
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
    return SIMD.max(s1, s2)


fn minimum[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Element wise minimum of two tensors.
    Parameters:
        dtype: The element type.

    Args:
        tensor1: A Tensor.
        tensor2: A Tensor.
    Returns:
        The element wise minimum of the two tensors as a Tensor of `dtype`.
    """
    var result: Tensor[dtype] = Tensor[dtype](tensor1.shape())
    alias nelts = simdwidthof[dtype]()
    if tensor1.shape() != tensor2.shape():
        raise Error("Tensor shapes are not the same")

    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            SIMD.min(
                tensor1.load[width=simd_width](idx),
                tensor2.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_min, nelts](tensor1.num_elements())
    return result


fn maximum[
    T: DType
](tensor1: Tensor[T], tensor2: Tensor[T]) raises -> Tensor[T]:
    """
    Element wise maximum of two tensors.
    Parameters:
        dtype: The element type.

    Args:
        tensor1: A Tensor.
        tensor2: A Tensor.
    Returns:
        The element wise maximum of the two tensors as a Tensor of `dtype`.
    """
    var result: Tensor[T] = Tensor[T](tensor1.shape())
    alias nelts = simdwidthof[T]()
    if tensor1.shape() != tensor2.shape():
        raise Error("Tensor shapes are not the same")

    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        result.store[width=simd_width](
            idx,
            SIMD.max(
                tensor1.load[width=simd_width](idx),
                tensor2.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized_max, nelts](tensor1.num_elements())
    return result


# * for loop version works fine for argmax and argmin, need to vectorize it
fn argmax[dtype: DType](tensor: Tensor[dtype]) raises -> Int:
    """
    Argmax of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The index of the maximum value of the tensor.
    """
    if tensor.num_elements() == 0:
        raise Error("Tensor is empty")

    var idx: Int = 0
    var max_val: Scalar[dtype] = tensor[0]
    for i in range(1, tensor.num_elements()):
        if tensor[i] > max_val:
            max_val = tensor[i]
            idx = i
    return idx


fn argmin[dtype: DType](tensor: Tensor[dtype]) raises -> Int:
    """
    Argmin of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The index of the minimum value of the tensor.
    """
    if tensor.num_elements() == 0:
        raise Error("Tensor is empty")

    var idx: Int = 0
    var min_val: Scalar[dtype] = tensor[0]

    for i in range(1, tensor.num_elements()):
        if tensor[i] < min_val:
            min_val = tensor[i]
            idx = i
    return idx
