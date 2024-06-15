import math

from algorithm import vectorize
from .ndarray import NDArray, NDArrayShape

"""
Basis of statistics module to be built later

###### TODO: ######
Right now, all stats functions have problems when the output datatype is different than input dataype, 
such as when input tensors have integer, but it's mean is float. In such cases, function returns 0 currently. 

1) Add support for type conversion
2) Add support for axis parameter
"""

# ===------------------------------------------------------------------------===#
# Sort
# ===------------------------------------------------------------------------===#


fn binary_sort[dtype: DType](tensor: NDArray[dtype]) -> NDArray[dtype]:
    var result: NDArray[dtype] = tensor
    var n = tensor.num_elements()
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


fn sum[dtype: DType](tensor: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Sum of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
    Returns:
        The cumulative sum of the tensor as a SIMD Value of `dtype`.
    """
    var result = Scalar[dtype]()
    alias simd_width: Int = simdwidthof[dtype]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = tensor.load[width=simd_width](idx)
        result += simd_data.reduce_add()

    vectorize[vectorize_sum, simd_width](tensor.num_elements())
    return result


fn prod[dtype: DType](tensor: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Product of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
    Returns:
        The cumulative product of the tensor as a SIMD Value of `dtype`.
    """

    var result: SIMD[dtype, 1] = SIMD[dtype, 1](1)
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(
        0, opt_nelts * (tensor.num_elements() // opt_nelts), opt_nelts
    ):
        var simd_data = tensor.load[width=opt_nelts](i)
        result *= simd_data.reduce_mul()

    if tensor.num_elements() % opt_nelts != 0:
        for i in range(
            opt_nelts * (tensor.num_elements() // opt_nelts),
            tensor.num_elements(),
        ):
            var simd_data = tensor.load[width=1](i)
            result *= simd_data.reduce_mul()
    return result


# ===------------------------------------------------------------------------===#
# Statistics Cumulative Operations
# ===------------------------------------------------------------------------===#


fn mean[dtype: DType](tensor: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Arithmatic Mean of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
    Returns:
        The mean of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return sum[dtype](tensor) / tensor.num_elements()


fn mode[dtype: DType](tensor: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Cumulative Mode of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
    Returns:
        The mode of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var sorted_tensor = binary_sort[dtype](tensor)
    var max_count = 0
    var mode_value = sorted_tensor[0]
    var current_count = 1

    for i in range(1, tensor.num_elements()):
        if sorted_tensor[i] == sorted_tensor[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                mode_value = sorted_tensor[i - 1]
            current_count = 1

    if current_count > max_count:
        mode_value = sorted_tensor[tensor.num_elements() - 1]

    return mode_value


# * IMPLEMENT median high and low
fn median[dtype: DType](tensor: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Median value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
    Returns:
        The median of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var sorted_tensor = binary_sort[dtype](tensor)
    var n = tensor.num_elements()
    if n % 2 == 1:
        return sorted_tensor[n // 2]
    else:
        return (sorted_tensor[n // 2 - 1] + sorted_tensor[n // 2]) / 2


# for max and min, I can later change to the latest reduce.max, reduce.min()
fn maxT[dtype: DType](tensor: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Maximum value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
    Returns:
        The maximum of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    # TODO: Test this
    alias nelts = simdwidthof[dtype]()
    var max_value = NDArray[dtype](NDArrayShape(nelts))
    for i in range(nelts):
        max_value[i] = tensor[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        max_value.store[width=simd_width](
            0,
            SIMD.max(
                max_value.load[width=simd_width](0),
                tensor.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized, nelts](tensor.num_elements())
    return SIMD.max(max_value[0], max_value[1])


fn minT[dtype: DType](tensor: NDArray[dtype]) -> SIMD[dtype, 1]:
    """
    Minimum value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.

    Returns:
        The minimum of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    alias nelts = simdwidthof[dtype]()
    var min_value = NDArray[dtype](NDArrayShape(nelts))
    for i in range(nelts):
        min_value[i] = tensor[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        min_value.store[width=simd_width](
            0,
            SIMD.min(
                min_value.load[width=simd_width](0),
                tensor.load[width=simd_width](idx),
            ),
        )

    vectorize[vectorized, nelts](tensor.num_elements())
    return SIMD.min(min_value[0], min_value[1])


fn pvariance[
    dtype: DType
](tensor: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[dtype, 1]:
    """
    Population variance of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
        mu: The mean of the tensor, if provided.
    Returns:
        The variance of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var mean_value: Scalar[dtype]

    if mu == Scalar[dtype]():
        mean_value = mean[dtype](tensor)
    else:
        mean_value = mu

    var result = Scalar[dtype]()
    for i in range(tensor.num_elements()):
        result += (tensor[i] - mean_value) ** 2
    return result / tensor.num_elements()


fn variance[
    dtype: DType
](tensor: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[dtype, 1]:
    """
    Variance of a tensor.

    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
        mu: The mean of the tensor, if provided.
        
    Returns:
        The variance of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var mean_value: Scalar[dtype]

    if mu == Scalar[dtype]():
        mean_value = mean[dtype](tensor)
    else:
        mean_value = mu

    var result = Scalar[dtype]()
    for i in range(tensor.num_elements()):
        result += (tensor[i] - mean_value) ** 2
    return result / (tensor.num_elements() - 1)


fn pstdev[
    dtype: DType
](tensor: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[dtype, 1]:
    """
    Population standard deviation of a tensor.

    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
        mu: The mean of the tensor, if provided.

    Returns:
        The standard deviation of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return math.sqrt(pvariance(tensor, mu))


fn stdev[
    dtype: DType
](tensor: NDArray[dtype], mu: Scalar[dtype] = Scalar[dtype]()) -> SIMD[dtype, 1]:
    """
    Standard deviation of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A NDArray.
        mu: The mean of the tensor, if provided.
    Returns:
        The standard deviation of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return math.sqrt(variance(tensor, mu))
