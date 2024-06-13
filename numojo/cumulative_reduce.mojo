import math
from tensor import Tensor, TensorShape

from algorithm import vectorize

"""
Basis of statistics module to be built later
TODO
Build functions for more generic use with axis parameter
max
min
mean
mode
std
var
"""

fn std[dtype: DType](tensor: Tensor[dtype]) -> SIMD[dtype, 1]:
    """
    Population standard deviation of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The standard deviation of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var sumv = sum[dtype](tensor)
    var n = tensor.num_elements()
    return math.sqrt[dtype](sum[dtype]((tensor - (sumv / n)) ** 2) / n)


fn variance[dtype: DType](tensor: Tensor[dtype]) -> SIMD[dtype, 1]:
    """
    Population variance of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The variance of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var sumv = sum[dtype](tensor)
    var n = tensor.num_elements()
    return sum[dtype]((tensor - (sumv / n)) ** 2) / n


fn binary_sort[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    var result:Tensor[dtype] = tensor
    var n = tensor.num_elements()
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i-1] > result[i]:
                var temp = result[i-1]
                result[i-1] = result[i]
                result[i] = temp
    return result

fn sum[dtype:DType](tensor:Tensor[dtype])->Scalar[dtype]:
    """
    Cumulative Sum of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The cumulative sum of the tensor as a SIMD Value of `dtype`.
    """
    var result = Scalar[dtype]()
    alias simd_width: Int = simdwidthof[dtype]()
    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = tensor.load[width = simd_width](idx)
        result += simd_data.reduce_add()
    vectorize[vectorize_sum, simd_width](tensor.num_elements())
    return result

fn prod[dtype:DType](tensor:Tensor[dtype])->Scalar[dtype]:
    """
    Cumulative Product of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The cumulative product of the tensor as a SIMD Value of `dtype`.
    """
    var result = Scalar[dtype]()
    alias simd_width: Int = simdwidthof[dtype]()
    @parameter
    fn vectorize_mul[simd_width: Int](idx: Int) -> None:
        var simd_data = tensor.load[width = simd_width](idx)
        result *= simd_data.reduce_mul()
    vectorize[vectorize_mul, simd_width](tensor.num_elements())
    return result

fn mean[T:DType](tensor:Tensor[T])->Scalar[T]:
    """
    Cumulative Arithmatic Mean of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The mean of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return sum[T](tensor) / tensor.num_elements()

fn mode[dtype:DType](tensor:Tensor[dtype])->Scalar[dtype]:
    """
    Cumulative Mode of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
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
fn median[dtype:DType](tensor:Tensor[dtype])->Scalar[dtype]:
    """
    Median value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The median of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var sorted_tensor = binary_sort[dtype](tensor)
    var n = tensor.num_elements()
    if n % 2 == 1:
        return sorted_tensor[n // 2]
    else:
        return (sorted_tensor[n // 2 - 1] + sorted_tensor[n // 2]) / 2

fn max[dtype:DType](tensor:Tensor[dtype])->Scalar[dtype]:
    """
    Maximum value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The maximum of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    # TODO: Test this
    alias nelts = simdwidthof[dtype]()
    var max_value = Tensor[dtype](TensorShape(nelts))
    for i in range(nelts):
        max_value[i] = tensor[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        max_value.store[width=simd_width](0, SIMD.max(max_value.load[width=simd_width](0), tensor.load[width=simd_width](idx)))
    vectorize[vectorized, nelts](tensor.num_elements())
    return SIMD.max(max_value[0], max_value[1])

fn min[dtype:DType](tensor:Tensor[dtype])->Scalar[dtype]:
    """
    Minimum value of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The minimum of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    alias nelts = simdwidthof[dtype]()
    var min_value = Tensor[dtype](TensorShape(nelts))
    for i in range(nelts):
        min_value[i] = tensor[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        min_value.store[width=simd_width](0, SIMD.min(min_value.load[width=simd_width](0), tensor.load[width=simd_width](idx)))
    vectorize[vectorized, nelts](tensor.num_elements())
    return SIMD.min(min_value[0], min_value[1])

fn pvariance[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    """
    Population variance of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
    Returns:
        The variance of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var mean_value:Scalar[T]

    if mu == Scalar[T]():
        mean_value = mean[T](tensor)
    else:
        mean_value = mu

    var sum = Scalar[T]()
    for i in range(tensor.num_elements()):
        sum += (tensor[i] - mean_value) ** 2
    return sum / tensor.num_elements()

fn variance[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    """
    Variance of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
        mu: The mean of the tensor, if provided.
    Returns:
        The variance of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    var mean_value:Scalar[T]

    if mu == Scalar[T]():
        mean_value = mean[T](tensor)
    else:
        mean_value = mu

    var sum = Scalar[T]()
    for i in range(tensor.num_elements()):
        sum += (tensor[i] - mean_value) ** 2
    return sum / (tensor.num_elements() -1)

fn pstdev[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    """
    Population standard deviation of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
        mu: The mean of the tensor, if provided.
    Returns:
        The standard deviation of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return math.sqrt(pvariance(tensor, mu))

fn stdev[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    """
    Standard deviation of a tensor.
    Parameters:
        dtype: The element type.

    Args:
        tensor: A Tensor.
        mu: The mean of the tensor, if provided.
    Returns:
        The standard deviation of all of the member values of tensor as a SIMD Value of `dtype`.
    """
    return math.sqrt(variance(tensor, mu))

