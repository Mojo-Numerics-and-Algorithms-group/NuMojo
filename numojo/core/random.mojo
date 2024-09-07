"""
Random values array generation.
"""
# ===----------------------------------------------------------------------=== #
# Implements RANDOM
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #

import math as mt
from random import random
from .ndarray import NDArray
from builtin.tuple import Tuple


fn rand[dtype: DType = DType.float64](*shape: Int) raises -> NDArray[dtype]:
    """
    Generate a random NDArray of the given shape and dtype.

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16](3,2,4)
        print(arr)
        ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.

    Returns:
        The generated NDArray of type `dtype` filled with random values.
    """
    var result: NDArray[dtype] = NDArray[dtype](shape)
    for i in range(result.ndshape.ndsize):
        var temp: Scalar[dtype] = random.random_float64(0, 1).cast[dtype]()
        result.__setitem__(i, temp)
    return result^


fn rand[
    dtype: DType = DType.float64
](*shape: Int, min: Scalar[dtype], max: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Generate a random NDArray of the given shape and dtype with values between `min` and `max`.

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16](3,2,4, min=0, max=100)
        print(arr)
        ```
    Raises:
        Error: If the dtype is not an integral or floating-point type.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        min: The minimum value of the random values.
        max: The maximum value of the random values.

    Returns:
        The generated NDArray of type `dtype` filled with random values between `min` and `max`.
    """
    var result: NDArray[dtype] = NDArray[dtype](shape)
    random.seed()
    # if dtype.is_integral():
    #     random.randint[dtype](
    #         ptr=result.data,
    #         size=result.ndshape.ndsize,
    #         low=int(min),
    #         high=int(max),
    #     )
    # elif dtype.is_floating_point():
    #     for i in range(result.ndshape.ndsize):
    #         var temp: Scalar[dtype] = random.random_float64(
    #             min.cast[f64](), max.cast[f64]()
    #         ).cast[dtype]()
    #         result.__setitem__(i, temp)
    # else:
    #     raise Error(
    #         "Invalid type provided. dtype must be either an integral or"
    #         " floating-point type."
    #     )
    for i in range(result.ndshape.ndsize):
        var temp: Scalar[dtype] = random.random_float64(
            min.cast[f64](), max.cast[f64]()
        ).cast[dtype]()
        result.data[i] = temp
    return result^


fn rand[
    dtype: DType = DType.float64
](shape: List[Int], min: Scalar[dtype], max: Scalar[dtype]) raises -> NDArray[
    dtype
]:
    """
    Generate a random NDArray of the given shape and dtype with values between `min` and `max`.

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16]((3,2,4), min=0, max=100)
        print(arr)
        ```

    Raises:
        Error: If the dtype is not an integral or floating-point type.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        min: The minimum value of the random values.
        max: The maximum value of the random values.

    Returns:
        The generated NDArray of type `dtype` filled with random values between `min` and `max`.
    """
    var result: NDArray[dtype] = NDArray[dtype](shape)
    random.seed()
    # if dtype.is_integral():
    #     random.randint[dtype](
    #         ptr=result.data,
    #         size=result.ndshape.ndsize,
    #         low=int(min),
    #         high=int(max),
    #     )
    # elif dtype.is_floating_point():
    #     for i in range(result.ndshape.ndsize):
    #         var temp: Scalar[dtype] = random.random_float64(
    #             min.cast[f64](), max.cast[f64]()
    #         ).cast[dtype]()
    #         result.data[i] = temp
    # else:
    #     raise Error(
    #         "Invalid type provided. dtype must be either an integral or"
    #         " floating-point type."
    #     )
    for i in range(result.ndshape.ndsize):
        var temp: Scalar[dtype] = random.random_float64(
            min.cast[f64](), max.cast[f64]()
        ).cast[dtype]()
        result.data[i] = temp
    return result


fn randn[
    dtype: DType = DType.float64
](
    *shape: Int, mean: Scalar[dtype] = 0, variance: Scalar[dtype] = 1
) raises -> NDArray[dtype]:
    """
    Generate a random NDArray of the given shape and dtype with values having a mean and variance.

    Example:
        ```py
        var arr = numojo.core.random.rand_meanvar[numojo.i16](3,2,4, mean=0, variance=1)
        print(arr)
        ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        mean: The mean value of the random values.
        variance: The variance of the random values.

    Returns:
        The generated NDArray of type `dtype` filled with random values having a mean and variance.
    """
    random.seed()
    var result: NDArray[dtype] = NDArray[dtype](shape)
    random.randn[dtype](
        ptr=result.data,
        size=result.ndshape.ndsize,
        mean=mean.cast[DType.float64](),
        variance=variance.cast[DType.float64](),
    )
    return result^


fn randn[
    dtype: DType = DType.float64
](
    shape: List[Int], mean: Scalar[dtype] = 0, variance: Scalar[dtype] = 1
) raises -> NDArray[dtype]:
    """
    Generate a random NDArray of the given shape and dtype with values having a mean and variance.

    Example:
        ```py
        var arr = numojo.core.random.rand_meanvar[numojo.i16](List[Int](3,2,4), mean=0, variance=1)
        print(arr)
        ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        mean: The mean value of the random values.
        variance: The variance of the random values.

    Returns:
        The generated NDArray of type `dtype` filled with random values having a mean and variance.
    """
    random.seed()
    var result: NDArray[dtype] = NDArray[dtype](shape)
    random.randn[dtype](
        ptr=result.data,
        size=result.ndshape.ndsize,
        mean=mean.cast[DType.float64](),
        variance=variance.cast[DType.float64](),
    )
    return result^


fn rand_exponential[
    dtype: DType = DType.float64
](*shape: Int, rate: Scalar[dtype] = 1.0) raises -> NDArray[dtype]:
    """
    Generate a random NDArray of the given shape and dtype with values from an exponential distribution.

    Example:
        ```py
        var arr = numojo.core.random.rand_exponential[numojo.f64](3, 2, 4, rate=2.0)
        print(arr)
        ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        rate: The rate parameter of the exponential distribution (lambda).

    Returns:
        The generated NDArray of type `dtype` filled with random values from an exponential distribution.
    """
    random.seed()
    var result = NDArray[dtype](NDArrayShape(shape))

    for i in range(result.num_elements()):
        var u = random.random_float64()
        result.data[i] = -mt.log(1 - u) / rate.cast[DType.float64]()

    return result^


fn rand_exponential[
    dtype: DType = DType.float64
](shape: List[Int], rate: Scalar[dtype] = 1.0) raises -> NDArray[dtype]:
    """
    Generate a random NDArray of the given shape and dtype with values from an exponential distribution.

    Example:
        ```py
        var arr = numojo.core.random.rand_exponential[numojo.f64](List[Int](3, 2, 4), rate=2.0)
        print(arr)
        ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray as a List[Int].
        rate: The rate parameter of the exponential distribution (lambda).

    Returns:
        The generated NDArray of type `dtype` filled with random values from an exponential distribution.
    """
    random.seed()
    var result = NDArray[dtype](shape)

    for i in range(result.num_elements()):
        var u = random.random_float64()
        result.data[i] = -mt.log(1 - u) / rate.cast[DType.float64]()

    return result^
