# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
numojo.routines.random
----------------------

Creates array of the given shape and populate it with random samples from
a certain distribution.

This module is similar to `numpy.random`. However, in this module, the shape is 
always appearing as the first argument.
"""

import math as mt
from random import random as builtin_random

from numojo.core.ndarray import NDArray

# ===----------------------------------------------------------------------=== #
# Uniform distribution
# ===----------------------------------------------------------------------=== #


fn rand[
    dtype: DType = DType.float64
](shape: NDArrayShape) raises -> NDArray[dtype]:
    """
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).

    Example:
    ```mojo
    var arr = numojo.core.random.rand[numojo.i16](Shape(3,2,4))
    print(arr)
    ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.

    Returns:
        The generated NDArray of type `dtype` filled with random values.
    """

    builtin_random.seed()

    @parameter
    if not dtype.is_floating_point():
        raise Error(
            "Invalid type provided. dtype must be a floating-point type."
        )

    var result: NDArray[dtype] = NDArray[dtype](shape)

    for i in range(result.size):
        var temp: Scalar[dtype] = builtin_random.random_float64(0, 1).cast[
            dtype
        ]()
        (result._buf.ptr + i).init_pointee_copy(temp)

    return result^


fn rand[dtype: DType = DType.float64](*shape: Int) raises -> NDArray[dtype]:
    """
    Overloads the function `rand(shape: NDArrayShape)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).
    """
    return rand[dtype](NDArrayShape(shape))


fn rand[
    dtype: DType = DType.float64
](shape: List[Int]) raises -> NDArray[dtype]:
    """
    Overloads the function `rand(shape: NDArrayShape)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).
    """
    return rand[dtype](NDArrayShape(shape))


fn rand[
    dtype: DType = DType.float64
](shape: VariadicList[Int]) raises -> NDArray[dtype]:
    """
    Overloads the function `rand(shape: NDArrayShape)`
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).
    """
    return rand[dtype](NDArrayShape(shape))


fn rand[
    dtype: DType = DType.float64
](
    shape: NDArrayShape, min: Scalar[dtype], max: Scalar[dtype]
) raises -> NDArray[dtype]:
    """
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [min, max). This is equivalent to
    `min + rand() * (max - min)`.

    Example:
    ```mojo
    var arr = numojo.core.random.rand[numojo.i16](Shape(3,2,4), min=0, max=100)
    print(arr)
    ```

    Raises:
        Error: If the dtype is not a floating-point type.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        min: The minimum value of the random values.
        max: The maximum value of the random values.

    Returns:
        The generated NDArray of type `dtype` filled with random values
        between `min` and `max`.
    """

    @parameter
    if not dtype.is_floating_point():
        raise Error(
            "Invalid type provided. dtype must be a floating-point type."
        )

    var result: NDArray[dtype] = NDArray[dtype](shape)

    for i in range(result.size):
        (result._buf.ptr + i).init_pointee_copy(
            builtin_random.random_float64(
                min.cast[DType.float64](), max.cast[DType.float64]()
            ).cast[dtype]()
        )

    return result^


fn rand[
    dtype: DType = DType.float64
](*shape: Int, min: Scalar[dtype], max: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Overloads the function `rand(shape: NDArrayShape, min, max)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [min, max). This is equivalent to
    `min + rand() * (max - min)`.
    """
    return rand[dtype](NDArrayShape(shape), min=min, max=max)


fn rand[
    dtype: DType = DType.float64
](shape: List[Int], min: Scalar[dtype], max: Scalar[dtype]) raises -> NDArray[
    dtype
]:
    """
    Overloads the function `rand(shape: NDArrayShape, min, max)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [min, max). This is equivalent to
    `min + rand() * (max - min)`.
    """
    return rand[dtype](NDArrayShape(shape), min=min, max=max)


# ===----------------------------------------------------------------------=== #
# Discrete integers
# ===----------------------------------------------------------------------=== #


fn randint[
    dtype: DType = DType.int64
](shape: NDArrayShape, low: Int, high: Int) raises -> NDArray[dtype]:
    """
    Return an array of random integers from low (inclusive) to high (exclusive).
    Note that it is different from the built-in `random.randint()` function
    which returns integer in range low (inclusive) to high (inclusive).

    Raises:
        Error: If the dtype is not a integer type.
        Error: If high is not greater than low.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        low: The minimum value of the random values.
        high: The maximum value of the random values.

    Returns:
        An array of random integers from low (inclusive) to high (exclusive).
    """

    @parameter
    if not dtype.is_integral():
        raise Error("Only Integral values can be sampled using this function.")

    if high <= low:
        raise Error("High must be greater than low.")

    var result: NDArray[dtype] = NDArray[dtype](shape)

    builtin_random.randint[dtype](
        ptr=result._buf.ptr, size=result.size, low=low, high=high - 1
    )

    return result^


fn randint[
    dtype: DType = DType.int64
](shape: NDArrayShape, high: Int) raises -> NDArray[dtype]:
    """
    Return an array of random integers from 0 (inclusive) to high (exclusive).

    Raises:
        Error: If the dtype is not a integer type.
        Error: If high <= 0.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        high: The maximum value of the random values.

    Returns:
        An array of random integers from [0, high).
    """

    @parameter
    if not dtype.is_integral():
        raise Error("Only Integral values can be sampled using this function.")

    if high <= 0:
        raise Error("High must be greater than 0.")

    var result: NDArray[dtype] = NDArray[dtype](shape)

    builtin_random.randint[dtype](
        ptr=result._buf.ptr, size=result.size, low=0, high=high - 1
    )

    return result^


# ===----------------------------------------------------------------------=== #
# Normal distribution
# ===----------------------------------------------------------------------=== #


fn randn[
    dtype: DType = DType.float64
](shape: NDArrayShape) raises -> NDArray[dtype]:
    """
    Creates an array of the given shape and populate it with random samples from
    a standard normal distribution.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.

    Returns:
        An array of the given shape and populate it with random samples from
        a standard normal distribution.
    """

    builtin_random.seed()

    var result: NDArray[dtype] = NDArray[dtype](shape)

    builtin_random.randn[dtype](
        ptr=result._buf.ptr,
        size=result.size,
    )

    return result^


fn randn[dtype: DType = DType.float64](*shape: Int) raises -> NDArray[dtype]:
    """
    Overloads the function `randn(shape: NDArrayShape)`.
    Creates an array of the given shape and populate it with random samples from
    a standard normal distribution.
    """
    return randn[dtype](NDArrayShape(shape))


fn randn[
    dtype: DType = DType.float64
](
    shape: NDArrayShape, mean: Scalar[dtype], variance: Scalar[dtype]
) raises -> NDArray[dtype]:
    """
    Creates an array of the given shape and populate it with random samples from
    a normal distribution with given mean and variance.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        mean: The mean value of the random values.
        variance: The variance of the random values.

    Returns:
        An array of the given shape and populate it with random samples from
        a normal distribution with given mean and variance.
    """

    builtin_random.seed()

    var result: NDArray[dtype] = NDArray[dtype](shape)

    builtin_random.randn[dtype](
        ptr=result._buf.ptr,
        size=result.size,
        mean=mean.cast[DType.float64](),
        variance=variance.cast[DType.float64](),
    )

    return result^


fn randn[
    dtype: DType = DType.float64
](*shape: Int, mean: Scalar[dtype], variance: Scalar[dtype]) raises -> NDArray[
    dtype
]:
    """
    Overloads the function `randn(shape: NDArrayShape, mean, variance)`.
    Creates an array of the given shape and populate it with random samples from
    a normal distribution with given mean and variance.
    """
    return randn[dtype](NDArrayShape(shape), mean=mean, variance=variance)


fn randn[
    dtype: DType = DType.float64
](
    shape: List[Int], mean: Scalar[dtype], variance: Scalar[dtype]
) raises -> NDArray[dtype]:
    """
    Overloads the function `randn(shape: NDArrayShape, mean, variance)`.
    Creates an array of the given shape and populate it with random samples from
    a normal distribution with given mean and variance.
    """
    return randn[dtype](NDArrayShape(shape), mean=mean, variance=variance)


# ===----------------------------------------------------------------------=== #
# Exponential distribution
# ===----------------------------------------------------------------------=== #


fn exponential[
    dtype: DType = DType.float64
](shape: NDArrayShape, scale: Scalar[dtype] = 1.0) raises -> NDArray[dtype]:
    """
    Creates an array of the given shape and populate it with random samples from
    an exponential distribution with given scale parameter.

    Example:
        ```py
        var arr = numojo.random.exponential(Shape(3, 2, 4), 2.0)
        print(arr)
        ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        scale: The scale parameter of the exponential distribution (lambda).

    Returns:
        An array of the given shape and populate it with random samples from
        an exponential distribution with given scale parameter.
    """

    @parameter
    if not dtype.is_floating_point():
        raise Error(
            "Invalid type provided. dtype must be a floating-point type."
        )

    builtin_random.seed()
    var result = NDArray[dtype](NDArrayShape(shape))

    for i in range(result.size):
        var u = builtin_random.random_float64().cast[dtype]()
        (result._buf.ptr + i).init_pointee_copy(-mt.log(u) / scale)

    return result^


fn exponential[
    dtype: DType = DType.float64
](*shape: Int, scale: Scalar[dtype] = 1.0) raises -> NDArray[dtype]:
    """
    Overloads the function `exponential(shape: NDArrayShape, rate)`.
    Creates an array of the given shape and populate it with random samples from
    an exponential distribution with given scale parameter.
    """

    return exponential[dtype](NDArrayShape(shape), scale=scale)


fn exponential[
    dtype: DType = DType.float64
](shape: List[Int], scale: Scalar[dtype] = 1.0) raises -> NDArray[dtype]:
    """
    Overloads the function `exponential(shape: NDArrayShape, rate)`.
    Creates an array of the given shape and populate it with random samples from
    an exponential distribution with given scale parameter.
    """

    return exponential[dtype](NDArrayShape(shape), scale=scale)


# ===----------------------------------------------------------------------=== #
# To be deprecated
# ===----------------------------------------------------------------------=== #


@parameter
fn _int_rand_func[
    dtype: DType
](mut result: NDArray[dtype], min: Scalar[dtype], max: Scalar[dtype]):
    """
    Generate random integers between `min` and `max` and store them in the given NDArray.

    Parameters:
        dtype: The data type of the random integers.

    Args:
        result: The NDArray to store the random integers.
        min: The minimum value of the random integers.
        max: The maximum value of the random integers.
    """
    builtin_random.randint[dtype](
        ptr=result._buf.ptr,
        size=result.size,
        low=int(min),
        high=int(max),
    )


@parameter
fn _float_rand_func[
    dtype: DType
](mut result: NDArray[dtype], min: Scalar[dtype], max: Scalar[dtype]):
    """
    Generate random floating-point numbers between `min` and `max` and
    store them in the given NDArray.

    Parameters:
        dtype: The data type of the random floating-point numbers.

    Args:
        result: The NDArray to store the random floating-point numbers.
        min: The minimum value of the random floating-point numbers.
        max: The maximum value of the random floating-point numbers.
    """
    for i in range(result.size):
        var temp: Scalar[dtype] = builtin_random.random_float64(
            min.cast[f64](), max.cast[f64]()
        ).cast[dtype]()
        (result._buf.ptr + i).init_pointee_copy(temp)
