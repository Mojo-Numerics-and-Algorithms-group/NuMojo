# ===----------------------------------------------------------------------=== #
# NuMojo: Random routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Random (numojo.routines.random).
================================
Random number generation and sampling.

Functions for creating arrays populated with random samples from various
distributions.

Exports
-------
- `seed`: Seed the global random number generator.
- `rand`, `uniform`: Uniform distribution [0, 1) (`uniform` is a
  `numpy.random`-named alias).
- `randint`: Random integers in range.
- `randn`, `normal`: Standard/general normal distribution (`normal` is a
  `numpy.random`-named alias).
- `exponential`: Exponential distribution.
- `randbool`: Random boolean values.
- `binomial`: Binomial distribution.
- `poisson`: Poisson distribution.
- `gamma`: Gamma distribution.
- `beta`: Beta distribution.
- `chisquare`: Chi-square distribution.
- `choice`: Sample elements from an array.
- `shuffle`: Shuffle an array in place along its first axis.
- `permutation`: Return a randomly permuted range or a shuffled copy of
  an array.

Notes:
    Similar to numpy.random but shape is always the first argument.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as mt
from std.random import random as builtin_random

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.core.dtype.default_dtype import f64
from numojo.core.layout.ndshape import NDArrayShape

# ===----------------------------------------------------------------------=== #
# Seeding
# ===----------------------------------------------------------------------=== #


def seed():
    """
    Seed the global random number generator using a time-based value.

    Examples:
        ```mojo
        import numojo as nm
        nm.random.seed()
        ```
    """
    builtin_random.seed()


def seed(a: Int):
    """
    Seed the global random number generator with the given value, so that
    subsequent calls to functions in `numojo.random` are reproducible
    across runs.

    Args:
        a: The seed value.

    Examples:
        ```mojo
        import numojo as nm
        nm.random.seed(42)
        var arr1 = nm.random.rand[nm.f64](nm.Shape(3))
        nm.random.seed(42)
        var arr2 = nm.random.rand[nm.f64](nm.Shape(3))
        # `arr1` and `arr2` contain identical values.
        ```
    """
    builtin_random.seed(a)


# ===----------------------------------------------------------------------=== #
# Uniform distribution
# ===----------------------------------------------------------------------=== #


def rand[
    dtype: DType = DType.float64
](shape: NDArrayShape) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).

    Example:
    ```mojo
    from numojo import Shape
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
    var result: NDArray[dtype] = NDArray[dtype](shape)

    for i in range(result.size):
        var temp: Scalar[dtype] = builtin_random.random_float64(0, 1).cast[
            dtype
        ]()
        result.unsafe_set(i, temp)

    return result^


def rand[
    dtype: DType = DType.float64
](*shape: Int) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Overloads the function `rand(shape: NDArrayShape)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).
    """
    return rand[dtype](NDArrayShape(shape))


def rand[
    dtype: DType = DType.float64
](shape: List[Int]) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Overloads the function `rand(shape: NDArrayShape)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).
    """
    return rand[dtype](NDArrayShape(shape))


def rand[
    dtype: DType = DType.float64
](shape: VariadicList[Int, _]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `rand(shape: NDArrayShape)`
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [0, 1).
    """
    return rand[dtype](NDArrayShape(shape))


def rand[
    dtype: DType = DType.float64
](
    shape: NDArrayShape, min: Scalar[dtype], max: Scalar[dtype]
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [min, max). This is equivalent to
    `min + rand() * (max - min)`.

    Example:
    ```mojo
    from numojo import Shape
    var arr = numojo.core.random.rand[numojo.i16](Shape(3,2,4), min=0, max=100)
    print(arr)
    ```

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

    var result: NDArray[dtype] = NDArray[dtype](shape)

    for i in range(result.size):
        result.unsafe_set(
            i,
            builtin_random.random_float64(
                min.cast[DType.float64](), max.cast[DType.float64]()
            ).cast[dtype](),
        )

    return result^


def rand[
    dtype: DType = DType.float64
](*shape: Int, min: Scalar[dtype], max: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `rand(shape: NDArrayShape, min, max)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [min, max). This is equivalent to
    `min + rand() * (max - min)`.
    """
    return rand[dtype](NDArrayShape(shape), min=min, max=max)


def rand[
    dtype: DType = DType.float64
](shape: List[Int], min: Scalar[dtype], max: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `rand(shape: NDArrayShape, min, max)`.
    Creates an array of the given shape and populate it with random samples from
    a uniform distribution over [min, max). This is equivalent to
    `min + rand() * (max - min)`.
    """
    return rand[dtype](NDArrayShape(shape), min=min, max=max)


def uniform[
    dtype: DType = DType.float64
](
    shape: NDArrayShape, low: Scalar[dtype] = 0, high: Scalar[dtype] = 1
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Creates an array of the given shape and populates it with random
    samples from a uniform distribution over [low, high).

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        low: The lower bound of the range (inclusive). Defaults to 0.
        high: The upper bound of the range (exclusive). Defaults to 1.

    Returns:
        The generated NDArray of type `dtype` filled with random values
        between `low` and `high`.

    Notes:
        Equivalent to `rand(shape, min=low, max=high)`, provided under
        the name `numpy.random.uniform` uses.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.uniform[nm.f64](nm.Shape(3, 2), low=-1.0, high=1.0)
        print(arr)
        ```
    """
    return rand[dtype](shape, min=low, max=high)


def uniform[
    dtype: DType = DType.float64
](
    *shape: Int, low: Scalar[dtype] = 0, high: Scalar[dtype] = 1
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Overloads the function `uniform(shape: NDArrayShape, low, high)`.
    """
    return uniform[dtype](NDArrayShape(shape), low=low, high=high)


def uniform[
    dtype: DType = DType.float64
](
    shape: List[Int], low: Scalar[dtype] = 0, high: Scalar[dtype] = 1
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Overloads the function `uniform(shape: NDArrayShape, low, high)`.
    """
    return uniform[dtype](NDArrayShape(shape), low=low, high=high)


# ===----------------------------------------------------------------------=== #
# Discrete integers
# ===----------------------------------------------------------------------=== #


def randint[
    dtype: DType = DType.int64
](shape: NDArrayShape, low: Int, high: Int) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Return an array of random integers from low (inclusive) to high (exclusive).
    Note that it is different from the built-in `random.randint()` function
    which returns integer in range low (inclusive) to high (inclusive).

    Raises:
        NumojoError: If high is not greater than low.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        low: The minimum value of the random values.
        high: The maximum value of the random values.

    Returns:
        An array of random integers from low (inclusive) to high (exclusive).
    """

    if high <= low:
        raise Error(
            NumojoError(
                category="value",
                message="High must be greater than low.",
                location="randint",
            )
        )

    var result: NDArray[dtype] = NDArray[dtype](shape)

    builtin_random.randint[dtype](
        ptr=result.unsafe_ptr(), size=result.size, low=low, high=high - 1
    )

    return result^


def randint[
    dtype: DType = DType.int64
](*shape: Int, low: Int, high: Int) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Overloads the function `randint(shape: NDArrayShape, low, high)`.
    Return an array of random integers from low (inclusive) to high (exclusive).
    Note that it is different from the built-in `random.randint()` function
    which returns integer in range low (inclusive) to high (inclusive).
    """

    return randint[dtype](NDArrayShape(shape), low=low, high=high)


def randint[
    dtype: DType = DType.int64
](shape: NDArrayShape, high: Int) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Return an array of random integers from 0 (inclusive) to high (exclusive).

    Raises:
        NumojoError: If the dtype is not a integer type.
        NumojoError: If high <= 0.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        high: The maximum value of the random values.

    Returns:
        An array of random integers from [0, high).
    """

    if high <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="High must be greater than 0.",
                location="randint",
            )
        )

    var result: NDArray[dtype] = NDArray[dtype](shape)

    builtin_random.randint[dtype](
        ptr=result.unsafe_ptr(), size=result.size, low=0, high=high - 1
    )

    return result^


def randint[
    dtype: DType = DType.int64
](*shape: Int, high: Int) raises -> NDArray[dtype] where dtype.is_integral():
    """
    Overloads the function `randint(shape: NDArrayShape, high)`.
    Return an array of random integers from 0 (inclusive) to high (exclusive).
    """

    return randint[dtype](NDArrayShape(shape), high=high)


# ===----------------------------------------------------------------------=== #
# Normal distribution
# ===----------------------------------------------------------------------=== #


def randn[
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

    var result: NDArray[dtype] = NDArray[dtype](shape)

    builtin_random.randn[dtype](
        ptr=result.unsafe_ptr(),
        size=result.size,
    )

    return result^


def randn[dtype: DType = DType.float64](*shape: Int) raises -> NDArray[dtype]:
    """
    Overloads the function `randn(shape: NDArrayShape)`.
    Creates an array of the given shape and populate it with random samples from
    a standard normal distribution.
    """
    return randn[dtype](NDArrayShape(shape))


def randn[
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

    return randn[dtype](shape) * mt.sqrt(variance) + mean


def randn[
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


def randn[
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


def normal[
    dtype: DType = DType.float64
](
    shape: NDArrayShape, loc: Scalar[dtype] = 0, scale: Scalar[dtype] = 1
) raises -> NDArray[dtype]:
    """
    Creates an array of the given shape and populates it with random
    samples from a normal distribution with mean `loc` and standard
    deviation `scale`.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        loc: The mean of the distribution. Defaults to 0.
        scale: The standard deviation of the distribution. Defaults to 1.

    Returns:
        The generated NDArray of type `dtype` filled with random values
        from `Normal(loc, scale)`.

    Notes:
        Equivalent to `randn(shape, mean=loc, variance=scale**2)`,
        provided under the name `numpy.random.normal` uses.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.normal[nm.f64](nm.Shape(3, 2), loc=1.0, scale=2.0)
        print(arr)
        ```
    """
    return randn[dtype](shape, mean=loc, variance=scale * scale)


def normal[
    dtype: DType = DType.float64
](
    *shape: Int, loc: Scalar[dtype] = 0, scale: Scalar[dtype] = 1
) raises -> NDArray[dtype]:
    """
    Overloads the function `normal(shape: NDArrayShape, loc, scale)`.
    """
    return normal[dtype](NDArrayShape(shape), loc=loc, scale=scale)


def normal[
    dtype: DType = DType.float64
](
    shape: List[Int], loc: Scalar[dtype] = 0, scale: Scalar[dtype] = 1
) raises -> NDArray[dtype]:
    """
    Overloads the function `normal(shape: NDArrayShape, loc, scale)`.
    """
    return normal[dtype](NDArrayShape(shape), loc=loc, scale=scale)


# ===----------------------------------------------------------------------=== #
# Exponential distribution
# ===----------------------------------------------------------------------=== #


def exponential[
    dtype: DType = DType.float64
](shape: NDArrayShape, scale: Scalar[dtype] = 1.0) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
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

    var result = NDArray[dtype](NDArrayShape(shape))

    for i in range(result.size):
        var u = builtin_random.random_float64().cast[dtype]()
        result.unsafe_set(i, -mt.log(u) / scale)

    return result^


def exponential[
    dtype: DType = DType.float64
](*shape: Int, scale: Scalar[dtype] = 1.0) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `exponential(shape: NDArrayShape, rate)`.
    Creates an array of the given shape and populate it with random samples from
    an exponential distribution with given scale parameter.
    """

    return exponential[dtype](NDArrayShape(shape), scale=scale)


def exponential[
    dtype: DType = DType.float64
](shape: List[Int], scale: Scalar[dtype] = 1.0) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `exponential(shape: NDArrayShape, rate)`.
    Creates an array of the given shape and populate it with random samples from
    an exponential distribution with given scale parameter.
    """

    return exponential[dtype](NDArrayShape(shape), scale=scale)


# ===----------------------------------------------------------------------=== #
# Random booleans
# ===----------------------------------------------------------------------=== #


def randbool(
    shape: NDArrayShape, p: Float64 = 0.5
) raises -> NDArray[DType.bool]:
    """
    Creates an array of the given shape and populates it with random boolean
    values where each element is `True` with probability `p` and `False`
    with probability `1 - p`.

    Example:
        ```py
        var arr = numojo.random.randbool(Shape(3, 4))
        var biased = numojo.random.randbool(Shape(10, 10), p=0.8)
        ```

    Args:
        shape: The shape of the NDArray.
        p: Probability of `True` for each element. Must be in [0.0, 1.0].
           Defaults to 0.5.

    Returns:
        An NDArray of dtype `bool` filled with random boolean values.

    Raises:
        NumojoError: If `p` is not in the range [0.0, 1.0].
    """

    if p < 0.0 or p > 1.0:
        raise Error(
            NumojoError(
                category="value",
                message="p must be in the range [0.0, 1.0], got " + String(p),
                location="randbool",
            )
        )

    var result = NDArray[DType.bool](shape)

    for i in range(result.size):
        var val = builtin_random.random_float64(0.0, 1.0) < p
        result.unsafe_set(i, val)

    return result^


def randbool(*shape: Int, p: Float64 = 0.5) raises -> NDArray[DType.bool]:
    """
    Overloads the function `randbool(shape: NDArrayShape, p)`.
    Creates an array of the given shape and populates it with random boolean
    values where each element is `True` with probability `p`.
    """
    return randbool(NDArrayShape(shape), p=p)


def randbool(shape: List[Int], p: Float64 = 0.5) raises -> NDArray[DType.bool]:
    """
    Overloads the function `randbool(shape: NDArrayShape, p)`.
    Creates an array of the given shape and populates it with random boolean
    values where each element is `True` with probability `p`.
    """
    return randbool(NDArrayShape(shape), p=p)


# ===----------------------------------------------------------------------=== #
# Binomial distribution
# ===----------------------------------------------------------------------=== #


def binomial[
    dtype: DType = DType.int64
](shape: NDArrayShape, n: Int, p: Float64) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Creates an array of the given shape and populates it with random
    samples from a binomial distribution: the number of successes in `n`
    independent trials, each succeeding with probability `p`.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        n: The number of trials.
        p: The probability of success of each trial.

    Returns:
        An array of the given shape filled with samples from
        `Binomial(n, p)`.

    Raises:
        NumojoError: If `n` is negative or `p` is not in [0.0, 1.0].

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.binomial(nm.Shape(3, 2), n=10, p=0.5)
        print(arr)
        ```
    """
    if n < 0:
        raise Error(
            NumojoError(
                category="value",
                message="`n` must be non-negative.",
                location="binomial",
            )
        )
    if p < 0.0 or p > 1.0:
        raise Error(
            NumojoError(
                category="value",
                message="`p` must be in the range [0.0, 1.0].",
                location="binomial",
            )
        )

    var result = NDArray[dtype](shape)
    for i in range(result.size):
        var successes = 0
        for _ in range(n):
            if builtin_random.random_float64(0.0, 1.0) < p:
                successes += 1
        result.unsafe_set(i, Scalar[dtype](successes))

    return result^


def binomial[
    dtype: DType = DType.int64
](*shape: Int, n: Int, p: Float64) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Overloads the function `binomial(shape: NDArrayShape, n, p)`.
    """
    return binomial[dtype](NDArrayShape(shape), n=n, p=p)


def binomial[
    dtype: DType = DType.int64
](shape: List[Int], n: Int, p: Float64) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Overloads the function `binomial(shape: NDArrayShape, n, p)`.
    """
    return binomial[dtype](NDArrayShape(shape), n=n, p=p)


# ===----------------------------------------------------------------------=== #
# Poisson distribution
# ===----------------------------------------------------------------------=== #


def poisson[
    dtype: DType = DType.int64
](shape: NDArrayShape, lam: Float64 = 1.0) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Creates an array of the given shape and populates it with random
    samples from a Poisson distribution with rate parameter `lam`, using
    Knuth's algorithm.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        lam: The rate (expected number of occurrences). Defaults to 1.0.

    Returns:
        An array of the given shape filled with samples from
        `Poisson(lam)`.

    Raises:
        NumojoError: If `lam` is negative.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.poisson(nm.Shape(3, 2), lam=4.0)
        print(arr)
        ```
    """
    if lam < 0.0:
        raise Error(
            NumojoError(
                category="value",
                message="`lam` must be non-negative.",
                location="poisson",
            )
        )

    var result = NDArray[dtype](shape)
    var l = mt.exp(-lam)
    for i in range(result.size):
        var k = 0
        var p: Float64 = 1.0
        while True:
            k += 1
            p *= builtin_random.random_float64(0.0, 1.0)
            if p <= l:
                break
        result.unsafe_set(i, Scalar[dtype](k - 1))

    return result^


def poisson[
    dtype: DType = DType.int64
](*shape: Int, lam: Float64 = 1.0) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Overloads the function `poisson(shape: NDArrayShape, lam)`.
    """
    return poisson[dtype](NDArrayShape(shape), lam=lam)


def poisson[
    dtype: DType = DType.int64
](shape: List[Int], lam: Float64 = 1.0) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Overloads the function `poisson(shape: NDArrayShape, lam)`.
    """
    return poisson[dtype](NDArrayShape(shape), lam=lam)


# ===----------------------------------------------------------------------=== #
# Gamma distribution
# ===----------------------------------------------------------------------=== #


def _sample_gamma[
    dtype: DType
](k: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Sample a single value from `Gamma(k, 1)` using the Marsaglia-Tsang
    method (for `k >= 1`; boosted via `Gamma(k + 1, 1) * U^(1/k)` for
    `k < 1`).

    Parameters:
        dtype: The data type of the sample. Should be floating-point.

    Args:
        k: The shape parameter of the distribution. Must be positive.

    Returns:
        A single sample from `Gamma(k, 1)`.
    """
    if k < 1:
        var u = builtin_random.random_float64(0.0, 1.0)
        return _sample_gamma[dtype](k + 1) * Scalar[dtype](
            mt.pow(u, 1.0 / Float64(k))
        )

    var d = k - Scalar[dtype](1.0 / 3.0)
    var c = Scalar[dtype](1.0) / mt.sqrt(9 * d)
    while True:
        var x = Scalar[dtype](builtin_random.randn_float64())
        var v = 1 + c * x
        v = v * v * v
        if v <= 0:
            continue
        var u = Scalar[dtype](builtin_random.random_float64(0.0, 1.0))
        if mt.log(u) < 0.5 * x * x + d - d * v + d * mt.log(v):
            return d * v


def gamma[
    dtype: DType = DType.float64
](
    shape: NDArrayShape, k: Scalar[dtype], scale: Scalar[dtype] = 1
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Creates an array of the given shape and populates it with random
    samples from a gamma distribution with shape parameter `k` (matching
    `numpy.random.gamma`'s `shape` parameter, renamed here to avoid
    clashing with the array `shape` argument) and scale parameter
    `scale`.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        k: The shape parameter of the distribution. Must be positive.
        scale: The scale parameter of the distribution. Defaults to 1.

    Returns:
        An array of the given shape filled with samples from
        `Gamma(k, scale)`.

    Raises:
        NumojoError: If `k` or `scale` is not positive.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.gamma[nm.f64](nm.Shape(3, 2), k=2.0, scale=2.0)
        print(arr)
        ```
    """
    if k <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`k` must be positive.",
                location="gamma",
            )
        )
    if scale <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`scale` must be positive.",
                location="gamma",
            )
        )

    var result = NDArray[dtype](shape)
    for i in range(result.size):
        result.unsafe_set(i, _sample_gamma[dtype](k) * scale)

    return result^


def gamma[
    dtype: DType = DType.float64
](*shape: Int, k: Scalar[dtype], scale: Scalar[dtype] = 1) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `gamma(shape: NDArrayShape, k, scale)`.
    """
    return gamma[dtype](NDArrayShape(shape), k=k, scale=scale)


def gamma[
    dtype: DType = DType.float64
](
    shape: List[Int], k: Scalar[dtype], scale: Scalar[dtype] = 1
) raises -> NDArray[dtype] where dtype.is_floating_point():
    """
    Overloads the function `gamma(shape: NDArrayShape, k, scale)`.
    """
    return gamma[dtype](NDArrayShape(shape), k=k, scale=scale)


# ===----------------------------------------------------------------------=== #
# Beta distribution
# ===----------------------------------------------------------------------=== #


def beta[
    dtype: DType = DType.float64
](shape: NDArrayShape, a: Scalar[dtype], b: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Creates an array of the given shape and populates it with random
    samples from a beta distribution with shape parameters `a` and `b`,
    derived from two independent gamma samples as `X / (X + Y)` with
    `X ~ Gamma(a, 1)` and `Y ~ Gamma(b, 1)`.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        a: The first shape parameter. Must be positive.
        b: The second shape parameter. Must be positive.

    Returns:
        An array of the given shape filled with samples from
        `Beta(a, b)`.

    Raises:
        NumojoError: If `a` or `b` is not positive.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.beta[nm.f64](nm.Shape(3, 2), a=2.0, b=5.0)
        print(arr)
        ```
    """
    if a <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`a` must be positive.",
                location="beta",
            )
        )
    if b <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`b` must be positive.",
                location="beta",
            )
        )

    var result = NDArray[dtype](shape)
    for i in range(result.size):
        var x = _sample_gamma[dtype](a)
        var y = _sample_gamma[dtype](b)
        result.unsafe_set(i, x / (x + y))

    return result^


def beta[
    dtype: DType = DType.float64
](*shape: Int, a: Scalar[dtype], b: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `beta(shape: NDArrayShape, a, b)`.
    """
    return beta[dtype](NDArrayShape(shape), a=a, b=b)


def beta[
    dtype: DType = DType.float64
](shape: List[Int], a: Scalar[dtype], b: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `beta(shape: NDArrayShape, a, b)`.
    """
    return beta[dtype](NDArrayShape(shape), a=a, b=b)


# ===----------------------------------------------------------------------=== #
# Chi-square distribution
# ===----------------------------------------------------------------------=== #


def chisquare[
    dtype: DType = DType.float64
](shape: NDArrayShape, df: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Creates an array of the given shape and populates it with random
    samples from a chi-square distribution with `df` degrees of freedom,
    equivalent to `Gamma(df / 2, 2)`.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.
        df: The degrees of freedom. Must be positive.

    Returns:
        An array of the given shape filled with samples from
        `ChiSquare(df)`.

    Raises:
        NumojoError: If `df` is not positive.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.chisquare[nm.f64](nm.Shape(3, 2), df=4.0)
        print(arr)
        ```
    """
    if df <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`df` must be positive.",
                location="chisquare",
            )
        )

    return gamma[dtype](shape, k=df / 2, scale=2)


def chisquare[
    dtype: DType = DType.float64
](*shape: Int, df: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `chisquare(shape: NDArrayShape, df)`.
    """
    return chisquare[dtype](NDArrayShape(shape), df=df)


def chisquare[
    dtype: DType = DType.float64
](shape: List[Int], df: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Overloads the function `chisquare(shape: NDArrayShape, df)`.
    """
    return chisquare[dtype](NDArrayShape(shape), df=df)


# ===----------------------------------------------------------------------=== #
# Shuffling and permutations
# ===----------------------------------------------------------------------=== #


def shuffle[dtype: DType](mut A: NDArray[dtype]) raises:
    """
    Shuffle the elements of an array in place along its first axis. For a
    1-D array, individual elements are shuffled. For an N-D array, whole
    sub-arrays along axis 0 are permuted (the array must be
    C-contiguous).

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        A: The array to shuffle in place.

    Raises:
        NumojoError: If the array has more than one dimension and is not
            C-contiguous.

    Notes:
        Matches `numpy.random.shuffle`.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.arange[nm.f64](6)
        nm.random.shuffle(arr)
        print(arr)
        ```
    """
    if A.ndim == 0:
        return

    var n = A.shape[0]

    if A.ndim == 1:
        for i in range(n - 1, 0, -1):
            var j = Int(builtin_random.random_ui64(0, UInt64(i)))
            var tmp = A.unsafe_get(i)
            A.unsafe_set(i, A.unsafe_get(j))
            A.unsafe_set(j, tmp)
        return

    if not A.is_c_contiguous():
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "`shuffle` requires a C-contiguous array for N-D input."
                ),
                location="shuffle",
            )
        )

    var row_size = A.size // n
    for i in range(n - 1, 0, -1):
        var j = Int(builtin_random.random_ui64(0, UInt64(i)))
        if i == j:
            continue
        for k in range(row_size):
            var tmp = A.unsafe_get(i * row_size + k)
            A.unsafe_set(i * row_size + k, A.unsafe_get(j * row_size + k))
            A.unsafe_set(j * row_size + k, tmp)


def permutation[
    dtype: DType = DType.int64
](n: Int) raises -> NDArray[dtype] where dtype.is_integral():
    """
    Return a randomly permuted range, equivalent to shuffling
    `arange(n)`.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        n: The length of the range to permute.

    Returns:
        A randomly permuted array containing `[0, 1, ..., n - 1]`.

    Notes:
        Matches `numpy.random.permutation` when called with an integer.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.random.permutation(5)
        print(arr)
        ```
    """
    var result = NDArray[dtype](NDArrayShape(n))
    for i in range(n):
        result.unsafe_set(i, Scalar[dtype](i))
    shuffle(result)

    return result^


def permutation[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Return a shuffled copy of `A` along its first axis. Unlike `shuffle`,
    `A` itself is not modified.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        A: The array to permute.

    Returns:
        A shuffled copy of `A`.

    Notes:
        Matches `numpy.random.permutation` when called with an array.

    Examples:
        ```mojo
        import numojo as nm
        var arr = nm.arange[nm.f64](6)
        var permuted = nm.random.permutation(arr)
        print(permuted)
        ```
    """
    var result = A.copy() if A.is_c_contiguous() else A.contiguous()
    shuffle(result)

    return result^


def choice[
    dtype: DType
](
    shape: NDArrayShape, A: NDArray[dtype], replace: Bool = True
) raises -> NDArray[dtype]:
    """
    Randomly sample elements from `A` (treated as flattened), with or
    without replacement.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the output array.
        A: The array to sample from.
        replace: Whether sampling is with replacement. Defaults to
            `True`.

    Returns:
        An array of the given shape filled with elements sampled from
        `A`.

    Raises:
        NumojoError: If `A` is empty, or if `replace` is `False` and the
            requested sample size exceeds `A.size`.

    Notes:
        Matches `numpy.random.choice` without weights (no `p` parameter).

    Examples:
        ```mojo
        import numojo as nm
        var pool = nm.arange[nm.f64](10)
        var sample = nm.random.choice(nm.Shape(5), pool, replace=False)
        print(sample)
        ```
    """
    if A.size == 0:
        raise Error(
            NumojoError(
                category="value",
                message="`A` must be non-empty.",
                location="choice",
            )
        )

    var result = NDArray[dtype](shape)

    if replace:
        for i in range(result.size):
            var idx = Int(builtin_random.random_ui64(0, UInt64(A.size - 1)))
            result.unsafe_set(i, A.unsafe_get(idx))
    else:
        if result.size > A.size:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Cannot take a sample larger than the population"
                        " when `replace=False`."
                    ),
                    location="choice",
                )
            )
        var pool = A.copy() if A.is_c_contiguous() else A.contiguous()
        shuffle(pool)
        for i in range(result.size):
            result.unsafe_set(i, pool.unsafe_get(i))

    return result^


def choice[
    dtype: DType
](*shape: Int, A: NDArray[dtype], replace: Bool = True) raises -> NDArray[
    dtype
]:
    """
    Overloads the function `choice(shape: NDArrayShape, A, replace)`.
    """
    return choice[dtype](NDArrayShape(shape), A, replace=replace)


def choice[
    dtype: DType
](shape: List[Int], A: NDArray[dtype], replace: Bool = True) raises -> NDArray[
    dtype
]:
    """
    Overloads the function `choice(shape: NDArrayShape, A, replace)`.
    """
    return choice[dtype](NDArrayShape(shape), A, replace=replace)


# ===----------------------------------------------------------------------=== #
# To be deprecated
# ===----------------------------------------------------------------------=== #


@parameter
def _int_rand_func[
    dtype: DType
](
    mut result: NDArray[dtype], min: Scalar[dtype], max: Scalar[dtype]
) where dtype.is_integral():
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
        ptr=result.unsafe_ptr(),
        size=result.size,
        low=Int(min),
        high=Int(max),
    )


@parameter
def _float_rand_func[
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
        result.unsafe_set(i, temp)
