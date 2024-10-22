"""
Array creation routine.
"""
# ===----------------------------------------------------------------------=== #
# ARRAY CREATION ROUTINES
# Last updated: 2024-09-08
# ===----------------------------------------------------------------------=== #


"""
# TODO (In order of priority)
1) Implement axis argument for the NDArray creation functions
2) Use `Shapelike` trait to replace `NDArrayShape`, `List`, `VariadicList` and 
    reduce the number of function reloads.
"""

from algorithm import parallelize
from builtin.math import pow
from sys import simdwidthof

from .ndarray import NDArray
from .ndshape import NDArrayShape
from .utility import _get_index


# ===------------------------------------------------------------------------===#
# Numerical ranges
# ===------------------------------------------------------------------------===#
fn arange[
    dtype: DType = DType.float64
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype] = Scalar[dtype](1),
) raises -> NDArray[dtype]:
    """
    Function that computes a series of values starting from "start" to "stop" with given "step" size.

    Raises:
        Error if both dtype and dtype are integers or if dtype is a float and dtype is an integer.

    Parameters:
        dtype: Datatype of the output array.

    Args:
        start: Scalar[dtype] - Start value.
        stop: Scalar[dtype]  - End value.
        step: Scalar[dtype]  - Step size between each element (default 1).

    Returns:
        A NDArray of datatype `dtype` with elements ranging from `start` to `stop` incremented with `step`.
    """
    var num: Int = ((stop - start) / step).__int__()
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num, size=num))
    for idx in range(num):
        result._buf[idx] = start + step * idx

    return result


# ===------------------------------------------------------------------------===#
# Linear Spacing NDArray Generation
# ===------------------------------------------------------------------------===#
fn linspace[
    dtype: DType = DType.float64
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    num: Int = 50,
    endpoint: Bool = True,
    parallel: Bool = False,
) raises -> NDArray[dtype]:
    """
    Function that computes a series of linearly spaced values starting from "start" to "stop" with given size. Wrapper function for _linspace_serial, _linspace_parallel.

    Raises:
        Error if dtype is an integer.

    Parameters:
        dtype: Datatype of the output array.

    Args:
        start: Start value.
        stop: End value.
        num: No of linearly spaced elements.
        endpoint: Specifies whether to include endpoint in the final NDArray, defaults to True.
        parallel: Specifies whether the linspace should be calculated using parallelization, deafults to False.

    Returns:
        A NDArray of datatype `dtype` with elements ranging from `start` to `stop` with num elements.

    """
    constrained[not dtype.is_integral()]()
    if parallel:
        return _linspace_parallel[dtype](start, stop, num, endpoint)
    else:
        return _linspace_serial[dtype](start, stop, num, endpoint)


fn _linspace_serial[
    dtype: DType = DType.float64
](
    start: SIMD[dtype, 1],
    stop: SIMD[dtype, 1],
    num: Int,
    endpoint: Bool = True,
) raises -> NDArray[dtype]:
    """
    Generate a linearly spaced NDArray of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        dtype: Datatype of the output NDArray elements.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
        A NDArray of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))

    if endpoint:
        var step: SIMD[dtype, 1] = (stop - start) / (num - 1)
        for i in range(num):
            result._buf[i] = start + step * i

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num
        for i in range(num):
            result._buf[i] = start + step * i

    return result


fn _linspace_parallel[
    dtype: DType = DType.float64
](
    start: SIMD[dtype, 1], stop: SIMD[dtype, 1], num: Int, endpoint: Bool = True
) raises -> NDArray[dtype]:
    """
    Generate a linearly spaced NDArray of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
        A NDArray of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))
    alias nelts = simdwidthof[dtype]()

    if endpoint:
        var denominator: SIMD[dtype, 1] = Scalar[dtype](num) - 1.0
        var step: SIMD[dtype, 1] = (stop - start) / denominator

        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            result._buf[idx] = start + step * idx

        parallelize[parallelized_linspace](num)

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num

        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            result._buf[idx] = start + step * idx

        parallelize[parallelized_linspace1](num)

    return result


# ===------------------------------------------------------------------------===#
# Logarithmic Spacing NDArray Generation
# ===------------------------------------------------------------------------===#
fn logspace[
    dtype: DType = DType.float64
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    num: Int,
    endpoint: Bool = True,
    base: Scalar[dtype] = 10.0,
    parallel: Bool = False,
) raises -> NDArray[dtype]:
    """
    Generate a logrithmic spaced NDArray of `num` elements between `start` and `stop`. Wrapper function for _logspace_serial, _logspace_parallel functions.

    Raises:
        Error if dtype is an integer.

    Parameters:
        dtype: Datatype of the output array.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.
        base: Base value of the logarithm, defaults to 10.
        parallel: Specifies whether to calculate the logarithmic spaced values using parallelization.

    Returns:
    - A NDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    constrained[not dtype.is_integral()]()
    if parallel:
        return _logspace_parallel[dtype](
            start,
            stop,
            num,
            base,
            endpoint,
        )
    else:
        return _logspace_serial[dtype](
            start,
            stop,
            num,
            base,
            endpoint,
        )


fn _logspace_serial[
    dtype: DType = DType.float64
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    num: Int,
    base: Scalar[dtype],
    endpoint: Bool = True,
) raises -> NDArray[dtype]:
    """
    Generate a logarithmic spaced NDArray of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        base: Base value of the logarithm, defaults to 10.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
        A NDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)
        for i in range(num):
            result._buf[i] = base ** (start + step * i)
    else:
        var step: Scalar[dtype] = (stop - start) / num
        for i in range(num):
            result._buf[i] = base ** (start + step * i)
    return result


fn _logspace_parallel[
    dtype: DType = DType.float64
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    num: Int,
    base: Scalar[dtype],
    endpoint: Bool = True,
) raises -> NDArray[dtype]:
    """
    Generate a logarithmic spaced NDArray of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        base: Base value of the logarithm, defaults to 10.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
        A NDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)

        @parameter
        fn parallelized_logspace(idx: Int) -> None:
            result._buf[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace](num)

    else:
        var step: Scalar[dtype] = (stop - start) / num

        @parameter
        fn parallelized_logspace1(idx: Int) -> None:
            result._buf[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace1](num)

    return result


# ! Outputs wrong values for Integer type, works fine for float type.
fn geomspace[
    dtype: DType = DType.float64
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    num: Int,
    endpoint: Bool = True,
) raises -> NDArray[dtype]:
    """
    Generate a NDArray of `num` elements between `start` and `stop` in a geometric series.

    Raises:
        Error if dtype is an integer.

    Parameters:
        dtype: Datatype of the input values.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
        A NDArray of `dtype` with `num` geometrically spaced elements between `start` and `stop`.
    """
    constrained[
        not dtype.is_integral(), "Int type will result to precision errors."
    ]()
    var a: Scalar[dtype] = start

    if endpoint:
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))
        var base: Scalar[dtype] = (stop / start)
        var power: Scalar[dtype] = 1 / Scalar[dtype](num - 1)
        var r: Scalar[dtype] = base**power
        for i in range(num):
            result._buf[i] = a * r**i
        return result

    else:
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))
        var base: Scalar[dtype] = (stop / start)
        var power: Scalar[dtype] = 1 / Scalar[dtype](num)
        var r: Scalar[dtype] = base**power
        for i in range(num):
            result._buf[i] = a * r**i
        return result


# ===------------------------------------------------------------------------===#
# Commonly used NDArray Generation routines
# ===------------------------------------------------------------------------===#
fn empty[
    dtype: DType = DType.float64
](shape: NDArrayShape) raises -> NDArray[dtype]:
    """
    Generate an empty NDArray of given shape with arbitrary values.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        shape: Shape of the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.
    """
    return NDArray[dtype](shape=shape)


fn empty[
    dtype: DType = DType.float64
](shape: List[Int]) raises -> NDArray[dtype]:
    """Overload of function `empty` that reads a list of ints."""
    return empty[dtype](shape=Shape(shape))


fn empty[
    dtype: DType = DType.float64
](shape: VariadicList[Int]) raises -> NDArray[dtype]:
    """Overload of function `empty` that reads a variadic list of ints."""
    return empty[dtype](shape=Shape(shape))


fn empty_like[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Generate an empty NDArray of the same shape as `array`.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        array: NDArray to be used as a reference for the shape.

    Returns:
        A NDArray of `dtype` with the same shape as `array`.
    """
    return NDArray[dtype](shape=array.ndshape)


fn eye[dtype: DType = DType.float64](N: Int, M: Int) raises -> NDArray[dtype]:
    """
    Return a 2-D NDArray with ones on the diagonal and zeros elsewhere.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        N: Number of rows in the matrix.
        M: Number of columns in the matrix.

    Returns:
        A NDArray of `dtype` with size N x M and ones on the diagonals.
    """
    var result: NDArray[dtype] = NDArray[dtype](
        NDArrayShape(N, M), fill=SIMD[dtype, 1](0)
    )
    var one: Scalar[dtype] = Scalar[dtype](1)
    for i in range(min(N, M)):
        result.store[1](i, i, val=one)
    return result^


fn identity[dtype: DType = DType.float64](N: Int) raises -> NDArray[dtype]:
    """
    Generate an identity matrix of size N x N.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        N: Size of the matrix.

    Returns:
        A NDArray of `dtype` with size N x N and ones on the diagonals.
    """
    var result: NDArray[dtype] = NDArray[dtype](
        NDArrayShape(N, N), fill=SIMD[dtype, 1](0)
    )
    var one: Scalar[dtype] = Scalar[dtype](1)
    for i in range(N):
        result.store[1](i, i, val=one)
    return result^


fn ones[
    dtype: DType = DType.float64
](shape: NDArrayShape) raises -> NDArray[dtype]:
    """
    Generate a NDArray of ones with given shape filled with ones.

    It calls the function `full`.

    Parameters:
        dtype: Datatype of the NDArray.

    Args:
        shape: Shape of the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.
    """
    return full[dtype](shape=shape, fill_value=1)


fn ones[
    dtype: DType = DType.float64
](shape: List[Int]) raises -> NDArray[dtype]:
    """Overload of function `ones` that reads a list of ints."""
    return ones[dtype](shape=Shape(shape))


fn ones[
    dtype: DType = DType.float64
](shape: VariadicList[Int]) raises -> NDArray[dtype]:
    """Overload of function `ones` that reads a variadic of ints."""
    return ones[dtype](shape=Shape(shape))


fn ones_like[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Generate a NDArray of the same shape as `a` filled with ones.

    Parameters:
        dtype: Datatype of the NDArray.

    Args:
        array: NDArray to be used as a reference for the shape.

    Returns:
        A NDArray of `dtype` with the same shape as `a` filled with ones.
    """
    return NDArray[dtype](shape=array.ndshape, fill=SIMD[dtype, 1](1))


fn zeros[
    dtype: DType = DType.float64
](shape: NDArrayShape) raises -> NDArray[dtype]:
    """
    Generate a NDArray of zeros with given shape.

    It calls the function `full`.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        shape: Shape of the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.

    of NDArray.
    """
    return full[dtype](shape=shape, fill_value=0)


fn zeros[
    dtype: DType = DType.float64
](shape: List[Int]) raises -> NDArray[dtype]:
    """Overload of function `zeros` that reads a list of ints."""
    return zeros[dtype](shape=Shape(shape))


fn zeros[
    dtype: DType = DType.float64
](shape: VariadicList[Int]) raises -> NDArray[dtype]:
    """Overload of function `zeros` that reads a variadic list of ints."""
    return zeros[dtype](shape=Shape(shape))


fn zeros_like[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Generate a NDArray of the same shape as `a` filled with zeros.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        array: NDArray to be used as a reference for the shape.

    Returns:
        A NDArray of `dtype` with the same shape as `a` filled with zeros.
    """
    return full[dtype](shape=array.ndshape, fill_value=0)


fn full[
    dtype: DType = DType.float64
](shape: NDArrayShape, fill_value: Scalar[dtype]) raises -> NDArray[dtype]:
    """Overload of function `full` that reads a list of ints."""
    return NDArray[dtype](shape=shape, fill=fill_value)


fn full[
    dtype: DType = DType.float64
](shape: List[Int], fill_value: Scalar[dtype]) raises -> NDArray[dtype]:
    """Overload of function `full` that reads a list of ints."""
    return full[dtype](shape=Shape(shape), fill_value=fill_value)


fn full[
    dtype: DType = DType.float64
](shape: VariadicList[Int], fill_value: Scalar[dtype]) raises -> NDArray[dtype]:
    """Overload of function `full` that reads a variadic list of ints."""
    return full[dtype](shape=Shape(shape), fill_value=fill_value)


fn full_like[
    dtype: DType = DType.float64
](array: NDArray[dtype], fill_value: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Generate a NDArray of the same shape as `a` filled with `fill_value`.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        array: NDArray to be used as a reference for the shape.
        fill_value: Value to fill the NDArray with.

    Returns:
        A NDArray of `dtype` with the same shape as `a` filled with `fill_value`.
    """
    return NDArray[dtype](shape=array.ndshape, fill=fill_value)


# ===------------------------------------------------------------------------===#
# Building matrices
# ===------------------------------------------------------------------------===#
fn diag[
    dtype: DType = DType.float64
](v: NDArray[dtype], k: Int = 0) raises -> NDArray[dtype]:
    """
    Extract a diagonal or construct a diagonal NDArray.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        v: NDArray to extract the diagonal from.
        k: Diagonal offset.

    Returns:
        A 1-D NDArray with the diagonal of the input NDArray.
    """
    if v.ndim == 1:
        var n: Int = v.ndshape.ndsize
        var result: NDArray[dtype] = zeros[dtype](shape(n + abs(k), n + abs(k)))
        if k >= 0:
            for i in range(n):
                result.data[i * (n + abs(k) + 1) + k] = v.data[i]
            return result^
        else:
            for i in range(n):
                result.data[
                    result.ndshape.ndsize - 1 - i * (result.ndshape[1] + 1) + k
                ] = v.data[n - 1 - i]
        return result^
    elif v.ndim == 2:
        var m: Int = v.ndshape[0]
        var n: Int = v.ndshape[1]
        var result: NDArray[dtype] = NDArray[dtype](n - abs(k))
        if k >= 0:
            for i in range(n - abs(k)):
                result.data[i] = v.data[i * (n + 1) + k]
        else:
            for i in range(n - abs(k)):
                result.data[m - abs(k) - 1 - i] = v.data[
                    v.ndshape.ndsize - 1 - i * (v.ndshape[1] + 1) + k
                ]
        return result^
    else:
        raise Error("Arrays bigger than 2D are not supported")


fn diagflat[
    dtype: DType = DType.float64
](inout v: NDArray[dtype], k: Int = 0) raises -> NDArray[dtype]:
    """
    Generate a 2-D NDArray with the flattened input as the diagonal.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        v: NDArray to be flattened and used as the diagonal.
        k: Diagonal offset.

    Returns:
        A 2-D NDArray with the flattened input as the diagonal.
    """
    var n: Int = v.ndshape.ndsize
    var result: NDArray[dtype] = zeros[dtype](shape(n + abs(k), n + abs(k)))
    if k >= 0:
        for i in range(n):
            result.store((n + k + 1) * i + k, v.data[i])
    else:
        for i in range(n):
            result.store(
                result.ndshape.ndsize - 1 - (n + abs(k) + 1) * i + k,
                v.data[v.ndshape.ndsize - 1 - i],
            )
    return result^


fn tri[
    dtype: DType = DType.float64
](N: Int, M: Int, k: Int = 0) raises -> NDArray[dtype]:
    """
    Generate a 2-D NDArray with ones on and below the k-th diagonal.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        N: Number of rows in the matrix.
        M: Number of columns in the matrix.
        k: Diagonal offset.

    Returns:
        A 2-D NDArray with ones on and below the k-th diagonal.
    """
    var result: NDArray[dtype] = NDArray[dtype](
        NDArrayShape(N, M), fill=SIMD[dtype, 1](0)
    )
    for i in range(N):
        for j in range(M):
            if j <= i + k:
                result.store(i, j, val=Scalar[dtype](1))
    return result^


fn tril[
    dtype: DType = DType.float64
](m: NDArray[dtype], k: Int = 0) raises -> NDArray[dtype]:
    """
    Zero out elements above the k-th diagonal.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        m: NDArray to be zeroed out.
        k: Diagonal offset.

    Returns:
        A NDArray with elements above the k-th diagonal zeroed out.
    """
    var initial_offset: Int = 1
    var final_offset: Int = 1
    var result: NDArray[dtype] = m
    if m.ndim == 2:
        for i in range(m.ndshape[0]):
            for j in range(i + 1 + k, m.ndshape[1]):
                result.data[i * m.ndshape[1] + j] = Scalar[dtype](0)
    elif m.ndim >= 2:
        for i in range(m.ndim - 2):
            initial_offset *= m.ndshape[i]
        for i in range(m.ndim - 2, m.ndim):
            final_offset *= m.ndshape[i]
        for offset in range(initial_offset):
            offset = offset * final_offset
            for i in range(m.ndshape[-2]):
                for j in range(i + 1 + k, m.ndshape[-1]):
                    result.data[offset + j + i * m.ndshape[-1]] = Scalar[dtype](
                        0
                    )
    else:
        raise Error(
            "Arrays smaller than 2D are not supported for this operation."
        )
    return result^


fn triu[
    dtype: DType = DType.float64
](m: NDArray[dtype], k: Int = 0) raises -> NDArray[dtype]:
    """
    Zero out elements below the k-th diagonal.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        m: NDArray to be zeroed out.
        k: Diagonal offset.

    Returns:
        A NDArray with elements below the k-th diagonal zeroed out.
    """
    var initial_offset: Int = 1
    var final_offset: Int = 1
    var result: NDArray[dtype] = m
    if m.ndim == 2:
        for i in range(m.ndshape[0]):
            for j in range(0, i + k):
                result.data[i * m.ndshape[1] + j] = Scalar[dtype](0)
    elif m.ndim >= 2:
        for i in range(m.ndim - 2):
            initial_offset *= m.ndshape[i]
        for i in range(m.ndim - 2, m.ndim):
            final_offset *= m.ndshape[i]
        for offset in range(initial_offset):
            offset = offset * final_offset
            for i in range(m.ndshape[-2]):
                for j in range(0, i + k):
                    result.data[offset + j + i * m.ndshape[-1]] = Scalar[dtype](
                        0
                    )
    else:
        raise Error(
            "Arrays smaller than 2D are not supported for this operation."
        )
    return result^


fn vander[
    dtype: DType = DType.float64
](
    x: NDArray[dtype], N: Optional[Int] = None, increasing: Bool = False
) raises -> NDArray[dtype]:
    """
    Generate a Vandermonde matrix.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        x: 1-D input array.
        N: Number of columns in the output. If N is not specified, a square array is returned.
        increasing: Order of the powers of the columns. If True, the powers increase from left to right, if False (the default) they are reversed.

    Returns:
        A Vandermonde matrix.
    """
    if x.ndim != 1:
        raise Error("x must be a 1-D array")

    var n_rows = x.ndshape.ndsize
    var n_cols = N.value() if N else n_rows
    var result: NDArray[dtype] = NDArray[dtype](
        NDArrayShape(n_rows, n_cols), fill=SIMD[dtype, 1](1)
    )
    for i in range(n_rows):
        var x_i = x.data[i]
        if increasing:
            for j in range(n_cols):
                result.store(i, j, val=x_i**j)
        else:
            for j in range(n_cols - 1, -1, -1):
                result.store(i, n_cols - 1 - j, val=x_i**j)
    return result^


# ===------------------------------------------------------------------------===#
# Construct array from string representation or txt files
# ===------------------------------------------------------------------------===#
fn fromstring[
    dtype: DType = DType.float64
](text: String, order: String = "C",) raises -> NDArray[dtype]:
    """
    NDArray initialization from string representation of an ndarray.
    The shape can be inferred from the string representation.
    The literals will be casted to the dtype of the NDArray.

    Note:
    StringLiteral is also allowed as input as it is coerced to String type
    before it is passed into the function.

    Example:
    ```
    import numojo as nm

    fn main() raises:
        var A = nm.fromstring[DType.int8]("[[[1,2],[3,4]],[[5,6],[7,8]]]")
        var B = nm.fromstring[DType.float16]("[[1,2,3,4],[5,6,7,8]]")
        var C = nm.fromstring[DType.float32]("[0.1, -2.3, 41.5, 19.29145, -199]")
        var D = nm.fromstring[DType.int32]("[0.1, -2.3, 41.5, 19.29145, -199]")

        print(A)
        print(B)
        print(C)
        print(D)
    ```

    The output goes as follows. Note that the numbers are automatically
    casted to the dtype of the NDArray.

    ```console
    [[[     1       2       ]
     [     3       4       ]]
     [[     5       6       ]
     [     7       8       ]]]
    3-D array  Shape: [2, 2, 2]  DType: int8

    [[      1.0     2.0     3.0     4.0     ]
     [      5.0     6.0     7.0     8.0     ]]
    2-D array  Shape: [2, 4]  DType: float16

    [       0.10000000149011612     2.2999999523162842      41.5    19.291450500488281      199.0   ]
    1-D array  Shape: [5]  DType: float32

    [       0       2       41      19      199     ]
    1-D array  Shape: [5]  DType: int32
    ```

    Args:
        text: String representation of an ndarray.
        order: Memory order C or F.
    """

    var data = List[Scalar[dtype]]()
    """Inferred data buffer of the array"""
    var shape = List[Int]()
    """Inferred shape of the array"""
    var bytes = text.as_bytes()
    var ndim = 0
    """Inferred number_as_str of dimensions."""
    var level = 0
    """Current level of the array."""
    var number_as_str: String = ""
    for i in range(len(bytes)):
        var b = bytes[i]
        if chr(int(b)) == "[":
            level += 1
            ndim = max(ndim, level)
            if len(shape) < ndim:
                shape.append(0)
            shape[level - 1] = 0

        if isdigit(b) or (chr(int(b)) == ".") or (chr(int(b)) == "-"):
            number_as_str = number_as_str + chr(int(b))
        if (chr(int(b)) == ",") or (chr(int(b)) == "]"):
            if number_as_str != "":
                var number = atof(number_as_str).cast[dtype]()
                data.append(number)  # Add the number to the data buffer
                number_as_str = ""  # Clean the number cache
                shape[-1] = shape[-1] + 1
        if chr(int(b)) == "]":
            level = level - 1
            if level < 0:
                raise ("Unmatched left and right brackets!")
            if level > 0:
                shape[level - 1] = shape[level - 1] + 1
    var result: NDArray[dtype] = NDArray[dtype](
        data=data, shape=shape, order=order
    )
    return result^


# ===------------------------------------------------------------------------===#
# Construct array from various objects.
# It can be reloaded to allow different types of input.
# ===------------------------------------------------------------------------===#
fn array[
    dtype: DType = DType.float64
](text: String, order: String = "C",) raises -> NDArray[dtype]:
    """
    This reload is an alias of `fromstring`.
    """
    return fromstring[dtype](text, order)


fn array[
    dtype: DType = DType.float64
](
    data: List[Scalar[dtype]], shape: List[Int], order: String = "C"
) raises -> NDArray[dtype]:
    """
    Array creation with given data, shape and order.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        data: List of data.
        shape: List of shape.
        order: Memory order C or F.

    Example:
        ```mojo
        import numojo as nm
        nm.array[f16](data=List[Scalar[f16]](1, 2, 3, 4), shape=List[Int](2, 2))
        ```

    Returns:
        An Array of given data, shape and order.
    """
    return NDArray[dtype](data=data, shape=shape, order=order)


fn array[
    dtype: DType = DType.float64
](data: PythonObject, order: String = "C") raises -> NDArray[dtype]:
    """
    Array creation with given data, shape and order.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        data: A Numpy array (PythonObject).
        order: Memory order C or F.

    Example:
        ```mojo
        import numojo as nm
        var np = Python.import_module("numpy")
        var np_arr = np.array([1, 2, 3, 4])
        nm.array[f16](data=np_arr, order="C")
        ```

    Returns:
        An Array of given data, shape and order.
    """
    return NDArray[dtype](data=data, order=order)
