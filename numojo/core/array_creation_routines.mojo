"""
# ===----------------------------------------------------------------------=== #
# ARRAY CREATION ROUTINES
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

"""
# TODO (In order of priority)
1) Add function overload for List, VariadicList types 
2) Implement axis argument for the NDArray creation functions
3) Implement Row/Column Major option
"""

from algorithm import parallelize
from builtin.math import pow

from .ndarray import NDArray, NDArrayShape
from .utility_funcs import is_inttype, is_floattype


# ===------------------------------------------------------------------------===#
# Arranged Value NDArray generation
# ===------------------------------------------------------------------------===#
fn arange[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    start: Scalar[in_dtype],
    stop: Scalar[in_dtype],
    step: Scalar[in_dtype] = Scalar[in_dtype](1),
) raises -> NDArray[out_dtype]:
    """
    Function that computes a series of values starting from "start" to "stop" with given "step" size.

    Raises:
        Error if both in_dtype and out_dtype are integers or if in_dtype is a float and out_dtype is an integer.

    Parameters:
        in_dtype: Input datatype of the input values.
        out_dtype: Output datatype of the output NDArray.

    Args:
        start: Scalar[in_dtype] - Start value.
        stop: Scalar[in_dtype]  - End value.
        step: Scalar[in_dtype]  - Step size between each element (default 1).

    Returns:
        A NDArray of datatype `out_dtype` with elements ranging from `start` to `stop` incremented with `step`.
    """
    if (is_floattype[in_dtype]() and is_inttype[out_dtype]()) or (
        is_inttype[in_dtype]() and is_inttype[out_dtype]()
    ):
        raise Error(
            "Both input and output datatypes cannot be integers. If the input is a float, the output must also be a float."
        )

    var num: Int = ((stop - start) / step).__int__()
    var result: NDArray[out_dtype] = NDArray[out_dtype](
        NDArrayShape(num, size=num)
    )
    for idx in range(num):
        result.data[idx] = start.cast[out_dtype]() + step.cast[out_dtype]() * idx

    return result


# ===------------------------------------------------------------------------===#
# Linear Spacing NDArray Generation
# ===------------------------------------------------------------------------===#


# I think defaulting parallelization to False is better
fn linspace[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    start: Scalar[in_dtype],
    stop: Scalar[in_dtype],
    num: Int = 50,
    endpoint: Bool = True,
    parallel: Bool = False,
) raises -> NDArray[out_dtype]:
    """
    Function that computes a series of linearly spaced values starting from "start" to "stop" with given size. Wrapper function for _linspace_serial, _linspace_parallel.

    Raises:
        Error if both in_dtype and out_dtype are integers or if in_dtype is a float and out_dtype is an integer.

    Parameters:
        in_dtype: Datatype of the input values.
        out_dtype: Datatype of the output NDArray.

    Args:
        start: Start value.
        stop: End value.
        num: No of linearly spaced elements.
        endpoint: Specifies whether to include endpoint in the final NDArray, defaults to True.
        parallel: Specifies whether the linspace should be calculated using parallelization, deafults to False.

    Returns:
        A NDArray of datatype `out_dtype` with elements ranging from `start` to `stop` with num elements.

    """
    if (is_inttype[in_dtype]() and is_inttype[out_dtype]()) or (
        is_floattype[in_dtype]() and is_inttype[out_dtype]()
    ):
        raise Error(
            "Both input and output datatypes cannot be integers. If the input is a float, the output must also be a float."
        )

    if parallel:
        return _linspace_parallel[out_dtype](
            start.cast[out_dtype](), stop.cast[out_dtype](), num, endpoint
        )
    else:
        return _linspace_serial[out_dtype](
            start.cast[out_dtype](), stop.cast[out_dtype](), num, endpoint
        )


fn _linspace_serial[
    dtype: DType
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
            result.data[i] = start + step * i

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num
        for i in range(num):
            result.data[i] = start + step * i

    return result


fn _linspace_parallel[
    dtype: DType
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
        var step: SIMD[dtype, 1] = (stop - start) / (num - 1.0)

        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            result.data[idx] = start + step * idx

        parallelize[parallelized_linspace](num)

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num

        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            result.data[idx] = start + step * idx

        parallelize[parallelized_linspace1](num)

    return result


# ===------------------------------------------------------------------------===#
# Logarithmic Spacing NDArray Generation
# ===------------------------------------------------------------------------===#
fn logspace[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    start: Scalar[in_dtype],
    stop: Scalar[in_dtype],
    num: Int,
    endpoint: Bool = True,
    base: Scalar[in_dtype] = 10.0,
    parallel: Bool = False,
) raises -> NDArray[out_dtype]:
    """
    Generate a logrithmic spaced NDArray of `num` elements between `start` and `stop`. Wrapper function for _logspace_serial, _logspace_parallel functions.

    Raises:
        Error if both in_dtype and out_dtype are integers or if in_dtype is a float and out_dtype is an integer.

    Parameters:
        in_dtype: Datatype of the input values.
        out_dtype: Datatype of the output NDArray.

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
    if (is_inttype[in_dtype]() and is_inttype[out_dtype]()) or (
        is_floattype[in_dtype]() and is_inttype[out_dtype]()
    ):
        raise Error(
            "Both input and output datatypes cannot be integers. If the input is a float, the output must also be a float."
        )
    if parallel:
        return _logspace_parallel[out_dtype](
            start.cast[out_dtype](),
            stop.cast[out_dtype](),
            num,
            base.cast[out_dtype](),
            endpoint,
        )
    else:
        return _logspace_serial[out_dtype](
            start.cast[out_dtype](),
            stop.cast[out_dtype](),
            num,
            base.cast[out_dtype](),
            endpoint,
        )


fn _logspace_serial[
    dtype: DType
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
            result.data[i] = base ** (start + step * i)
    else:
        var step: Scalar[dtype] = (stop - start) / num
        for i in range(num):
            result.data[i] = base ** (start + step * i)
    return result


fn _logspace_parallel[
    dtype: DType
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
            result.data[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace](num)

    else:
        var step: Scalar[dtype] = (stop - start) / num

        @parameter
        fn parallelized_logspace1(idx: Int) -> None:
            result.data[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace1](num)

    return result


# ! Outputs wrong values for Integer type, works fine for float type.
fn geomspace[
    in_dtype: DType, out_dtype: DType = DType.float64
](
    start: Scalar[in_dtype],
    stop: Scalar[in_dtype],
    num: Int,
    endpoint: Bool = True,
) raises -> NDArray[out_dtype]:
    """
    Generate a NDArray of `num` elements between `start` and `stop` in a geometric series.

    Raises:
        Error if both in_dtype and out_dtype are integers or if in_dtype is a float and out_dtype is an integer.

    Parameters:
        in_dtype: Datatype of the input values.
        out_dtype: Datatype of the output NDArray.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
        A NDArray of `dtype` with `num` geometrically spaced elements between `start` and `stop`.
    """

    if (is_inttype[in_dtype]() and is_inttype[out_dtype]()) or (
        is_floattype[in_dtype]() and is_inttype[out_dtype]()
    ):
        raise Error(
            "Both input and output datatypes cannot be integers. If the input is a float, the output must also be a float."
        )

    var a: Scalar[out_dtype] = start.cast[out_dtype]()

    if endpoint:
        var result: NDArray[out_dtype] = NDArray[out_dtype](NDArrayShape(num))
        var r: Scalar[out_dtype] = (
            stop.cast[out_dtype]() / start.cast[out_dtype]()
        ) ** (1 / (num - 1)).cast[out_dtype]()
        for i in range(num):
            result.data[i] = a * r**i
        return result

    else:
        var result: NDArray[out_dtype] = NDArray[out_dtype](NDArrayShape(num))
        var r: Scalar[out_dtype] = (
            stop.cast[out_dtype]() / start.cast[out_dtype]()
        ) ** (1 / (num)).cast[out_dtype]()
        for i in range(num):
            result.data[i] = a * r**i
        return result


# ===------------------------------------------------------------------------===#
# Commonly used NDArray Generation routines
# ===------------------------------------------------------------------------===#


# empty basically has to be either random or zero, can't return a purely empty matrix I think.
fn empty[dtype: DType](*shape: Int) raises -> NDArray[dtype]:
    """
    Generate a NDArray of given shape with arbitrary values.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        shape: Shape of the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.
    """
    return NDArray[dtype](shape, fill=0)


fn zeros[dtype: DType](*shape: Int) raises -> NDArray[dtype]:
    """
    Generate a NDArray of zeros with given shape.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        shape: Shape of the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.
    """
    return NDArray[dtype](shape, random=False)


fn eye[dtype: DType](N: Int, M: Int) raises -> NDArray[dtype]:
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
    var result: NDArray[dtype] = NDArray[dtype](N, M, random=False)
    var one = Scalar[dtype](1)
    for i in range(min(N, M)):
        result.store[1](i, i, val=one)
    return result


fn identity[dtype: DType](N: Int) raises -> NDArray[dtype]:
    """
    Generate an identity matrix of size N x N.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        N: Size of the matrix.

    Returns:
        A NDArray of `dtype` with size N x N and ones on the diagonals.
    """
    var result: NDArray[dtype] = NDArray[dtype](N, N, random=False)
    var one = Scalar[dtype](1)
    for i in range(N):
        result.store[1](i, i, val=one)
    return result


fn ones[dtype: DType](*shape: Int) raises -> NDArray[dtype]:
    """
    Generate a NDArray of ones with given shape filled with ones.

    Parameters:
        dtype: Datatype of the NDArray.

    Args:
        shape: Shape of the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.
    """
    var tens_shape: VariadicList[Int] = shape
    var res = NDArray[dtype](tens_shape)
    for i in range(res.num_elements()):
        res.store(i, SIMD[dtype, 1](1))
    return res


fn full[
    dtype: DType
](*shape: Int, fill_value: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Generate a NDArray of `fill_value` with given shape.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        shape: Shape of the NDArray.
        fill_value: Value to be splatted over the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.
    """
    return NDArray[dtype](shape, fill=fill_value)


fn full[
    dtype: DType
](shape: VariadicList[Int], fill_value: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Generate a NDArray of `fill_value` with given shape.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        shape: Shape of the NDArray.
        fill_value: Value to be splatted over the NDArray.

    Returns:
        A NDArray of `dtype` with given `shape`.
    """
    var tens_value: SIMD[dtype, 1] = SIMD[dtype, 1](fill_value).cast[dtype]()
    return NDArray[dtype](shape, fill=tens_value)


fn diagflat():
    pass


fn tri():
    pass


fn tril():
    pass


fn triu():
    pass
