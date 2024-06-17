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
    Indtype: DType, Outdtype: DType = DType.float64
](
    start: SIMD[Indtype, 1],
    stop: SIMD[Indtype, 1],
    step: SIMD[Indtype, 1] = SIMD[Indtype, 1](1),
) raises -> NDArray[Outdtype]:
    """
    Function that computes a series of values starting from "start" to "stop" with given "step" size.

    Parameters:
        Indtype: DType  - datatype of the input values.
        Outdtype: DType  - datatype of the output NDArray.

    Args:
        start: SIMD[Indtype, 1] - Start value.
        stop: SIMD[Indtype, 1]  - End value.
        step: SIMD[Indtype, 1]  - Step size between each element defualt 1.

    Returns:
        NDArray[dtype] - NDArray of datatype T with elements ranging from "start" to "stop" incremented with "step".
    """
    if is_inttype[Indtype]() and is_inttype[Outdtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )

    var num: Int = ((stop - start) / step).__int__()
    var result: NDArray[Outdtype] = NDArray[Outdtype](NDArrayShape(num))
    for idx in range(num):
        result[idx] = (start + step * idx).cast[Outdtype]()
    return result


# ===------------------------------------------------------------------------===#
# Linear Spacing NDArray Generation
# ===------------------------------------------------------------------------===#


# I think defaulting parallelization to False is better
fn linspace[
    Indtype: DType, Outdtype: DType = DType.float64
](
    start: SIMD[Indtype, 1],
    stop: SIMD[Indtype, 1],
    num: Int = 50,
    endpoint: Bool = True,
    parallel: Bool = False,
) raises -> NDArray[Outdtype]:
    """
    Function that computes a series of linearly spaced values starting from "start" to "stop" with given size. Wrapper function for _linspace_serial, _linspace_parallel.

    Parameters:
        Indtype: DType - datatype of the input values.
        Outdtype: DType - datatype of the output NDArray.

    Args:
        start: SIMD[Indtype, 1] - Start value.
        stop: SIMD[Indtype, 1]  - End value.
        num: Int  - No of linearly spaced elements.
        endpoint: Bool - Specifies whether to include endpoint in the final NDArray, defaults to True.
        parallel: Bool - Specifies whether the linspace should be calculated using parallelization, deafults to False.

    Returns:
        NDArray[dtype] - NDArray of datatype T with elements ranging from "start" to "stop" with num elements.

    """
    if is_inttype[Indtype]() and is_inttype[Outdtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )

    if parallel:
        return _linspace_parallel[Outdtype](
            start.cast[Outdtype](), stop.cast[Outdtype](), num, endpoint
        )
    else:
        return _linspace_serial[Outdtype](
            start.cast[Outdtype](), stop.cast[Outdtype](), num, endpoint
        )


fn _linspace_serial[
    dtype: DType
](
    start: SIMD[dtype, 1],
    stop: SIMD[dtype, 1],
    num: Int,
    endpoint: Bool = True,
) -> NDArray[dtype]:
    """
    Generate a linearly spaced NDArray of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        dtype: DType - datatype of the output NDArray.

    Args:
        start: SIMD[dtype, 1] - The starting value of the NDArray.
        stop: SIMD[dtype, 1] - The ending value of the NDArray.
        num: Int - The number of elements in the NDArray.
        endpoint: Bool - Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
    - A NDArray of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))

    if endpoint:
        var step: SIMD[dtype, 1] = (stop - start) / (num - 1)
        for i in range(num):
            result[i] = start + step * i

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num
        for i in range(num):
            result[i] = start + step * i

    return result


fn _linspace_parallel[
    dtype: DType
](
    start: SIMD[dtype, 1], stop: SIMD[dtype, 1], num: Int, endpoint: Bool = True
) -> NDArray[dtype]:
    """
    Generate a linearly spaced NDArray of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        start: SIMD[dtype, 1] - The starting value of the NDArray.
        stop: SIMD[dtype, 1] - The ending value of the NDArray.
        num: Int - The number of elements in the NDArray.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
    - A NDArray of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))
    alias nelts = simdwidthof[dtype]()

    if endpoint:
        var step: SIMD[dtype, 1] = (stop - start) / (num - 1.0)

        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            result[idx] = start + step * idx

        parallelize[parallelized_linspace](num)

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num

        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            result[idx] = start + step * idx

        parallelize[parallelized_linspace1](num)

    return result


# ===------------------------------------------------------------------------===#
# Logarithmic Spacing NDArray Generation
# ===------------------------------------------------------------------------===#
fn logspace[
    Indtype: DType, Outdtype: DType = DType.float64
](
    start: SIMD[Indtype, 1],
    stop: SIMD[Indtype, 1],
    num: Int,
    endpoint: Bool = True,
    base: SIMD[Indtype, 1] = 10.0,
    parallel: Bool = False,
) raises -> NDArray[Outdtype]:
    """
    Generate a logrithmic spaced NDArray of `num` elements between `start` and `stop`. Wrapper function for _logspace_serial, _logspace_parallel functions.

    Parameters:
        Indtype: DType - datatype of the input values.
        Outdtype: DType - datatype of the output NDArray.

    Args:
        start: SIMD[dtype, 1] - The starting value of the NDArray.
        stop: SIMD[dtype, 1] - The ending value of the NDArray.
        num: Int - The number of elements in the NDArray.
        endpoint: Bool - Whether to include the `stop` value in the NDArray. Defaults to True.
        base: SIMD[Indtype, 1] - Base value of the logarithm, defaults to 10.
        parallel: Bool - Specifies whether to calculate the logarithmic spaced values using parallelization.

    Returns:
    - A NDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    if is_inttype[Indtype]() and is_inttype[Outdtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )
    if parallel:
        return _logspace_parallel[Outdtype](
            start.cast[Outdtype](),
            stop.cast[Outdtype](),
            num,
            base.cast[Outdtype](),
            endpoint,
        )
    else:
        return _logspace_serial[Outdtype](
            start.cast[Outdtype](),
            stop.cast[Outdtype](),
            num,
            base.cast[Outdtype](),
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
) -> NDArray[dtype]:
    """
    Generate a logarithmic spaced NDArray of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        start: SIMD[dtype, 1] - The starting value of the NDArray.
        stop: SIMD[dtype, 1] - The ending value of the NDArray.
        num: Int - The number of elements in the NDArray.
        base: SIMD[dtype, 1] - Base value of the logarithm, defaults to 10.
        endpoint: Bool - Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
    - A NDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)
        for i in range(num):
            result[i] = base ** (start + step * i)
    else:
        var step: Scalar[dtype] = (stop - start) / num
        for i in range(num):
            result[i] = base ** (start + step * i)
    return result


fn _logspace_parallel[
    dtype: DType
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    num: Int,
    base: Scalar[dtype],
    endpoint: Bool = True,
) -> NDArray[dtype]:
    """
    Generate a logarithmic spaced NDArray of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        start: SIMD[dtype, 1] - The starting value of the NDArray.
        stop: SIMD[dtype, 1] - The ending value of the NDArray.
        num: Int - The number of elements in the NDArray.
        base: SIMD[dtype, 1] - Base value of the logarithm, defaults to 10.
        endpoint: Bool - Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
    - A NDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))

    if endpoint:
        var step: Scalar[dtype] = (stop - start) / (num - 1)

        @parameter
        fn parallelized_logspace(idx: Int) -> None:
            result[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace](num)

    else:
        var step: Scalar[dtype] = (stop - start) / num

        @parameter
        fn parallelized_logspace1(idx: Int) -> None:
            result[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace1](num)

    return result


# ! Outputs wrong values for Integer type, works fine for float type.
fn geomspace[
    Indtype: DType, Outdtype: DType = DType.float64
](
    start: SIMD[Indtype, 1],
    stop: SIMD[Indtype, 1],
    num: Int,
    endpoint: Bool = True,
) raises -> NDArray[Outdtype]:
    """
    Generate a NDArray of `num` elements between `start` and `stop` in a geometric series.

    Parameters:
        Indtype: DType - datatype of the input values.
        Outdtype: DType - datatype of the output NDArray.

    Args:
        start: SIMD[Indtype, 1] - The starting value of the NDArray.
        stop: SIMD[Indtype, 1] - The ending value of the NDArray.
        num: Int - The number of elements in the NDArray.
        endpoint: Bool - Whether to include the `stop` value in the NDArray. Defaults to True.

    Constraints:
        `Outdtype` must be a float type

    Returns:
    - A NDArray of `dtype` with `num` geometrically spaced elements between `start` and `stop`.
    """

    if is_inttype[Indtype]() and is_inttype[Outdtype]():
        raise Error(
            "Input and output cannot be `Int` datatype as it may lead to"
            " precision errors"
        )

    var a: Scalar[Outdtype] = start.cast[Outdtype]()

    if endpoint:
        var result: NDArray[Outdtype] = NDArray[Outdtype](NDArrayShape(num))
        var r: Scalar[Outdtype] = (stop / start) ** (1 / (num - 1))
        for i in range(num):
            result[i] = a * r**i
        return result

    else:
        var result: NDArray[Outdtype] = NDArray[Outdtype](NDArrayShape(num))
        var r: Scalar[Outdtype] = (stop / start) ** (1 / (num - 1))
        for i in range(num):
            result[i] = a * r**i
        return result


# ===------------------------------------------------------------------------===#
# Commonly used NDArray Generation routines
# ===------------------------------------------------------------------------===#


fn empty[dtype: DType](*shape: Int) -> NDArray[dtype]:
    """
    Generate a NDArray of given shape with arbitrary values.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        shape: VariadicList[Int] - Shape of the NDArray.

    Returns:
    - A NDArray of `dtype` with given `shape`.
    """
    var tens_shape: NDArrayShape = NDArrayShape(shape)
    return NDArray[dtype](shape=tens_shape, random=True)


fn zeros[dtype: DType](*shape: Int) -> NDArray[dtype]:
    """
    Generate a NDArray of zeros with given shape.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        shape: VariadicList[Int] - Shape of the NDArray.

    Returns:
    - A NDArray of `dtype` with given `shape`.
    """
    var tens_shape: VariadicList[Int] = shape
    return NDArray[dtype](tens_shape)


fn eye[dtype: DType](N: Int, M: Int) -> NDArray[dtype]:
    """
    Return a 2-D NDArray with ones on the diagonal and zeros elsewhere.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        N: Int - Number of rows in the matrix.
        M: Int - Number of columns in the matrix.

    Returns:
    - A NDArray of `dtype` with size N x M and ones on the diagonals.
    """
    var result: NDArray[dtype] = NDArray[dtype](N, M)
    var one = Scalar[dtype](1)
    for idx in range(M * N):
        if (idx + 1 - M) // N == 0:
            result[idx] = one
        else:
            continue
    return result


fn identity[dtype: DType](n: Int) -> NDArray[dtype]:
    """
    Generate an identity matrix of size N x N.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        n: Int - Size of the matrix.

    Returns:
    - A NDArray of `dtype` with size N x N and ones on the diagonals.
    """
    return eye[dtype](n, n)


fn ones[dtype: DType](*shape: Int) -> NDArray[dtype]:
    """
    Generate a NDArray of ones with given shape filled with ones.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        shape: VariadicList[Int] - Shape of the NDArray.

    Returns:
    - A NDArray of `dtype` with given `shape`.
    """
    var tens_shape: VariadicList[Int] = shape
    return NDArray[dtype](tens_shape, Scalar[dtype](1))


fn full[dtype: DType](fill_value: Scalar[dtype], *shape: Int) -> NDArray[dtype]:
    """
    Generate a NDArray of `fill_value` with given shape.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        fill_value: Scalar[dtype] - value to be splatted over the NDArray.
        shape: VariadicList[Int] - Shape of the NDArray.

    Returns:
    - A NDArray of `dtype` with given `shape`.
    """
    # var tens_shape: VariadicList[Int] = shape
    var tens_shape: NDArrayShape = NDArrayShape(shape)
    return NDArray[dtype](shape=tens_shape, value=fill_value)


fn full[
    dtype: DType
](fill_value: Scalar[dtype], shape: VariadicList[Int]) -> NDArray[dtype]:
    """
    Generate a NDArray of `fill_value` with given shape.

    Parameters:
        dtype: DType - datatype of the NDArray.

    Args:
        fill_value: Scalar[dtype] - value to be splatted over the NDArray.
        shape: VariadicList[Int] - Shape of the NDArray.

    Returns:
    - A NDArray of `dtype` with given `shape`.
    """
    var tens_shape: NDArrayShape = NDArrayShape(shape)
    var tens_value: SIMD[dtype, 1] = SIMD[dtype, 1](fill_value).cast[dtype]()
    return NDArray[dtype](shape=tens_shape, value=tens_value)


fn diagflat():
    pass


fn tri():
    pass


fn tril():
    pass


fn triu():
    pass
