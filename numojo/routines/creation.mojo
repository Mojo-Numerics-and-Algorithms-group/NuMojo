# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Array creation routine.

# TODO (In order of priority)
1) Implement axis argument for the NDArray creation functions
2) Separate `array(object)` and `NDArray.__init__(shape)`.
3) Use `Shapelike` trait to replace `NDArrayShape`, `List`, `VariadicList` and reduce the number of function reloads.
4) Simplify complex overloads into sum of real methods.

---

Use more uniformed way of calling functions, i.e., using one specific
Use more uniformed way of calling functions, i.e., using one specific
overload for each function. This makes maintenance easier. Example:

- `NDArray.__init__` takes in `ShapeLike` and initialize an `NDArray` container.
- `full` calls `NDArray.__init__`.
- `zeros`, `ones` calls `full`.
- Other functions calls `zeros`, `ones`, `full`.

If overloads are needed, it is better to call the default signature in other overloads. Example: `zeros(shape: NDArrayShape)`. All other overloads call this function. So it is easy for modification.

"""

from algorithm import parallelize, vectorize
from algorithm import parallelize, vectorize
from builtin.math import pow
from collections import Dict
from collections.optional import Optional
from memory import UnsafePointer, memset_zero, memset, memcpy
from algorithm.memory import parallel_memcpy
from python import PythonObject, Python
from sys import simd_width_of


from numojo.core.flags import Flags
from numojo.core.ndarray import NDArray
from numojo.core.complex import ComplexScalar
from numojo.core.ndshape import NDArrayShape
from numojo.core.utility import _get_offset
from numojo.core.own_data import OwnData


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
    Function that computes a series of values starting from "start" to "stop"
    with given "step" size.

    Raises:
        Error if both dtype and dtype are integers or if dtype is a float and
        dtype is an integer.

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
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))
    for idx in range(num):
        result._buf.ptr[idx] = start + step * idx

    return result^


fn arange[
    dtype: DType = DType.float64
](stop: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    (Overload) When start is 0 and step is 1.
    """

    var size: Int = Int(stop)  # TODO: handle negative values.
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(size))
    for i in range(size):
        (result._buf.ptr + i).init_pointee_copy(Scalar[dtype](i))

    return result^


fn arange[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    step: ComplexSIMD[cdtype] = ComplexSIMD[cdtype](1, 1),
) raises -> ComplexNDArray[cdtype]:
    """
    Function that computes a series of values starting from "start" to "stop"
    with given "step" size.

    Raises:
        Error if both dtype and dtype are integers or if dtype is a float and
        dtype is an integer.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: ComplexSIMD[cdtype] - Start value.
        stop: ComplexSIMD[cdtype]  - End value.
        step: ComplexSIMD[cdtype]  - Step size between each element (default 1).

    Returns:
        A ComplexNDArray of datatype `dtype` with elements ranging from `start` to `stop` incremented with `step`.
    """
    var num_re: Int = ((stop.re - start.re) / step.re).__int__()
    var num_im: Int = ((stop.im - start.im) / step.im).__int__()
    if num_re != num_im:
        raise Error(
            String(
                "Number of real and imaginary parts are not equal {} != {}"
            ).format(num_re, num_im)
        )
    var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](Shape(num_re))
    for idx in range(num_re):
        result.store[width=1](
            idx,
            ComplexSIMD[cdtype](
                start.re + step.re * idx, start.im + step.im * idx
            ),
        )

    return result^


fn arange[
    cdtype: ComplexDType = ComplexDType.float64,
](stop: ComplexSIMD[cdtype]) raises -> ComplexNDArray[cdtype]:
    """
    (Overload) When start is 0 and step is 1.
    """
    var size_re = Int(stop.re)
    var size_im = Int(stop.im)
    if size_re != size_im:
        raise Error(
            String(
                "Number of real and imaginary parts are not equal {} != {}"
            ).format(size_re, size_im)
        )

    var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](Shape(size_re))
    for i in range(size_re):
        result.store[width=1](
            i,
            ComplexSIMD[cdtype](Scalar[cdtype._dtype](i)),
        )

    return result^


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
            result._buf.ptr[i] = start + step * i

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num
        for i in range(num):
            result._buf.ptr[i] = start + step * i

    return result^


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
    alias nelts = simd_width_of[dtype]()

    if endpoint:
        var denominator: SIMD[dtype, 1] = Scalar[dtype](num) - 1.0
        var step: SIMD[dtype, 1] = (stop - start) / denominator

        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            result._buf.ptr[idx] = start + step * idx

        parallelize[parallelized_linspace](num)

    else:
        var step: SIMD[dtype, 1] = (stop - start) / num

        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            result._buf.ptr[idx] = start + step * idx

        parallelize[parallelized_linspace1](num)

    return result^


fn linspace[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    num: Int = 50,
    endpoint: Bool = True,
    parallel: Bool = False,
) raises -> ComplexNDArray[cdtype]:
    """
    Function that computes a series of linearly spaced values starting from "start" to "stop" with given size. Wrapper function for _linspace_serial, _linspace_parallel.

    Raises:
        Error if dtype is an integer.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: Start value.
        stop: End value.
        num: No of linearly spaced elements.
        endpoint: Specifies whether to include endpoint in the final ComplexNDArray, defaults to True.
        parallel: Specifies whether the linspace should be calculated using parallelization, deafults to False.

    Returns:
        A ComplexNDArray of `dtype` with `num` linearly spaced elements between `start` and `stop`.

    """
    constrained[not cdtype.is_integral()]()
    if parallel:
        return _linspace_parallel[cdtype](start, stop, num, endpoint)
    else:
        return _linspace_serial[cdtype](start, stop, num, endpoint)


fn _linspace_serial[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    num: Int,
    endpoint: Bool = True,
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a linearly spaced NDArray of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: The starting value of the NDArray.
        stop: The ending value of the NDArray.
        num: The number of elements in the NDArray.
        endpoint: Whether to include the `stop` value in the NDArray. Defaults to True.

    Returns:
        A ComplexNDArray of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    alias dtype: DType = cdtype._dtype
    var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](Shape(num))

    if endpoint:
        var step_re: Scalar[dtype] = (stop.re - start.re) / (num - 1)
        var step_im: Scalar[dtype] = (stop.im - start.im) / (num - 1)
        for i in range(num):
            result.store[width=1](
                i,
                ComplexSIMD[cdtype](
                    start.re + step_re * i, start.im + step_im * i
                ),
            )

    else:
        var step_re: Scalar[dtype] = (stop.re - start.re) / num
        var step_im: Scalar[dtype] = (stop.im - start.im) / num
        for i in range(num):
            result.store[width=1](
                i,
                ComplexSIMD[cdtype](
                    start.re + step_re * i, start.im + step_im * i
                ),
            )

    return result^


fn _linspace_parallel[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    num: Int,
    endpoint: Bool = True,
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a linearly spaced ComplexNDArray of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: The starting value of the ComplexNDArray.
        stop: The ending value of the ComplexNDArray.
        num: The number of elements in the ComplexNDArray.
        endpoint: Whether to include the `stop` value in the ComplexNDArray. Defaults to True.

    Returns:
        A ComplexNDArray of `dtype` with `num` linearly spaced elements between `start` and `stop`.
    """
    alias dtype: DType = cdtype._dtype
    alias nelts = simd_width_of[dtype]()
    var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](Shape(num))

    if endpoint:
        var denominator: Scalar[dtype] = Scalar[dtype](num) - 1.0
        var step_re: Scalar[dtype] = (stop.re - start.re) / denominator
        var step_im: Scalar[dtype] = (stop.im - start.im) / denominator

        # need better error handling here later
        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            try:
                result.store[width=1](
                    idx,
                    ComplexSIMD[cdtype](
                        start.re + step_re * idx, start.im + step_im * idx
                    ),
                )
            except:
                print("Error in parallelized_linspace")

        parallelize[parallelized_linspace](num)

    else:
        var step_re: Scalar[dtype] = (stop.re - start.re) / num
        var step_im: Scalar[dtype] = (stop.im - start.im) / num

        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            try:
                result.store[width=1](
                    idx,
                    ComplexSIMD[cdtype](
                        start.re + step_re * idx, start.im + step_im * idx
                    ),
                )
            except:
                print("Error in parallelized_linspace1")

        parallelize[parallelized_linspace1](num)

    return result^


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
            result._buf.ptr[i] = base ** (start + step * i)
    else:
        var step: Scalar[dtype] = (stop - start) / num
        for i in range(num):
            result._buf.ptr[i] = base ** (start + step * i)
    return result^


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
            result._buf.ptr[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace](num)

    else:
        var step: Scalar[dtype] = (stop - start) / num

        @parameter
        fn parallelized_logspace1(idx: Int) -> None:
            result._buf.ptr[idx] = base ** (start + step * idx)

        parallelize[parallelized_logspace1](num)

    return result^


fn logspace[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    num: Int,
    endpoint: Bool = True,
    base: ComplexSIMD[cdtype] = ComplexSIMD[cdtype](10.0, 10.0),
    parallel: Bool = False,
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a logrithmic spaced ComplexNDArray of `num` elements between `start` and `stop`. Wrapper function for _logspace_serial, _logspace_parallel functions.

    Raises:
        Error if dtype is an integer.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: The starting value of the ComplexNDArray.
        stop: The ending value of the ComplexNDArray.
        num: The number of elements in the ComplexNDArray.
        endpoint: Whether to include the `stop` value in the ComplexNDArray. Defaults to True.
        base: Base value of the logarithm, defaults to 10.
        parallel: Specifies whether to calculate the logarithmic spaced values using parallelization.

    Returns:
    - A ComplexNDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    constrained[not cdtype.is_integral()]()
    if parallel:
        return _logspace_parallel[cdtype](
            start,
            stop,
            num,
            base,
            endpoint,
        )
    else:
        return _logspace_serial[cdtype](
            start,
            stop,
            num,
            base,
            endpoint,
        )


fn _logspace_serial[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    num: Int,
    base: ComplexSIMD[cdtype],
    endpoint: Bool = True,
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a logarithmic spaced ComplexNDArray of `num` elements between `start` and `stop` using naive for loop.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: The starting value of the ComplexNDArray.
        stop: The ending value of the ComplexNDArray.
        num: The number of elements in the ComplexNDArray.
        base: Base value of the logarithm, defaults to 10.
        endpoint: Whether to include the `stop` value in the ComplexNDArray. Defaults to True.

    Returns:
        A ComplexNDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    alias dtype: DType = cdtype._dtype
    var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](
        NDArrayShape(num)
    )

    if endpoint:
        var step_re: Scalar[dtype] = (stop.re - start.re) / (num - 1)
        var step_im: Scalar[dtype] = (stop.im - start.im) / (num - 1)
        for i in range(num):
            result.store[1](
                i,
                ComplexSIMD[cdtype](
                    base.re ** (start.re + step_re * i),
                    base.im ** (start.im + step_im * i),
                ),
            )
    else:
        var step_re: Scalar[dtype] = (stop.re - start.re) / num
        var step_im: Scalar[dtype] = (stop.im - start.im) / num
        for i in range(num):
            result.store[1](
                i,
                ComplexSIMD[cdtype](
                    base.re ** (start.re + step_re * i),
                    base.im ** (start.im + step_im * i),
                ),
            )
    return result^


fn _logspace_parallel[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    num: Int,
    base: ComplexSIMD[cdtype],
    endpoint: Bool = True,
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a logarithmic spaced ComplexNDArray of `num` elements between `start` and `stop` using parallelization.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: The starting value of the ComplexNDArray.
        stop: The ending value of the ComplexNDArray.
        num: The number of elements in the ComplexNDArray.
        base: Base value of the logarithm, defaults to 10.
        endpoint: Whether to include the `stop` value in the ComplexNDArray. Defaults to True.

    Returns:
        A ComplexNDArray of `dtype` with `num` logarithmic spaced elements between `start` and `stop`.
    """
    alias dtype: DType = cdtype._dtype
    var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](
        NDArrayShape(num)
    )

    if endpoint:
        var step_re: Scalar[dtype] = (stop.re - start.re) / (num - 1)
        var step_im: Scalar[dtype] = (stop.im - start.im) / (num - 1)

        @parameter
        fn parallelized_logspace(idx: Int) -> None:
            try:
                result.store[1](
                    idx,
                    ComplexSIMD[cdtype](
                        base.re ** (start.re + step_re * idx),
                        base.im ** (start.im + step_im * idx),
                    ),
                )
            except:
                print("Error in parallelized_logspace")

        parallelize[parallelized_logspace](num)

    else:
        var step_re: Scalar[dtype] = (stop.re - start.re) / num
        var step_im: Scalar[dtype] = (stop.im - start.im) / num

        @parameter
        fn parallelized_logspace1(idx: Int) -> None:
            try:
                result.store[1](
                    idx,
                    ComplexSIMD[cdtype](
                        base.re ** (start.re + step_re * idx),
                        base.im ** (start.im + step_im * idx),
                    ),
                )
            except:
                print("Error in parallelized_logspace")

        parallelize[parallelized_logspace1](num)

    return result^


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
        var base: Scalar[dtype] = stop / start
        var power: Scalar[dtype] = 1 / Scalar[dtype](num - 1)
        var r: Scalar[dtype] = base**power
        for i in range(num):
            result._buf.ptr[i] = a * r**i
        return result^

    else:
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(num))
        var base: Scalar[dtype] = stop / start
        var power: Scalar[dtype] = 1 / Scalar[dtype](num)
        var r: Scalar[dtype] = base**power
        for i in range(num):
            result._buf.ptr[i] = a * r**i
        return result^


fn geomspace[
    cdtype: ComplexDType = ComplexDType.float64,
](
    start: ComplexSIMD[cdtype],
    stop: ComplexSIMD[cdtype],
    num: Int,
    endpoint: Bool = True,
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a ComplexNDArray of `num` elements between `start` and `stop` in a geometric series.

    Raises:
        Error if dtype is an integer.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        start: The starting value of the ComplexNDArray.
        stop: The ending value of the ComplexNDArray.
        num: The number of elements in the ComplexNDArray.
        endpoint: Whether to include the `stop` value in the ComplexNDArray. Defaults to True.

    Returns:
        A ComplexNDArray of `dtype` with `num` geometrically spaced elements between `start` and `stop`.
    """
    constrained[
        not cdtype.is_integral(), "Int type will result to precision errors."
    ]()
    alias dtype: DType = cdtype._dtype
    var a: ComplexSIMD[cdtype] = start

    if endpoint:
        var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](
            NDArrayShape(num)
        )
        var base: ComplexSIMD[cdtype] = stop / start
        var power: Scalar[dtype] = 1 / Scalar[dtype](num - 1)
        var r: ComplexSIMD[cdtype] = base**power
        for i in range(num):
            result.store[1](
                i,
                ComplexSIMD[cdtype](a.re * r.re**i, a.im * r.im**i),
            )
        return result^

    else:
        var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](
            NDArrayShape(num)
        )
        var base: ComplexSIMD[cdtype] = stop / start
        var power: Scalar[dtype] = 1 / Scalar[dtype](num)
        var r: ComplexSIMD[cdtype] = base**power
        for i in range(num):
            result.store[1](
                i,
                ComplexSIMD[cdtype](a.re * r.re**i, a.im * r.im**i),
            )
        return result^


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
    return empty[dtype](shape=NDArrayShape(shape))


fn empty[
    dtype: DType = DType.float64
](shape: VariadicList[Int]) raises -> NDArray[dtype]:
    """Overload of function `empty` that reads a variadic list of ints."""
    return empty[dtype](shape=NDArrayShape(shape))


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
    return NDArray[dtype](shape=array.shape)


fn empty[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: NDArrayShape) raises -> ComplexNDArray[cdtype]:
    """
    Generate an empty ComplexNDArray of given shape with arbitrary values.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        shape: Shape of the ComplexNDArray.

    Returns:
        A ComplexNDArray of `dtype` with given `shape`.
    """
    return ComplexNDArray[cdtype](shape=shape)


fn empty[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: List[Int]) raises -> ComplexNDArray[cdtype]:
    """Overload of function `empty` that reads a list of ints."""
    return empty[cdtype](shape=NDArrayShape(shape))


fn empty[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: VariadicList[Int]) raises -> ComplexNDArray[cdtype]:
    """Overload of function `empty` that reads a variadic list of ints."""
    return empty[cdtype](shape=NDArrayShape(shape))


fn empty_like[
    cdtype: ComplexDType = ComplexDType.float64,
](array: ComplexNDArray[cdtype]) raises -> ComplexNDArray[cdtype]:
    """
    Generate an empty ComplexNDArray of the same shape as `array`.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        array: ComplexNDArray to be used as a reference for the shape.

    Returns:
        A ComplexNDArray of `dtype` with the same shape as `array`.
    """
    return ComplexNDArray[cdtype](shape=array.shape)


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
    var result: NDArray[dtype] = zeros[dtype](NDArrayShape(N, M))
    var one: Scalar[dtype] = Scalar[dtype](1)
    for i in range(min(N, M)):
        result.store[1](i, i, val=one)
    return result^


fn eye[
    cdtype: ComplexDType = ComplexDType.float64,
](N: Int, M: Int) raises -> ComplexNDArray[cdtype]:
    """
    Return a 2-D ComplexNDArray with ones on the diagonal and zeros elsewhere.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        N: Number of rows in the matrix.
        M: Number of columns in the matrix.

    Returns:
        A ComplexNDArray of `dtype` with size N x M and ones on the diagonals.
    """
    var result: ComplexNDArray[cdtype] = zeros[cdtype](NDArrayShape(N, M))
    var one: ComplexSIMD[cdtype] = ComplexSIMD[cdtype](1, 1)
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
    var result: NDArray[dtype] = zeros[dtype](NDArrayShape(N, N))
    var one: Scalar[dtype] = Scalar[dtype](1)
    for i in range(N):
        result.store[1](i, i, val=one)
    return result^


fn identity[
    cdtype: ComplexDType = ComplexDType.float64,
](N: Int) raises -> ComplexNDArray[cdtype]:
    """
    Generate an Complex identity matrix of size N x N.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        N: Size of the matrix.

    Returns:
        A ComplexNDArray of `dtype` with size N x N and ones on the diagonals.
    """
    var result: ComplexNDArray[cdtype] = zeros[cdtype](NDArrayShape(N, N))
    var one: ComplexSIMD[cdtype] = ComplexSIMD[cdtype](1, 1)
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
    return ones[dtype](shape=NDArrayShape(shape))


fn ones[
    dtype: DType = DType.float64
](shape: VariadicList[Int]) raises -> NDArray[dtype]:
    """Overload of function `ones` that reads a variadic of ints."""
    return ones[dtype](shape=NDArrayShape(shape))


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
    return ones[dtype](shape=array.shape)


fn ones[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: NDArrayShape) raises -> ComplexNDArray[cdtype]:
    """
    Generate a ComplexNDArray of ones with given shape filled with ones.

    It calls the function `full`.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        shape: Shape of the ComplexNDArray.

    Returns:
        A ComplexNDArray of `dtype` with given `shape`.
    """
    return full[cdtype](shape=shape, fill_value=ComplexSIMD[cdtype](1, 1))


fn ones[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: List[Int]) raises -> ComplexNDArray[cdtype]:
    """Overload of function `ones` that reads a list of ints."""
    return ones[cdtype](shape=NDArrayShape(shape))


fn ones[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: VariadicList[Int]) raises -> ComplexNDArray[cdtype]:
    """Overload of function `ones` that reads a variadic of ints."""
    return ones[cdtype](shape=NDArrayShape(shape))


fn ones_like[
    cdtype: ComplexDType = ComplexDType.float64,
](array: ComplexNDArray[cdtype]) raises -> ComplexNDArray[cdtype]:
    """
    Generate a ComplexNDArray of the same shape as `a` filled with ones.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        array: ComplexNDArray to be used as a reference for the shape.

    Returns:
        A ComplexNDArray of `dtype` with the same shape as `a` filled with ones.
    """
    return full[cdtype](shape=array.shape, fill_value=ComplexSIMD[cdtype](1, 1))


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
    return zeros[dtype](shape=NDArrayShape(shape))


fn zeros[
    dtype: DType = DType.float64
](shape: VariadicList[Int]) raises -> NDArray[dtype]:
    """Overload of function `zeros` that reads a variadic list of ints."""
    return zeros[dtype](shape=NDArrayShape(shape))


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
    return full[dtype](shape=array.shape, fill_value=0)


fn zeros[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: NDArrayShape) raises -> ComplexNDArray[cdtype]:
    """
    Generate a ComplexNDArray of zeros with given shape.

    It calls the function `full` with `fill_value` set to `ComplexSIMD[cdtype](0, 0)`.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        shape: Shape of the ComplexNDArray.

    Returns:
        A ComplexNDArray of `dtype` with given `shape`.

    """
    return full[cdtype](shape=shape, fill_value=ComplexSIMD[cdtype](0, 0))


fn zeros[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: List[Int]) raises -> ComplexNDArray[cdtype]:
    """Overload of function `zeros` that reads a list of ints."""
    return zeros[cdtype](shape=NDArrayShape(shape))


fn zeros[
    cdtype: ComplexDType = ComplexDType.float64,
](shape: VariadicList[Int]) raises -> ComplexNDArray[cdtype]:
    """Overload of function `zeros` that reads a variadic list of ints."""
    return zeros[cdtype](shape=NDArrayShape(shape))


fn zeros_like[
    cdtype: ComplexDType = ComplexDType.float64,
](array: ComplexNDArray[cdtype]) raises -> ComplexNDArray[cdtype]:
    """
    Generate a ComplexNDArray of the same shape as `a` filled with zeros.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        array: ComplexNDArray to be used as a reference for the shape.

    Returns:
        A ComplexNDArray of `dtype` with the same shape as `a` filled with zeros.
    """
    return full[cdtype](shape=array.shape, fill_value=ComplexSIMD[cdtype](0, 0))


fn full[
    dtype: DType = DType.float64
](
    shape: NDArrayShape, fill_value: Scalar[dtype], order: String = "C"
) raises -> NDArray[dtype]:
    """Initialize an NDArray of certain shape fill it with a given value.

    Args:
        shape: Shape of the array.
        fill_value: Set all the values to this.
        order: Memory order C or F.

    Example:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var a = nm.full(Shape(2,3,4), fill_value=10)
        ```
    """

    var A = NDArray[dtype](shape=shape, order=order)
    for i in range(A.size):
        A._buf.ptr[i] = fill_value
    return A^


fn full[
    dtype: DType = DType.float64
](
    shape: List[Int], fill_value: Scalar[dtype], order: String = "C"
) raises -> NDArray[dtype]:
    """Overload of function `full` that reads a list of ints."""
    return full[dtype](
        shape=NDArrayShape(shape), fill_value=fill_value, order=order
    )


fn full[
    dtype: DType = DType.float64
](
    shape: VariadicList[Int], fill_value: Scalar[dtype], order: String = "C"
) raises -> NDArray[dtype]:
    """Overload of function `full` that reads a variadic list of ints."""
    return full[dtype](
        shape=NDArrayShape(shape), fill_value=fill_value, order=order
    )


fn full_like[
    dtype: DType = DType.float64
](
    array: NDArray[dtype], fill_value: Scalar[dtype], order: String = "C"
) raises -> NDArray[dtype]:
    """
    Generate a NDArray of the same shape as `a` filled with `fill_value`.

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        array: NDArray to be used as a reference for the shape.
        fill_value: Value to fill the NDArray with.
        order: Memory order C or F.

    Returns:
        A NDArray of `dtype` with the same shape as `a` filled with `fill_value`.
    """
    return full[dtype](shape=array.shape, fill_value=fill_value, order=order)


fn full[
    cdtype: ComplexDType = ComplexDType.float64
](
    shape: NDArrayShape,
    fill_value: ComplexSIMD[cdtype],
    order: String = "C",
) raises -> ComplexNDArray[cdtype]:
    """Initialize an ComplexNDArray of certain shape fill it with a given value.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        shape: Shape of the ComplexNDArray.
        fill_value: Set all the values to this.
        order: Memory order C or F.

    Example:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var a = nm.full[nm.cf32](Shape(2,3,4), fill_value=CScalar[nm.cf32](10, 10))
        ```
    """
    var A = ComplexNDArray[cdtype](shape=shape, order=order)
    for i in range(A.size):
        A._re._buf.ptr.store(i, fill_value.re)
        A._im._buf.ptr.store(i, fill_value.im)
    return A^


fn full[
    cdtype: ComplexDType = ComplexDType.float64
](
    shape: List[Int],
    fill_value: ComplexSIMD[cdtype],
    order: String = "C",
) raises -> ComplexNDArray[cdtype]:
    """Overload of function `full` that reads a list of ints."""
    return full[cdtype](
        shape=NDArrayShape(shape), fill_value=fill_value, order=order
    )


fn full[
    cdtype: ComplexDType = ComplexDType.float64
](
    shape: VariadicList[Int],
    fill_value: ComplexSIMD[cdtype],
    order: String = "C",
) raises -> ComplexNDArray[cdtype]:
    """Overload of function `full` that reads a variadic list of ints."""
    return full[cdtype](
        shape=NDArrayShape(shape), fill_value=fill_value, order=order
    )


fn full_like[
    cdtype: ComplexDType = ComplexDType.float64
](
    array: ComplexNDArray[cdtype],
    fill_value: ComplexSIMD[cdtype],
    order: String = "C",
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a ComplexNDArray of the same shape as `a` filled with `fill_value`.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        array: ComplexNDArray to be used as a reference for the shape.
        fill_value: Value to fill the ComplexNDArray with.
        order: Memory order C or F.

    Returns:
        A ComplexNDArray of `dtype` with the same shape as `a` filled with `fill_value`.
    """
    return full[cdtype](shape=array.shape, fill_value=fill_value, order=order)


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
        var n: Int = v.size
        var result: NDArray[dtype] = zeros[dtype](
            NDArrayShape(n + abs(k), n + abs(k))
        )
        if k >= 0:
            for i in range(n):
                result._buf.ptr[i * (n + abs(k) + 1) + k] = v._buf.ptr[i]
            return result^
        else:
            for i in range(n):
                result._buf.ptr[
                    result.size - 1 - i * (result.shape[1] + 1) + k
                ] = v._buf.ptr[n - 1 - i]
        return result^
    elif v.ndim == 2:
        var m: Int = v.shape[0]
        var n: Int = v.shape[1]
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(n - abs(k)))
        if k >= 0:
            for i in range(n - abs(k)):
                result._buf.ptr[i] = v._buf.ptr[i * (n + 1) + k]
        else:
            for i in range(n - abs(k)):
                result._buf.ptr[m - abs(k) - 1 - i] = v._buf.ptr[
                    v.size - 1 - i * (v.shape[1] + 1) + k
                ]
        return result^
    else:
        raise Error("Arrays bigger than 2D are not supported")


fn diag[
    cdtype: ComplexDType = ComplexDType.float64,
](v: ComplexNDArray[cdtype], k: Int = 0) raises -> ComplexNDArray[cdtype]:
    """
    Extract a diagonal or construct a diagonal ComplexNDArray.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        v: ComplexNDArray to extract the diagonal from.
        k: Diagonal offset.

    Returns:
        A 1-D ComplexNDArray with the diagonal of the input ComplexNDArray.
    """
    return ComplexNDArray[cdtype](
        re=diag[cdtype._dtype](v._re, k),
        im=diag[cdtype._dtype](v._im, k),
    )


fn diagflat[
    dtype: DType = DType.float64
](v: NDArray[dtype], k: Int = 0) raises -> NDArray[dtype]:
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
    var n: Int = v.size
    var result: NDArray[dtype] = zeros[dtype](
        NDArrayShape(n + abs(k), n + abs(k))
    )
    if k >= 0:
        for i in range(n):
            result.store((n + k + 1) * i + k, v._buf.ptr[i])
    else:
        for i in range(n):
            result.store(
                result.size - 1 - (n + abs(k) + 1) * i + k,
                v._buf.ptr[v.size - 1 - i],
            )
    return result^


fn diagflat[
    cdtype: ComplexDType = ComplexDType.float64,
](v: ComplexNDArray[cdtype], k: Int = 0) raises -> ComplexNDArray[cdtype]:
    """
    Generate a 2-D ComplexNDArray with the flattened input as the diagonal.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        v: ComplexNDArray to be flattened and used as the diagonal.
        k: Diagonal offset.

    Returns:
        A 2-D ComplexNDArray with the flattened input as the diagonal.
    """
    return ComplexNDArray[cdtype](
        re=diagflat[cdtype._dtype](v._re, k),
        im=diagflat[cdtype._dtype](v._im, k),
    )


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
    var result: NDArray[dtype] = zeros[dtype](NDArrayShape(N, M))
    for i in range(N):
        for j in range(M):
            if j <= i + k:
                result.store(i, j, val=Scalar[dtype](1))
    return result^


fn tri[
    cdtype: ComplexDType = ComplexDType.float64,
](N: Int, M: Int, k: Int = 0) raises -> ComplexNDArray[cdtype]:
    """
    Generate a 2-D ComplexNDArray with ones on and below the k-th diagonal.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        N: Number of rows in the matrix.
        M: Number of columns in the matrix.
        k: Diagonal offset.

    Returns:
        A 2-D ComplexNDArray with ones on and below the k-th diagonal.
    """
    return ComplexNDArray[cdtype](
        re=tri[cdtype._dtype](N, M, k),
        im=tri[cdtype._dtype](N, M, k),
    )


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
    var result: NDArray[
        dtype
    ] = m.copy()  # * We should move this to be inplace operation perhaps.
    if m.ndim == 2:
        for i in range(m.shape[0]):
            for j in range(i + 1 + k, m.shape[1]):
                result._buf.ptr[i * m.shape[1] + j] = Scalar[dtype](0)
    elif m.ndim >= 2:
        for i in range(m.ndim - 2):
            initial_offset *= m.shape[i]
        for i in range(m.ndim - 2, m.ndim):
            final_offset *= m.shape[i]
        for offset in range(initial_offset):
            offset = offset * final_offset
            for i in range(m.shape[-2]):
                for j in range(i + 1 + k, m.shape[-1]):
                    result._buf.ptr[offset + j + i * m.shape[-1]] = Scalar[
                        dtype
                    ](0)
    else:
        raise Error(
            "Arrays smaller than 2D are not supported for this operation."
        )
    return result^


fn tril[
    cdtype: ComplexDType = ComplexDType.float64,
](m: ComplexNDArray[cdtype], k: Int = 0) raises -> ComplexNDArray[cdtype]:
    """
    Zero out elements above the k-th diagonal.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        m: ComplexNDArray to be zeroed out.
        k: Diagonal offset.

    Returns:
        A ComplexNDArray with elements above the k-th diagonal zeroed out.
    """
    return ComplexNDArray[cdtype](
        re=tril[cdtype._dtype](m._re, k),
        im=tril[cdtype._dtype](m._im, k),
    )


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
    var result: NDArray[dtype] = m.copy()
    if m.ndim == 2:
        for i in range(m.shape[0]):
            for j in range(0, i + k):
                result._buf.ptr[i * m.shape[1] + j] = Scalar[dtype](0)
    elif m.ndim >= 2:
        for i in range(m.ndim - 2):
            initial_offset *= m.shape[i]
        for i in range(m.ndim - 2, m.ndim):
            final_offset *= m.shape[i]
        for offset in range(initial_offset):
            offset = offset * final_offset
            for i in range(m.shape[-2]):
                for j in range(0, i + k):
                    result._buf.ptr[offset + j + i * m.shape[-1]] = Scalar[
                        dtype
                    ](0)
    else:
        raise Error(
            "Arrays smaller than 2D are not supported for this operation."
        )
    return result^


fn triu[
    cdtype: ComplexDType = ComplexDType.float64,
](m: ComplexNDArray[cdtype], k: Int = 0) raises -> ComplexNDArray[cdtype]:
    """
    Zero out elements below the k-th diagonal.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        m: ComplexNDArray to be zeroed out.
        k: Diagonal offset.

    Returns:
        A ComplexNDArray with elements below the k-th diagonal zeroed out.
    """
    return ComplexNDArray[cdtype](
        re=triu[cdtype._dtype](m._re, k),
        im=triu[cdtype._dtype](m._im, k),
    )


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

    var n_rows = x.size
    var n_cols = N.value() if N else n_rows
    var result: NDArray[dtype] = ones[dtype](NDArrayShape(n_rows, n_cols))
    for i in range(n_rows):
        var x_i = x._buf.ptr[i]
        if increasing:
            for j in range(n_cols):
                result.store(i, j, val=x_i**j)
        else:
            for j in range(n_cols - 1, -1, -1):
                result.store(i, n_cols - 1 - j, val=x_i**j)
    return result^


fn vander[
    cdtype: ComplexDType = ComplexDType.float64,
](
    x: ComplexNDArray[cdtype],
    N: Optional[Int] = None,
    increasing: Bool = False,
) raises -> ComplexNDArray[cdtype]:
    """
    Generate a Complex Vandermonde matrix.

    Parameters:
        cdtype: Complex datatype of the output array.

    Args:
        x: 1-D input array.
        N: Number of columns in the output. If N is not specified, a square array is returned.
        increasing: Order of the powers of the columns. If True, the powers increase from left to right, if False (the default) they are reversed.

    Returns:
        A Complex Vandermonde matrix.
    """
    return ComplexNDArray[cdtype](
        re=vander[cdtype._dtype](x._re, N, increasing),
        im=vander[cdtype._dtype](x._im, N, increasing),
    )


# ===------------------------------------------------------------------------===#
# Construct array by changing the data type
# ===------------------------------------------------------------------------===#


# TODO: Check whether inplace cast is needed.
fn astype[
    dtype: DType, //, target: DType
](a: NDArray[dtype]) raises -> NDArray[target]:
    """
    Cast an NDArray to a different dtype.

    Parameters:
        dtype: Data type of the input array, always inferred.
        target: Data type to cast the NDArray to.

    Args:
        a: NDArray to be casted.

    Returns:
        A NDArray with the same shape and strides as `a`
        but with elements casted to `target`.
    """
    var array_order: String = "C" if a.flags.C_CONTIGUOUS else "F"
    var result: NDArray[target] = NDArray[target](a.shape, order=array_order)

    @parameter
    if target == DType.bool:

        @parameter
        fn vectorized_astype[simd_width: Int](idx: Int) -> None:
            (result.unsafe_ptr() + idx).strided_store[width=simd_width](
                a._buf.ptr.load[width=simd_width](idx).cast[target](), 1
            )

        vectorize[vectorized_astype, a.width](a.size)

    else:

        @parameter
        if target == DType.bool:

            @parameter
            fn vectorized_astypenb_from_b[simd_width: Int](idx: Int) -> None:
                result._buf.ptr.store(
                    idx,
                    (a._buf.ptr + idx)
                    .strided_load[width=simd_width](1)
                    .cast[target](),
                )

            vectorize[vectorized_astypenb_from_b, a.width](a.size)

        else:

            @parameter
            fn vectorized_astypenb[simd_width: Int](idx: Int) -> None:
                result._buf.ptr.store(
                    idx, a._buf.ptr.load[width=simd_width](idx).cast[target]()
                )

            vectorize[vectorized_astypenb, a.width](a.size)

    return result^


fn astype[
    cdtype: ComplexDType, //,
    target: ComplexDType,
](a: ComplexNDArray[cdtype]) raises -> ComplexNDArray[target]:
    """
    Cast a ComplexNDArray to a different dtype.

    Parameters:
        cdtype: Complex datatype of the input array.
        target: Complex datatype of the output array.

    Args:
        a: ComplexNDArray to be casted.

    Returns:
        A ComplexNDArray with the same shape and strides as `a`
        but with elements casted to `target`.
    """
    alias target_dtype: DType = target._dtype
    return ComplexNDArray[target](
        re=astype[target_dtype](a._re),
        im=astype[target_dtype](a._im),
    )


# ===------------------------------------------------------------------------===#
# Construct array from other objects
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

    var data = List[Scalar[dtype]]()  # Inferred data buffer of the array
    var shape = List[Int]()  # Inferred shape of the array
    var bytes = text.as_bytes()
    var ndim = 0  # Inferred number_as_str of dimensions.
    var level = 0  # Current level of the array.
    var number_as_str: String = ""
    for i in range(len(bytes)):
        var b = bytes[i]
        if chr(Int(b)) == "[":
            level += 1
            ndim = max(ndim, level)
            if len(shape) < ndim:
                shape.append(0)
            shape[level - 1] = 0

        if (
            chr(Int(b)).isdigit()
            or (chr(Int(b)) == ".")
            or (chr(Int(b)) == "-")
        ):
            number_as_str = number_as_str + chr(Int(b))
        if (chr(Int(b)) == ",") or (chr(Int(b)) == "]"):
            if number_as_str != "":
                var number = atof(number_as_str).cast[dtype]()
                data.append(number)  # Add the number to the data buffer
                number_as_str = ""  # Clean the number cache
                shape[-1] = shape[-1] + 1
        if chr(Int(b)) == "]":
            level = level - 1
            if level < 0:
                raise ("Unmatched left and right brackets!")
            if level > 0:
                shape[level - 1] = shape[level - 1] + 1
    var result: NDArray[dtype] = array[dtype](
        data=data, shape=shape, order=order
    )
    return result^


# fn from_tensor[
#     dtype: DType = DType.float64
# ](data: Tensor[dtype]) raises -> NDArray[dtype]:
#     """
#     Create array from tensor.

#     Parameters:
#         dtype: Datatype of the NDArray elements.

#     Args:
#         data: Tensor.

#     Returns:
#         NDArray.
#     """

#     var ndim = data.shape().rank()
#     var shape = NDArrayShape(ndim=ndim, initialized=False)
#     for i in range(ndim):
#         (shape._buf + i).init_pointee_copy(data.shape()[i])

#     var a = NDArray[dtype](shape=shape)

#     memcpy(a._buf.ptr, data._ptr, a.size)

#     return a


# fn from_tensorC[
#     dtype: DType = DType.float64
# ](real: Tensor[dtype], imag: Tensor[dtype]) raises -> ComplexNDArray[cdtype]:
#     """
#     Create array from tensor.

#     Parameters:
#         dtype: Datatype of the NDArray elements.

#     Args:
#         real: Tensor.
#         imag: Tensor.

#     Returns:
#         ComplexNDArray constructed from real and imaginary tensors.
#     """

#     var ndim = real.shape().rank()
#     if ndim != imag.shape().rank():
#         raise ("Real and imaginary tensors must have the same rank!")
#     var shape = NDArrayShape(ndim=ndim, initialized=False)
#     for i in range(ndim):
#         (shape._buf + i).init_pointee_copy(real.shape()[i])

#     var a = ComplexNDArray[cdtype](shape=shape)

#     memcpy(a._re._buf.ptr, real._ptr, a._re.size)
#     memcpy(a._im._buf.ptr, imag._ptr, a._im.size)

#     return a


# ===------------------------------------------------------------------------===#
# Overloads of `array` function
# Construct array from various objects.
# - String
# - List of Scalars
# - Numpy array
# - Tensor
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
        from numojo.prelude import *
        var arr = nm.array[f16](data=List[Scalar[f16]](1, 2, 3, 4), shape=List[Int](2, 2))
        ```

    Returns:
        An Array of given data, shape and order.
    """
    var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(shape), order)
    for i in range(result.size):
        result._buf.ptr[i] = data[i]
    return result^


fn array[
    cdtype: ComplexDType = ComplexDType.float64,
](
    data: List[ComplexScalar[cdtype]],
    shape: List[Int],
    order: String = "C",
) raises -> ComplexNDArray[cdtype]:
    """
    Array creation with given data, shape and order.

    Parameters:
        cdtype: Complex datatype of the ComplexNDArray elements.

    Args:
        data: List of complex data.
        shape: List of shape.
        order: Memory order C or F.

    Example:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var array = nm.array[cf64](
            data=List[CScalar[cf64]](CScalar[cf64](1, 1),
            CScalar[cf64](2, 2),
            CScalar[cf64](3, 3),
            CScalar[cf64](4, 4)),
            shape=List[Int](2, 2),
        )
        ```

    Returns:
        A ComplexNDArray constructed from real and imaginary data, shape and order.
    """
    var size: Int = 1
    for i in range(len(shape)):
        size = size * shape[i]
    if len(data) != size:
        raise Error(
            "Error in array: Real and imaginary data must have the same length!"
        )
    A = ComplexNDArray[cdtype](shape=shape, order=order)
    for i in range(A.size):
        A._re._buf.ptr[i] = data[i].re
        A._im._buf.ptr[i] = data[i].im
    return A^


fn array[
    dtype: DType = DType.float64
](data: PythonObject, order: String = "C") raises -> NDArray[dtype]:
    """
    Array creation with given data, shape and order.

    Example:
    ```mojo
    import numojo as nm
    from numojo.prelude import *
    from python import Python
    var np = Python.import_module("numpy")
    var np_arr = np.array([1, 2, 3, 4])
    A = nm.array[f16](data=np_arr, order="C")
    ```

    Parameters:
        dtype: Datatype of the NDArray elements.

    Args:
        data: A Numpy array (PythonObject).
        order: Memory order C or F.

    Returns:
        An Array of given data, shape and order.
    """

    var len = Int(len(data.shape))
    var shape: List[Int] = List[Int]()
    for i in range(len):
        if Int(data.shape[i]) == 1:
            continue
        shape.append(Int(data.shape[i]))

    var np = Python.import_module("numpy")
    var np_dtype = np.float64
    if dtype == DType.float16:
        np_dtype = np.float16
    elif dtype == DType.float32:
        np_dtype = np.float32
    elif dtype == DType.int64:
        np_dtype = np.int64
    elif dtype == DType.int32:
        np_dtype = np.int32
    elif dtype == DType.int16:
        np_dtype = np.int16
    elif dtype == DType.int8:
        np_dtype = np.int8
    elif dtype == DType.index:
        np_dtype = np.intp
    elif dtype == DType.uint64:
        np_dtype = np.uint64
    elif dtype == DType.uint32:
        np_dtype = np.uint32
    elif dtype == DType.uint16:
        np_dtype = np.uint16
    elif dtype == DType.uint8:
        np_dtype = np.uint8
    elif dtype == DType.bool:
        np_dtype = np.bool_

    var array_shape: NDArrayShape = NDArrayShape(shape)
    var np_arr = np.array(data, dtype=np_dtype, order=order.upper())
    var pointer = np_arr.__array_interface__["data"][0].unsafe_get_as_pointer[
        dtype
    ]()
    var A: NDArray[dtype] = NDArray[dtype](array_shape, order)
    memcpy[Scalar[dtype]](A._buf.ptr, pointer, A.size)
    return A^


fn array[
    cdtype: ComplexDType = ComplexDType.float64
](
    real: PythonObject, imag: PythonObject, order: String = "C"
) raises -> ComplexNDArray[cdtype]:
    """
    Array creation with given real and imaginary data, shape and order.

    Example:
    ```mojo
    import numojo as nm
    from numojo.prelude import *
    from python import Python
    var np = Python.import_module("numpy")
    var np_arr = np.array([1, 2, 3, 4])
    A = nm.array[f32](real=np_arr, imag=np_arr, order="C")
    ```

    Parameters:
        cdtype: Complex datatype of the NDArray elements.

    Args:
        real: A Numpy array (PythonObject).
        imag: A Numpy array (PythonObject).
        order: Memory order C or F.

    Returns:
        A ComplexNDArray constructed from real and imaginary data, shape and order.
    """
    alias dtype: DType = cdtype._dtype
    var len = Int(len(real.shape))
    var shape: List[Int] = List[Int]()
    if real.shape != imag.shape:
        raise Error(
            "Error in array: Real and imaginary data must have the same shape!"
        )
    for i in range(len):
        if Int(real.shape[i]) == 1:
            continue
        shape.append(Int(real.shape[i]))

    var np = Python.import_module("numpy")
    var np_dtype = np.float64
    if dtype == DType.float16:
        np_dtype = np.float16
    elif dtype == DType.float32:
        np_dtype = np.float32
    elif dtype == DType.int64:
        np_dtype = np.int64
    elif dtype == DType.int32:
        np_dtype = np.int32
    elif dtype == DType.int16:
        np_dtype = np.int16
    elif dtype == DType.int8:
        np_dtype = np.int8
    elif dtype == DType.index:
        np_dtype = np.intp
    elif dtype == DType.uint64:
        np_dtype = np.uint64
    elif dtype == DType.uint32:
        np_dtype = np.uint32
    elif dtype == DType.uint16:
        np_dtype = np.uint16
    elif dtype == DType.uint8:
        np_dtype = np.uint8
    elif dtype == DType.bool:
        np_dtype = np.bool_

    var array_shape: NDArrayShape = NDArrayShape(shape)
    var np_arr = np.array(real, dtype=np_dtype, order=order.upper())
    var np_arr_imag = np.array(imag, dtype=np_dtype, order=order.upper())
    var pointer = np_arr.__array_interface__["data"][0].unsafe_get_as_pointer[
        dtype
    ]()
    var pointer_imag = np_arr_imag.__array_interface__["data"][
        0
    ].unsafe_get_as_pointer[dtype]()
    var A: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](array_shape, order)
    memcpy[Scalar[dtype]](A._re._buf.ptr, pointer, A._re.size)
    memcpy[Scalar[dtype]](A._im._buf.ptr, pointer_imag, A._im.size)
    return A^


# fn array[
#     dtype: DType = DType.float64
# ](data: Tensor[dtype]) raises -> NDArray[dtype]:
#     """
#     Create array from tensor.

#     Example:
#     ```mojo
#     import numojo as nm
#     from tensor import Tensor, TensorShape
#     from numojo.prelude import *

#     fn main() raises:
#         height = 256
#         width = 256
#         channels = 3
#         image = Tensor[DType.float32].rand(TensorShape(height, width, channels))
#         print(image)
#         print(nm.array(image))
#     ```

#     Parameters:
#         dtype: Datatype of the NDArray elements.

#     Args:
#         data: Tensor.

#     Returns:
#         NDArray.
#     """

#     return from_tensor(data)


# fn array[
#     dtype: DType = DType.float64
# ](real: Tensor[dtype], imag: Tensor[dtype]) raises -> ComplexNDArray[cdtype]:
#     """
#     Create array from tensor.

#     Example:
#     ```mojo
#     import numojo as nm
#     from tensor import Tensor, TensorShape
#     from numojo.prelude import *

#     fn main() raises:
#         height = 256
#         width = 256
#         channels = 3
#         image = Tensor[DType.float32].rand(TensorShape(height, width, channels))
#         print(nm.array(real=image, imag=image))
#     ```

#     Parameters:
#         dtype: Datatype of the NDArray elements.

#     Args:
#         real: Tensor.
#         imag: Tensor.

#     Returns:
#         ComplexNDArray.
#     """

#     return from_tensorC(real, imag)


# ===----------------------------------------------------------------------=== #
# Internal functions
# ===----------------------------------------------------------------------=== #
# for creating a 0darray (only for internal use)
fn _0darray[
    dtype: DType
](val: Scalar[dtype],) raises -> NDArray[dtype]:
    """
    Initialize an special 0darray (numojo scalar).
    The ndim is 0.
    The shape is unitialized (0-element shape).
    The strides is unitialized (0-element strides).
    The size is 1 (`=0!`).
    """

    var b = NDArray[dtype](
        shape=NDArrayShape(ndim=0, initialized=False),
        strides=NDArrayStrides(ndim=0, initialized=False),
        ndim=0,
        size=1,
        flags=Flags(
            c_contiguous=True, f_contiguous=True, owndata=True, writeable=False
        ),
    )
    b._buf = OwnData[dtype](1)
    b._buf.ptr.init_pointee_copy(val)
    b.flags.OWNDATA = True
    return b^


fn _0darray[
    cdtype: ComplexDType
](val: ComplexSIMD[cdtype],) raises -> ComplexNDArray[cdtype]:
    """
    Initialize an special 0d complexarray (numojo scalar).
    The ndim is 0.
    The shape is unitialized (0-element shape).
    The strides is unitialized (0-element strides).
    The size is 1 (`=0!`).
    """
    var b = ComplexNDArray[cdtype](
        shape=NDArrayShape(ndim=0, initialized=False),
        strides=NDArrayStrides(ndim=0, initialized=False),
        ndim=0,
        size=1,
        flags=Flags(
            c_contiguous=True, f_contiguous=True, owndata=True, writeable=False
        ),
    )
    b._re._buf = OwnData[cdtype._dtype](1)
    b._im._buf = OwnData[cdtype._dtype](1)
    b._re._buf.ptr.init_pointee_copy(val.re)
    b._im._buf.ptr.init_pointee_copy(val.im)
    b.flags.OWNDATA = True
    return b^
