"""
Implements arithmetic operations functions
"""
# ===----------------------------------------------------------------------=== #
# Arithmetic operations functions
# ===----------------------------------------------------------------------=== #


import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from utils import Variant

import numojo.core._math_funcs as _mf
from numojo.core.ndarray import NDArray


fn add[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on two arrays.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The elementwise sum of `array1` and`array2`.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, SIMD.__add__](
        array1, array2
    )


fn add[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The elementwise sum of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__add__
    ](array, scalar)


fn add[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The elementwise sum of `array1` and`array2`.
    """
    return add[dtype, backend=backend](array, scalar)


fn add[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](owned *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[
    dtype
]:
    """
    Perform addition on a list of arrays and a scalars.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        values: A list of arrays or Scalars to be added.

    Returns:
        The elementwise sum of `array1` and`array2`.
    """
    var array_list: List[NDArray[dtype]] = List[NDArray[dtype]]()
    var scalar_part: Scalar[dtype] = 0
    for val in values:
        if val[].isa[NDArray[dtype]]():
            array_list.append(val[].take[NDArray[dtype]]())
        elif val[].isa[Scalar[dtype]]():
            scalar_part += val[].take[Scalar[dtype]]()
    if len(array_list) == 0:
        raise Error(
            "math:arithmetic:add(*values:Variant[NDArray[dtype],Scalar[dtype]]):"
            " No arrays in arguaments"
        )
    var result_array: NDArray[dtype] = NDArray[dtype](array_list[0].shape)
    for array in array_list:
        result_array = add[dtype, backend=backend](result_array, array)
    result_array = add[dtype, backend=backend](result_array, scalar_part)

    return result_array


fn sub[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on two arrays.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The elementwise difference of `array1` and`array2`.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, SIMD.__sub__](
        array1, array2
    )


fn sub[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The elementwise difference of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__sub__
    ](array, scalar)


fn sub[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The elementwise difference of `array1` and`array2`.
    """
    return sub[dtype, backend=backend](array, scalar)


fn diff[
    dtype: DType = DType.float64
](array: NDArray[dtype], n: Int) raises -> NDArray[dtype]:
    """
    Compute the n-th order difference of the input array.

    Parameters:
        dtype: The element type.

    Args:
        array: A array.
        n: The order of the difference.

    Returns:
        The n-th order difference of the input array.
    """

    var array1: NDArray[dtype] = NDArray[dtype](
        NDArrayShape(array.num_elements())
    )
    for i in range(array.num_elements()):
        array1.store(i, array.load(i))

    for num in range(n):
        var result: NDArray[dtype] = NDArray[dtype](
            NDArrayShape(array.num_elements() - (num + 1))
        )
        for i in range(array1.num_elements() - 1):
            result.store(i, (array1.load[1](i + 1) - array1.load[1](i)))
        array1 = result
    return array1


fn mod[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise modulo of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1 % array2.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, SIMD.__mod__](
        array1, array2
    )


fn mod[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The elementwise difference of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__mod__
    ](array, scalar)


fn mod[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The elementwise difference of `array1` and`array2`.
    """
    return mod[dtype, backend=backend](array, scalar)


fn mul[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise product of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1*array2.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, SIMD.__mul__](
        array1, array2
    )


fn mul[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The elementwise product of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__mul__
    ](array, scalar)


fn mul[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The elementwise product of `array1` and`array2`.
    """
    return mul[dtype, backend=backend](array, scalar)


fn mul[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](owned *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[
    dtype
]:
    """
    Perform multiplication on a list of arrays an arrays and a scalars.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        values: A list of arrays or Scalars to be added.

    Returns:
        The elementwise product of `array1` and`array2`.
    """
    var array_list: List[NDArray[dtype]] = List[NDArray[dtype]]()
    var scalar_part: Scalar[dtype] = 0
    for val in values:
        if val[].isa[NDArray[dtype]]():
            array_list.append(val[].take[NDArray[dtype]]())
        elif val[].isa[Scalar[dtype]]():
            scalar_part += val[].take[Scalar[dtype]]()
    if len(array_list) == 0:
        raise Error(
            "math:arithmetic:mul(*values:Variant[NDArray[dtype],Scalar[dtype]]):"
            " No arrays in arguaments"
        )
    var result_array: NDArray[dtype] = NDArray[dtype](array_list[0].shape)
    for array in array_list:
        result_array = mul[dtype, backend=backend](result_array, array)
    result_array = mul[dtype, backend=backend](result_array, scalar_part)

    return result_array


fn div[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise quotent of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1/array2.
    """
    return backend().math_func_2_array_in_one_array_out[
        dtype, SIMD.__truediv__
    ](array1, array2)


fn div[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The elementwise quotient of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__truediv__
    ](array, scalar)


fn div[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The elementwise quotient of `array1` and`array2`.
    """
    return div[dtype, backend=backend](array, scalar)


fn floor_div[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise quotent of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1/array2.
    """
    return backend().math_func_2_array_in_one_array_out[
        dtype, SIMD.__floordiv__
    ](array1, array2)


fn floor_div[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The elementwise quotient of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__floordiv__
    ](array, scalar)


fn floor_div[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The elementwise quotient of `array1` and`array2`.
    """
    return floor_div[dtype, backend=backend](array, scalar)


fn fma[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](
    array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        array3: A NDArray.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return backend().math_func_fma(array1, array2, array3)


fn fma[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](
    array1: NDArray[dtype], array2: NDArray[dtype], simd: SIMD[dtype, 1]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        simd: A SIMD[dtype,1] value to be added.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return backend().math_func_fma(array1, array2, simd)


fn remainder[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise remainders of NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1//array2.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, math.remainder](
        array1, array2
    )


# fn reciprocal[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](NDArray: NDArray[dtype]) -> NDArray[dtype]:
#     """
#     Elementwise reciprocals of array1 and array2.

#     Constraints:
#         Both arrays must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         NDArray: A NDArray.

#     Returns:
#         A NDArray equal to 1/NDArray.
#     """
#     return backend().math_func_1_array_in_one_array_out[
#         dtype, math.reciprocal
#     ](NDArray)
