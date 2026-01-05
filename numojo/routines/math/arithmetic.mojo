"""
Implements arithmetic operations functions
"""
# ===----------------------------------------------------------------------=== #
# Arithmetic operations functions
# ===----------------------------------------------------------------------=== #


from algorithm import parallelize, Static2DTileUnitFunc as Tile2DFunc
import math
from utils import Variant

import numojo.routines.math._math_funcs as _mf
from numojo.core.traits.backend import Backend
from numojo.core.ndarray import NDArray

# from numojo.core.datatypes import TypeCoercion


fn add[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on two arrays.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, SIMD.__add__](
        array1, array2
    )


# fn add[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array1: NDArray[dtype], array2: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform addition on two arrays.

#     Constraints:
#         Both arrays must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array1: A NDArray.
#         array2: A NDArray.

#     Returns:
#         The elementwise sum of `array1` and`array2`.
#     """
#     return backend().math_func_2_array_in_one_array_out[
#         ResultDType, SIMD.__add__
#     ](array1.astype[ResultDType](), array2.astype[ResultDType]())


fn add[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__add__
    ](array, scalar)


# fn add[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array: NDArray[dtype], scalar: Scalar[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform addition on two arrays.

#     Constraints:
#         Both arrays must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array: A NDArray.
#         scalar: A NDArray.

#     Returns:
#         The elementwise sum of `array1` and`array2`.
#     """
#     return backend().math_func_1_array_1_scalar_in_one_array_out[
#         ResultDType, SIMD.__add__
#     ](array.astype[ResultDType](), scalar.cast[ResultDType]())


fn add[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    return add[dtype, backend=backend](array, scalar)


# fn add[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](scalar: Scalar[dtype], array: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform addition on between an array and a scalar.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         scalar: A NDArray.
#         array: A NDArray.

#     Returns:
#         The elementwise sum of `array1` and`array2`.
#     """
#     return add[ResultDType, backend=backend](
#         array.astype[ResultDType](), scalar.cast[ResultDType]()
#     )


fn add[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform addition on a list of arrays and a scalars.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        values: A list of arrays or Scalars to be added.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    var array_list: List[NDArray[dtype]] = List[NDArray[dtype]]()
    var scalar_part: Scalar[dtype] = 0
    for i in range(len(values)):
        if values[i].isa[NDArray[dtype]]():
            array_list.append(values[i].take[NDArray[dtype]]())
        elif values[i].isa[Scalar[dtype]]():
            scalar_part += values[i].take[Scalar[dtype]]()
    if len(array_list) == 0:
        raise Error(
            "math:arithmetic:add(*values:Variant[NDArray[dtype],Scalar[dtype]]):"
            " No arrays in arguaments"
        )
    var result_array: NDArray[dtype] = NDArray[dtype](array_list[0].shape)
    for array in array_list:
        result_array = add[dtype, backend=backend](result_array, array)
    result_array = add[dtype, backend=backend](result_array, scalar_part)

    return result_array^


fn sub[
    dtype: DType, backend: Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on two arrays.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The element-wise difference of `array1` and`array2`.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, SIMD.__sub__](
        array1, array2
    )


# fn sub[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array1: NDArray[dtype], array2: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform subtraction on two arrays.

#     Constraints:
#         Both arrays must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array1: A NDArray.
#         array2: A NDArray.

#     Returns:
#         The elementwise difference of `array1` and`array2`.
#     """
#     return backend().math_func_2_array_in_one_array_out[
#         ResultDType, SIMD.__sub__
#     ](array1.astype[ResultDType](), array2.astype[ResultDType]())


fn sub[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The element-wise difference of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__sub__
    ](array, scalar)


# fn sub[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array: NDArray[dtype], scalar: Scalar[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform subtraction on between an array and a scalar.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array: A NDArray.
#         scalar: A NDArray.

#     Returns:
#         The elementwise difference of `array` and`scalar`.
#     """
#     return backend().math_func_1_array_1_scalar_in_one_array_out[
#         ResultDType, SIMD.__sub__
#     ](array.astype[ResultDType](), scalar.cast[ResultDType]())


fn sub[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The element-wise difference of `array1` and`array2`.
    """
    return sub[dtype, backend=backend](array, scalar)


# fn sub[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](scalar: Scalar[dtype], array: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform subtraction on between an array and a scalar.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         scalar: A NDArray.
#         array: A NDArray.

#     Returns:
#         The elementwise difference of `array` and`scalar`.
#     """
#     return sub[ResultDType, backend=backend](
#         array.astype[ResultDType](), scalar.cast[ResultDType]()
#     )


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

    var current: NDArray[dtype] = array.copy()

    for _ in range(n):
        var result: NDArray[dtype] = NDArray[dtype](
            NDArrayShape(current.size - 1)
        )
        for i in range(current.size - 1):
            result.store(i, current.load(i + 1) - current.load(i))
        current = result^
    return current^


fn mod[
    dtype: DType, backend: Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise modulo of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

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
    backend: Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The element-wise difference of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__mod__
    ](array, scalar)


fn mod[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The element-wise difference of `array1` and`array2`.
    """
    return mod[dtype, backend=backend](array, scalar)


fn mul[
    dtype: DType, backend: Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise product of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1*array2.
    """
    return backend().math_func_2_array_in_one_array_out[dtype, SIMD.__mul__](
        array1, array2
    )


# fn mul[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array1: NDArray[dtype], array2: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform multiplication on between two arrays.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array1: A NDArray.
#         array2: A NDArray.

#     Returns:
#         The element-wise product of `array1` and`array2`.
#     """
#     return backend().math_func_2_array_in_one_array_out[
#         ResultDType, SIMD.__mul__
#     ](array1.astype[ResultDType](), array2.astype[ResultDType]())


fn mul[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The element-wise product of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__mul__
    ](array, scalar)


# fn mul[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array: NDArray[dtype], scalar: Scalar[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform multiplication on between an array and a scalar.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array: A NDArray.
#         scalar: A NDArray.

#     Returns:
#         The element-wise product of `array` and`scalar`.
#     """
#     return backend().math_func_1_array_1_scalar_in_one_array_out[
#         ResultDType, SIMD.__mul__
#     ](array.astype[ResultDType](), scalar.cast[ResultDType]())


fn mul[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The element-wise product of `array1` and`array2`.
    """
    return mul[dtype, backend=backend](array, scalar)


# fn mul[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](scalar: Scalar[dtype], array: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform multiplication on between an array and a scalar.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         scalar: A NDArray.
#         array: A NDArray.

#     Returns:
#         The element-wise product of `array` and`scalar`.
#     """
#     return backend().math_func_1_array_1_scalar_in_one_array_out[
#         ResultDType, SIMD.__mul__
#     ](array.astype[ResultDType](), scalar.cast[ResultDType]())


fn mul[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform multiplication on a list of arrays an arrays and a scalars.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        values: A list of arrays or Scalars to be added.

    Returns:
        The element-wise product of `array1` and`array2`.
    """
    var array_list: List[NDArray[dtype]] = List[NDArray[dtype]]()
    var scalar_part: Scalar[dtype] = 0
    for i in range(len(values)):
        if values[i].isa[NDArray[dtype]]():
            array_list.append(values[i].take[NDArray[dtype]]())
        elif values[i].isa[Scalar[dtype]]():
            scalar_part += values[i].take[Scalar[dtype]]()
    if len(array_list) == 0:
        raise Error(
            "math:arithmetic:mul(*values:Variant[NDArray[dtype],Scalar[dtype]]):"
            " No arrays in arguaments"
        )
    var result_array: NDArray[dtype] = NDArray[dtype](array_list[0].shape)
    for array in array_list:
        result_array = mul[dtype, backend=backend](result_array, array)
    result_array = mul[dtype, backend=backend](result_array, scalar_part)

    return result_array^


fn div[
    dtype: DType, backend: Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise quotient of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1/array2.
    """
    return backend().math_func_2_array_in_one_array_out[
        dtype, SIMD.__truediv__
    ](array1, array2)


# fn div[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array1: NDArray[dtype], array2: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform true division on between two arrays.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array1: A NDArray.
#         array2: A NDArray.

#     Returns:
#         The element-wise quotient of `array1` and`array2`.
#     """
#     return backend().math_func_2_array_in_one_array_out[
#         ResultDType, SIMD.__truediv__
#     ](array1.astype[ResultDType](), array2.astype[ResultDType]())


fn div[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The element-wise quotient of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__truediv__
    ](array, scalar)


# fn div[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](array: NDArray[dtype], scalar: Scalar[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform true division on between an array and a scalar.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         array: A NDArray.
#         scalar: A NDArray.

#     Returns:
#         The element-wise quotient of `array` and`scalar`.
#     """
#     return backend().math_func_1_array_1_scalar_in_one_array_out[
#         ResultDType, SIMD.__truediv__
#     ](array.astype[ResultDType](), scalar.cast[ResultDType]())


fn div[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The element-wise quotient of `array1` and`array2`.
    """
    return backend().math_func_1_scalar_1_array_in_one_array_out[
        dtype, SIMD.__truediv__
    ](scalar, array)


# fn div[
#     dtype: DType,
#     backend: Backend = _mf.Vectorized,
#     *,
#     OtherDType: DType,
#     ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
# ](scalar: Scalar[dtype], array: NDArray[OtherDType]) raises -> NDArray[
#     ResultDType
# ]:
#     """
#     Perform true division on between an array and a scalar.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.
#         OtherDType: The element type of the second array.
#         ResultDType: The element type of the result array.

#     Args:
#         scalar: A NDArray.
#         array: A NDArray.

#     Returns:
#         The element-wise quotient of `array` and`scalar`.
#     """
#     return backend().math_func_1_array_1_scalar_in_one_array_out[
#         ResultDType, SIMD.__truediv__
#     ](array.astype[ResultDType](), scalar.cast[ResultDType]())


fn floor_div[
    dtype: DType, backend: Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise quotient of array1 and array2.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

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
    backend: Backend = _mf.Vectorized,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.
        scalar: A NDArray.

    Returns:
        The element-wise quotient of `array1` and`array2`.
    """
    return backend().math_func_1_array_1_scalar_in_one_array_out[
        dtype, SIMD.__floordiv__
    ](array, scalar)


fn floor_div[
    dtype: DType,
    backend: Backend = _mf.Vectorized,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        scalar: A NDArray.
        array: A NDArray.

    Returns:
        The element-wise quotient of `array1` and`array2`.
    """
    return floor_div[dtype, backend=backend](array, scalar)


fn fma[
    dtype: DType, backend: Backend = _mf.Vectorized
](
    array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multiply add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        array3: A NDArray.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return backend().math_func_fma(array1, array2, array3)


fn fma[
    dtype: DType, backend: Backend = _mf.Vectorized
](
    array1: NDArray[dtype], array2: NDArray[dtype], simd: SIMD[dtype, 1]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multiply add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        simd: A SIMD[dtype,1] value to be added.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return backend().math_func_fma(array1, array2, simd)


# fn pow[dtype: DType,
#     backend: _mf.Backend = _mf.Vectorized](array1: NDArray[dtype], intval: Int) -> NDArray[dtype]:
#     """
#     Element-wise NDArray to the power of intval.

#     Constraints:
#         Both arrays must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.

#     Args:
#         array1: A NDArray.
#         intval: An integer.

#     Returns:
#         A NDArray equal to NDArray**intval.
#     """
#     return backend().math_func_simd_int[dtype, math.pow](array1, intval)


fn remainder[
    dtype: DType, backend: Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remainders of NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

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
#     dtype: DType, backend: Backend = _mf.Vectorized
# ](NDArray: NDArray[dtype]) -> NDArray[dtype]:
#     """
#     Element-wise reciprocals of array1 and array2.

#     Constraints:
#         Both arrays must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defaults to `Vectorized`.

#     Args:
#         NDArray: A NDArray.

#     Returns:
#         A NDArray equal to 1/NDArray.
#     """
#     return backend().math_func_1_array_in_one_array_out[
#         dtype, math.reciprocal
#     ](NDArray)
