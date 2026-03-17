# ===----------------------------------------------------------------------=== #
# NuMojo: Arithmetic routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Arithmetic routines for NuMojo (numojo.routines.math.arithmetic).

Implements addition, subtraction, multiplication, division, floor division, fused multiply-add, and remainder helpers for NDArrays.
"""

from std.utils import Variant
from std.builtin.simd import FastMathFlag

from numojo.routines import HostExecutor
from numojo.core.ndarray import NDArray

# ===------------------------------------------------------------------------===#
# Addition
# ===------------------------------------------------------------------------===#


fn add[
    dtype: DType,
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__add__](array1, array2)


fn add[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise sum of array and scalar.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__add__](array, scalar)


fn add[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise sum of scalar and array.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__add__](scalar, array)


fn add[
    dtype: DType,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform addition on a list of arrays and a scalars.

    Parameters:
        dtype: The element type.

    Args:
        values: A list of arrays or Scalars to be added.

    Raises:
        Error: If there are no arrays in the input values.

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
        result_array = add[dtype](result_array, array)
    result_array = add[dtype](result_array, scalar_part)

    return result_array^


# ===------------------------------------------------------------------------===#
# Subtraction
# ===------------------------------------------------------------------------===#


fn sub[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The element-wise difference of `array1` and`array2`.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__sub__](array1, array2)


fn sub[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise difference of array and scalar.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__sub__](array, scalar)


fn sub[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise difference of scalar and array.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__sub__](scalar, array)


# ===------------------------------------------------------------------------===#
# Modulo
# ===------------------------------------------------------------------------===#


# ===------------------------------------------------------------------------===#
# Modulo
# ===------------------------------------------------------------------------===#


fn mod[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise modulo of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1 % array2.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mod__](array1, array2)


fn mod[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise modulo between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        A NDArray equal to array % scalar.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mod__](array, scalar)


fn mod[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise modulo between a scalar and an array.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        A NDArray equal to scalar % array.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mod__](scalar, array)


# ===------------------------------------------------------------------------===#
# Multiplication
# ===------------------------------------------------------------------------===#


fn mul[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise product of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1*array2.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mul__](array1, array2)


fn mul[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise product of array and scalar.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mul__](array, scalar)


fn mul[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise product of scalar and array.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mul__](scalar, array)


fn mul[
    dtype: DType,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform multiplication on a list of arrays an arrays and a scalars.

    Parameters:
        dtype: The element type.

    Args:
        values: A list of arrays or Scalars to be added.

    Raises:
        Error: If there are no arrays in the input values.

    Returns:
        The element-wise product of `array1` and`array2`.
    """
    var array_list: List[NDArray[dtype]] = List[NDArray[dtype]]()
    var scalar_part: Scalar[dtype] = 1
    for i in range(len(values)):
        if values[i].isa[NDArray[dtype]]():
            array_list.append(values[i].take[NDArray[dtype]]())
        elif values[i].isa[Scalar[dtype]]():
            scalar_part *= values[i].take[Scalar[dtype]]()
    if len(array_list) == 0:
        raise Error(
            "math:arithmetic:mul(*values:Variant[NDArray[dtype],Scalar[dtype]]):"
            " No arrays in arguments"
        )
    var result_array: NDArray[dtype] = array_list[0].deep_copy()
    for i in range(1, len(array_list)):
        result_array = mul[dtype](result_array, array_list[i])
    result_array = mul[dtype](result_array, scalar_part)

    return result_array^


# ===------------------------------------------------------------------------===#
# Division
# ===------------------------------------------------------------------------===#


fn div[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise quotient of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1/array2.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__truediv__](array1, array2)


fn div[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise quotient of array and scalar.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__truediv__](array, scalar)


fn div[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division between a scalar and an array.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise quotient of scalar and array.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__truediv__](scalar, array)


# ===------------------------------------------------------------------------===#
# Floor Division
# ===------------------------------------------------------------------------===#


fn floor_div[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise quotient of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1/array2.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__floordiv__](array1, array2)


fn floor_div[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise quotient of array and scalar.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__floordiv__](array, scalar)


fn floor_div[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise quotient of scalar and array.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__floordiv__](scalar, array)


# ===------------------------------------------------------------------------===#
# Fused Multiply-Add
# ===------------------------------------------------------------------------===#


fn fma[
    dtype: DType
](
    array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multiply add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape.

    Parameters:
        dtype: The element type.


    Args:
        array1: A NDArray.
        array2: A NDArray.
        array3: A NDArray.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    # TODO: Support passing through the FastMathFlag parameter
    # For now, FastMathFlag.CONTRACT is was default prior to this error.
    return HostExecutor.apply_ternary[dtype, SIMD.fma[FastMathFlag.CONTRACT]](
        array1, array2, array3
    )


fn fma[
    dtype: DType
](
    array1: NDArray[dtype], array2: NDArray[dtype], simd: SIMD[dtype, 1]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multiply add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        simd: A SIMD[dtype,1] value to be added.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return HostExecutor.apply_ternary[dtype, SIMD.fma[FastMathFlag.CONTRACT]](
        array1, array2, simd
    )


# ===------------------------------------------------------------------------===#
# Remainder
# ===------------------------------------------------------------------------===#


fn remainder[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remainders of NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1//array2.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mod__](array1, array2)


fn remainder[
    dtype: DType
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remainders of NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A scalar.

    Returns:
        A NDArray equal to array//scalar.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mod__](array, scalar)


fn remainder[
    dtype: DType
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remainders of NDArray.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A scalar.
        array: A NDArray.

    Returns:
        A NDArray equal to scalar//array.
    """
    return HostExecutor.apply_binary[dtype, SIMD.__mod__](scalar, array)
