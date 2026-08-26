# ===----------------------------------------------------------------------=== #
# NuMojo: Arithmetic operations
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Arithmetic routines (numojo.routines.math.arithmetic).
======================================================
Basic arithmetic operations: addition, subtraction, multiplication, division, and related functions.

This module provides element-wise arithmetic operations for NDArrays supporting both
array-array and array-scalar operations.

Exports
-------
- `add`: Element-wise addition.
- `sub`: Element-wise subtraction.
- `mul`: Element-wise multiplication.
- `div`: Element-wise division.
- `floor_div`: Element-wise floor division.
- `mod`: Element-wise modulo.
- `remainder`: Element-wise remainder.
- `fma`: Fused multiply-add.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.builtin.simd import FastMathFlag
from std.utils import Variant

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor

# ===------------------------------------------------------------------------===#
# Addition
# ===------------------------------------------------------------------------===#


def add[
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

    @parameter
    def add_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 + simd2

    return HostExecutor.apply_binary[dtype, add_kernel](array1, array2)


def add[
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

    @parameter
    def add_kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w], scalar_simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd + scalar_simd

    return HostExecutor.apply_binary[dtype, add_kernel](array, scalar)


def add[
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

    @parameter
    def add_kernel[
        dtype: DType, simd_w: Int
    ](scalar_simd: SIMD[dtype, simd_w], simd: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return scalar_simd + simd

    return HostExecutor.apply_binary[dtype, add_kernel](scalar, array)


def add[
    dtype: DType,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform addition on a list of arrays and a scalars.

    Parameters:
        dtype: The element type.

    Args:
        values: A list of arrays or Scalars to be added.

    Raises:
        NumojoError: If there are no arrays in the input values.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    var array_list: List[NDArray[dtype]] = List[NDArray[dtype]]()
    var scalar_part: Scalar[dtype] = 0
    for i in range(len(values)):
        if values[i].isa[NDArray[dtype]]():
            # TODO: Figure out how to remove this unnecessary copy here (even though we take values as owned.
            array_list.append(values[i].copy().unsafe_unwrap[NDArray[dtype]]())
        elif values[i].isa[Scalar[dtype]]():
            scalar_part += values[i].copy().unsafe_unwrap[Scalar[dtype]]()
    if len(array_list) == 0:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "math:arithmetic:add(*values:Variant[NDArray[dtype],Scalar[dtype]]):"
                    " No arrays in arguaments"
                ),
                location="add",
            )
        )
    var result_array: NDArray[dtype] = NDArray[dtype](array_list[0].shape)
    for array in array_list:
        result_array = add[dtype](result_array, array)
    result_array = add[dtype](result_array, scalar_part)

    return result_array^


# ===------------------------------------------------------------------------===#
# Subtraction
# ===------------------------------------------------------------------------===#


def sub[
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

    @parameter
    def sub_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 - simd2

    return HostExecutor.apply_binary[dtype, sub_kernel](array1, array2)


def sub[
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

    @parameter
    def sub_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 - simd2

    return HostExecutor.apply_binary[dtype, sub_kernel](array, scalar)


def sub[
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

    @parameter
    def sub_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 - simd2

    return HostExecutor.apply_binary[dtype, sub_kernel](scalar, array)


# ===------------------------------------------------------------------------===#
# Modulo
# ===------------------------------------------------------------------------===#


def mod[
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

    @parameter
    def mod_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 % simd2

    return HostExecutor.apply_binary[dtype, mod_kernel](array1, array2)


def mod[
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

    @parameter
    def mod_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 % simd2

    return HostExecutor.apply_binary[dtype, mod_kernel](array, scalar)


def mod[
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

    @parameter
    def mod_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 % simd2

    return HostExecutor.apply_binary[dtype, mod_kernel](scalar, array)


# ===------------------------------------------------------------------------===#
# Multiplication
# ===------------------------------------------------------------------------===#


def mul[
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

    @parameter
    def mul_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 * simd2

    return HostExecutor.apply_binary[dtype, mul_kernel](array1, array2)


def mul[
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

    @parameter
    def mul_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 * simd2

    return HostExecutor.apply_binary[dtype, mul_kernel](array, scalar)


def mul[
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

    @parameter
    def mul_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 * simd2

    return HostExecutor.apply_binary[dtype, mul_kernel](scalar, array)


def mul[
    dtype: DType,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform multiplication on a list of arrays an arrays and a scalars.

    Parameters:
        dtype: The element type.

    Args:
        values: A list of arrays or Scalars to be added.

    Raises:
        NumojoError: If there are no arrays in the input values.

    Returns:
        The element-wise product of `array1` and`array2`.
    """
    var array_list: List[NDArray[dtype]] = List[NDArray[dtype]]()
    var scalar_part: Scalar[dtype] = 1
    for i in range(len(values)):
        if values[i].isa[NDArray[dtype]]():
            array_list.append(values[i].copy().unsafe_unwrap[NDArray[dtype]]())
        elif values[i].isa[Scalar[dtype]]():
            scalar_part *= values[i].copy().unsafe_unwrap[Scalar[dtype]]()
    if len(array_list) == 0:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "math:arithmetic:mul(*values:Variant[NDArray[dtype],Scalar[dtype]]):"
                    " No arrays in arguments"
                ),
                location="mul",
            )
        )
    var result_array: NDArray[dtype] = array_list[0].copy()
    for i in range(1, len(array_list)):
        result_array = mul[dtype](result_array, array_list[i])
    result_array = mul[dtype](result_array, scalar_part)

    return result_array^


# ===------------------------------------------------------------------------===#
# Division
# ===------------------------------------------------------------------------===#


def div[
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

    @parameter
    def truediv_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 / simd2

    return HostExecutor.apply_binary[dtype, truediv_kernel](array1, array2)


def div[
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

    @parameter
    def truediv_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 / simd2

    return HostExecutor.apply_binary[dtype, truediv_kernel](array, scalar)


def div[
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

    @parameter
    def truediv_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 / simd2

    return HostExecutor.apply_binary[dtype, truediv_kernel](scalar, array)


# ===------------------------------------------------------------------------===#
# Floor Division
# ===------------------------------------------------------------------------===#


def floor_div[
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

    @parameter
    def floordiv_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 // simd2

    return HostExecutor.apply_binary[dtype, floordiv_kernel](array1, array2)


def floor_div[
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

    @parameter
    def floordiv_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 // simd2

    return HostExecutor.apply_binary[dtype, floordiv_kernel](array, scalar)


def floor_div[
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

    @parameter
    def floordiv_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 // simd2

    return HostExecutor.apply_binary[dtype, floordiv_kernel](scalar, array)


# ===------------------------------------------------------------------------===#
# Fused Multiply-Add
# ===------------------------------------------------------------------------===#


def fma[
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

    @parameter
    def fma_kernel[
        dtype: DType, simd_w: Int
    ](
        simd1: SIMD[dtype, simd_w],
        simd2: SIMD[dtype, simd_w],
        simd3: SIMD[dtype, simd_w],
    ) -> SIMD[dtype, simd_w]:
        return simd1.fma(simd2, simd3)

    return HostExecutor.apply_ternary[dtype, fma_kernel](array1, array2, array3)


def fma[
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

    @parameter
    def fma_kernel[
        dtype: DType, simd_w: Int
    ](
        simd1: SIMD[dtype, simd_w],
        simd2: SIMD[dtype, simd_w],
        simd3: SIMD[dtype, simd_w],
    ) -> SIMD[dtype, simd_w]:
        return simd1.fma(simd2, simd3)

    return HostExecutor.apply_ternary[dtype, fma_kernel](array1, array2, simd)


# ===------------------------------------------------------------------------===#
# Remainder
# ===------------------------------------------------------------------------===#


def remainder[
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

    @parameter
    def mod_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 % simd2

    return HostExecutor.apply_binary[dtype, mod_kernel](array1, array2)


def remainder[
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

    @parameter
    def mod_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 % simd2

    return HostExecutor.apply_binary[dtype, mod_kernel](array, scalar)


def remainder[
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

    @parameter
    def mod_kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return simd1 % simd2

    return HostExecutor.apply_binary[dtype, mod_kernel](scalar, array)
