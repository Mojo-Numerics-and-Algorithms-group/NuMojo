# ===----------------------------------------------------------------------=== #
# NuMojo: Routines module
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Bitwise operations (numojo.routines.bitwise).
=============================================
Bitwise operations for integer arrays.

Element-wise bitwise operations for integer NDArrays.

Exports
-------
- `bitwise_and`, `bitwise_or`, `bitwise_xor`: Binary bitwise operations.
- `left_shift`, `right_shift`: Binary shift operations.
- `invert`: Bitwise NOT operation.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.routines.manipulation import broadcast_to

# ===----------------------------------------------------------------------=== #
# Bitwise operations
# ===----------------------------------------------------------------------=== #


def bitwise_and[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
    dtype
] where (dtype.is_integral() or dtype == DType.bool):
    """
    Compute the element-wise bitwise AND of two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: The first input array.
        array2: The second input array.

    Returns:
        An array containing the element-wise bitwise AND.

    Constraints:
        Both arrays must have the same integral or boolean element type.

    Examples:
        `bitwise_and(a, b)` computes the bitwise AND of `a` and `b`.
    """

    var shape = array1.shape.broadcast(array2.shape)
    var a = broadcast_to(array1, shape).contiguous()
    var b = broadcast_to(array2, shape).contiguous()
    var result = NDArray[dtype](shape)
    for i in range(result.size):
        result.unsafe_set(i, a.unsafe_get(i) & b.unsafe_get(i))
    return result^


def bitwise_or[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
    dtype
] where (dtype.is_integral() or dtype == DType.bool):
    """
    Compute the element-wise bitwise OR of two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: The first input array.
        array2: The second input array.

    Returns:
        An array containing the element-wise bitwise OR.

    Constraints:
        Both arrays must have the same integral or boolean element type.

    Examples:
        `bitwise_or(a, b)` computes the bitwise OR of `a` and `b`.
    """

    var shape = array1.shape.broadcast(array2.shape)
    var a = broadcast_to(array1, shape).contiguous()
    var b = broadcast_to(array2, shape).contiguous()
    var result = NDArray[dtype](shape)
    for i in range(result.size):
        result.unsafe_set(i, a.unsafe_get(i) | b.unsafe_get(i))
    return result^


def bitwise_xor[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
    dtype
] where (dtype.is_integral() or dtype == DType.bool):
    """
    Compute the element-wise bitwise XOR of two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: The first input array.
        array2: The second input array.

    Returns:
        An array containing the element-wise bitwise XOR.

    Constraints:
        Both arrays must have the same integral or boolean element type.

    Examples:
        `bitwise_xor(a, b)` computes the bitwise XOR of `a` and `b`.
    """

    var shape = array1.shape.broadcast(array2.shape)
    var a = broadcast_to(array1, shape).contiguous()
    var b = broadcast_to(array2, shape).contiguous()
    var result = NDArray[dtype](shape)
    for i in range(result.size):
        result.unsafe_set(i, a.unsafe_get(i) ^ b.unsafe_get(i))
    return result^


def left_shift[
    dtype: DType
](array: NDArray[dtype], shift: Int) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Shift each array element left by an integer number of bits.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.
        shift: The number of bits to shift each element.

    Returns:
        An array containing the element-wise left shift.

    Constraints:
        The array must have an integral element type.

    Examples:
        `left_shift(a, 2)` shifts every element of `a` left by two bits.
    """

    var source = array.contiguous()
    var result = NDArray[dtype](array.shape)
    var amount = Scalar[dtype](shift)
    for i in range(result.size):
        result.unsafe_set(i, source.unsafe_get(i) << amount)
    return result^


def right_shift[
    dtype: DType
](array: NDArray[dtype], shift: Int) raises -> NDArray[
    dtype
] where dtype.is_integral():
    """
    Shift each array element right by an integer number of bits.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.
        shift: The number of bits to shift each element.

    Returns:
        An array containing the element-wise right shift.

    Constraints:
        The array must have an integral element type.

    Examples:
        `right_shift(a, 2)` shifts every element of `a` right by two bits.
    """

    var source = array.contiguous()
    var result = NDArray[dtype](array.shape)
    var amount = Scalar[dtype](shift)
    for i in range(result.size):
        result.unsafe_set(i, source.unsafe_get(i) >> amount)
    return result^


def invert[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[dtype] where (
    dtype.is_integral() or dtype == DType.bool
):
    """
    Element-wise invert of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to the bitwise inversion of array.

    Constraints:
        The array must be either a boolean or integral array.

    Examples:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        from numojo.routines.bitwise import invert

        var arr1 = nm.array[nm.i8]([1, 2, 3], shape=[3])
        var result1 = invert(arr1) # result1 is [-2, -3, -4]

        var arr2 = nm.array[nm.boolean]([True, False, True], shape=[3])
        var result2 = invert(arr2) # result2 is [false, true, false
        ```
    """

    var source = array.contiguous()
    var result = NDArray[dtype](array.shape)
    for i in range(result.size):
        result.unsafe_set(i, ~source.unsafe_get(i))
    return result^
