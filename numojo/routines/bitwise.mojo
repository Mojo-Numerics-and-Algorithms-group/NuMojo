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
- `bitwise_count`: Population count (number of set bits).
- `packbits`, `unpackbits`: Packing/unpacking boolean arrays into bytes.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.collections.optional import Optional
from std.sys import size_of

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.routines.manipulation import broadcast_to, ravel

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


def bitwise_count[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    DType.uint8
] where dtype.is_integral():
    """
    Computes the number of 1-bits in the two's-complement representation of
    each element.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.

    Returns:
        A `uint8` array containing the population count of each element.

    Constraints:
        The array must have an integral element type.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.bitwise import bitwise_count

        var arr = nm.array[nm.i32]([0, 1, 3, 7], shape=[4])
        print(bitwise_count(arr))  # [0, 1, 2, 3]
        ```
    """
    var source = array.contiguous()
    var result = NDArray[DType.uint8](array.shape)
    comptime bits = size_of[Scalar[dtype]]() * 8
    for i in range(result.size):
        var v = source.unsafe_get(i)
        var count: UInt8 = 0
        for _ in range(bits):
            if (v & Scalar[dtype](1)) != 0:
                count += 1
            v = v >> Scalar[dtype](1)
        result.unsafe_set(i, count)
    return result^


def packbits[
    dtype: DType
](
    a: NDArray[dtype], axis: Optional[Int] = None, bitorder: String = "big"
) raises -> NDArray[DType.uint8] where (
    dtype.is_integral() or dtype == DType.bool
):
    """
    Packs the elements of a boolean/integer array into bits in a `uint8`
    array, grouping every 8 elements along `axis` into one byte (padded
    with zero bits at the end of each group when the axis length is not a
    multiple of 8).

    Parameters:
        dtype: The element type of the input array.

    Args:
        a: The input array. Non-zero elements are treated as `1` bits.
        axis: The axis along which to pack bits. If not given, `a` is
            flattened first.
        bitorder: Either `"big"` (first element becomes the most
            significant bit of each byte) or `"little"` (first element
            becomes the least significant bit).

    Returns:
        A `uint8` array with `axis`'s length replaced by
        `ceil(a.shape[axis] / 8)`.

    Raises:
        NumojoError: If `bitorder` is not `"big"` or `"little"`, or if
            `axis` is out of bound.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.bitwise import packbits

        var arr = nm.array[nm.boolean](
            [True, False, True, True, False, False, False, False, True],
            shape=[9],
        )
        print(packbits(arr))  # [178, 128]
        ```
    """
    if (bitorder != "big") and (bitorder != "little"):
        raise Error(
            NumojoError(
                category="value",
                message="`bitorder` must be 'big' or 'little'.",
                location="packbits",
            )
        )

    if not axis:
        var flat = ravel(a, order="C")
        return packbits(flat, axis=0, bitorder=bitorder)

    var ax = axis.value()
    if ax < 0:
        ax += a.ndim
    if (ax < 0) or (ax >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Axis out of range: got {}, expected {} <= axis < {}."
                ).format(axis.value(), -a.ndim, a.ndim),
                location="packbits",
            )
        )

    var A = a.contiguous()
    var n = A.shape[ax]
    var packed_n = (n + 7) // 8

    var new_shape_list = List[Int]()
    for d in range(A.ndim):
        if d == ax:
            new_shape_list.append(packed_n)
        else:
            new_shape_list.append(A.shape[d])
    var result = NDArray[DType.uint8](NDArrayShape(new_shape_list))

    var big_endian = bitorder == "big"
    for flat_idx in range(result.size):
        var remainder = flat_idx
        var src_base = 0
        var out_byte_idx = 0
        for d in range(A.ndim):
            var coord = remainder // result.strides[d]
            remainder = remainder % result.strides[d]
            if d == ax:
                out_byte_idx = coord
            else:
                src_base += coord * A.strides[d]

        var byte_val: UInt8 = 0
        for bit in range(8):
            var pos = out_byte_idx * 8 + bit
            if pos < n:
                var src_flat = src_base + pos * A.strides[ax]
                if A.unsafe_get(src_flat) != 0:
                    if big_endian:
                        byte_val |= UInt8(1) << UInt8(7 - bit)
                    else:
                        byte_val |= UInt8(1) << UInt8(bit)
        result.unsafe_set(flat_idx, byte_val)

    return result^


def unpackbits(
    a: NDArray[DType.uint8],
    axis: Optional[Int] = None,
    count: Optional[Int] = None,
    bitorder: String = "big",
) raises -> NDArray[DType.uint8]:
    """
    Unpacks a `uint8` array into a `uint8` array of `0`/`1` bits, expanding
    each byte along `axis` into 8 elements.

    Parameters:

    Args:
        a: The input `uint8` array.
        axis: The axis along which to unpack bits. If not given, `a` is
            flattened first.
        count: The number of bits to keep along `axis` after unpacking.
            A non-negative value truncates (or, if larger than
            `8 * a.shape[axis]`, raises); a negative value trims that
            many bits from the end. If not given, the full
            `8 * a.shape[axis]` bits are kept.
        bitorder: Either `"big"` (most significant bit unpacked first) or
            `"little"` (least significant bit unpacked first).

    Returns:
        A `uint8` array of `0`/`1` values with `axis`'s length replaced by
        the resolved bit count.

    Raises:
        NumojoError: If `bitorder` is not `"big"` or `"little"`, if `axis`
            is out of bound, or if `count` is out of range.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.routines.bitwise import unpackbits

        var arr = nm.array[nm.u8]([178], shape=[1])
        print(unpackbits(arr))  # [1, 0, 1, 1, 0, 0, 1, 0]
        ```
    """
    if (bitorder != "big") and (bitorder != "little"):
        raise Error(
            NumojoError(
                category="value",
                message="`bitorder` must be 'big' or 'little'.",
                location="unpackbits",
            )
        )

    if not axis:
        var flat = ravel(a, order="C")
        return unpackbits(flat, axis=0, count=count, bitorder=bitorder)

    var ax = axis.value()
    if ax < 0:
        ax += a.ndim
    if (ax < 0) or (ax >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Axis out of range: got {}, expected {} <= axis < {}."
                ).format(axis.value(), -a.ndim, a.ndim),
                location="unpackbits",
            )
        )

    var A = a.contiguous()
    var n = A.shape[ax]
    var full_n = n * 8

    var out_n = full_n
    if count:
        var c = count.value()
        if c < 0:
            out_n = full_n + c
        else:
            out_n = c
        if (out_n < 0) or (out_n > full_n):
            raise Error(
                NumojoError(
                    category="value",
                    message=String(
                        "`count` must resolve to a value in [0, {}]."
                    ).format(full_n),
                    location="unpackbits",
                )
            )

    var new_shape_list = List[Int]()
    for d in range(A.ndim):
        if d == ax:
            new_shape_list.append(out_n)
        else:
            new_shape_list.append(A.shape[d])
    var result = NDArray[DType.uint8](NDArrayShape(new_shape_list))

    var big_endian = bitorder == "big"
    for flat_idx in range(result.size):
        var remainder = flat_idx
        var src_base = 0
        var bit_pos = 0
        for d in range(A.ndim):
            var coord = remainder // result.strides[d]
            remainder = remainder % result.strides[d]
            if d == ax:
                bit_pos = coord
            else:
                src_base += coord * A.strides[d]

        var byte_idx = bit_pos // 8
        var bit_in_byte = bit_pos % 8
        var src_flat = src_base + byte_idx * A.strides[ax]
        var byte_val = A.unsafe_get(src_flat)

        var bit_val: UInt8
        if big_endian:
            bit_val = (byte_val >> UInt8(7 - bit_in_byte)) & UInt8(1)
        else:
            bit_val = (byte_val >> UInt8(bit_in_byte)) & UInt8(1)
        result.unsafe_set(flat_idx, bit_val)

    return result^
