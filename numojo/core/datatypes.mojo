"""
Datatypes Module - Implements datatypes aliases, conversions
"""
# ===----------------------------------------------------------------------=== #
# Datatypes Module - Implements datatypes aliases, conversions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #


# Rust-like data type alias
"""alias for `DType.int8`"""
alias i8 = DType.int8
"""Data type alias for DType.int8"""
alias i16 = DType.int16
"""Data type alias for DType.int16"""
alias i32 = DType.int32
"""Data type alias for DType.int32"""
alias i64 = DType.int64
"""Data type alias for DType.int64"""
alias u8 = DType.uint8
"""Data type alias for DType.uint8"""
alias u16 = DType.uint16
"""Data type alias for DType.uint16"""
alias u32 = DType.uint32
"""Data type alias for DType.uint32"""
alias u64 = DType.uint64
"""Data type alias for DType.uint64"""
alias f16 = DType.float16
"""Data type alias for DType.float16"""
alias f32 = DType.float32
"""Data type alias for DType.float32"""
alias f64 = DType.float64
"""Data type alias for DType.float64"""


fn cvtdtype[
    in_dtype: DType, out_dtype: DType, width: Int = 1
](value: SIMD[in_dtype, width]) -> SIMD[out_dtype, width]:
    """
    Converts datatype of a value from in_dtype to out_dtype at run time.

    Parameters:
        in_dtype: The input datatype.
        out_dtype: The output dataytpe.
        width: The width of the SIMD vector.

    Args:
        value: The SIMD value to be converted.

    Returns:
        The `value` with its dtype cast as out_dtype.

    """
    return value.cast[out_dtype]()


fn cvtdtype[
    in_dtype: DType,
    out_dtype: DType,
    width: Int = 1,
    value: SIMD[in_dtype, width] = SIMD[in_dtype](),
]() -> SIMD[out_dtype, width]:
    """
    Converts datatype of a value from in_dtype to out_dtype at compile time.

    Parameters:
        in_dtype: The input datatype.
        out_dtype: The output dataytpe.
        width: The width of the SIMD vector.
        value: The SIMD value to be converted.

    Returns:
        The `value` with its dtype cast as out_dtype.

    """
    return value.cast[out_dtype]()
