"""
Datatypes Module - Implements datatypes comptimees, conversions
"""
# ===----------------------------------------------------------------------=== #
# Datatypes Module - Implements datatypes comptimees, conversions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #

# Rust-like or numpy-like data type comptime
comptime i8 = DType.int8
"""Data type comptime for DType.int8."""
comptime i16 = DType.int16
"""Data type comptime for DType.int16."""
comptime i32 = DType.int32
"""Data type comptime for DType.int32."""
comptime i64 = DType.int64
"""Data type comptime for DType.int64."""
comptime i128 = DType.int128
"""Data type comptime for DType.int128."""
comptime i256 = DType.int256
"""Data type comptime for DType.int256."""
comptime int = DType.int
"""Data type comptime for DType.int."""
comptime uint = DType.int
"""Data type comptime for DType.uint."""
comptime u8 = DType.uint8
"""Data type comptime for DType.uint8."""
comptime u16 = DType.uint16
"""Data type comptime for DType.uint16."""
comptime u32 = DType.uint32
"""Data type comptime for DType.uint32."""
comptime u64 = DType.uint64
"""Data type comptime for DType.uint64."""
comptime u128 = DType.uint128
"""Data type comptime for DType.uint128."""
comptime u256 = DType.uint256
"""Data type comptime for DType.uint256."""
comptime f16 = DType.float16
"""Data type comptime for DType.float16."""
comptime bf16 = DType.bfloat16
"""Data type comptime for DType.bfloat16."""
comptime f32 = DType.float32
"""Data type comptime for DType.float32."""
comptime f64 = DType.float64
"""Data type comptime for DType.float64."""
comptime boolean = DType.bool
"""Data type comptime for DType.bool."""

# ===----------------------------------------------------------------------=== #

fn _concise_dtype_str(dtype: DType) -> String:
    """Returns a concise string representation of the data type."""
    if dtype == i8:
        return "i8"
    elif dtype == i64:
        return "i64"
    elif dtype == i128:
        return "i128"
    elif dtype == i256:
        return "i256"
    elif dtype == int:
        return "int"
    elif dtype == u8:
        return "u8"
    elif dtype == u16:
        return "u16"
    elif dtype == u32:
        return "u32"
    elif dtype == u64:
        return "u64"
    elif dtype == u128:
        return "u128"
    elif dtype == u256:
        return "u256"
    elif dtype == uint:
        return "uint"
    elif dtype == bf16:
        return "bf16"
    elif dtype == f16:
        return "f16"
    elif dtype == f32:
        return "f32"
    elif dtype == f64:
        return "f64"
    elif dtype == boolean:
        return "boolean"
    else:
        return "Unknown"
