"""
# ===----------------------------------------------------------------------=== #
# Implements Utility functions
# Last updated: 2024-06-15
# ===----------------------------------------------------------------------=== #
"""


fn is_inttype[dtype: DType]() -> Bool:
    if (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    ):
        return True
    return False


fn is_inttype(dtype: DType) -> Bool:
    if (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    ):
        return True
    return False

fn is_floattype[dtype: DType]() -> Bool:
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        return True
    return False
