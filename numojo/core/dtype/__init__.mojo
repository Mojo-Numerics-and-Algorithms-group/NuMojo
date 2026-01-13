"""
=====================================
Dtype (numojo.core.dtype)
=====================================

Dtype aliases and dtype-related utilities used across NuMojo.
"""

from .default_dtype import (
    i8,
    i16,
    i32,
    i64,
    i128,
    i256,
    int,
    u8,
    u16,
    u32,
    u64,
    u128,
    u256,
    uint,
    bf16,
    f16,
    f32,
    f64,
    boolean,
)

from .complex_dtype import (
    ComplexDType,
    ci8,
    ci16,
    ci32,
    ci64,
    ci128,
    ci256,
    cint,
    cu8,
    cu16,
    cu32,
    cu64,
    cu128,
    cu256,
    cuint,
    cbf16,
    cf16,
    cf32,
    cf64,
    cboolean,
    cinvalid,
)
