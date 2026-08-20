# ===----------------------------------------------------------------------=== #
# NuMojo: Dtype aliases and utilities
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Dtype (numojo.core.dtype).
=========================
Dtype aliases and dtype-related utilities used across NuMojo.

Exports
-------
Real dtype aliases: `i8`, `i16`, `i32`, `i64`, `i128`, `i256`, `int`, `u8`, `u16`, `u32`, `u64`, `u128`, `u256`, `uint`, `bf16`, `f16`, `f32`, `f64`, `boolean`.

Complex dtype aliases: `ci8`, `ci16`, `ci32`, `ci64`, `ci128`, `ci256`, `cint`, `cu8`, `cu16`, `cu32`, `cu64`, `cu128`, `cu256`, `cuint`, `cbf16`, `cf16`, `cf32`, `cf64`, `cboolean`, `cinvalid`.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .complex_dtype import (
    cbf16,
    cboolean,
    cf16,
    cf32,
    cf64,
    ci128,
    ci16,
    ci256,
    ci32,
    ci64,
    ci8,
    cint,
    cinvalid,
    ComplexDType,
    cu128,
    cu16,
    cu256,
    cu32,
    cu64,
    cu8,
    cuint,
)
from .default_dtype import (
    bf16,
    boolean,
    f16,
    f32,
    f64,
    i128,
    i16,
    i256,
    i32,
    i64,
    i8,
    int,
    u128,
    u16,
    u256,
    u32,
    u64,
    u8,
    uint,
)
