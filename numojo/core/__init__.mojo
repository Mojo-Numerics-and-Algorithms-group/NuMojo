"""
=====================================
Core (numojo.core)
=====================================

Core types and utilities for NuMojo (arrays, matrices, layout, memory, dtypes, and errors).
"""

from .ndarray import NDArray

from .type_aliases import (
    Shape,
    Strides,
    ComplexScalar,
    CScalar,
    `1j`,
)

from .error import (
    terminate,
    NumojoError,
)

from .matrix import Matrix

from .layout import (
    NDArrayShape,
    NDArrayStrides,
    Flags,
)

from .dtype import (
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

from .complex import (
    ComplexSIMD,
    ComplexNDArray,
)

from .memory import DataContainer

from .indexing import Item, IndexMethods, TraverseMethods, Validator

import .dtype
import .layout
import .memory
import .matrix
import .complex
import .traits
import .accelerator
