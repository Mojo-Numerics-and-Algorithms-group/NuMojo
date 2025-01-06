# ARRAYS

# Structs
from .ndarray import NDArray
from .index import Idx
from .ndshape import NDArrayShape
from .ndstrides import NDArrayStrides

from .complex_dtype import CDType
from .complex_simd import ComplexSIMD, ComplexScalar
from .complex_ndarray import ComplexNDArray

from .datatypes import (
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    ci8,
    ci16,
    ci32,
    ci64,
    cu8,
    cu16,
    cu32,
    cu64,
    cf16,
    cf32,
    cf64,
)

# from .utility import

alias idx = Idx
alias shape = NDArrayShape
alias Shape = NDArrayShape
