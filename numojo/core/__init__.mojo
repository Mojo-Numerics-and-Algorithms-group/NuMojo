# ARRAYS

# Structs
from .ndarray import NDArray
from .item import Item
from .ndshape import NDArrayShape
from .ndstrides import NDArrayStrides
from .own_data import OwnData
from .ref_data import RefData

from .complex import (
    ComplexSIMD,
    ComplexScalar,
    CScalar,
    ComplexNDArray,
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

from .datatypes import (
    i8,
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

from .error import (
    ShapeError,
    IndexError,
    BroadcastError,
    MemoryError,
    ValueError,
    ArithmeticError,
)

alias idx = Item
alias shape = NDArrayShape
alias Shape = NDArrayShape
