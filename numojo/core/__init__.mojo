# ARRAYS

# Structs
from .ndarray import NDArray
from .item import Item
from .ndshape import NDArrayShape
from .ndstrides import NDArrayStrides

from .complex import (
    ComplexSIMD,
    ComplexScalar,
    ComplexNDArray,
)

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
