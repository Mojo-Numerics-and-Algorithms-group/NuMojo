# ===----------------------------------------------------------------------=== #
# Define `DataContainer` type
#
# TODO: fields in traits are not supported yet by Mojo
# Currently use `get_ptr()` to get pointer, in future, use `ptr` directly.
# var ptr: UnsafePointer[Scalar[dtype]]
# TODO: implement `Bufferable` trait.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from numojo.core.traits.bufferable import Bufferable



trait Buffered(ImplicitlyCopyable, Movable):
    fn __init__(out self):
        ...

    fn is_owned_data(self) -> Bool:
        ...

struct OwnData(Buffered, ImplicitlyCopyable, Movable):
    alias view: Bool = False

    fn __init__(out self):
        pass

    fn is_owned_data(self) -> Bool:
        return True


struct RefData[is_mutable: Bool, //, origin: Origin[is_mutable]](Buffered, ImplicitlyCopyable, Movable):
    alias view: Bool = True
    fn __init__(out self):
        pass

    fn is_owned_data(self) -> Bool:
        return False

struct DataContainer[dtype: DType]():
    var ptr: UnsafePointer[Scalar[dtype]]

    fn __init__(out self, size: Int):
        """
        Allocate given space on memory.
        The bytes allocated is `size` * `byte size of dtype`.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as True.
        The memory should be freed by `__del__`.
        """
        self.ptr = UnsafePointer[Scalar[dtype]]().alloc(size)

    fn __init__(out self, ptr: UnsafePointer[Scalar[dtype]]):
        """
        Do not use this if you know what it means.
        If the pointer is associated with another array, it might cause
        dangling pointer problem.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as False.
        The memory should not be freed by `__del__`.
        """
        self.ptr = ptr

    fn __moveinit__(out self, deinit other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.ptr
