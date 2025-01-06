# ===----------------------------------------------------------------------=== #
# Define `RefData` type
#
# TODO: fields in traits are not supported yet by Mojo
# Currently use `get_ptr()` to get pointer, in future, use `ptr` directly.
# var ptr: UnsafePointer[Float16]
# TODO: use parameterized trait.
# Replace `Float16` with `Scalar[dtype]`
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from numojo.core.traits.bufferable import Bufferable


struct RefData[is_mutable: Bool, //, origin: Origin[is_mutable]](Bufferable):
    var ptr: UnsafePointer[Float16]

    fn __init__(out self, size: Int):
        """
        Allocate given space on memory.
        The bytes allocated is `size` * `byte size of dtype`.

        Notes:
        Although it has the lifetime of another array, it owns the data.
        `ndarray.flags['OWN_DATA']` should be set as True.
        The memory should be freed by `__del__`.
        """
        self.ptr = UnsafePointer[Float16]().alloc(size)

    fn __init__(out self, ptr: UnsafePointer[Float16]):
        """
        Reads the underlying data of another array.

        Notes:
        `ndarray.flags['OWN_DATA']` should be set as False.
        The memory should not be freed by `__del__`.
        """
        self.ptr = ptr

    fn __moveinit__(out self, owned other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> UnsafePointer[Float16]:
        return self.ptr
