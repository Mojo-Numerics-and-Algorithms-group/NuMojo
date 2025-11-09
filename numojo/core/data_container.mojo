# ===----------------------------------------------------------------------=== #
# Define `DataContainer` type
#
# TODO: fields in traits are not supported yet by Mojo
# Currently use `get_ptr()` to get pointer, in future, use `ptr` directly.
# var ptr: UnsafePointer[Scalar[dtype]]
# ===----------------------------------------------------------------------===

from memory import UnsafePointer, UnsafePointerV2


struct DataContainer[dtype: DType](ImplicitlyCopyable):
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
