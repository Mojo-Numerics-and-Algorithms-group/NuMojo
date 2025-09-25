# ===----------------------------------------------------------------------=== #
# Define `Bufferable` traits
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer


trait Bufferable:
    """
    Data buffer types that can be used as a container of the underlying buffer.
    """

    # TODO: fields in traits are not supported yet by Mojo
    # Currently use `get_ptr()` to get pointer, in future, use `ptr` directly.
    # var ptr: UnsafePointer[Float16]
    # TODO: use parameterized trait.
    # Replace `Float16` with `Scalar[dtype]`

    fn __init__(out self, size: Int):
        ...

    fn __init__(out self, ptr: UnsafePointer[Float16]):
        ...

    fn __moveinit__(out self, deinit other: Self):
        ...

    fn get_ptr(self) -> UnsafePointer[Float16]:
        ...
