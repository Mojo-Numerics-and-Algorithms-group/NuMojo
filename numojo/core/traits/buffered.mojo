# ===----------------------------------------------------------------------=== #
# Define `Buffered` traits
# ===----------------------------------------------------------------------=== #
w
from memory import UnsafePointer


trait Buffered(ImplicitlyCopyable, Movable):
    fn __init__(out self):
        ...

    fn owns_data(self) -> Bool:
        ...
