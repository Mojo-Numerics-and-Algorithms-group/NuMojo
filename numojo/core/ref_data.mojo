# ===----------------------------------------------------------------------=== #
# Define `OwnData` type to denote arrays that do not own their data. It is used to represent views of an existing memory.
# ===----------------------------------------------------------------------===

from memory import UnsafePointer
from numojo.core.traits.buffered import Buffered


struct RefData[is_mutable: Bool, //, origin: Origin[is_mutable]](
    Buffered, ImplicitlyCopyable, Movable
):
    alias owns: Bool = False

    fn __init__(out self):
        pass

    fn owns_data(self) -> Bool:
        return False
