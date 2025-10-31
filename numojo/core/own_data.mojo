# ===----------------------------------------------------------------------=== #
# Define `OwnData` type to denote arrays that own their data.
# ===----------------------------------------------------------------------===

from numojo.core.traits.buffered import Buffered


struct OwnData(Buffered, ImplicitlyCopyable, Movable):
    alias owns: Bool = True

    fn __init__(out self):
        pass

    fn owns_data(self) -> Bool:
        return True
