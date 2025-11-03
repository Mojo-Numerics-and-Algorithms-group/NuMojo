# ===----------------------------------------------------------------------=== #
# Define `OwnData` type
# ===----------------------------------------------------------------------=== #

from numojo.core.traits.buffered import Buffered


struct OwnData(Buffered, ImplicitlyCopyable, Movable):
    """A type to denote arrays that own their data buffer."""

    fn __init__(out self):
        pass

    fn is_own_data(self) -> Bool:
        return True

    fn is_ref_data(self) -> Bool:
        return False
