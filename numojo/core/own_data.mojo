# ===----------------------------------------------------------------------=== #
# Define `OwnData` type
# ===----------------------------------------------------------------------=== #

from numojo.core.traits.buffered import Buffered


struct OwnData(Buffered, ImplicitlyCopyable, Movable):
    """A type to denote arrays that own their data buffer."""

    fn __init__(out self):
        pass

    @staticmethod
    fn is_own_data() -> Bool:
        return True

    @staticmethod
    fn is_ref_data() -> Bool:
        return False

    fn __str__(self) -> String:
        return "OwnData"
