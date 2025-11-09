# ===----------------------------------------------------------------------=== #
# Define `RefData` type
# ===----------------------------------------------------------------------===

from numojo.core.traits.buffered import Buffered


struct RefData[is_mutable: Bool, //, origin: Origin[is_mutable]](
    Buffered, ImplicitlyCopyable, Movable
):
    """A type to denote arrays that do not own their data.
    It is used to represent views of an existing memory.
    It records the parametric mutability of the origin array to ensure safety.
    The origin array will be kept alive as long as the ref array is alive.
    """

    fn __init__(out self):
        pass

    @staticmethod
    fn is_own_data() -> Bool:
        return False

    @staticmethod
    fn is_ref_data() -> Bool:
        return True

    fn __str__(self) -> String:
        return "RefData"
