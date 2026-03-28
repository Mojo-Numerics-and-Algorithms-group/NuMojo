# ===----------------------------------------------------------------------=== #
# Define `Buffered` traits
# ===----------------------------------------------------------------------=== #


trait Buffered(ImplicitlyCopyable, Movable):
    """A trait to denote whether the data buffer is owned or not.

    There will be two implementations:
    1. `OwnData`: for arrays that own their data buffer.
    2. `RefData`: for arrays that do not own their data buffer.

    The `RefData` type will record the origin of the data to ensure safety.
    """

    def __init__(out self):
        ...

    @staticmethod
    def is_own_data() -> Bool:
        ...

    @staticmethod
    def is_ref_data() -> Bool:
        ...

    def __str__(self) -> String:
        ...
