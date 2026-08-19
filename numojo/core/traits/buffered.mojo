# ===----------------------------------------------------------------------=== #
# NuMojo: Buffered trait
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Buffered Trait (numojo.core.traits.buffered)
==============================================

Trait for buffer ownership semantics.

Trait to denote whether a data buffer is owned or referenced. Implementations
distinguish between owned data (OwnData) and referenced data (RefData).

Exports
-------
- `Buffered`: Trait for buffer ownership.
"""


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
