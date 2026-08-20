# ===----------------------------------------------------------------------=== #
# NuMojo: Indexing and slicing utilities
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Indexing (numojo.core.indexing).
================================
Indexing-related helpers and types used by NuMojo core containers.

Exports
-------
- `IndexBuffer`: Buffer for storing array indices.
- `Item`: Single item indexing.
- `IndexMethods`: Offset-based indexing methods.
- `InternalSlice`: Internal slice representation.
- `TraverseMethods`: Array traversal methods.
- `Validator`: Index validation utilities.
- `newaxis`: Marker for expanding dimensions.
- `bool_to_numeric`, `to_numpy`: Conversion utilities.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .index_buffer import IndexBuffer
from .item import Item
from .offset import IndexMethods
from .slicing import InternalSlice
from .traversal import TraverseMethods
from .utility import (
    bool_to_numeric,
    newaxis,
    to_numpy,
)
from .validation import Validator
