# ===----------------------------------------------------------------------=== #
# NuMojo: Traits and protocol abstractions
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Traits (numojo.core.traits).
===========================

Trait and protocol abstractions used across NuMojo core containers and internals.

Exports
-------
- `Backend`: Protocol for backend implementations.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .backend import Backend
