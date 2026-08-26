# ===----------------------------------------------------------------------=== #
# NuMojo: Operation execution backends
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Operations (numojo.routines.operations).
========================================
Vectorized operation execution backends for unary, binary, and predicate operations.

Exports
-------
- `HostExecutor`: CPU execution backend for array operations.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .backend import HostExecutor
