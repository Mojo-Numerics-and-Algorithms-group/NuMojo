# ===----------------------------------------------------------------------=== #
# NuMojo: Linear algebra operations
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Linear algebra routines (numojo.routines.linalg).
=================================================
Linear algebra operations including matrix decompositions, norms, products, and linear system solving.

Exports
-------
- Decompositions: `lu_decomposition`.
- Norms: `det`, `trace`.
- Products: `dot`, `matmul`, `cross`.
- Solving: `solve`, `lstsq`, `inv`.
- Miscellaneous: `diagonal`.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .decompositions import lu_decomposition
from .misc import diagonal
from .norms import (
    det,
    trace,
)
from .products import (
    cross,
    dot,
    matmul,
)
from .solving import (
    inv,
    solve,
)
