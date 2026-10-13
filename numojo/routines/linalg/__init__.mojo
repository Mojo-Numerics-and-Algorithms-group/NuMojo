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
- Decompositions: `lu_decomposition`, `qr`, `cholesky`, `eig`, `eigh`, `svd`.
- Norms: `det`, `trace`, `norm`, `matrix_rank`, `cond`.
- Products: `dot`, `matmul`, `cross`, `outer`, `kron`, `tensordot`.
- Solving: `solve`, `lstsq`, `inv`, `pinv`.
- Miscellaneous: `diagonal`.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .decompositions import (
    cholesky,
    cond,
    eig,
    eigh,
    lu_decomposition,
    matrix_rank,
    qr,
    svd,
)
from .misc import diagonal
from .norms import (
    det,
    norm,
    trace,
)
from .products import (
    cross,
    dot,
    kron,
    matmul,
    outer,
    tensordot,
)
from .solving import (
    inv,
    lstsq,
    pinv,
    solve,
)
