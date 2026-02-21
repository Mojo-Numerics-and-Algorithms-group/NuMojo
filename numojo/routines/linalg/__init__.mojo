# ===----------------------------------------------------------------------=== #
# NuMojo: LinAlg submodule
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Linear algebra routines (numojo.routines.linalg)

This module provides functions for linear algebra operations, including matrix decompositions, norms, products, and solving linear systems etc.
"""
from .decompositions import lu_decomposition, qr, eig
from .norms import det, trace
from .products import cross, dot, matmul
from .solving import inv, solve, lstsq
from .misc import diagonal
