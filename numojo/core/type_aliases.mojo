# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""
Type aliases for commonly used data types in NuMojo.

This module provides convenient, user-friendly aliases for core types such as shapes,
strides, and complex scalars, as well as a symbolic constant for the imaginary unit.
"""

from numojo.core.complex.complex_simd import ComplexSIMD, ImaginaryUnit
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.layout.ndstrides import NDArrayStrides


comptime Shape = NDArrayShape
"""Alias for NDArrayShape, representing the shape of an n-dimensional array."""

comptime Strides = NDArrayStrides
"""Alias for NDArrayStrides, representing the memory strides of an n-dimensional array."""

comptime ComplexScalar = ComplexSIMD[_, width=1]
"""Alias for a scalar (width=1) complex SIMD value."""

comptime CScalar = ComplexSIMD[_, width=1]
"""Alias for a scalar complex number, equivalent to ComplexScalar."""

comptime `1j` = ImaginaryUnit()
"""Constant representing the imaginary unit (0 + 1j).

Allows Python-like syntax for complex numbers, e.g., (3 + 4 * `1j`).
"""
