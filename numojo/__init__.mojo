"""
NuMojo is a library for numerical computing in Mojo 🔥 similar to NumPy, SciPy in Python.
"""

alias __version__ = "V0.2"

# ===----------------------------------------------------------------------=== #
# Import core types
# ===----------------------------------------------------------------------=== #

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape, Shape
from numojo.core.index import Idx
from .core.datatypes import i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64

# Alias for users

# For ease of use, the name of the types may not follow the Mojo convention,
# e.g., lower case can also be used for alias of structs.
alias idx = core.index.Idx

# ===----------------------------------------------------------------------=== #
# Import routines and objects
# ===----------------------------------------------------------------------=== #

# Constants
alias pi = numojo.routines.constants.Constants.pi
alias e = numojo.routines.constants.Constants.e
alias c = numojo.routines.constants.Constants.c

# TODO Make explicit imports in future
from .routines import *
