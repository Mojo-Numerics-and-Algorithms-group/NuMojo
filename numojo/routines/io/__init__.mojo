# ===----------------------------------------------------------------------=== #
# NuMojo: I/O submodule
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""I/O routines (numojo.routines.io)

This module provides functions for reading and writing arrays to and from files, as well as formatting options for printing arrays.
"""
from .files import loadtxt, savetxt, load, save

from .formatting import (
    format_floating_scientific,
    PrintOptions,
    set_printoptions,
)
