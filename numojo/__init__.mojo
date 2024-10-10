"""
An Array Mathematics library for Mojo.
"""

from .core import *
from .math import *
from .math.statistics import stats
from .IO import *

alias __version__ = "V0.2"

# Constants
alias pi = core.constants.Constants.pi
alias e = core.constants.Constants.e
alias c = core.constants.Constants.c

# core alias
alias idx = Idx
alias ndarray = NDArray
alias ndshape = NDArrayShape
alias ndstride = NDArrayStride
