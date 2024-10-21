"""
An Array Mathematics library for Mojo.
"""

from .core import *
from .math import *
from .math.statistics import stats
from .io import *

alias __version__ = "V0.2"

# Constants
alias pi = core.constants.Constants.pi
alias e = core.constants.Constants.e
alias c = core.constants.Constants.c

# Alias for users
# For ease of use, the name of the types may not follow the Mojo convention,
# e.g., lower case can also be used for alias of structs.
alias idx = core.index.Idx
