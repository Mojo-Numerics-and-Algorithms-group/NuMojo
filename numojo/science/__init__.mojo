"""
NuMojo Science Package (`numojo.science`)
========================================

This package contains higher-level, domain-focused modules built on top of the
core array/matrix types and the routines layer.

It is intended to be similar in spirit to `scipy.*` in Python, providing
specialized functionality (e.g. signal processing, interpolation) while keeping
the foundational numerical building blocks in `numojo.core` and `numojo.routines`.

Submodules:
- `signal`: Signal processing utilities
- `interpolate`: Interpolation utilities
"""

from . import signal
from . import interpolate
