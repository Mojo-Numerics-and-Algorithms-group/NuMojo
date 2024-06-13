[Discord](https://discord.gg/NcnSH5n26F)

# Purpose
NuMojo is a library for numerical computing in Mojo similar to numpy in Python

NuMojo extends (most of) the standard library math functions to work on tensor inputs

NuMojo intends to capture a wide swath of numerics capability present in the Python packages numpy,scipy and scikit such as

* Linear/Tensor Algebra
* Integration
* Derivatives
* Optimizers
* Function approximators

NuMojo intends to try and get the most out of the capabilities of Mojo including vectorization, parallelization, and GPU acceleration(once available). Currently, NuMojo extends (most of) the standard library math functions to work on tensor inputs.

NuMojo intends to be a building block for other Mojo packages that need fast math under the hood without the added weight of a ML back and forward propagation system

# What NuMojo is not

NuMojo is not a machine learning library, it will never include back-propagation in the base library.

# How to install

Clone the repository and build

# Usage

For now

```mojo
import numojo
from tensor import Tensor
def main():
    var tens = Tensor[DType.float32](10,10)
    tens=tens+numojo.pi/2
    print(numojo.sin[DType.float32](tens))
```

## Available functions include:

abs, floor, ceil, trunc, round, roundeven, round_half_down, round_half_up, rsqrt, exp2, exp, log, log2, tanh, reciprocal, acos, asin, atan, cos, sin, tan, acosh, asinh, atanh, cosh, sinh, expm1, log10, log1p, cbrt, pow, mod, mul, sub, add, div, copysign, atan2, hypot, nextafter, scalb, remainder

## Warnings

This library is still very much a work in progress and may change at any time. Also, the standard tensor has many rough edges.

## Acknowledgements

* Built in native [Mojo](https://github.com/modularml/mojo) created by [Modular](https://github.com/modularml)