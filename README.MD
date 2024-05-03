# Purpose 
NuMojo is a libary for numerical computing in Mojo similiar to numpy in python

NuMojo extends (most of) the standard library math functions to work on tensor inputs

NuMojo intends to capture a wide swath of numerics capability present in the python packages numpy,scipy and scikit such as
* Linear/Tensor Algebra
* Integration
* Derivatives
* Optimizers
* Function approximators

NuMojo intends to try and get the most out of the capabilities of Mojo including vectorization, parallelization, and GPU acceleration(once available).

NuMojo intends to be a building block for other Mojo packages that need fast math under the hood without the added weight of a ML back and forward propigation system

# What NuMojo is not
NuMojo is not a machine learning library, it will never include backproppagation in the base library. In order to make it as fast as possible for calculation and simulation

# How to install
Clone the repository and build

# Usage
For now
```Mojo
import base as numojo
from Tensor import Tensor
def main():
    var tens = Tensor[DType.float32](10,10)
    tens=tens+numojo.pi/2
    print(numojo.sin[DType.float32](tens))
```
## Available functions include:
abs, floor, ceil, trunc, round, roundeven, round_half_down, round_half_up, rsqrt, exp2, exp, log, log2, erf, tanh, reciprocal, identity, acos, asin, atan, cos, sin, tan, acosh, asinh, atanh, cosh, sinh, expm1, log10, log1p, cbrt, erfc, lgamma, tgamma, nearbyint, rint, j0, j1, y0, y1, ulp, all_true, any_true, none_true
# Warnings

This library is still very much a work in progress and may change at any time. Also the standard tensor has many rough edges.