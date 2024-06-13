<br/>
<p align="center">
<!-- If we add a logo in future -->
  <!-- <a href=""> 
    <img src="" alt="Logo" width="200" height="200">
  </a> -->

  <h1 align="center">NuMojo</h1>

  <p align="center">
    NuMojo is a library for numerical computing in Mojo similar to numpy in Python.🔥
  </p>
</p>

## What NuMojo is
NuMojo intends to capture a wide swath of numerics capability present in the Python packages NumPy, SciPy and Scikit.

NuMojo intends to try and get the most out of the capabilities of Mojo including vectorization, parallelization, and GPU acceleration(once available). Currently, NuMojo extends (most of) the standard library math functions to work on tensor inputs.

NuMojo intends to be a building block for other Mojo packages that need fast math under the hood without the added weight of a ML back and forward propagation system

# What NuMojo is not

NuMojo is not a machine learning library, it will never include back-propagation in the base library.

## Goals / Roadmap

* Linear/Tensor Algebra
    * Native array types
    * Vectorized, Parallelized math operations
    * Array manipulation - vstack, slicing, concat etc. 
* Calculus
    * Integration
    * Derivates
* Optimizers
* Function approximators

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

Please find all the available functions [here](features.md)

# How to install

Clone the repository and build

# Warnings

This library is still very much a work in progress and may change at any time. Also, the standard tensor has many rough edges.

## Acknowledgements

* Built in native [Mojo](https://github.com/modularml/mojo) created by [Modular](https://github.com/modularml)