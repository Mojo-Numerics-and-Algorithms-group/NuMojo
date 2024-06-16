<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <!-- If we add a logo in future -->
  <!-- <a href="">
    <img src="" alt="Logo" width="80" height="80">
  </a> -->

  <h1 align="center" font-size>NuMojo</h1>
  <p align="center">
    NuMojo is a library for numerical computing in Mojo 🔥 similar to numpy in Python.
    <br />
    <!-- when we create docs -->
    <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a> -->
    <br />
    <!-- <br /> -->
    <!-- <a href="">View Demo</a>
    ·
    <a href="">Report Bug</a>
    ·
    <a href="">Request Feature</a> -->
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#what-numojo-is-not">What Numojo is not</a></li>
      </ul>
    </li>
    <a href="#goals-roadmap">Goals/Roadmap</a>
      <ul>
        <li><a href="#long-term-goals">Long term goals</a></li>
      </ul>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#how-to-install">How to install</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#warnings">Warnings</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the project
NuMojo intends to capture a wide swath of numerics capability present in the Python packages NumPy, SciPy and Scikit.

NuMojo intends to try and get the most out of the capabilities of Mojo including vectorization, parallelization, and GPU acceleration(once available). Currently, NuMojo extends (most of) the standard library math functions to work on tensor inputs.
NuMojo intends to try and get the most out of the capabilities of Mojo including vectorization, parallelization, and GPU acceleration(once available). Currently, NuMojo extends (most of) the standard library math functions to work on tensor inputs.

NuMojo intends to be a building block for other Mojo packages that need fast math under the hood without the added weight of a ML back and forward propagation system

## What NuMojo is not
## What NuMojo is not

NuMojo is not a machine learning library, it will never include back-propagation in the base library.

## Goals / Roadmap

For a detailed roadmap, please refer to the [Roadmap.md](Roadmap.md) file.

Our main goal is to implement a fast, comprehensive numerics library in Mojo. Following are some brief long term goals,

### Long term goals
* Linear Algebra
    * Native array types
    * Vectorized, Parallelized math operations
    * Array manipulation - vstack, slicing, concat etc. 
* Calculus
    * Integration & Derivatives etc
* Optimizers
* Function approximators

## Usage

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

## How to install

Clone the repository and build

## Contributing
Any contributions you make are **greatly appreciated**. For more details and guidelines on contributions, please check [here](CONTRIBUTING.md)

## Warnings

This library is still very much a work in progress and may change at any time. Also, the standard tensor has many rough edges.

## License

Distributed under the MIT License. See [License](LICENSE) for more information.

## Acknowledgements

* Built in native [Mojo](https://github.com/modularml/mojo) created by [Modular](https://github.com/modularml)