<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="./assets/numojo_logo.png" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">NuMojo</h1>

  <p align="center">
    NuMojo is a library for numerical computing in Mojo 🔥 similar to NumPy, SciPy in Python.
    <br />
    <!-- when we create docs -->
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md"><strong>Explore the docs» </strong></a> &nbsp; &nbsp;
        <a href="./docs/changelog.md"><strong>Changelog» </strong></a> &nbsp; &nbsp;
        <a href="https://discord.com/channels/1149778565756366939/1149778566603620455" ><strong>Check out our Discord» </strong></a>
    </div>
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="./docs/readme_zhs.md"><strong>中文·简» </strong></a> &nbsp; 
        <a href="./docs/readme_zht.md"><strong>中文·繁» </strong></a> &nbsp;
        <a href="./docs/readme_jp.md"><strong>日本語» </strong></a>
    </div>
    <!-- <a href="./docs/readme_kr.md"><strong>한국어 문서» </strong></a> -->
    <!-- <br /> -->
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
        <li><a href="#what-numojo-is"> What NuMojo is </a></li>
        <li><a href="#what-numojo-is-not">What NuMojo is not</a></li>
      </ul>
    </li>
    <li><a href="#goals">Goals</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#how-to-install">How to install</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#warnings">Warnings</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#Contributors">Contributors</a></li>
  </ol>
</details>

## About the project

### What NuMojo is

NuMojo intends to capture a wide swath of numerics capability present in the Python packages NumPy, SciPy, and Scikit.

NuMojo intends to try and get the most out of the capabilities of Mojo including vectorization, parallelization, and GPU acceleration (once available). Currently, NuMojo extends (most of) the standard library math functions to work on array inputs.

We intend NuMojo to be a building block for other Mojo packages that need fast math under the hood without the added weight of a ML back and forward propagation system

### What NuMojo is not

NuMojo is not a machine learning library, it will never include back-propagation in the base library.

## Goals and features

Our main goal is to implement a fast, comprehensive numerics library in Mojo.

Following are some brief long-term goals. Some of them have already been implemented or partially implemented.

Core data types:

- Native n-dimensional array (`numojo.core.ndarray.NDArray`).
- Native 2-dimensional array, i.e., matrix (`numojo.mat.matrix.Matrix`).
- Native fixed-dimension array (to be implemented when trait parameterization is available).

Routines and objects:

- Array creation routines (`numojo.routines.creation`)
- Array manipulation routines (`numojo.routines.manipulation`)
- Bit-wise operations (`numojo.routines.bitwise`)
- Constants (`numojo.routines.constants`)
- Input and output (`numojo.routines.io`)
- Text files (`numojo.routines.files`)
- Text formatting options (`numojo.routines.formatting`)
- Linear algebra (`numojo.routines.linalg`)
- Decompositions (`numojo.routines.decompositions`)
- Products of matrices and vectors (`numojo.routines.products`)
- Solving (`numojo.routines.solving`)
- Logic functions (`numojo.routines.logic`)
- Comparison (`numojo.routines.comparison`)
- Array contents (`numojo.routines.contents`)
- Truth value testing (`numojo.routines.truth`)
- Mathematical functions (`numojo.routines.math`)
- Arithmetic operations (`numojo.routines.arithmetic`)
- Exponents and logarithms (`numojo.routines.exponents`)
- Extrema finding (`numojo.routines.extrema`)
- Floating point routines (`numojo.routines.floating`)
- Hyperbolic functions (`numojo.routines.hyper`)
- Indexing (`numojo.routines.indexing`)
- Miscellaneous (`numojo.routines.misc`)
- Rounding (`numojo.routines.rounding`)
- Sums, products, differences (`numojo.routines.sums`, `numojo.routines.products`, `numojo.routines.differences`)
- Trigonometric functions (`numojo.routines.trig`)
- Random sampling (`numojo.routines.random`)
- Sorting, searching, and counting (`numojo.routines.sorting`, `numojo.routines.searching`)
- Statistics (`numojo.routines.statistics`)
- Averages and variances (`numojo.routines.averages`)
- Calculus, Integration & Derivatives etc
- Optimizers
- Function approximators

Please find all the available functions and objects [here](docs/features.md).

For a detailed roadmap, please refer to the [docs/roadmap.md](docs/roadmap.md) file.

## Usage

An example of n-dimensional array (`NDArray` type) goes as follows.

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = nm.random.randn(shape=Shape(1000, 1000))
    var B = nm.random.randn(shape=Shape(1000, 1000))

    # Generate a 3x2 matrix from string representation
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # Print array
    print(A)

    # Array multiplication
    var C = A @ B

    # Array inversion
    var I = nm.inv(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Get scalar from array
    var A_item = A[item(291, 141)]
    var A_item_2 = A.item(291, 141)
```

An example of matrix (`Matrix` type) goes as follows.

```mojo
from numojo import Matrix
from numojo.prelude import *


fn main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = Matrix.rand(shape=(1000, 1000))
    var B = Matrix.rand(shape=(1000, 1000))

    # Generate 1000x1 matrix (column vector) with random float64 values
    var C = Matrix.rand(shape=(1000, 1))

    # Generate a 4x3 matrix from string representation
    var F = Matrix.fromstring[i8](
        "[[12,11,10],[9,8,7],[6,5,4],[3,2,1]]", shape=(4, 3)
    )

    # Matrix slicing
    var A_slice = A[1:3, 4:19]
    var B_slice = B[255, 103:241:2]

    # Get scalar from matrix
    var A_item = A[291, 141]

    # Flip the column vector
    print(C[::-1, :])

    # Sort and argsort along axis
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # Sum the matrix
    print(nm.sum(B))
    print(nm.sum(B, axis=1))

    # Matrix multiplication
    print(A @ B)

    # Matrix inversion
    print(A.inv())

    # Solve linear algebra
    print(nm.solve(A, B))

    # Least square
    print(nm.lstsq(A, C))
```

## How to install

There are two approach to install and use the Numojo package.

### Build package

This approach invovles building a standalone package file `mojopkg`.

1. Clone the repository.
2. Build the package using `mojo package numojo`
3. Move the numojo.mojopkg into the directory containing the your code.

### Include NuMojo's path for compiler and LSP

This approach does not require buiding a package file. Instead, when you compile your code, you can include the path of NuMojo reporsitory with the following command:

```console
mojo run -I "../NuMojo" example.mojo
```

This is more flexible as you are able to edit the NuMojo source files when testing your code.

In order to allow VSCode LSP to resolve the imported `numojo` package, you can:

1. Go to preference page of VSCode.
2. Go to `Mojo › Lsp: Include Dirs`
3. Click `add item` and write the path where the Numojo repository is located, e.g. `/Users/Name/Programs/NuMojo`.
4. Restart the Mojo LSP server.

Now VSCode can show function hints for the Numojo package!

## Contributing

Any contributions you make are **greatly appreciated**. For more details and guidelines on contributions, please check [here](CONTRIBUTING.md)

## Warnings

This library is still very much a work in progress and may change at any time.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See [LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE) and the LLVM [License](https://llvm.org/LICENSE.txt) for more information.

This project includes code from [Mojo Standard Library](https://github.com/modularml/mojo), licensed under the Apache License v2.0 with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)). MAX and Mojo usage and distribution are licensed under the [MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license).

## Acknowledgements

Built in native [Mojo](https://github.com/modularml/mojo) which was created by [Modular](https://github.com/modularml).

## Contributors

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>