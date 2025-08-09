# NuMojo

![logo](./assets/numojo_logo_360x360.png)

NuMojo is a library for numerical computing in Mojo 🔥 similar to NumPy, SciPy in Python.

**[Explore the docs»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md)**  |  **[Changelog»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/changelog.md)**  |  **[Check out our Discord»](https://discord.gg/NcnSH5n26F)**

**[中文·简»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_zhs.md)**  |  **[中文·繁»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_zht.md)**  |  **[日本語»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_jp.md)**    |
**[한국어»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_kr.md)** 

**Table of Contents**

1. [About The Project](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#about-the-project)
2. [Goals](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#goals)
3. [Usage](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#usage)
4. [How to install](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#how-to-install)
5. [Contributing](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#contributing)
6. [Warnings](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#warnings)
7. [License](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#license)
8. [Acknowledgements](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#acknowledgments)
9. [Contributors](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#Contributors)

## About the project

NuMojo aims to encompass the extensive numerics capabilities found in Python packages such as NumPy, SciPy, and Scikit-learn.
***What NuMojo is***

We seek to harness the full potential of Mojo, including vectorization, parallelization, and GPU acceleration (when available). Currently, NuMojo extends most (if not all) standard library math functions to support array inputs.

Our vision for NuMojo is to serve as an essential building block for other Mojo packages needing fast math operations, without the additional weight of a machine learning back-propagation system.

***What NuMojo is not***

NuMojo is not a machine learning library and will never include back-propagation as part of the base library.

## Features and goals

Our primary objective is to develop a fast, comprehensive numerics library in Mojo. Below are some features and long-term goals. Some have already been implemented (fully or partially).

Core data types:

- Native n-dimensional array (`numojo.NDArray`).
- Native 2-dimensional array, i.e., matrix (`numojo.Matrix`).
- Native n-dimensional complex array (`numojo.ComplexNDArray`)
- Native fixed-dimension array (to be implemented when trait parameterization is available).

Routines and objects:

- Array creation routines (`numojo.creation`)
- Array manipulation routines (`numojo.manipulation`)
- Input and output (`numojo.io`)
- Linear algebra (`numojo.linalg`)
- Logic functions (`numojo.logic`)
- Mathematical functions (`numojo.math`)
- Exponents and logarithms (`numojo.exponents`)
- Extrema finding (`numojo.extrema`)
- Rounding (`numojo.rounding`)
- Trigonometric functions (`numojo.trig`)
- Random sampling (`numojo.random`)
- Sorting and searching (`numojo.sorting`, `numojo.searching`)
- Statistics (`numojo.statistics`)
- etc...

Please find all the available functions and objects [here](docs/features.md). A living roadmap is maintained in [docs/roadmap.md](docs/roadmap.md).

For a detailed roadmap, please refer to the [docs/roadmap.md](docs/roadmap.md) file.

## Usage

An example of n-dimensional array (`NDArray` type) goes as follows.

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = nm.random.randn(Shape(1000, 1000))
    var B = nm.random.randn(Shape(1000, 1000))

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

An example of `ComplexNDArray` is as follows:

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # Create a complex scalar 5 + 5j
    var complexscalar = ComplexSIMD[f32](re=5, im=5)
    # Create complex arrays
    var A = nm.full[f32](Shape(1000, 1000), fill_value=complexscalar)  # (5+5j)
    var B = nm.ones[f32](Shape(1000, 1000))                            # (1+1j)

    # Print array
    print(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Array multiplication
    var C = A * B

    # Get scalar from array
    var A_item = A[item(291, 141)]
    # Set an element of the array
    A[item(291, 141)] = complexscalar
```

## How to install

There are three approaches to install and use the NuMojo package.

### Add `numojo` in `pixi.toml`

You can add the package `numojo` (pin to an exact version for reproducibility) in the dependencies section of your `pixi.toml` file.

```toml
[dependencies]
numojo = "=0.7.0"
```

Then run `pixi install` to install the package.

The following table shows the version of `numojo` and the corresponding version of `mojo` that is required.

| `numojo` | `mojo` |
| -------- | ------ |
| v0.7.0   | ==25.3 |
| v0.6.1   | ==25.2 |
| v0.6.0   | ==25.2 |

### Build package

This approach builds a standalone package file `numojo.mojopkg` that you can copy into other projects (useful for offline or hermetic builds and for using the latest NuMojo branch).

1. Clone the repository.
2. Build the package using `pixi run package`.
3. Move `numojo.mojopkg` into the directory containing your code (or add its parent directory to your include paths).

### Include NuMojo's path for compiler and LSP

This approach does not require building a package file. When compiling, include the NuMojo source path directly:

```console
mojo run -I "../NuMojo" example.mojo
```

This is more flexible as you are able to edit the NuMojo source files when testing your code.

To allow VSCode's Mojo LSP to resolve the imported `numojo` package:

1. Go to preference page of VSCode.
2. Go to `Mojo › Lsp: Include Dirs`
3. Click `add item` and write the path where the Numojo repository is located, e.g. `/Users/Name/Programs/NuMojo`.
4. Restart the Mojo LSP server.

Now VSCode can show function hints for the Numojo package!

## Contributing

Any contributions you make are **greatly appreciated**. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coding style, testing, documentation, release cadence).

## Warnings

This library is still early and may introduce breaking changes between minor versions. Pin versions in production or research code.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See [LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE) and the LLVM [License](https://llvm.org/LICENSE.txt) for more information.

This project includes code from [Mojo Standard Library](https://github.com/modularml/mojo), licensed under the Apache License v2.0 with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)). MAX and Mojo usage and distribution are licensed under the [MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license).

## Acknowledgements

Built in native [Mojo](https://github.com/modularml/mojo) which was created by [Modular](https://github.com/modularml).

## Contributors

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
