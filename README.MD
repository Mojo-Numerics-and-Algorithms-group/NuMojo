# NuMojo

![logo](./assets/numojo_logo_360x360.png)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Mojo](https://img.shields.io/badge/mojo-%3E%3D1.0.0-orange.svg)](https://www.modular.com/mojo)

NuMojo is a library for numerical computing in Mojo 🔥, similar to NumPy in Python.

**[Explore the examples»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md)**  |  **[Changelog»](docs/user-guide/changelog.md)**  |  **[Check out our Discord»](https://discord.gg/NcnSH5n26F)**

**[中文·简»](docs/getting-started/readme_zhs.md)**  |  **[中文·繁»](docs/getting-started/readme_zht.md)**  |  **[日本語»](docs/getting-started/readme_jp.md)**  |  **[한국어»](docs/getting-started/readme_kr.md)**

**Table of Contents**

1. [About the project](#about-the-project)
2. [Why NuMojo](#why-numojo)
3. [Features and goals](#features-and-goals)
4. [Usage](#usage)
5. [Installation](#installation)
6. [Contributing](#contributing)
7. [Warnings](#warnings)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)
10. [Contributors](#contributors)

## About the project

NuMojo aims to encompass the extensive numerics capabilities found in NumPy.

**What NuMojo is**

We seek to harness the full potential of Mojo, including vectorization, parallelization, and GPU acceleration. Currently, NuMojo extends most (if not all) standard library math functions to support array inputs.

Our vision for NuMojo is to serve as a familiar and essential building block for other Mojo libraries needing fast math operations, without the additional weight of a machine learning back-propagation system.

## Why NuMojo

- **Native to Mojo.** NuMojo's `NDArray` is a Mojo-native SIMD-backed type, not a binding around NumPy or MAX's tensor types, so it compiles into your program with no Python interop overhead.
- **NumPy-familiar API.** Slicing, broadcasting, `@` for matrix multiplication, and function names mirror NumPy where it makes sense, so existing intuition carries over.
- **Built for Mojo's strengths.** Vectorization and parallelism are used throughout the routines, with GPU and other accelerator support (`AcceleratorNDArray`) landing as Mojo's own device support matures.

## Features and goals

Our primary objective is to develop a fast, comprehensive numerics library in Mojo. Below are some features and long-term goals. Some have already been implemented (fully or partially).

Core data types:

- Native n-dimensional array (`numojo.NDArray`).
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

Please find all the available functions and objects [here](docs/user-guide/features.md). A living roadmap is maintained in [docs/user-guide/roadmap.md](docs/user-guide/roadmap.md).

## Usage

Runnable examples are available in `examples/` (e.g., `examples/quickstart.mojo`).

An example of n-dimensional array (`NDArray` type) goes as follows.

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = nm.random.randn(Shape(1000, 1000)) # Shape is used for all shape related operations in numojo. 
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
    var A_item = A[Item(291, 141)] # Item() is used to define coordinates of an ndarray in numojo. 
    var A_item_2 = A.item(291, 141)

    # Sort and argsort along axis
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # Sum along axis
    print(nm.sum(A))
    print(nm.sum(A, axis=1))

    # Solve a linear system
    print(nm.solve(A, B))
```

An example of `ComplexNDArray` is as follows:

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # Create a complex scalar 5 + 5j
    # cf32 is the complex version of f32 (DType.float32) used to identify complex types in numojo.
    var complexscalar = CScalar[cf32](5) # Equivalently ComplexSIMD[cf32](5, 5)
    # Also can be define as simple as  5 + 5*`1j`!
  
    # Create complex arrays
    var A = nm.full[cf32](Shape(1000, 1000), fill_value=complexscalar)  # filled with (5+5j)
    var B = nm.ones[cf32](Shape(1000, 1000))                            # filled with (1+1j)

    # Print array
    print(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Array multiplication
    var C = A * B

    # Get scalar from array
    var A_item = A[Item(291, 141)]
    # Set an element of the array
    A[item(291, 141)] = complexscalar
```

## Installation

NuMojo offers several installation methods to suit different development needs. Choose the method that best fits your workflow:

### Method 1: Git Installation with pixi-build-mojo (Recommended)

Install NuMojo directly from the GitHub repository to access both stable releases and cutting-edge features. This method is perfect for developers who want the latest functionality or need to work with the most recent stable version.

Add the following to your existing `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project_name"
version = "0.1.0"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[package.build.config.pkg]
name = "your_package_name"

[package.host-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"

[package.build-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[package.run-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[dependencies]
mojo = ">=1.0.0, <1.1.0"
max-core = ">=26.5.0,<27"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}
```

Then run:
```bash
pixi install
```

**Branch Selection:**
- **`main` branch**: Provides the latest stable release. Currently NuMojo v0.10.0, compatible with Mojo >=1.0.0, <1.1.0. For earlier NuMojo versions, use Method 2.
- **`pre-x.y` branches**: Active development branch for the next release. Note that this branch receives frequent updates and may have breaking changes in features and syntax.

The package will be automatically available in your Pixi environment, and VSCode LSP will provide intelligent code hints.

### Method 2: Stable Release via Pixi (prefix.dev)

For most users, we recommend installing a stable release through Pixi for guaranteed compatibility and reproducibility.

Add the following to your `pixi.toml` file:

```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]

[dependencies]
numojo = "=0.10.0"
```

Then run:
```bash
pixi install
```

**Version Compatibility:**

| NuMojo Version | Required Mojo Version |
| -------------- | --------------------- |
| v0.10.0        | ==1.0.0               |
| v0.9.0         | ==26.2                |
| v0.8.0         | ==25.7                |
| v0.7.0         | ==25.3                |
| v0.6.1         | ==25.2                |
| v0.6.0         | ==25.2                |

### Method 3: Build Standalone Package

This method creates a portable `numojo.mojopkg` file that you can use across multiple projects, perfect for offline development or hermetic builds.

1. Clone the repository:
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   cd NuMojo
   ```

2. Build the package:
   ```bash
   pixi run package
   ```

3. Copy `numojo.mojopkg` to your project directory or add its parent directory to your include paths.

### Method 4: Direct Source Integration

For maximum flexibility and the ability to modify NuMojo source code during development:

1. Clone the repository to your desired location:
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   ```

2. When compiling your code, include the NuMojo source path:
   ```bash
   mojo run -I "/path/to/NuMojo" your_program.mojo
   ```

3. **VSCode LSP Setup** (for code hints and autocompletion):
   - Open VSCode preferences
   - Navigate to `Mojo › Lsp: Include Dirs`
   - Click `Add Item` and enter the full path to your NuMojo directory (e.g., `/Users/YourName/Projects/NuMojo`)
   - Restart the Mojo LSP server

After setup, VSCode will provide intelligent code completion and hints for NuMojo functions!

## Contributing

Any contributions you make are **greatly appreciated**. NuMojo is early-stage and there's a lot of room to help: implementing routines, writing tests, improving docs, or just filing issues for things that feel off.

Quick start for contributors:

```bash
git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
cd NuMojo
pixi install
pixi run final   # format + test
```

See [docs/developer-guide/contributing.md](docs/developer-guide/contributing.md) for full guidelines (coding style, directory structure, PR process), [docs/developer-guide/style-guide.md](docs/developer-guide/style-guide.md) for docstring/formatting conventions, and [docs/developer-guide/pre-pr-checks.md](docs/developer-guide/pre-pr-checks.md) for what to run before opening a PR.

## Warnings

This library is still early and may introduce breaking changes between minor versions. Pin versions in production or research code.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See [LICENSE](LICENSE) and the LLVM [License](https://llvm.org/LICENSE.txt) for more information.

This project includes code from [Mojo Standard Library](https://github.com/modularml/mojo), licensed under the Apache License v2.0 with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)). MAX and Mojo usage and distribution are licensed under the [MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license).

## Acknowledgements

Built in native [Mojo](https://github.com/modularml/mojo) which was created by [Modular](https://github.com/modularml).

## Contributors

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
