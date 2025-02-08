# Roadmap

Our primary objective is to develop a robust and comprehensive library for numerical and scientific computing. To facilitate a smooth transition from Python, we will primarily adhere to the conventions of `numpy` and `scipy`, while allowing deviations when they:

- Align with the design philosophy of the Mojo programming language.
- Significantly enhance performance, such as through the use of generics.
- Improve consistency in conventions, such as naming parameters and arguments.

NuMojo is currently in its early development stages. At this point, our focus is to ***stabilize the API*** and ***implement the majority of functionalities***. If you notice any missing features, please consider developing them and submitting a pull request or creating a feature request.

## Core Tasks

- Implement the n-dimensional array type and support SIMD-compatible standard library math functions[^stdlib].
- Develop `numpy`-like functions for mathematical, statistical, linear algebra, searching, sorting, etc.
- Create `scipy`-like functions for scientific purposes, such as optimizers, function approximators, and FFT.

### N-dimensional Arrays

We have implemented basic functions and methods for the N-dimensional array `NDArray` (and also `ComplexNDArray` and `Matrix`). We are working on incorporating additional essential features similar to those in `numpy`.

Currently, operations on an array return a copy. When the Mojo programming language supports parameterized traits, some operations (e.g., slicing and transpose) will return a view of the array. This will avoid excessive copying of data, increase memory reuse, and potentially enhance performance.

In the future, when the Mojo programming language supports GPU functionality as it currently does with SIMD, NuMojo will also provide an option to use the GPU for calculations.

### Implement Basic Numeric Functions

We are currently working on implementing basic numeric functions into NuMojo. The scope is similar to `numpy`. Functions on [this page](https://numpy.org/doc/stable/reference/routines.html) will be considered for gradual implementation in NuMojo.

### Implement Advanced Functions

We also aim to implement advanced functions into NuMojo. The scope is similar to `scipy`.

## Internal Organization of Objects and Functions

NuMojo organizes modules internally according to the following structure[^numpy]:

1. A `routines` folder is created under `/numojo`. Functions covered by [this page](https://numpy.org/doc/stable/reference/routines.html) will be considered for implementation in this folder.
2. Sub-folders[^single] will be created under `/routines` for each topic [on this page](https://numpy.org/doc/stable/reference/routines.html). Examples include:
   - `/creation` (Array creation routines)
   - `/logic` (Logic functions)
   - `/mathematics` (Mathematical functions)
   - ...
3. In each sub-folder, functions are grouped by topics into single Mojo files. For example, in the `/mathematics` folder, the following files will be created [(as classified by NumPy on this page)](https://numpy.org/doc/stable/reference/routines.math.html):
   - `trig.mojo` (Trigonometric functions)
   - `hyperbolic.mojo` (Hyperbolic functions)
   - `exp_log.mojo` (Exponents and logarithms)
   - `other.mojo` (Other special functions)
   - `arithmetic.mojo` (Arithmetic operations)
   - ...
4. In each file, functions are sorted alphabetically.
5. The `__init__.mojo` files of parent folders import functions from their child modules explicitly, avoiding `import *` to prevent polluting the namespace.

Additionally, a `science` folder is created under `/numojo`. It is similar to the `routines` folder but contains sub-packages for features present in `scipy`[^science]. For example:

Users can access the functions either directly at the top level or via sub-packages.

1. Most common functions can be called from the top level, e.g., `numojo.sort()`.
2. Advanced features (e.g., those listed as sub-packages in `numpy` or `scipy`) need to be called via their own namespaces. For example:
   - Random array generators, e.g., `numojo.random.randint()`.
   - Linear algebra, e.g., `numojo.linalg.solve()`.
   - FFT, e.g., `numojo.fft()`.
   - Ordinary differential equations.
   - Optimizers, e.g., `numojo.optimize`.

[^stdlib]: Standard library functions that are SIMD-compatible.
[^numpy]: The structure is inspired by the organization of functions in NumPy.
[^single]: If a topic has only a few functions, they can be grouped into a single Mojo file instead of creating a separate folder.
[^science]: This folder can be further integrated into the `routines` folder if necessary.
