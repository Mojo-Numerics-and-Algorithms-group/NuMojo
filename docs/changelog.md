# NuMojo released changelog

This is a list of RELEASED changes for the NuMojo Package.

## xx/xx/2025 (v0.x)

### ⭐️ New

- Add an additional dtype mapping for `NDArray.to_numpy`: `DType.index` -> `numpy.intp` ([PR #157](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/157)).
- Add method `NDArray.resize` that reshapes the array in-place ([PR #158](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/158)).

### 🦋 Changed

- Update the following functions to allow operation by any axis:
  - `argsort` ([PR #157](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/157)).
  - `cumsum` and `cumprod` ([PR #160](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/160)).
- Refine the function `reshape` so that it is working on any dimensions and is working on both row-major and col-major. This also allows us to change the order with the code ```A.reshape(A.shape, "F"))```. Also refine functions `flatten`, `ravel` ([PR #158](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/158)).
- Update `matmul` to enable multiplication between two arrays of any dimensions ([PR #159](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/159)).

### ❌ Removed

### 🛠️ Fixed

- Removes `mut` keyword before `self` from `NDArray.__add__`.

### 📚 Documentatory and testing

- Add `magic run t` and `magic run f` for magic CLI that clear the terminal before running `test` or `final`.

## 22/12/2024 (v0.4)

### ⭐️ New

- Implement a static-sized 2D Array type (`Matrix` type) in `numojo.mat` sub-package. `Matrix` is a special case of `NDArray` but has been optimized since the number of dimensions is known at the compile time. It is useful when users only want to work with 2-dimensional arrays. The indexing and slicing is also more consistent with `numpy` ([PR #138](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/138), [PR #141](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/142), [PR #141](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/142)). It provides:
  - `Matrix` type (2D array) with basic dunder methods and core methods.
  - Function to construct `Matrix` from other data objects, e.g., `List`, `NDArray`, `String`, and `numpy` array.
  - Arithmetic functions for item-wise calculation and broadcasting.
  - Linear algebra: matrix mutiplication, decomposition, inverse of matrix, solve of linear system, Ordinary Least Square, etc.
  - Auxilliary types, e.g., `_MatrixIter`.
- Implement more the array creation routines from numpy and expand the NuMojo array creation functionality ([PR #137](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/137)).
- Add `convolve2d` and in the `science.signal` module ([PR #135](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/135)).
- Add more detailed error messages for `NDArray` type ([PR #140](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/140)).

### 🦋 Changed

- Adapt the code to [the latest update of Mojo to V24.6](https://docs.modular.com/mojo/changelog/) ([PR #148](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/148)).
  - `Slice.step` is now returning `Optional[int]`. Thus, it is fixed by using `Slice.step.else_to(1)`.
  - `Formattable` is now renamed to `Writable` (same applies to `write_to` and `string.write`).
  - `width` is now inferred from the SIMD's width. So this parameter must be removed when we call `UnSafePointer`'s `load` and `store` methods. Due to this, the function `fill_pointer`, which fill in a width of array with a scale, no longer work. It is replaced by copying with loop.
  - `Lifetime` is renamed as `Origin` (same applies to the function `__origin_of`).
  - `inout` is renamed as `mut`.
- Rename the data buffer from `data` to `_buf` ([PR #136](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/136), [PR #137](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/137))
- To make `matmul` flexible for different shapes of input arrays ([PR #137](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/137)).
- Change the way to get the shape of the array: `array.shape` returns the shape of array as `NDArrayShape` ([PR #137](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/137)).
- The array creation functions are unified in such a way ([PR #139](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/139)).
  - `NDAarray.__init__()` reads in shape information and initializes an empty ndarray.
  - All other creation routines are implemented by the functions in the `array_creation_routine` module. For example, to create an array with filled value, the function `full` should be used. To create an array from a list, the function `array` should be used.
- Re-organize the functions and modules by topic, so that it is more consistent with `numpy` ([Issue 144](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/issues/144), [PR #146](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/146)).
- Rename some attributes of `NDArray` and make `size` an attribute instead of a method ([PR #145](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/145)).
- Completely remove `import *` in __init__ files to fix namespace leak ([PR #151](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/151)).
- Update function `sum` ([PR #149](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/149)).
- `transpose` now allows arbitrary dimensions and permutations of axes ([PR #152](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/152)).
- Change buffer type of `NDArrayShape` and `NDArrayStrides` to `Int` ([PR #153](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/153)).
- `sort` now allows sorting by any axis for high dimensional arrays ([PR #154](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/154)).

### ❌ Removed

- Temporarily removed negative indexing support in slices since it causes error. Will add just feature in later updates ([PR #133](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/133)).
- Remove `inout` before `self` for `NDArray.__getitem__` ([PR #137](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/137)).

### 🛠️ Fixed

- Fixed and rewrote the `adjust_slice` function that was causing errors ([PR #133](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/133)).
- There is an error in fromstring that the negative signs are not read. It is fixed. Now a valid numeric should start with a digit, a dot, or a hyhen ([PR #134](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/134)).

### 📚 Documentary and testing

- Added test for slicing in getters. Complete tests for setters will be added later. This is due to python interop limitation ([PR #133](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/133)).
- Move documents from root docs. Delete unused files ([PR #147](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/147)).
- Update the readme.md and features.md to reflect current progress ([PR #150](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/150)).

## 19/10/2024 (v0.3.1)

### 🛠️ Fixed

- Fix the `matmul` function which returns wrong results for small matrices (PR #129).
- Correct the getter and setter functions, specifically the `_adjust_slice` (PR #130).

## 14/10/2024 (v0.3)

### ⭐️ New

- Add support for `magic` system and MAX 24.5 (PR #91 #109 by @shivasankarka).
- Add some basic functions, e.g., `diagflat`, `tri`, `trace`, `T` (PR #91 by @shivasankarka).
- Add a constructor which reads arrays from numpy arrays (PR #91 by @shivasankarka).
- Add functions `solve` and `inv` for solving linear algebra `AX = Y` for `X`, finding inverse of a matrix, and conducting LU decomposition (PR #101 #104 #105 by @forFudan).
- Add `itemset` method to fill a scalar into an `NDArray` (PR #102 by @forFudan).
- Add `Idx` struct to represent the index of an `NDArray` (PR #118 by @shivasankarka).
- Add NDArray initialization from numpy arrays (PR #118 by @shivasankarka).
- Created a new `io` module with some basic functions, e.g., `format_float_scientific`, `loadtxt`, `savetxt` (PR #118 by @shivasankarka).

### 🦋 Changed

- Make some methods, e.g., `sort`, `flatten`, `inplace` (Issue #87 by @mmenendezg, PR #91 by @shivasankarka).
- Modify initialization of NDArray (PR #97 by @MadAlex1997)
- Added `Formattable` trait and fixed the `print` function (PR #108 by @shivasankarka)
- Refine the `NDArray` initializers and array creation routines (Discussion #90, Issue #110).
  - Remove `random` argument from `NDArray` constructors. Make random initializer a standalone functions (Issue #96 by @MadAlex1997, PR #97 by @MadAlex1997, PR #98 by @shivasankarka).
  - Remove initializers from `String`. Make `fromstring` a standalone function (#113 by @forFudan).
  - Add several `array` overloads for initializing `NDArray` (PR #118 by @shivasankarka).
- Modify the behavior of `__get__` and `__set__`. Passing in a sequence of `Int` or `Slice` returns an `NDArray`. Passing in an `Idx` returns a scalar. Allow users to set a multiple items in one `NDArray` with another `NDArray`, using `__set__` (PR #118 by @shivasankarka, Discussion #70).

### ❌ Removed

- Removed all instances of getters and setters with `List[Int]`, `VariadicList[Int]` (PR #118 by @shivasankarka).

### 🛠️ Fixed

- Fix the issues in parallelization (due to Max 24.5) for some linear algebra functions, e.g, `matmul`, `solve`, `inv`, etc (PR #115 #117 by @forFudan).

### 📚 Documentary and testing

- Add workflows with unit tests and linting (PR #95 by @sandstromviktor).
- Add multi-lingual support (Chinese, Japanese) for the readme file (PR #99 #120 by @forFudan, PR #100 by @shivasankarka).
- Update the test flow file to accommodate MAX 24.5 and the `magic` system (PR #116 by @forFudan).

## 17/08/2024 (V0.2)

### ⭐️ New

Array operations:

- Introduced `diagflat()` method for creating diagonal arrays
- Implemented most basic array creation routines
- Implemented initializer from string `numojo.NDArray("[[1,2,3],[4,5,6]]")`
- Enhanced `NDArray` constructor methods
- Added boolean masking for NDArrays
- Introduced new mathematical methods: `floordiv`, `mod`, and more
- `__getitem__` and `__setitem__` now fully compatible with NumPy behavior

Others:

- Introduced Rust-like data type aliases (e.g., `DType.float64` → `f64`)
- Implemented function overloading for core mathematical operations (add, sub, etc.)
- Added a new differentiation module

### 🦋 Changed

- Improved slicing functionality to match NumPy behavior

### ❌ Removed

- Removed `in_dtype` and `out_dtype` parameters for simplified type handling

### 📚 Documentatory and testing

Documentation updates:

- Expanded and updated README
- Improved docstrings for functions
- Added style guide and examples

Testing updates:

- Introduced new test files that are compatible with `mojo test`.
