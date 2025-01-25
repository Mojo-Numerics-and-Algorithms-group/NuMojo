# NuMojo released changelog

This is a list of RELEASED changes for the NuMojo Package.

## xx/xx/2025 (v0.5)

### ⭐️ New

- Add support for complex arrays `ComplexNDArray`, `ComplexSIMD`, `CDType` ([PR #165](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/165)).
- Add `TypeCoercion` struct that calculates the resultant type based on two initial data types. Apply type coercion to math functions ([PR #164](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/164), [PR #189](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/189)).
- Add `OwnData` type as container of data buffer for `NDArray` and `Matrix`. The property `_buf` is changed from `UnsafePointer` to `OwnData`. Introduced `RefData` struct and `Bufferable` trait. This step prepares for future support of array views and facilitates an easy transition once parameterized traits are integrated into Mojo ([PR #175](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/175), [PR #170](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/170), [PR #178](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/178)).
- Add `NDIter` type as a iterator over the array items according to a certain memory layout. Use `NDArray.nditer()` to construct the iterator ([PR #188](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/188)).
- Add an additional data type mapping for `NDArray.to_numpy`, i.e., `DType.index` is mapped to `numpy.intp` ([PR #157](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/157)).
- Add method `NDArray.resize` that reshapes the array in-place ([PR #158](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/158)).
- Add a new property `flags` for `NDArray` and `Matrix` to store memory layout of the array, e.g., c-continuous, f-continuous, own data. The memory layout is determined by both strides and shape of the array ([PR #170](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/170), [PR #178](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/178)).
- Add functions `to_tensor` and `from_tensor` (also an overload of `array`) that convert between `NDArray` and MAX `Tensor` ([Issue #183](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/183), [PR #184](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/184)).
- Add several functions for linear algebra:
  - Add `det` function calculating the determinant of array ([PR #171](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/171)).
  - Add Householder-based QR decomposition for matrix ([PR #172](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/172)).

### 🦋 Changed

- Update several methods of `NDArray` type:
  - Make `load` and `store` methods safer by imposing boundary checks. Rename `get` as `load` and rename `set` as `store` ([PR #166](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/166)).
  - Update `item` method to allow negative index and conduct boundary checks on that ([PR #167](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/167)).
  - Update `item` function so that it allows both setting and getting values, e.g., `a.item(1,2)` and `a.item(1,2) = 10` ([PR #176](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/176)).
  - Refine the `__init__` overloads ([PR #170](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/170)).
- Allow getting items from `NDArray` with a list or array of indices, while the array can be multi-dimensional. Allow getting items from `NDArray` with a list or array of boolean values, while the array can be both one-dimensional or multi-dimensional ([PR #180](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/180), [PR #182](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/182)).
- Rename `Idx` struct as `Item`.You can get scalar from array using `a[item(1,2)]` ([PR #176](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/176)).
- Update the constructor of the `Item` type to allow passing int-like values in ([PR #185](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/185)).
- Integrate `mat` sub-package into `core` and `routines` modules, so that users can use a uniformed way to call functions for both `Matrix` and `NDArray` types ([PR #177](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/177)).
- Update the following functions to allow operation by any axis:
  - `argsort` ([PR #157](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/157)).
  - `cumsum` and `cumprod` ([PR #160](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/160)).
  - `flip` ([PR #163](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/163)).
- Update `matmul` to enable multiplication between two arrays of any dimensions ([PR #159](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/159)).
- Refine the function `reshape` so that it is working on any dimensions and is working on both row-major and col-major. This also allows us to change the order with the code ```A.reshape(A.shape, "F"))```. Also refine functions `flatten`, `ravel` ([PR #158](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/158)).
- Remove `size` property from `NDArrayShape` and add `size_of_array` method to get the size of the corresponding array ([PR #181](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/181)).
- Add `PrintOption` type to customize string representation of `NDArray`, `NDArrayShape`, `NDArrayStrides`, and `Item`, e.g., `str()`, `repr()`, `print()`. Allow customized separators, paddings, number of items to display, width of formatting, etc, for `NDArray._array_to_string` method. Auto-adjust width of formatted values and auto-determine wether scientific notations are needed ([PR #185](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/185), [PR #186](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/186), [PR #190](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/190), [PR #191](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/191), [PR #192](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/192)).
- Rename the auxiliary function `_get_index` as `_get_offset` ([PR #173](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/173)).
- Rename the underlying buffer of `Idx` type to `_buf` ([PR #173](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/173)).
- Return a view instead of copy for iterator of `NDArray` ([PR #174](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/174)).

### ❌ Removed

- Remove `order` property and `datatype` property from `NDArray`. You can use `a.flags` and `a.dtype` instead ([PR #170](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/170)).
- Remove the property `coefficient` of `NDArray` type ([PR #166](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/166)).
- Remove `offset` property from the `NDArrayStrides` type ([PR #173](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/173)).

### 🛠️ Fixed

- Removes `mut` keyword before `self` from `NDArray.__add__`.
- The methods `get` and `set` of `NDArray` does not check against big negative index values and this leads to overflows. This is fixed ([PR #162](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/162)).
- Fix `__setitem__` method so that it can read in variant of int and slices, e.g., `a[0, Slice(2, 4)] = a[3, Slice(0, 2)]` ([PR #176](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/176)).

### 📚 Documentatory and testing

- Add `readthedocs` pages to the repo under `/docs/readthedocs` ([PR #194](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/194)).
- Add `magic run t`, `magic run f`, and `magic run p` for the magic CLI. They first clear the terminal before running `test`,`final`, and `package`.
- Allow partial testing via command, e.g., `magic run test_creation`, to avoid overheat.
- Convert the readme file into pure markdown syntax ([PR #187](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/187)).

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
