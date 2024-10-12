# NuMojo UNRELEASED changelog

This is a list of UNRELEASED changes for the NuMojo Package.

When we make a release, the items in this file will be edited and moved to `changelog.md`.

## dd/mm/yyyy (v0.3)

### ⭐️ New

- Add support for `magic` system and MAX 24.5 (PR #91, #109 by @shivasankarka).
- Add some basic functions, e.g., `diagflat`, `tri`, `trace`, `T` (PR #91 by @shivasankarka).
- Add a constructor which reads arrays from numpy arrays (PR #91 by @shivasankarka).
- Add functions `solve` and `inv` for solving linear algebra `AX = Y` for `X`, finding inverse of a matrix, and conducting LU decomposition (PR #101 #104 #105 by @forFudan).
- Add `itemset` method to fill a scalar into an `NDArray` (PR #102 by @forFudan).
- Add `Idx` struct to represent the index of an `NDArray` (PR #118 by @shivasankarka).
- Add NDArray initialization from numpy arrays (PR #118 by @shivasankarka).
- Created a new IO module with some basic functions, e.g., `format_float_scientific`, `loadtxt`, `savetxt` (PR #118 by @shivasankarka).

### 🦋 Changed

- Make some methods, e.g., `sort`, `flatten`, `inplace` (Issue #87 by @mmenendezg, PR #91 by @shivasankarka).
- Modify initialization of NDArray (PR #97 by @MadAlex1997)
- Added `Formattable` trait and fixed the `print` function (PR #108 by @shivasankarka)
- Refine the `NDArray` initializers and array creation routines (Discussion #90 and Issue #110).
  - Remove `random` argument from `NDArray` constructors. Make random initializer a standalone functions (Issue #96 by @MadAlex1997, PR #97 by @MadAlex1997, PR #98 by @shivasankarka).
  - Remove initializers from `String`. Make `fromstring` a standalone function (#113 by @forFudan).
- Modify the behavior of `__get__` and `__set__`. Passing in a sequence of `Int` or `Slice` returns an `NDArray`. Passing in an `Idx` returns a scalar. Allow users to set a multiple items in one `NDArray` with another `NDArray`, using `__set__` (PR #118 by @shivasankarka, Discussion #70).

### ❌ Removed

- Removed all instances of getters and setters with `List[Int]`, `VariadicList[Int]` (PR #118 by @shivasankarka).

### 🛠️ Fixed

- Fix the issues in parallelization for some linear algebra functions, e.g, `matmul`, `solve`, `inv`, etc (PR #115, #117 by @forFudan).

### 📚 Documentary and testing

- Add workflows with unit tests and linting (PR #95 by @sandstromviktor).
- Add multi-lingual support (Chinese, Japanese) for the readme file (PR #99 by @forFudan, PR #100 by @shivasankarka).
- Update the test flow file to accommodate MAX 24.5 and the `magic` system (PR #116 by @forFudan).
