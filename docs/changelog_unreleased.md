# NuMojo UNRELEASED changelog

This is a list of UNRELEASED changes (not yet merged to the nightly branch) for the NuMojo Package.

When a PR is merged to the nightly branch, the items in this file will be moved to `changelog.md`.

## dd/mm/yyyy (v0.3)

### ⭐️ New

- Add `Idx` struct to represent the index of an `NDArray` (PR #118 by @shivasankarka).
- Add NDArray initialization from numpy arrays (PR #118 by @shivasankarka).
- Created a new IO module with some basic functions, e.g., `format_float_scientific`, `loadtxt`, `savetxt` (PR #118 by @shivasankarka).

### 🦋 Changed

- Modify the behavior of `__get__` and `__set__`. Passing in a sequence of `Int` or `Slice` returns an `NDArray`. Passing in an `Idx` returns a scalar. Allow users to set a multiple items in one `NDArray` with another `NDArray`, using `__set__` (PR #118 by @shivasankarka, Discussion #70).

### ❌ Removed

- Removed all instances of getters and setters with `List[Int]`, `VariadicList[Int]` (PR #118 by @shivasankarka).

### 🛠️ Fixed

### 📚 Documentary and testing
