# NuMojo UNRELEASED changelog

This is a list of UNRELEASED changes for the NuMojo Package.

When we make a release, the items in this file will be edited and moved to `changelog.md`.

## dd/mm/yyyy (v0.3)

### ⭐️ New

- Add functions for calculating the inverse of a matrix and LU decomposition (PR #101 by @forFudan).
- Add `itemset` method to fill a scalar into an ndarray (PR #102 by @forFudan).

### 🦋 Changed

- Modify initialization of NDArray (PR #97 by @MadAlex1997)

### ❌ Removed

- Refine the `NDArray` initializers. Remove `random` argument from `NDArray` constructors. Make random initilizer a standalone functions (Issue #96 by @MadAlex1997, PR #97 by @MadAlex1997, PR #98 by @shivasankarka).

### 🛠️ Fixed

- Make `sort` method an inplace method (Issue #87 by @mmenendezg, PR #91 by @shivasankarka).

### 📚 Documentatory and testing

- Add workflows with unit tests and linting (PR #95 by @sandstromviktor).
- Add multi-langual support (Chinese, Japanese, and Korean) for the readme file (PR #99 by @forFudan, PR #100 by @shivasankarka).