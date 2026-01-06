# NuMojo released changelog

This is a list of RELEASED changes for the NuMojo Package.

## (v0.8.0)

### ⭐️ New
- Introduced a Python-like imaginary literal via the new `ImaginaryUnit` type and the `1j` alias, enabling natural complex-number expressions across scalars, SIMD vectors, and arrays.
  #### Example: Python-like complex literals
  ```mojo
  from numojo import `1j`
  # Scalar complex numbers
  var c1 = 3 + 4 * `1j`                    # ComplexScalar[cint]: (3 + 4j)
  var c2 = 2.0 * `1j`                      # ComplexScalar[cf64]: (0 + 2j)
  var c3 = 5 - `1j`                        # ComplexScalar[cint]: (5 - 1j)
  
  # SIMD complex vectors  
  var c4 = SIMD[f32, 4](1.0) + `1j` * SIMD[f32, 4](2.0)  # ComplexSIMD[cf32, 4]
  var c5 = SIMD[f64, 2](3.0, 4.0) + `1j`                 # ComplexSIMD[cf64, 2]
   var d = SIMD[f32, 2](1) + SIMD[f32, 2](2) * `1j` # creates [( 1 + 2 j) (1 + 2 j)]
  
  # Mathematical properties
  var c6 = `1j` * `1j`                     # -1 (Scalar[f64])
  var c7 = `1j` ** 3                       # (0 - 1j) (ComplexScalar[cf64])
  var c8 = (1 + `1j`) / `1j`               # (1 - 1j) (ComplexScalar[cf64])
  ```
  - Refined the behavior of ComplexSIMD accessors and mutators (`__getitem__, __setitem__, item, itemset`) to improve clarity and flexibility.
  - Updated ComplexSIMD access patterns to clearly support:
    - Lane-wise access
    - Component-wise access (re / im)
    - Bulk access of real and imaginary components
    ##### Example: Updated ComplexSIMD access patterns
    ```mojo
    var complex_simd = ComplexSIMD[cf32, 4](1.0, 2.0)  # All lanes set to 1+2i
    
    # Lane-wise access
    var lane2 = complex_simd[2]  # Get ComplexScalar at lane 2
    complex_simd[1] = ComplexScalar[cf32](3.0, 4.0)  # Set lane 1 to 3+4i
    
    # Component access
    var real_part = complex_simd.item["re"](2)  # Get real part of lane 2
    complex_simd.itemset["im"](1, 5.0)  # Set imaginary part of lane 1 to 5.0
    
    # Bulk access
    var all_reals = complex_simd.re  # Get all real parts as SIMD vector
    var all_imags = complex_simd.im  # Get all imaginary parts as SIMD vector
    ```
- Added multiple convenience APIs for complex workflows:
  - Convenience constructors such as `zero()`, `one()`, `I()`, and `from_polar()` for creating `ComplexSIMD` instances. 
  - New ComplexSIMD helpers (`component_bitwidth, elem_pow, all_close)
  - Broadcasting support for scalar complex values.
- `Matrix` views are finally here! 🥁 [PR #280](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/280), [PR #281](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/281), [PR #282](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/282), [PR #283](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/283), [PR #284](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/284)
  - Added full support for view returns in `Matrix`.
  - Laid the groundwork for view-based returns in both Matrix and NDArray, following the design outlined in NuMojo Enhancement Proposal #279.
  - Temporarily renamed Matrix methods that return views from `__getitem__ / __setitem__` to `get() / set()` due to a Mojo compiler bug.
  - Default `__getitem__` and `__setitem__` currently return copies while `get()`, `set()` work with both copies and views. Use get() and set() to obtain and modify views until the Mojo issue is resolved. We might consider keeping both in future too if it seems useful to have the distinction.
  - Full NDArray view support and adoption of the new UnsafePointer model will follow in subsequent PRs to keep changes incremental.
- Introduced a new `TestSuite`-based testing infrastructure to replace the deprecated `mojo test` command. [PR #280](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/280), [PR #281](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/281), [PR #282](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/282), [PR #283](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/283), [PR #284](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/284)
- Updated GitHub workflows to run tests using mojo run and TestSuite. [PR #280](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/280), [PR #281](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/281), [PR #282](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/282), [PR #283](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/283), [PR #284](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/284)
- Extended ComplexNDArray with magnitude-based comparison operators (`__lt__, __le__, __gt__, __ge__`). [PR #275](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/275)
- Added a comprehensive set of statistical and reduction methods to ComplexNDArray, including `all, any, sum, prod, mean, max, min, argmax, argmin, cumsum, and cumprod`. [PR #275](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/275)
- Added array manipulation utilities for ComplexNDArray, such as `flatten, fill, row, col, clip, round, T, diagonal, trace, tolist, and resize`. [PR #275](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/275)
- Implemented SIMD load/store APIs (`load, store, unsafe_load, unsafe_store`) for `Item, Shape`, and `Strides` to support vectorized operations. [PR #275](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/275)
- Added new dtype aliases. [PR #274](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/274)
- Unified creation and manipulation APIs for Item, Shape, and Strides. [PR #274](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/274)
- Introduced explicit copy semantics for large data structures in alignment with Mojo 0.25.6 copy rules. [PR #270](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/270)
  - Implemented the Copyable trait for large container types such as NDArray and Matrix, enabling explicit duplication via `.copy()` which returns an array with same origin. To get an instance with without an origin referencing previous memory, use `.deep_copy()`.
- Reintroduced parameter-based type distinctions between real and complex values across NuMojo APIs. [PR #269](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/269)
  - Added ComplexDType variants for all supported real DType values by prefixing with c (e.g., `i8 → ci8, i32 → ci32, u32 → cu32, f64 → cf64`). 
- Added parallel function overloads that accept `ComplexDType`, ensuring strict separation between real and complex workflows.
- Enabled scalar and SIMD creation utilities for complex values, including `CScalar` and `ComplexSIMD`. [PR #269](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/269)
- Introduced the pixi-build-mojo backend for NuMojo, enabling installation directly from the NuMojo GitHub repository without relying on Modular Community, Prefix.dev, or Conda channels. Check out the PR for more details on how to use it. [PR #268](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/268)
  - Automatically adds the NuMojo package to the path when using pixi shell.
  - Provides LSP support in VS Code.
- Enhanced slicing support for `NDArray` and `ComplexNDArray` to achieve closer NumPy compatibility. [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Added full negative indexing and reverse slicing support (e.g., `arr[::-1], arr[5:1:-1]`). [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Introduced automatic out-of-bounds clamping for slice ranges, matching NumPy behavior (e.g., arr[100:200]). [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Added a `CScalar` alias for simpler creation of ComplexSIMD values with width 1, mirroring Mojo’s default Scalar. [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Added configurable printing support for NDArray and ComplexNDArray via a new internal `print_options: PrintOptions` field. [PR #264](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/264)
- Enabled runtime customization of array printing behavior (e.g., floating-point precision). [PR #264](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/264)
- Added slicing support via `__getitem__(slice: Slice) -> Self` for NDArrayShape, enabling more flexible shape manipulation in select use cases. [PR #263](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/263)
- Introduced a structured error handling framework for NuMojo centered around a unified base type, NumojoError. [PR #256](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/256)
- Added commonly used derived error types (e.g., IndexError, ShapeError, IOError) to provide clearer semantics and more consistent error handling across the codebase. [PR #256](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/256)
- Added an experimental native array serialization method, savenpy, enabling file output without relying on the NumPy backend. [PR #256](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/256)

### 🦋 Changed
- Enhanced ComplexSIMD arithmetic support with additional operator overloads.
- Replaced UnsafePointer usages with LegacyUnsafePointer to temporarily retain existing pointer semantics. Fixed and standardized import names related to UnsafePointer. We will slowly implement support for new UnsafePointer in all structs in upcoming PRs. [PR #285](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/285)
- Migrated all test files to the TestSuite.discover_tests pattern with explicit main() entry points. [PR #280](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/280), [PR #281](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/281), [PR #282](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/282), [PR #283](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/283), [PR #284](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/284)
- Matrix internals have been reworked. [PR #280](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/280), [PR #281](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/281), [PR #282](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/282), [PR #283](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/283), [PR #284](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/284)  
  - Expanded Matrix setter support by adding additional `__setitem__` and `set()` overloads.
  - Improved Matrix slicing and indexing behavior.
  - Refactored cumsum and cumprod implementations to better support both C- and F-contiguous matrices.
- Updated internal buffer representations in NDArrayShape, NDArrayStrides, and Item from `UnsafePointer[Int]` to `UnsafePointer[Scalar[DType.int]]`. [PR #275](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/275)
- Introduced temporary Int conversions to support stride and shape-related operations. [PR #275](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/275)
- Standardized type annotations by adding explicit type information across multiple code paths. [PR #274](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/274)
- Replaced legacy integer types (isize, intp) with int to align with current Mojo support. [PR #274](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/274)
- Updated NuMojo internals to be compatible with Mojo 0.25.6, which introduces a clearer distinction between implicit and explicit copies. [PR #270](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/270)
  - Reduced unnecessary data copies by favoring in-place operations where possible and improving handling of mutable versus immutable references.
  - Designated lightweight structs (e.g., Item, NDArrayShape, NDArrayStrides, _NDIter) as implicitly copyable to balance performance and usability.
- Functions parameterized by DType now work with real-valued inputs (`NDArray, Scalar`), while functions parameterized by ComplexDType work with complex-valued inputs (`ComplexNDArray, CScalar, ComplexSIMD`). There might be cases in future where this might not necessarily hold. [PR #269](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/269)
- Creation routines (e.g., arange) now return either `NDArray` or `ComplexNDArray` based on whether a real or complex dtype is specified.[PR #269](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/269)
- Reworked slicing internals to improve performance and reduce memory usage. [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Improved edge case handling to ensure consistent behavior across all slicing operations. [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Updated __getitem__ implementations to support the following overloads: [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
  ```mojo
  __getitem__(slice_list: List[Slice])
  __getitem__(*slices: Slice)
  __getitem__(*slices: Variant[Slice, Int])
  ```
  #### Supported slicing behavior
    - Forward slicing: arr[1:5], arr[:3], arr[2:] (with bounds clamping)
    - Reverse slicing: full negative stride support (e.g., arr[::-1], arr[5:1:-1])
    - Out-of-bounds slicing: automatically clamped to valid ranges
  #### Known limitation: mixed Int and Slice usage
  ```mojo
    import numojo as nm
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(nm.Shape(2, 3, 4))
    
    nm_slice1 = nm_arr[0, 0:3, 0:3]                 # ❌ compiler cannot resolve this
    nm_slice1 = nm_arr[0, Slice(0,3), Slice(0,4)]   # ✅ works (Ints + Slice need explicit Slice)
    nm_slice1 = nm_arr[0:1, 0:3, 0:4]               # ✅ works as expected
  ```
- Updated NDArray and ComplexNDArray printing logic to respect per-instance print_options. [PR #264](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/264)
- Printing configuration is currently stored as an internal field due to the lack of global variable support in Mojo; this will be migrated to a global configuration once supported. [PR #264](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/264)
  ```mojo
  # Example: Customizing print options
  var arr = nm.zeros[nm.f32](nm.Shape(3, 4))
  print(arr) # prints with default values
  arr.print_options.set_options(precision = 2)
  print(arr) # prints with precision 2 for floating values
  ```
- Improved __getitem__(idx: Int) -> Self and __setitem__(idx: Int, val: Self) implementations for both NDArray and ComplexNDArray. [PR #263](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/263)
- Added comprehensive edge-case validation to getter and setter methods, providing clearer and more consistent error reporting. [PR #263](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/263)
- Optimized C-contiguous and F-contiguous index calculation paths for improved performance. [PR #263](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/263)
- Errors now include structured diagnostic information via the Error struct: [PR #258](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/258)
  - category: Error classification (e.g., IndexError, ShapeError)
  - message: Description of the failure
  - location (optional): Source location where the error occurred
  - suggestion (optional): Guidance on how to resolve the issue

  #### Example: Index validation in NDArray.store
  ```mojo
  fn store[
      width: Int = 1
  ](mut self, *indices: Int, val: SIMD[dtype, width]) raises:
      if len(indices) != self.ndim:
          raise Error(
              IndexError(
                  message=String(
                      "Mismatch in number of indices: expected {} indices"
                      " (one per dimension) but received {}."
                  ).format(self.ndim, len(indices)),
                  suggestion=String(
                      "Provide exactly {} indices to correctly index into the"
                      " array."
                  ).format(self.ndim),
                  location=String(
                      "NDArray.store[width: Int](*indices: Int, val:"
                      " SIMD[dtype, width])"
                  ),
              )
          )
  
      for i in range(self.ndim):
          if (indices[i] < 0) or (indices[i] >= self.shape[i]):
              raise Error(
                  IndexError(
                      message=String(
                          "Invalid index at dimension {}: index {} is out of"
                          " bounds [0, {})."
                      ).format(i, indices[i], self.shape[i]),
                      suggestion=String(
                          "Ensure that index is within the valid range"
                          " [0, {})"
                      ).format(self.shape[i]),
                      location=String(
                          "NDArray.store[width: Int](*indices: Int, val:"
                          " SIMD[dtype, width])"
                      ),
                  )
              )
  ```
- Updated NDArray.store to perform explicit index validation and raise structured errors with detailed diagnostics, including category, message, location, and resolution suggestions. [PR #256](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/256)

### ❌ Removed
- Removed usage of deprecated `isize` and `intp` types. [PR #274](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/274)

### 🛠️ Fixed
- Restored compatibility with existing pointer-based code paths to allow interim releases before migrating to the new UnsafePointer model. [PR #285](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/285)
- Resolved multiple correctness issues related to slicing, indexing, and cumulative operations. [PR #280](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/280), [PR #281](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/281), [PR #282](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/282), [PR #283](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/283), [PR #284](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/284)
- Implemented correct and consistent error types for Item, Shape, and Strides. [PR #274](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/274)
  - Fixed memory-related issues in the above structs.
- Corrected outdated docstrings that referenced legacy code. [PR #274](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/274)
- Fixed incompatibilities with Mojo 0.25.6 related to copy semantics and reference handling. [PR #270](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/270)
- Improved slicing-related error messages for clarity and explicitness. [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Invalid indexing operations now produce actionable diagnostics rather than generic failures. [PR #258](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/258)
Example: Structured runtime error output
  ```console
  Unhandled exception caught during execution: NuMojo Error
          Category  : IndexError
          Message   : Invalid index at dimension 2: index 5 is out of bounds [0, 3).
          Location  : NDArray.store[width: Int](*indices: Int, val: SIMD[dtype, width])
          Suggestion: Ensure that index is within the valid range [0, 3)
  ```
- Improved error traceability and developer feedback for invalid indexing operations by replacing ad-hoc errors with the new structured error system. [PR #256](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/256)

### 📚 Documentatory and testing
- Significantly expanded and clarified documentation across the complex-number ecosystem, including ComplexDType, ComplexSIMD, ComplexNDArray, and ImaginaryUnit. Added practical usage examples to reduce ambiguity and improve discoverability.
- Improve docstrings of all internal methods of `MatrixBase`. [PR #287](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/287)
- Updated test infrastructure documentation to reflect the new TestSuite workflow. [PR #280](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/280), [PR #281](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/281), [PR #282](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/282), [PR #283](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/283), [PR #284](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/284)
- Update the github issue templates to streamline issue and PRs. [PR #277](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/277)
- Updated documentation to reflect the expanded ComplexNDArray API and new SIMD-based internal operations. [PR #275](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/275)
- Documented the new copy behavior and clarified which types require explicit .copy() calls versus those that are implicitly copyable. [PR #270](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/270)
- Documented the real vs. complex type distinction with concrete examples for scalars, SIMD values, and arrays. [PR #269](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/269)
  ```mojo
  # Example: Real vs. complex scalars
  import numojo as nm
  from numojo.prelude import *
  
  var scalar = Scalar[f32](1.0)
  print(scalar) # 1.0
  
  var complex_scalar = CScalar[cf32](1.0, 2.0)
  print(complex_scalar) # 1.0 + 2.0 j
  
  var complex_simd = ComplexSIMD[cf32](1.0, 2.0)
  var complex_simd_width_2 = ComplexSIMD[cf32, 2](
      SIMD[f32](1.0, 1.0),
      SIMD[f32](2.0, 2.0)
  )
  
  # Example: Array creation with real and complex dtypes
  import numojo as nm
  from numojo.prelude import *
  
  var array = nm.arange[f32](1.0, 10.0, 1.0)  # NDArray
  print(array)
  
  var complex_array = nm.arange[cf32](
      CScalar[cf32](1.0),
      CScalar[cf32](10.0),
      CScalar[cf32](1.0)
  )  # ComplexNDArray
  print(complex_array)
  ```
- Updated README files with the new Pixi-based installation workflow. [PR #268](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/268)
- Added a Korean version of the README. [PR #268](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/268)
- Updated the project roadmap. [PR #268](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/268)
- Expanded test coverage for slicing across edge cases, negative strides, and mixed slicing patterns. [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Documented current compiler limitations when mixing Int and Slice in __getitem__ overloads. [PR #266](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/266)
- Documented the new printing configuration workflow and its current limitations, including verbosity and lack of context manager support. [PR #264](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/264)
- Added test coverage for the updated getter and setter implementations to validate correctness across edge cases. [PR #263](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/263)
- Documented the new error conventions to support gradual migration of existing NuMojo errors to the unified system. [PR #258](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/258)
- Added illustrative examples demonstrating structured error output and usage patterns for the new error handling system. [PR #256](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/256)

## 01/06/2025 (v0.7.0)

### ⭐️ New

- Implement the `take_along_axis()` method. This method allows you to take elements from an array along a specified axis, using the indices provided in another array ([PR #226](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/226)).
- Add support for column-major memory layout for the `Matrix` type and for all matrix routines ([PR #232](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/232), [PR #233](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/233), [PR #234](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/234)).
- Add eigenvalue decomposition for symmetric matrices ([PR #238](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/238)).

### 🦋 Changed

- Update the syntax to accommodate to Mojo 25.3 ([PR #245](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/245)).
- Migrate Magic to Pixi ([PR #250](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/250)).
- Improve the getter methods for `ComplexNDArray` ([PR #229](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/229)).
- Re-write the `argmax()` and `argmin()` methods to return indices along given axis ([PR #230](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/230)).
- Replaced IO backend with NumPy ([PR #250](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/250)).

### ❌ Removed

- Remove the `numojo.CDType` (Complex Data Type) ([PR #231](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/231)). We will use the standard `CDType` from Mojo instead.
- Temporarily remove type coercion (`TypeCoercion`) until a better solution is available ([PR #242](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/242)). For now, please use implicit casting to convert arrays to targeted data types.
- Remove redundant `self: Self` ([PR #246](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/246)).

### 🛠️ Fixed

- Fixed broken links in zhs and zht readme files ([Issue #239](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/issues/239), [PR #240](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/240)).
- Fix error in division of an array and a scalar ([PR #244](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/244)).

### 📚 Documentatory and testing

## 08/03/2025 (v0.6.1)

### 🛠️ Fixed

Fix the bug that numojo crashes on "mojopkg" ([PR #227](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/227)).

## 28/02/2025 (v0.6)

### ⭐️ New

- Implement the `broadcast_to()` method for `NDArray`. This function broadcasts an array to any compatible shape ([PR #202](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/202)).
- Add the `apply_along_axis()` function that executes a function working on 1-d arrays on n-d arrays along the given axis ([PR #213](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/213), [PR #218](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/218)).
- Implement the `diagonal()` function and the `NDArray.diagonal()` method ([PR #217](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/217)).
- Implement the `compress()` function and the `NDArray.compress()` method ([PR #219](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/219)).
- Implement the `clip()` function and the `NDArray.clip()` method  ([PR #220](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/220)).
- Add the `_NDAxisIter` type as a iterator that returns, in each iteration, a 1-d array along that axis. The iterator traverse the array either by C-order or F-order ([PR #212](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/212), [PR #214](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/214)).
- Add the `ith()` method to the `_NDArrayIter` type and to the `_NDIter` type to get the i-th item ([PR #219](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/219), [PR #221](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/221)).
- Add the `Flags` type for storing information on memory layout of arrays. The `Flags` type replaces the current `Dict[String, Bool]` type ([PR #210](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/210), [PR #214](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/214)).
- Add the `swapaxes()` methods for the `NDArrayShape` type and the `NDArrayStrides` type ([PR #221](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/221)).
- Add the `offset()` methods for the `Item` type to get the offset of an index in the underlying buffer. Allow the `Item` object to be constructed from index and shape ([PR #221](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/221)).

### 🦋 Changed

- Update the syntax to accommodate to Mojo 25.1 ([PR #211](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/211)).
  - Change constructors, e.g., `str()` to `String()`.
  - Change `index()` function to `Int()`.
  - Change the function `isdigit()` to method.
  - Stop using `NDArray.__init__()` to construct arrays but `NDArray()`.
- Update functions in the `random` module:
  - Add `randint()`, and accept `Shape` as the first argument ([PR #199](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/199)).
- Update functions in the `statistics` module:
  - Add the parameter `returned_dtype` to functions which defaults to `f64` ([PR #200](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/200)).
  - Add `variance()` and `std()` ([PR #200](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/200)). Allow calculating variance and std of an array by axis ([PR #207](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/207)).
  - Allow `median()` and `mode()` functions to work on any axis.
- Update functions in the `sorting` module:
  - Considerably improve the performance of `sort()` function ([PR #213](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/213), [PR #214](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/214)).
  - Allow `argsort` by any axis for both C-order and F-order arrays ([PR #214](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/214)).
- Update function in the `math.extrema` module ([PR #216](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/216)):
  - Allow the `max()` and `min()` functions to work on any axis.
  - Update the `max()` and `min()` methods for the `NDArray` type.
- Update the behaviors of 0-d array (numojo scalar). Although the syntax `a.item(0)` or `a[Item(0)]` is always preferred, we also allow some basic operations on 0-d array. 0-d array can now be unpacked to get the corresponding mojo scalar either by `[]` or by `item()` ([PR #209](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/209)).
- Add boundary checks for `NDArrayShape` and `NDArrayStrides` to ensure safe use. Improve the docstring ([PR #205](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/205), [PR #206](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/206)).
- Significantly increase the speed of printing large arrays ([PR #215](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/215)).
- Replace the `NDArray.num_elements()` method by the `NDArray.size` attribute for all modules ([PR #216](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/216)).

### ❌ Removed

- Remove the `cumvariance`, `cumstd`, `cumpvariance`, `cumpstd` functions ([PR #200](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/200)).
- Remove the `maxT()` and `minT()` functions ([PR #216](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/216)).

### 🛠️ Fixed

- Re-write the `ravel()` function so that it will not break for F-order arrays ([PR #214](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/214)).
- Fix the `NDArray.sort()` method (in-place sort). The default axis is changed to `-1` ([PR #217](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/217)).
- Fix the `NDArray.__bool__()` method which may returns incorrect results ([PR #219](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/219)).

### 📚 Documentatory and testing

- Update the docstring of all methods belonging to the `NDArray` type, following the Mojo Docstring Style Guide. Provide more detailed error messages in the internal functions of `NDArray` to enhance clarity and traceability of errors ([PR #222](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/222)).
- Updates the roadmap document according to our current progress ([PR #208](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/208)).

## 26/01/2025 (v0.5)

### ⭐️ New

- Add support for complex arrays `ComplexNDArray`, `ComplexSIMD`, `CDType` ([PR #165](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/165)). Add creation routines for complex arrays ([PR #195](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/pull/195)).
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
