# Roadmap

Our primary objective is to develop a robust and comprehensive library for numerical and scientific computing. To facilitate a smooth transition from Python, we will primarily adhere to the conventions of `numpy` and `scipy`, while allowing deviations when they:

- Align with the design philosophy of the Mojo programming language.
- Significantly enhance performance, such as through the use of generics.
- Improve consistency in conventions, such as naming parameters and arguments.

NuMojo is currently in its early development stages. At this point, our focus is to ***stabilize the API*** and ***implement the majority of functionalities***. If you notice any missing features, please consider developing them and submitting a pull request or creating a feature request.

## Core Tasks

- ✅ Implement the n-dimensional array type and support SIMD-compatible standard library math functions[^stdlib].
- 🔄 Develop `numpy`-like functions for mathematical, statistical, linear algebra, searching, sorting, etc.
- 🔄 Create `scipy`-like functions for scientific purposes, such as optimizers, function approximators, and FFT.

### N-dimensional Arrays

✅ **Completed:**
- Basic `NDArray`, `ComplexNDArray`, and `Matrix` types with comprehensive arithmetic operations
- Full indexing and slicing support including negative indices
- Broadcasting support for operations between arrays and scalars
- Memory-efficient operations with contiguous and strided array support
- Printing and formatting system with configurable options
- Complex number operations with full arithmetic support

🔄 **In Progress:**
- View-based operations (awaiting Mojo language support for parameterized traits)
- GPU acceleration (awaiting Mojo language GPU support)

🔄 **Planned:**
- Fixed-dimension arrays (awaiting trait parameterization)
- More advanced indexing features (boolean masking, fancy indexing)

### Implement Basic Numeric Functions

We are currently working on implementing basic numeric functions into NuMojo. The scope is similar to `numpy`. Functions on [this page](https://numpy.org/doc/stable/reference/routines.html) will be considered for gradual implementation in NuMojo.

✅ **Implemented Modules:**

**Array Creation:**
- `arange`, `linspace`, `logspace` (with complex variants)
- `zeros`, `ones`, `full`, `empty`, `eye`, `identity` (with complex variants) 
- `*_like` functions for creating arrays with same shape as existing arrays

**Mathematical Functions:**
- **Trigonometric:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `hypot`
- **Hyperbolic:** Full suite of hyperbolic functions
- **Exponential/Logarithmic:** `exp`, `log`, `log10`, `log2`, power functions
- **Arithmetic:** `add`, `subtract`, `multiply`, `divide`, `fma` with broadcasting
- **Extrema:** `min`, `max`, `argmin`, `argmax`
- **Rounding:** `round`, `floor`, `ceil`, `trunc`
- **Floating Point:** `isnan`, `isinf`, `isfinite`
- **Products/Sums:** Element-wise and axis-based operations

**Linear Algebra:**
- **Matrix Operations:** `matmul` (`@` operator), `inv`, `transpose`
- **Decompositions:** `lu_decomposition`, `qr`, `eig` (eigenvalues)
- **Solving:** `solve`, `lstsq` (least squares)
- **Norms:** `det` (determinant), `trace`

**Logic Functions:**
- **Comparison:** Element-wise comparisons (`equal`, `not_equal`, `less`, etc.)
- **Array Contents:** `all`, `any`, content checking functions
- **Truth Testing:** Boolean array operations

**Array Manipulation:**
- **Reshaping:** `reshape`, `transpose`, `squeeze`
- **Joining/Splitting:** `concatenate`, `stack`, `split`
- **Indexing:** Advanced slicing and indexing routines

**Statistics:**
- **Averages:** `mean`, `median`, variance calculations
- Basic statistical functions

**Input/Output:**
- **File Operations:** Text file reading/writing
- **Formatting:** Array display and string conversion

**Sorting/Searching:**
- `sort`, `argsort` with axis support
- Search functions for finding elements

**Random Sampling:**
- Random number generation for arrays
- Various probability distributions

🔄 **In Progress:**
- More statistical functions (standard deviation, correlation, etc.)
- Advanced signal processing functions
- More comprehensive I/O support

### Implement Advanced Functions

We also aim to implement advanced functions into NuMojo. The scope is similar to `scipy`.

✅ **Implemented Science Modules:**
- **Interpolation:** Basic interpolation functions
- **Signal Processing:** Signal processing utilities

🔄 **Planned Science Features:**
- FFT (Fast Fourier Transform)
- Optimization algorithms
- ODE (Ordinary Differential Equation) solvers
- Numerical integration
- Special functions
- Sparse matrix support

## Internal Organization of Objects and Functions

✅ **Current Implementation Status:**

NuMojo has successfully implemented the planned organizational structure with the following hierarchy:

### Core Infrastructure (`/numojo/core/`)
- **Data Types:** `NDArray`, `ComplexNDArray`, `Matrix` with full operator support
- **Shape/Strides:** Efficient memory layout handling (`ndshape.mojo`, `ndstrides.mojo`)
- **Memory Management:** `own_data.mojo`, `ref_data.mojo` for flexible memory handling
- **Complex Numbers:** Dedicated complex array support with full arithmetic
- **Traits:** Array-like interfaces and backend abstractions
- **Utilities:** Helper functions for array operations

### Routines (`/numojo/routines/`)
Functions are organized by topic following NumPy's structure:

1. **Array Creation** (`creation.mojo`): `arange`, `linspace`, `zeros`, `ones`, `full`, `eye`, etc.
2. **Mathematical Functions** (`/math/`):
   - `arithmetic.mojo`: Basic arithmetic operations
   - `trig.mojo`: Trigonometric functions (`sin`, `cos`, `tan`, etc.)
   - `hyper.mojo`: Hyperbolic functions
   - `exponents.mojo`: Exponential and logarithmic functions
   - `extrema.mojo`: Min/max and related functions
   - `rounding.mojo`: Rounding operations
   - `floating.mojo`: Floating-point utilities
   - `misc.mojo`: Miscellaneous mathematical functions
   - `products.mojo`, `sums.mojo`, `differences.mojo`: Aggregate operations
3. **Linear Algebra** (`/linalg/`):
   - `products.mojo`: Matrix multiplication and related operations
   - `decompositions.mojo`: LU, QR, eigenvalue decompositions
   - `solving.mojo`: Linear system solving
   - `norms.mojo`: Matrix norms, determinant, trace
4. **Logic Functions** (`/logic/`):
   - `comparison.mojo`: Element-wise comparisons
   - `contents.mojo`: Array content checking
   - `truth.mojo`: Boolean operations
5. **Input/Output** (`/io/`):
   - `files.mojo`: File reading/writing
   - `formatting.mojo`: Array display formatting
6. **Statistics** (`/statistics/`):
   - `averages.mojo`: Mean, median, variance calculations
7. **Array Manipulation** (`manipulation.mojo`): Reshape, transpose, concatenate
8. **Indexing** (`indexing.mojo`): Advanced indexing operations
9. **Sorting/Searching** (`sorting.mojo`, `searching.mojo`): Sort and search functions
10. **Random Sampling** (`random.mojo`): Random number generation
11. **Bitwise Operations** (`bitwise.mojo`): Bit manipulation functions
12. **Constants** (`constants.mojo`): Mathematical constants

### Scientific Computing (`/numojo/science/`)
Advanced functions similar to SciPy:
- `interpolate.mojo`: Interpolation functions
- `signal.mojo`: Signal processing utilities

### Access Patterns
The implementation supports both access patterns as planned:

1. **Top-level access:** `numojo.sort()`, `numojo.sin()`, etc.
2. **Namespace access:** `numojo.linalg.solve()`, `numojo.random.randn()`, etc.

### Code Organization Principles
✅ **Successfully Implemented:**
- Functions within each file are organized logically and alphabetically where appropriate
- `__init__.mojo` files properly expose functions without namespace pollution
- Clear separation between core data structures and computational routines
- Consistent API design across all modules
- Comprehensive documentation and examples

The current implementation has achieved the organizational goals set in the original roadmap, providing a clean, scalable structure that mirrors NumPy/SciPy conventions while leveraging Mojo's performance capabilities.

## Next Steps and Future Development

### Immediate Priorities (v0.8+)
1. **Complete Statistics Module:** Expand beyond averages to include standard deviation, correlation, percentiles
2. **Enhanced I/O Support:** Better file format support (CSV, HDF5, JSON)
3. **Performance Optimization:** Further SIMD optimization and memory efficiency improvements
4. **Testing Coverage:** Comprehensive test suite expansion for all implemented functions

### Medium-term Goals (v1.0)
1. **GPU Support:** Implement GPU acceleration when Mojo language support becomes available
2. **Advanced Linear Algebra:** Singular value decomposition (SVD), Cholesky decomposition
3. **Signal Processing:** FFT implementation and advanced signal processing functions
4. **Optimization:** Implement scipy.optimize equivalent functions

### Long-term Vision (v1.5+)
1. **Machine Learning Foundation:** While avoiding ML algorithms in core, provide efficient primitives
2. **Sparse Arrays:** Support for sparse matrix operations
3. **Distributed Computing:** Multi-node array operations
4. **Advanced Scientific Computing:** ODE solvers, numerical integration, special functions

### Language Feature Dependencies
- **Parameterized Traits:** Required for view-based operations and zero-copy slicing
- **GPU Support:** Required for GPU acceleration features
- **Advanced Memory Management:** For more sophisticated memory optimization

The roadmap reflects NuMojo's current mature state with a solid foundation of core functionality and a clear path toward becoming a comprehensive scientific computing platform for Mojo.

[^stdlib]: Standard library functions that are SIMD-compatible.
[^numpy]: The structure is inspired by the organization of functions in NumPy.
[^single]: If a topic has only a few functions, they can be grouped into a single Mojo file instead of creating a separate folder.
[^science]: This folder can be further integrated into the `routines` folder if necessary.
