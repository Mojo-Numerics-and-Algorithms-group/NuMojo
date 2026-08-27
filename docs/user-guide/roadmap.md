# Roadmap

Our primary objective is to develop a robust and comprehensive library for numerical and scientific computing. To facilitate a smooth transition from Python, we will primarily adhere to the conventions of `numpy` and `scipy`, while allowing deviations when they:

- Align with the design philosophy of the Mojo programming language.
- Significantly enhance performance, such as through the use of generics.
- Improve consistency in conventions, such as naming parameters and arguments.

NuMojo is currently in its early development stages. At this point, our focus is to ***stabilize the API*** and ***implement the majority of functionalities***. If you notice any missing features, please consider developing them and submitting a pull request or creating a feature request.

## Core Tasks

- ✅ Implement the n-dimensional array type and support SIMD-compatible standard library math functions[^stdlib].
- 🔄 Develop `numpy`-like functions for mathematical, statistical, linear algebra, searching, sorting, etc.

### N-dimensional Arrays

✅ **Completed:**
- Basic `NDArray` and `ComplexNDArray` types with comprehensive arithmetic operations
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
- **Matrix Operations:** `matmul` (`@` operator), `inv`
- **Decompositions:** `lu_decomposition`
- **Solving:** `solve`
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
- More comprehensive I/O support

## Architecture

For a concise overview of the directory layout and how core, routines, and operations fit together, see [architecture.md](../developer-guide/architecture.md).

## Next Steps and Future Development

### Immediate Priorities (v0.8+)
1. **Enhanced I/O Support:** Better file format support (CSV, HDF5, JSON)
2. **Performance Optimization:** Further SIMD optimization and memory efficiency improvements
3. **Testing Coverage:** Comprehensive test suite expansion for all implemented functions

### Medium-term Goals (v1.0)
1. **GPU Support:** Implement GPU acceleration when Mojo language support becomes available
2. **Advanced Linear Algebra:** Singular value decomposition (SVD), Cholesky decomposition

### Long-term Vision (v1.5+)
1. **Machine Learning Foundation:** While avoiding ML algorithms in core, provide efficient primitives
2. **Sparse Arrays:** Support for sparse matrix operations
3. **Distributed Computing:** Multi-node array operations

### Language Feature Dependencies
- **Parameterized Traits:** Required for view-based operations and zero-copy slicing
- **GPU Support:** Required for GPU acceleration features
- **Advanced Memory Management:** For more sophisticated memory optimization

This roadmap is a living document and will evolve as the library and Mojo language capabilities mature.

[^stdlib]: Standard library functions that are SIMD-compatible.
[^numpy]: The structure is inspired by the organization of functions in NumPy.
[^single]: If a topic has only a few functions, they can be grouped into a single Mojo file instead of creating a separate folder.
