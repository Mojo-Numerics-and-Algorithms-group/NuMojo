# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------===#
# SECTIONS OF THE FILE:

# `ComplexNDArray` type
# 1. Life cycle methods.
# 2. Indexing and slicing (get and set dunders and relevant methods).
# 3. Operator dunders.
# 4. IO, trait, and iterator dunders.
# 5. Other methods (Sorted alphabetically).
# ===----------------------------------------------------------------------===#

# ===----------------------------------------------------------------------===#
# FORMAT FOR DOCSTRING (See "Mojo docstring style guide" for more information)
# 1. Description *
# 2. Parameters *
# 3. Args *
# 4. Constraints *
# 4) Returns *
# 5) Raises *
# 6) SEE ALSO
# 7) NOTES
# 8) REFERENCES
# 9) Examples *
# (Items marked with * are flavored in "Mojo docstring style guide")
# ===----------------------------------------------------------------------===#

# ===----------------------------------------------------------------------===#
# === Stdlib ===
# ===----------------------------------------------------------------------===#
from algorithm import parallelize, vectorize
import builtin.bool as builtin_bool
import builtin.math as builtin_math
from builtin.type_aliases import Origin
from collections.optional import Optional
from math import log10, sqrt
from memory import memset_zero, memcpy
from memory import LegacyUnsafePointer
from python import PythonObject
from sys import simd_width_of
from utils import Variant

# ===----------------------------------------------------------------------===#
# === numojo core ===
# ===----------------------------------------------------------------------===#
from numojo.core.complex.complex_dtype import _concise_dtype_str
from numojo.core.flags import Flags
from numojo.core.item import Item
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides
from numojo.core.complex.complex_simd import ComplexSIMD, ComplexScalar, CScalar
from numojo.core.complex.complex_dtype import ComplexDType
from numojo.core.data_container import DataContainer
from numojo.core.utility import (
    _get_offset,
    _transfer_offset,
    _traverse_iterative,
    _traverse_iterative_setter,
    to_numpy,
    bool_to_numeric,
)
from numojo.core.error import (
    IndexError,
    ShapeError,
    BroadcastError,
    MemoryError,
    ValueError,
    ArithmeticError,
)

# ===----------------------------------------------------------------------===#
# === numojo routines (creation / io / logic) ===
# ===----------------------------------------------------------------------===#
import numojo.routines.creation as creation
from numojo.routines.io.formatting import (
    format_value,
    PrintOptions,
)
import numojo.routines.logic.comparison as comparison

# ===----------------------------------------------------------------------===#
# === numojo routines (math / bitwise / searching) ===
# ===----------------------------------------------------------------------===#
import numojo.routines.bitwise as bitwise
import numojo.routines.math._array_funcs as _af
from numojo.routines.math._math_funcs import Vectorized
import numojo.routines.math.arithmetic as arithmetic
import numojo.routines.math.rounding as rounding
import numojo.routines.math.trig as trig
import numojo.routines.math.exponents as exponents
import numojo.routines.math.misc as misc
import numojo.routines.searching as searching


# ===----------------------------------------------------------------------=== #
# Implements N-Dimensional Complex Array
# ===----------------------------------------------------------------------=== #
struct ComplexNDArray[cdtype: ComplexDType = ComplexDType.float64](
    Copyable,
    FloatableRaising,
    IntableRaising,
    Movable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """
    N-dimensional Complex array.

    ComplexNDArray represents an N-dimensional array whose elements are complex numbers, supporting efficient storage, indexing, and mathematical operations. Each element consists of a real and imaginary part, stored in separate buffers.

    Parameters:
        cdtype: The complex data type of the array elements (default: ComplexDType.float64).

    Attributes:
        - _re: NDArray[Self.dtype]
            Buffer for real parts.
        - _im: NDArray[Self.dtype]
            Buffer for imaginary parts.
        - ndim: Int
            Number of dimensions.
        - shape: NDArrayShape
            Shape of the array.
        - size: Int
            Total number of elements.
        - strides: NDArrayStrides
            Stride information for each dimension.
        - flags: Flags
            Memory layout information.
        - print_options: PrintOptions
            Formatting options for display.

    Notes:
        - The array is uniquely defined by its data buffers, shape, strides, and element datatype.
        - Supports both row-major (C) and column-major (F) memory order.
        - Provides rich indexing, slicing, and broadcasting semantics.
        - ComplexNDArray should be created using factory functions in `nomojo.routines.creation` module for convenience.
    """

    # --- Aliases ---
    alias dtype: DType = cdtype._dtype
    """corresponding real data type"""

    # --- FIELDS ---
    var _re: NDArray[Self.dtype]
    """Buffer for real parts."""
    var _im: NDArray[Self.dtype]
    """Buffer for imaginary parts."""

    # TODO: add methods to for users to access the following properties directly from _re, _im and remove them from here.
    var ndim: Int
    """Number of Dimensions."""
    var shape: NDArrayShape
    """Size and shape of ComplexNDArray."""
    var size: Int
    """Size of ComplexNDArray."""
    var strides: NDArrayStrides
    """Contains offset, strides."""
    var flags: Flags
    "Information about the memory layout of the array."
    var print_options: PrintOptions
    """Per-instance print options (formerly global)."""

    # --- Life cycle methods ---
    @always_inline("nodebug")
    fn __init__(
        out self, var re: NDArray[Self.dtype], var im: NDArray[Self.dtype]
    ) raises:
        """
        Initialize a ComplexNDArray with given real and imaginary parts.

        Args:
            re: Real part of the complex array.
            im: Imaginary part of the complex array.
        """
        if re.shape != im.shape:
            raise Error(
                ShapeError(
                    message=String(
                        "Real and imaginary array parts must have identical"
                        " shapes; got re={} vs im={}."
                    ).format(re.shape, im.shape),
                    suggestion=String(
                        "Ensure both NDArray arguments are created with the"
                        " same shape before constructing ComplexNDArray."
                    ),
                    location=String("ComplexNDArray.__init__(re, im)"),
                )
            )
        self._re = re^
        self._im = im^
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=80, formatted_width=6
        )

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: NDArrayShape,
        order: String = "C",
    ) raises:
        """
        Initialize a ComplexNDArray with given shape. The memory is not filled with values.

        Args:
            shape: Variadic shape.
            order: Memory order C or F.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = nm.ComplexNDArray[cf32](Shape(2,3,4))
            ```

        Notes:
            This constructor should not be used by users directly. Use factory functions in `numojo.routines.creation` module instead.
        """
        self._re = NDArray[Self.dtype](shape, order)
        self._im = NDArray[Self.dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: List[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initialize a ComplexNDArray with given shape (list of integers).

        Args:
            shape: List of shape.
            order: Memory order C or F.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = nm.ComplexNDArray[cf32](List[Int](2,3,4))
            ```

        Notes:
            This constructor should not be used by users directly. Use factory functions in `numojo.routines.creation` module instead.
        """
        self._re = NDArray[Self.dtype](shape, order)
        self._im = NDArray[Self.dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: VariadicList[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initialize a ComplexNDArray with given shape (variadic list of integers).

        Args:
            shape: Variadic List of shape.
            order: Memory order C or F.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = nm.ComplexNDArray[cf32](VariadicList(2,3,4))
            ```

        Notes:
            This constructor should not be used by users directly. Use factory functions in `numojo.routines.creation` module instead.
        """
        self._re = NDArray[Self.dtype](shape, order)
        self._im = NDArray[Self.dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    fn __init__(
        out self,
        shape: List[Int],
        offset: Int,
        strides: List[Int],
    ) raises:
        """
        Initialize a ComplexNDArray with a specific shape, offset, and strides.

        Args:
            shape: List of integers specifying the shape of the array.
            offset: Integer offset into the underlying buffer.
            strides: List of integers specifying the stride for each dimension.

        Example:
            ```mojo
            from numojo.prelude import *
            var shape = List[Int](2, 3)
            var offset = 0
            var strides = List[Int](3, 1)
            var arr = ComplexNDArray[cf32](shape, offset, strides)
            ```

        Notes:
            - This constructor is intended for advanced use cases requiring precise control over memory layout.
            - The resulting array is uninitialized and should be filled before use.
            - Both real and imaginary buffers are created with the same shape, offset, and strides.
        """
        self._re = NDArray[Self.dtype](shape, offset, strides)
        self._im = NDArray[Self.dtype](shape, offset, strides)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    fn __init__(
        out self,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        ndim: Int,
        size: Int,
        flags: Flags,
    ):
        """
        Initialize a ComplexNDArray with explicit shape, strides, number of dimensions, size, and flags. This constructor creates an uninitialized ComplexNDArray with the provided properties. No compatibility checks are performed between shape, strides, ndim, size, or flags. This allows construction of arrays with arbitrary metadata, including 0-D arrays (scalars).

        Args:
            shape: Shape of the array.
            strides: Strides for each dimension.
            ndim: Number of dimensions.
            size: Total number of elements.
            flags: Memory layout flags.

        Notes:
            - This constructor is intended for advanced or internal use cases requiring manual control.
            - The resulting array is uninitialized; values must be set before use.
            - No validation is performed on the consistency of the provided arguments.
        """

        self.shape = shape
        self.strides = strides
        self.ndim = ndim
        self.size = size
        self.flags = flags
        self._re = NDArray[Self.dtype](shape, strides, ndim, size, flags)
        self._im = NDArray[Self.dtype](shape, strides, ndim, size, flags)
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    fn __init__(
        out self,
        shape: NDArrayShape,
        ref buffer_re: LegacyUnsafePointer[Scalar[Self.dtype]],
        ref buffer_im: LegacyUnsafePointer[Scalar[Self.dtype]],
        offset: Int,
        strides: NDArrayStrides,
    ) raises:
        """
        Initialize a ComplexNDArray view with explicit shape, raw buffers, offset, and strides.

        This constructor creates a view over existing memory buffers for the real and imaginary parts,
        using the provided shape, offset, and stride information. It is intended for advanced or internal
        use cases where direct control over memory layout is required.

        ***Unsafe!*** This function is unsafe and should only be used internally. The caller is responsible
        for ensuring that the buffers are valid and that the shape, offset, and strides are consistent.

        Args:
            shape: NDArrayShape specifying the dimensions of the array.
            buffer_re: Unsafe pointer to the buffer containing the real part data.
            buffer_im: Unsafe pointer to the buffer containing the imaginary part data.
            offset: Integer offset into the buffers.
            strides: NDArrayStrides specifying the stride for each dimension.

        Notes:
            - No validation is performed on the buffers or metadata.
            - The resulting ComplexNDArray shares memory with the provided buffers.
            - Incorrect usage may lead to undefined behavior.
        """
        self._re = NDArray(shape, buffer_re, offset, strides)
        self._im = NDArray(shape, buffer_im, offset, strides)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Copy other into self.
        """
        self._re = other._re.copy()
        self._im = other._im.copy()
        self.ndim = other.ndim
        self.shape = other.shape
        self.size = other.size
        self.strides = other.strides
        self.flags = other.flags
        self.print_options = other.print_options

    @always_inline("nodebug")
    fn __moveinit__(out self, deinit existing: Self):
        """
        Move other into self.
        """
        self._re = existing._re^
        self._im = existing._im^
        self.ndim = existing.ndim
        self.shape = existing.shape
        self.size = existing.size
        self.strides = existing.strides
        self.flags = existing.flags
        self.print_options = existing.print_options

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing
    # Getter dunders and other getter methods
    # FIXME: currently most of the getitem and setitem methods don't match exactly between NDArray and ComplexNDArray in it's implementation, docstring, argument mutability etc. Fix this.

    # 1. Basic Indexing Operations
    # fn _getitem(self, *indices: Int) -> ComplexSIMD[cdtype]                         # Direct unsafe getter
    # fn _getitem(self, indices: List[Int]) -> ComplexSIMD[cdtype]                         # Direct unsafe getter
    # fn __getitem__(self) raises -> ComplexSIMD[cdtype]                             # Get 0d array value
    # fn __getitem__(self, index: Item) raises -> ComplexSIMD[cdtype]                # Get by coordinate list
    #
    # 2. Single Index Slicing
    # fn __getitem__(self, idx: Int) raises -> Self                             # Get by single index
    #
    # 3. Multi-dimensional Slicing
    # fn __getitem__(self, *slices: Slice) raises -> Self                       # Get by variable slices
    # fn __getitem__(self, slice_list: List[Slice]) raises -> Self              # Get by list of slices
    # fn __getitem__(self, *slices: Variant[Slice, Int]) raises -> Self         # Get by mix of slices/ints
    #
    # 4. Advanced Indexing
    # fn __getitem__(self, indices: NDArray[DType.int]) raises -> Self        # Get by index array
    # fn __getitem__(self, indices: List[Int]) raises -> Self                   # Get by list of indices
    # fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self            # Get by boolean mask
    # fn __getitem__(self, mask: List[Bool]) raises -> Self                     # Get by boolean list
    #
    # 5. Low-level Access
    # fn item(self, var index: Int) raises -> ComplexSIMD[Self.dtype]                   # Get item by linear index
    # fn item(self, *index: Int) raises -> ComplexSIMD[Self.dtype]                        # Get item by coordinates
    # fn load(self, var index: Int) raises -> ComplexSIMD[Self.dtype]                   # Load with bounds check
    # fn load[width: Int](self, index: Int) raises -> ComplexSIMD[Self.dtype, width]        # Load SIMD value
    # fn load[width: Int](self, *indices: Int) raises -> ComplexSIMD[Self.dtype, width]     # Load SIMD at coordinates
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn normalize(self, idx: Int, dim: Int) -> Int:
        """
        Normalize a potentially negative index to its positive equivalent
        within the bounds of the given dimension.

        Args:
            idx: The index to normalize. Can be negative to indicate indexing
                 from the end (e.g., -1 refers to the last element).
            dim: The size of the dimension to normalize against.

        Returns:
            The normalized index as a non-negative integer.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix[f32](shape=(3, 4))
            var norm_idx = mat.normalize(-1, mat.shape[0])  # Normalize -1 to 2
            ```
        """
        var idx_norm = idx
        if idx_norm < 0:
            idx_norm = dim + idx_norm
        return idx_norm

    fn _getitem(self, *indices: Int) -> ComplexSIMD[cdtype]:
        """
        Get item at indices and bypass all boundary checks.
        ***UNSAFE!*** No boundary checks made, for internal use only.

        Args:
            indices: Indices to get the value.

        Returns:
            The element of the array at the indices.

        Examples:
            ```mojo
            import numojo as nm
            var A = nm.ones[nm.cf32](nm.Shape(2,3,4))
            print(A._getitem(1,2,3))
            ```

        Notes:
            This function is unsafe and should be used only on internal use.
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * Int(self.strides._buf[i])
        return ComplexSIMD[cdtype](
            re=self._re._buf.ptr[index_of_buffer],
            im=self._im._buf.ptr[index_of_buffer],
        )

    fn _getitem(self, indices: List[Int]) -> ComplexScalar[cdtype]:
        """
        Get item at indices and bypass all boundary checks.
        ***UNSAFE!*** No boundary checks made, for internal use only.

        Args:
            indices: Indices to get the value.

        Returns:
            The element of the array at the indices.

        Examples:
            ```mojo
            import numojo as nm
            var A = nm.ones[nm.cf32](numojo.Shape(2,3,4))
            print(A._getitem(List[Int](1,2,3)))
            ```

        Notes:
            This function is unsafe and should be used only on internal use.
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * Int(self.strides._buf[i])
        return ComplexSIMD[cdtype](
            re=self._re._buf.ptr[index_of_buffer],
            im=self._im._buf.ptr[index_of_buffer],
        )

    fn __getitem__(self) raises -> ComplexSIMD[cdtype, 1]:
        """
        Gets the value of the 0-D Complex array.

        Returns:
            The value of the 0-D Complex array.

        Raises:
            Error: If the array is not 0-d.

        Examples:

        ```mojo
        import numojo as nm
        var a = nm.arange[nm.cf32](3)[0]
        print(a[]) # gets values of the 0-D complex array.
        ```.
        """
        if self.ndim != 0:
            raise Error(
                IndexError(
                    message=String(
                        "Cannot read a scalar value from a non-0D"
                        " ComplexNDArray without indices."
                    ),
                    suggestion=String(
                        "Use `A[]` only for 0D arrays (scalars). For higher"
                        " dimensions supply indices, e.g. `A[i,j]`."
                    ),
                    location=String("ComplexNDArray.__getitem__()"),
                )
            )
        return ComplexSIMD[cdtype](
            re=self._re._buf.ptr[],
            im=self._im._buf.ptr[],
        )

    fn __getitem__(self, index: Item) raises -> ComplexSIMD[cdtype, 1]:
        """
        Get the value at the index list.

        Args:
            index: Index list.

        Returns:
            The value at the index list.

        Raises:
            Error: If the length of `index` does not match the number of dimensions.
            Error: If any of the index elements exceeds the size of the dimension of the array.

        Examples:

        ```console
        >>>import numojo as nm
        >>>var A = nm.full[nm.f32](nm.Shape(2, 5), ComplexSIMD[nm.f32](1.0, 1.0))
        >>>print(A[Item(1, 2)]) # gets values of the element at (1, 2).
        ```.
        """
        if index.__len__() != self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Expected {} indices (ndim) but received {}."
                    ).format(self.ndim, index.__len__()),
                    suggestion=String(
                        "Provide one index per dimension for shape {}."
                    ).format(self.shape),
                    location=String("ComplexNDArray.__getitem__(index: Item)"),
                )
            )

        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                raise Error(
                    IndexError(
                        message=String(
                            "Index {} out of range for dimension {} (size {})."
                        ).format(index[i], i, self.shape[i]),
                        suggestion=String(
                            "Valid indices for this dimension are in [0, {})."
                        ).format(self.shape[i]),
                        location=String(
                            "ComplexNDArray.__getitem__(index: Item)"
                        ),
                    )
                )

        var idx: Int = _get_offset(index, self.strides)
        return ComplexSIMD[cdtype](
            re=self._re._buf.ptr.load[width=1](idx),
            im=self._im._buf.ptr.load[width=1](idx),
        )

    fn __getitem__(self, idx: Int) raises -> Self:
        """Single-axis integer slice (first dimension).
        Returns a slice of the complex array taken at axis 0 position `idx`.
        Dimensionality is reduced by exactly one; a 1-D source produces a
        0-D ComplexNDArray (scalar wrapper). Negative indices are supported
        and normalized. The result preserves the source memory order (C/F).

        Args:
            idx: Integer index along the first (axis 0) dimension. Supports
                negative indices in [-shape[0], shape[0]).

        Returns:
            ComplexNDArray with shape `self.shape[1:]` when `self.ndim > 1`,
            otherwise a 0-D ComplexNDArray scalar wrapper.

        Raises:
            IndexError: If the array is 0-D.
            IndexError: If `idx` (after normalization) is out of bounds.

        Notes:
            Performance fast path: For C-contiguous arrays the slice for both
            real and imaginary parts is copied with single `memcpy` calls.
            For F-contiguous or arbitrary stride layouts, a generic
            stride-based copier is used for both components. (Future: return
            a non-owning view).

        Example:
            ```mojo
            import numojo as nm
            from numojo.prelude import *
            var a = nm.arange[cf32](CScalar[cf32](0), CScalar[cf32](12), CScalar[cf32](1)).reshape(Shape(3, 4))
            print(a.shape)        # (3,4)
            print(a[1].shape)     # (4,)  -- 1-D slice
            print(a[-1].shape)    # (4,)  -- negative index
            var b = nm.arange[cf32](CScalar[cf32](6)).reshape(nm.Shape(6))
            print(b[2])           # 0-D array (scalar wrapper)
            ```
        """
        if self.ndim == 0:
            raise Error(
                IndexError(
                    message=String(
                        "Cannot slice a 0D ComplexNDArray (scalar)."
                    ),
                    suggestion=String(
                        "Use `A[]` or `A.item(0)` to read its value."
                    ),
                    location=String("ComplexNDArray.__getitem__(idx: Int)"),
                )
            )

        var norm = idx
        if norm < 0:
            norm += self.shape[0]
        if (norm < 0) or (norm >= self.shape[0]):
            raise Error(
                IndexError(
                    message=String(
                        "Index {} out of bounds for axis 0 (size {})."
                    ).format(idx, self.shape[0]),
                    suggestion=String(
                        "Valid indices: 0 <= i < {} or -{} <= i < 0 (negative"
                        " wrap)."
                    ).format(self.shape[0], self.shape[0]),
                    location=String("ComplexNDArray.__getitem__(idx: Int)"),
                )
            )

        # 1-D -> complex scalar (0-D ComplexNDArray wrapper)
        if self.ndim == 1:
            return creation._0darray[cdtype](
                ComplexSIMD[cdtype](
                    re=self._re._buf.ptr[norm],
                    im=self._im._buf.ptr[norm],
                )
            )

        var out_shape: NDArrayShape = self.shape[1:]
        var alloc_order: String = String("C")
        if self.flags.F_CONTIGUOUS:
            alloc_order = String("F")
        var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](
            shape=out_shape, order=alloc_order
        )

        # Fast path for C-contiguous
        if self.flags.C_CONTIGUOUS:
            var block: Int = self.size // self.shape[0]
            memcpy(
                dest=result._re._buf.ptr,
                src=self._re._buf.ptr + norm * block,
                count=block,
            )
            memcpy(
                dest=result._im._buf.ptr,
                src=self._im._buf.ptr + norm * block,
                count=block,
            )
            return result^
        else:
            # F layout
            self[Self.dtype]._re._copy_first_axis_slice(
                self._re, norm, result._re
            )
            self[Self.dtype]._im._copy_first_axis_slice(
                self._im, norm, result._im
            )
            return result^

    fn __getitem__(self, var *slices: Slice) raises -> Self:
        """
        Retrieves a slice or sub-array from the current array using variadic slice arguments.

        Args:
            slices: Variadic list of `Slice` objects, one for each dimension to be sliced.

        Constraints:
            - The number of slices provided must not exceed the number of array dimensions.
            - Each slice must be valid for its corresponding dimension.

        Returns:
            Self: A new array instance representing the sliced view of the original array.

        Raises:
            IndexError: If any slice is out of bounds for its corresponding dimension.
            ValueError: If the number of slices does not match the array's dimensions.

        NOTES:
            - This method creates a new array; Views are not currently supported.
            - Negative indices and step sizes are supported as per standard slicing semantics.

        Examples:
            ```mojo
            import numojo as nm
            var a = numojo.arange(10).reshape(nm.Shape(2, 5))
            var b = a[:, 2:4]
            print(b) # Output: 2x2 sliced array corresponding to columns 2 and 3 of each row.
            ```
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Too many slices provided: expected at most {} but"
                        " got {}."
                    ).format(self.ndim, n_slices),
                    suggestion=String(
                        "Provide at most {} slices for an array with {}"
                        " dimensions."
                    ).format(self.ndim, self.ndim),
                    location=String("NDArray.__getitem__(slices: Slice)"),
                )
            )
        var slice_list: List[Slice] = List[Slice](capacity=self.ndim)
        for i in range(len(slices)):
            slice_list.append(slices[i])

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        var narr: Self = self[slice_list^]
        return narr^

    fn _calculate_strides(self, shape: List[Int]) -> List[Int]:
        var strides = List[Int](capacity=len(shape))

        if self.flags.C_CONTIGUOUS:  # C_CONTIGUOUS
            var temp_strides = List[Int](capacity=len(shape))
            var stride = 1
            for i in range(len(shape) - 1, -1, -1):
                temp_strides.append(stride)
                stride *= shape[i]

            for i in range(len(temp_strides) - 1, -1, -1):
                strides.append(temp_strides[i])
        else:  # F_CONTIGUOUS
            var stride = 1
            for i in range(len(shape)):
                strides.append(stride)
                stride *= shape[i]

        return strides^

    fn __getitem__(self, var slice_list: List[Slice]) raises -> Self:
        """
        Retrieves a sub-array from the current array using a list of slice objects, enabling advanced slicing operations across multiple dimensions.

        Args:
            slice_list: List of Slice objects, where each Slice defines the start, stop, and step for the corresponding dimension.

        Constraints:
            - The length of slice_list must not exceed the number of dimensions in the array.
            - Each Slice in slice_list must be valid for its respective dimension.

        Returns:
            Self: A new array instance representing the sliced view of the original array.

        Raises:
            Error: If slice_list is empty or contains invalid slices.

        NOTES:
            - This method supports advanced slicing similar to NumPy's multi-dimensional slicing.
            - The returned array shares data with the original array if possible.

        Example:
            ```mojo
            import numojo as nm
            from numojo.prelude import *
            var a = nm.arange[cf32](CScalar[cf32](10.0, 10.0)).reshape(nm.Shape(2, 5))
            var b = a[List[Slice](Slice(0, 2, 1), Slice(2, 4, 1))]  # Equivalent to arr[:, 2:4], returns a 2x2 sliced array.
            print(b)
            ```
        """
        var n_slices: Int = slice_list.__len__()
        # Check error cases
        # I think we can remove this since it seems redundant.
        if n_slices == 0:
            raise Error(
                IndexError(
                    message=String(
                        "Empty slice list provided to"
                        " ComplexNDArray.__getitem__."
                    ),
                    suggestion=String(
                        "Provide a List with at least one slice to index the"
                        " array."
                    ),
                    location=String(
                        "ComplexNDArray.__getitem__(slice_list: List[Slice])"
                    ),
                )
            )

        var slices: List[Slice] = self._adjust_slice(slice_list)
        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slices.append(Slice(0, self.shape[i], 1))

        var ndims: Int = 0
        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var noffset: Int = 0

        for i in range(self.ndim):
            var start: Int = slices[i].start.value()
            var end: Int = slices[i].end.value()
            var step: Int = slices[i].step.or_else(1)

            var slice_len: Int
            if step > 0:
                slice_len: Int = max(0, (end - start + (step - 1)) // step)
            else:
                slice_len: Int = max(0, (start - end - step - 1) // (-step))
            # if slice_len >= 1: # remember to remove this behaviour and reduce dimension when user gives integer instead of slices
            nshape.append(slice_len)
            ncoefficients.append(self.strides[i] * step)
            ndims += 1
            noffset += start * self.strides[i]

        if len(nshape) == 0:
            nshape.append(1)
            ncoefficients.append(1)

        # only C & F order are supported
        var nstrides: List[Int] = self._calculate_strides(
            nshape,
        )
        var narr = ComplexNDArray[cdtype](
            offset=noffset, shape=nshape, strides=nstrides
        )
        # TODO: combine the two traverses into one.
        var index_re: List[Int] = List[Int](length=ndims, fill=0)
        _traverse_iterative[Self.dtype](
            self._re,
            narr._re,
            nshape,
            ncoefficients,
            nstrides,
            noffset,
            index_re,
            0,
        )
        var index_im: List[Int] = List[Int](length=ndims, fill=0)
        _traverse_iterative[Self.dtype](
            self._im,
            narr._im,
            nshape,
            ncoefficients,
            nstrides,
            noffset,
            index_im,
            0,
        )

        return narr^

    fn __getitem__(self, var *slices: Variant[Slice, Int]) raises -> Self:
        """
        Get items of ComplexNDArray with a series of either slices or integers.

        Args:
            slices: A series of either Slice or Int.

        Returns:
            A slice of the ndarray with a smaller or equal dimension of the original one.

        Raises:
            Error: If the number of slices is greater than the number of dimensions of the array.

        Examples:

        ```mojo
            import numojo as nm
            from numojo.prelude import *
            var a = nm.full[cf32](nm.Shape(2, 5), CScalar[cf32](1.0, 1.0))
            var b = a[1, Slice(2,4)]
            print(b)
        ```
        """
        var n_slices: Int = len(slices)
        if n_slices > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Too many indices or slices: received {} but array has"
                        " only {} dimensions."
                    ).format(n_slices, self.ndim),
                    suggestion=String(
                        "Pass at most {} indices/slices (one per dimension)."
                    ).format(self.ndim),
                    location=String(
                        "NDArray.__getitem__(*slices: Variant[Slice, Int])"
                    ),
                )
            )
        var slice_list: List[Slice] = List[Slice]()
        var count_int: Int = 0  # Count the number of Int in the argument
        var indices: List[Int] = List[Int]()

        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i][Slice])
            elif slices[i].isa[Int]():
                var norm: Int = slices[i][Int]
                if norm >= self.shape[i] or norm < -self.shape[i]:
                    raise Error(
                        IndexError(
                            message=String(
                                "Integer index {} out of bounds for axis {}"
                                " (size {})."
                            ).format(slices[i][Int], i, self.shape[i]),
                            suggestion=String(
                                "Valid indices: 0 <= i < {} or negative -{}"
                                " <= i < 0 (negative indices wrap from the"
                                " end)."
                            ).format(self.shape[i], self.shape[i]),
                            location=String(
                                "ComplexNDArray.__getitem__(*slices:"
                                " Variant[Slice, Int])"
                            ),
                        )
                    )
                if norm < 0:
                    norm += self.shape[i]
                count_int += 1
                indices.append(norm)
                slice_list.append(Slice(norm, norm + 1, 1))

        var narr: Self
        if count_int == self.ndim:
            narr = creation._0darray[cdtype](self._getitem(indices))
            return narr^

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        narr = self.__getitem__(slice_list^)
        return narr^

    fn __getitem__(self, indices: NDArray[DType.int]) raises -> Self:
        """
        Get items from 0-th dimension of a ComplexNDArray of indices.
        If the original array is of shape (i,j,k) and
        the indices array is of shape (l, m, n), then the output array
        will be of shape (l,m,n,j,k).

        Args:
            indices: Array of indices.

        Returns:
            ComplexNDArray with items from the array of indices.

        Raises:
            Error: If the elements of indices are greater than size of the corresponding dimension of the array.
        """
        # Get the shape of resulted array
        var shape = indices.shape.join(self.shape._pop(0))

        var result: ComplexNDArray[cdtype] = ComplexNDArray[cdtype](shape)
        var size_per_item: Int = self.size // self.shape[0]

        # Fill in the values
        for i in range(indices.size):
            if indices.item(i) >= self.shape[0]:
                raise Error(
                    IndexError(
                        message=String(
                            "Index {} (value {}) out of range for first"
                            " dimension size {}."
                        ).format(i, indices.item(i), self.shape[0]),
                        suggestion=String(
                            "Ensure each index < {}. Consider clipping or"
                            " validating indices before indexing."
                        ).format(self.shape[0]),
                        location=String(
                            "ComplexNDArray.__getitem__(indices:"
                            " NDArray[index])"
                        ),
                    )
                )
            memcpy(
                dest=result._re._buf.ptr + i * size_per_item,
                src=self._re._buf.ptr + indices.item(i) * size_per_item,
                count=size_per_item,
            )
            memcpy(
                dest=result._im._buf.ptr + i * size_per_item,
                src=self._im._buf.ptr + indices.item(i) * size_per_item,
                count=size_per_item,
            )

        return result^

    fn __getitem__(self, indices: List[Int]) raises -> Self:
        """
        Get items from 0-th dimension of a ComplexNDArray of indices.
        It is an overload of
        `__getitem__(self, indices: NDArray[DType.int]) raises -> Self`.

        Args:
            indices: A list of Int.

        Returns:
            ComplexNDArray with items from the list of indices.

        Raises:
            Error: If the elements of indices are greater than size of the corresponding dimension of the array.

        """

        var indices_array = NDArray[DType.int](shape=Shape(len(indices)))
        for i in range(len(indices)):
            (indices_array._buf.ptr + i).init_pointee_copy(indices[i])

        return self[indices_array]

    fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self:
        """
        Get item from a ComplexNDArray according to a mask array.
        If array shape is equal to mask shape, it returns a flattened array of
        the values where mask is True.
        If array shape is not equal to mask shape, it returns items from the
        0-th dimension of the array where mask is True.

        Args:
            mask: NDArray with Dtype.bool.

        Returns:
            ComplexNDArray with items from the mask.

        Raises:
            Error: If the mask is not a 1-D array (Currently we only support 1-d mask array).

        """
        # CASE 1:
        # if array shape is equal to mask shape,
        # return a flattened array of the values where mask is True
        if mask.shape == self.shape:
            var len_of_result = 0

            # Count number of True
            for i in range(mask.size):
                if mask.item(i):
                    len_of_result += 1

            # Change the first number of the ndshape
            var result = ComplexNDArray[cdtype](
                shape=NDArrayShape(len_of_result)
            )

            # Fill in the values
            var offset = 0
            for i in range(mask.size):
                if mask.item(i):
                    (result._re._buf.ptr + offset).init_pointee_copy(
                        self._re._buf.ptr[i]
                    )
                    (result._im._buf.ptr + offset).init_pointee_copy(
                        self._im._buf.ptr[i]
                    )
                    offset += 1

            return result^

        # CASE 2:
        # if array shape is not equal to mask shape,
        # return items from the 0-th dimension of the array where mask is True
        if mask.ndim > 1:
            raise Error(
                ShapeError(
                    message=String(
                        "Boolean mask must be 1-D or match full array shape;"
                        " got ndim={} for mask shape {}."
                    ).format(mask.ndim, mask.shape),
                    suggestion=String(
                        "Use a 1-D mask of length {} for first-dimension"
                        " filtering or a full-shape mask {} for element-wise"
                        " selection."
                    ).format(self.shape[0], self.shape),
                    location=String(
                        "ComplexNDArray.__getitem__(mask: NDArray[bool])"
                    ),
                )
            )

        if mask.shape[0] != self.shape[0]:
            raise Error(
                ShapeError(
                    message=String(
                        "Mask length {} does not match first dimension size {}."
                    ).format(mask.shape[0], self.shape[0]),
                    suggestion=String(
                        "Provide mask of length {} to filter along first"
                        " dimension."
                    ).format(self.shape[0]),
                    location=String(
                        "ComplexNDArray.__getitem__(mask: NDArray[bool])"
                    ),
                )
            )

        var len_of_result = 0

        # Count number of True
        for i in range(mask.size):
            if mask.item(i):
                len_of_result += 1

        # Change the first number of the ndshape
        var shape = self.shape
        shape._buf[0] = len_of_result

        var result = ComplexNDArray[cdtype](shape)
        var size_per_item = self.size // self.shape[0]

        # Fill in the values
        var offset = 0
        for i in range(mask.size):
            if mask.item(i):
                memcpy(
                    dest=result._re._buf.ptr + offset * size_per_item,
                    src=self._re._buf.ptr + i * size_per_item,
                    count=size_per_item,
                )
                memcpy(
                    dest=result._im._buf.ptr + offset * size_per_item,
                    src=self._im._buf.ptr + i * size_per_item,
                    count=size_per_item,
                )
                offset += 1

        return result^

    fn __getitem__(self, mask: List[Bool]) raises -> Self:
        """
        Get items from 0-th dimension of a ComplexNDArray according to mask.

        Args:
            mask: A list of boolean values.

        Returns:
            ComplexNDArray with items from the mask.

        Raises:
            Error: If the mask is not a 1-D array (Currently we only support 1-d mask array).
        """

        var mask_array = NDArray[DType.bool](shape=Shape(len(mask)))
        for i in range(len(mask)):
            (mask_array._buf.ptr + i).init_pointee_copy(mask[i])

        return self[mask_array]

    fn item(self, var index: Int) raises -> ComplexSIMD[cdtype]:
        """
        Return the scalar at the coordinates.
        If one index is given, get the i-th item of the complex array (not buffer).
        It first scans over the first row, even it is a column-major array.
        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.
        If the ndim is 0 (0-D array), get the value as a mojo scalar.

        Args:
            index: Index of item, counted in row-major way.

        Returns:
            A ComplexSIMD matching the dtype of the complex array.

        Raises:
            Error if array is 0-D array (numojo scalar).
            Error if index is equal or larger than array size.

        Examples:

        ```console
        >>> import numojo as nm
        >>> var A = nm.full[nm.f32](Shape(2, 2, 2), ComplexSIMD[nm.f32](1.0, 1.0))
        >>> print(A.item(10)) # returns the 10-th item of the complex array.
        ```.
        """
        # For 0-D array, raise error
        if self.ndim == 0:
            raise Error(
                IndexError(
                    message=String(
                        "Cannot index into a 0D ComplexNDArray with a linear"
                        " position."
                    ),
                    suggestion=String(
                        "Call item() with no arguments or use A[] to read"
                        " scalar."
                    ),
                    location=String("ComplexNDArray.item(index: Int)"),
                )
            )

        index = self.normalize(index, self.size)

        if (index < 0) or (index >= self.size):
            raise Error(
                IndexError(
                    message=String(
                        "Linear index {} out of range for array size {}."
                    ).format(index, self.size),
                    suggestion=String(
                        "Valid linear indices: 0..{} (inclusive). Use negative"
                        " indices only where supported."
                    ).format(self.size - 1),
                    location=String("ComplexNDArray.item(index: Int)"),
                )
            )

        if self.flags.F_CONTIGUOUS:
            return ComplexSIMD[cdtype](
                re=(
                    self._re._buf.ptr + _transfer_offset(index, self.strides)
                )[],
                im=(
                    self._im._buf.ptr + _transfer_offset(index, self.strides)
                )[],
            )

        else:
            return ComplexSIMD[cdtype](
                re=(self._re._buf.ptr + index)[],
                im=(self._im._buf.ptr + index)[],
            )

    fn item(self, *index: Int) raises -> ComplexSIMD[cdtype]:
        """
        Return the scalar at the coordinates.
        If one index is given, get the i-th item of the complex array (not buffer).
        It first scans over the first row, even it is a colume-major array.
        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.
        For 0-D complex array (numojo scalar), return the scalar value.

        Args:
            index: The coordinates of the item.

        Returns:
            A ComplexSIMD matching the dtype of the complex array.

        Raises:
            Error: If the number of indices is not equal to the number of dimensions of the array.
            Error: If the index is equal or larger than size of dimension.

        Examples:

        ```console
        >>> import numojo as nm
        >>> var A = nm.full[nm.f32](Shape(2, 2, 2), ComplexSIMD[nm.f32](1.0, 1.0))
        >>> print(A.item(1, 1, 1)) # returns the 10-th item of the complex array.
        ```.
        """

        if len(index) != self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Expected {} indices (ndim) but got {}."
                    ).format(self.ndim, len(index)),
                    suggestion=String(
                        "Provide one coordinate per dimension for shape {}."
                    ).format(self.shape),
                    location=String("ComplexNDArray.item(*index: Int)"),
                )
            )

        if self.ndim == 0:
            return ComplexSIMD[cdtype](
                re=self._re._buf.ptr[],
                im=self._im._buf.ptr[],
            )

        var list_index = List[Int]()
        for i in range(len(index)):
            if index[i] < 0:
                list_index.append(index[i] + self.shape[i])
            else:
                list_index.append(index[i])
            if (list_index[i] < 0) or (list_index[i] >= self.shape[i]):
                raise Error(
                    IndexError(
                        message=String(
                            "Index {} out of range for dimension {} (size {})."
                        ).format(list_index[i], i, self.shape[i]),
                        suggestion=String(
                            "Valid range is [0, {}). Consider adjusting or"
                            " clipping."
                        ).format(self.shape[i]),
                        location=String("ComplexNDArray.item(*index: Int)"),
                    )
                )
        return ComplexSIMD[cdtype](
            re=(self._re._buf.ptr + _get_offset(index, self.strides))[],
            im=(self._im._buf.ptr + _get_offset(index, self.strides))[],
        )

    fn load(self, var index: Int) raises -> ComplexSIMD[cdtype]:
        """
        Safely retrieve i-th item from the underlying buffer.

        `A.load(i)` differs from `A._buf.ptr[i]` due to boundary check.

        Args:
            index: Index of the item.

        Returns:
            The value at the index.

        Raises:
            Index out of bounds.

        Examples:

        ```console
        >>> import numojo as nm
        >>> var A = nm.full[nm.f32](Shape(2, 2, 2), ComplexSIMD[nm.f32](1.0, 1.0))
        >>> print(A.load(10)) # returns the 10-th item of the complex array.
        ```.
        """

        index = self.normalize(index, self.size)

        if (index >= self.size) or (index < 0):
            raise Error(
                IndexError(
                    message=String("Index {} out of range for size {}.").format(
                        index, self.size
                    ),
                    suggestion=String(
                        "Use 0 <= i < {}. Adjust negatives manually; negative"
                        " indices are not supported here."
                    ).format(self.size),
                    location=String("ComplexNDArray.load(index: Int)"),
                )
            )

        return ComplexSIMD[cdtype](
            re=self._re._buf.ptr[index],
            im=self._im._buf.ptr[index],
        )

    fn load[
        width: Int = 1
    ](self, index: Int) raises -> ComplexSIMD[cdtype, width]:
        """
        Safely loads a ComplexSIMD element of size `width` at `index`
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Args:
            index: Index of the item.

        Returns:
            The ComplexSIMD element at the index.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                IndexError(
                    message=String("Index {} out of range for size {}.").format(
                        index, self.size
                    ),
                    suggestion=String(
                        "Use 0 <= i < {} when loading elements."
                    ).format(self.size),
                    location=String("ComplexNDArray.load[width](index: Int)"),
                )
            )

        return ComplexSIMD[cdtype, width](
            re=self._re._buf.ptr.load[width=1](index),
            im=self._im._buf.ptr.load[width=1](index),
        )

    fn load[
        width: Int = 1
    ](self, *indices: Int) raises -> ComplexSIMD[cdtype, width=width]:
        """
        Safely loads a ComplexSIMD element of size `width` at given variadic indices
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Args:
            indices: Variadic indices.

        Returns:
            The ComplexSIMD element at the indices.

        Raises:
            Error: If the length of indices does not match the number of dimensions.
            Error: If any of the indices is out of bound.

        Examples:

        ```console
        >>> import numojo as nm
        >>> var A = nm.full[nm.f32](Shape(2, 2, 2), ComplexSIMD[nm.f32](1.0, 1.0))
        >>> print(A.load(0, 1, 1))
        ```.
        """

        if len(indices) != self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Expected {} indices (ndim) but received {}."
                    ).format(self.ndim, len(indices)),
                    suggestion=String(
                        "Provide one index per dimension: shape {} needs {}"
                        " coordinates."
                    ).format(self.shape, self.ndim),
                    location=String(
                        "ComplexNDArray.load[width](*indices: Int)"
                    ),
                )
            )

        # NOTE: if we take in an owned instances of indices, we can modify it in place.
        var indices_list: List[Int] = List[Int](capacity=self.ndim)
        for i in range(self.ndim):
            var idx_i = indices[i]
            if idx_i < 0 or idx_i >= self.shape[i]:
                raise Error(
                    IndexError(
                        message=String(
                            "Index out of range at dim {}: got {}; valid range"
                            " is [0, {})."
                        ).format(i, idx_i, self.shape[i]),
                        suggestion=String(
                            "Clamp or validate indices against the dimension"
                            " size ({})."
                        ).format(self.shape[i]),
                        location=String(
                            "NDArray.load[width: Int = 1](*indices: Int) ->"
                            " SIMD[dtype, width]"
                        ),
                    )
                )
            idx_i = self.normalize(idx_i, self.shape[i])
            indices_list.append(idx_i)

        var idx: Int = _get_offset(indices_list, self.strides)
        return ComplexSIMD[cdtype, width=width](
            re=self._re._buf.ptr.load[width=width](idx),
            im=self._im._buf.ptr.load[width=width](idx),
        )

    fn _adjust_slice(self, slice_list: List[Slice]) raises -> List[Slice]:
        """
        Adjusts slice values to handle all possible slicing scenarios including:
        - Negative indices (Python-style wrapping)
        - Out-of-bounds clamping
        - Negative steps (reverse slicing)
        - Empty slices
        - Default start/end values based on step direction
        """
        var n_slices: Int = slice_list.__len__()
        if n_slices > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Too many slice dimensions: got {} but array has {}"
                        " dims."
                    ).format(n_slices, self.ndim),
                    suggestion=String(
                        "Provide at most {} slices for this array."
                    ).format(self.ndim),
                    location=String("ComplexNDArray._adjust_slice"),
                )
            )

        var slices = List[Slice](capacity=self.ndim)
        for i in range(n_slices):
            var dim_size = self.shape[i]
            var step = slice_list[i].step.or_else(1)

            if step == 0:
                raise Error(
                    ValueError(
                        message=String(
                            "Slice step cannot be zero (dimension {})."
                        ).format(i),
                        suggestion=String(
                            "Use positive or negative non-zero step."
                        ),
                        location=String("ComplexNDArray._adjust_slice"),
                    )
                )

            # defaults
            var start: Int
            var end: Int
            if step > 0:
                start = 0
                end = dim_size
            else:
                start = dim_size - 1
                end = -1

            # start
            if slice_list[i].start is not None:
                start = slice_list[i].start.value()
                if start < 0:
                    start += dim_size
                # Clamp to valid bounds once
                if step > 0:
                    start = 0 if start < 0 else (
                        dim_size if start > dim_size else start
                    )
                else:
                    start = -1 if start < -1 else (
                        dim_size - 1 if start >= dim_size else start
                    )

            # end
            if slice_list[i].end is not None:
                end = slice_list[i].end.value()
                if end < 0:
                    end += dim_size
                # NOTE: Clamp to valid bounds once. This is an implicit behavior right now instead of raising errors. not sure if this should be kept.
                if step > 0:
                    end = 0 if end < 0 else (
                        dim_size if end > dim_size else end
                    )
                else:
                    end = -1 if end < -1 else (
                        dim_size if end > dim_size else end
                    )

            slices.append(
                Slice(
                    start=Optional(start),
                    end=Optional(end),
                    step=Optional(step),
                )
            )

        return slices^

    fn _setitem(self, *indices: Int, val: ComplexSIMD[cdtype]):
        """
        (UNSAFE! for internal use only.)
        Set item at indices and bypass all boundary checks.

        Args:
            indices: Indices to set the value.
            val: Value to set.

        Notes:
            This function is unsafe and for internal use only.

        Examples:

        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var A = nm.full[cf32](Shape(2, 2), CScalar[cf32](1.0, 1.0))
        A._setitem(0, 1, val=CScalar[cf32](3.0, 4.0))
        ```
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * Int(self.strides._buf[i])
        self._re._buf.ptr[index_of_buffer] = val.re
        self._im._buf.ptr[index_of_buffer] = val.im

    fn __setitem__(mut self, idx: Int, val: Self) raises:
        """Assign a single first-axis slice.
        Replaces the sub-array at axis 0 position `idx` with `val`.
        The shape of `val` must exactly match `self.shape[1:]` and its
        dimensionality must be `self.ndim - 1` (or be a 0-D complex scalar
        when assigning into a 1-D array). Negative indices are supported.
        Fast path: contiguous memcpy for C-order; otherwise a stride-based
        generic copy is performed for both real and imaginary parts.

        Args:
            idx: Integer index along first dimension (supports negatives).
            val: ComplexNDArray slice data to assign.

        Raises:
            IndexError: If array is 0-D or idx out of bounds.
            ShapeError: If `val` shape/dim mismatch with target slice.
        """
        if self.ndim == 0:
            raise Error(
                IndexError(
                    message=String("Cannot assign slice on 0D ComplexNDArray."),
                    suggestion=String(
                        "Assign to its scalar value with `A[] = ...` once"
                        " supported."
                    ),
                    location=String(
                        "ComplexNDArray.__setitem__(idx: Int, val: Self)"
                    ),
                )
            )

        var norm = idx
        norm = self.normalize(norm, self.shape[0])
        if (norm < 0) or (norm >= self.shape[0]):
            raise Error(
                IndexError(
                    message=String(
                        "Index {} out of bounds for axis 0 (size {})."
                    ).format(idx, self.shape[0]),
                    suggestion=String(
                        "Valid indices: 0 <= i < {} or -{} <= i < 0."
                    ).format(self.shape[0], self.shape[0]),
                    location=String(
                        "ComplexNDArray.__setitem__(idx: Int, val: Self)"
                    ),
                )
            )

        # 1-D target: expect 0-D complex scalar wrapper (val.ndim == 0)
        if self.ndim == 1:
            if val.ndim != 0:
                raise Error(
                    ShapeError(
                        message=String(
                            "Shape mismatch: expected 0D value for 1D target"
                            " slice."
                        ),
                        suggestion=String(
                            "Provide a 0D ComplexNDArray (scalar wrapper)."
                        ),
                        location=String(
                            "ComplexNDArray.__setitem__(idx: Int, val: Self)"
                        ),
                    )
                )
            self._re._buf.ptr.store(norm, val._re._buf.ptr.load[width=1](0))
            self._im._buf.ptr.store(norm, val._im._buf.ptr.load[width=1](0))
            return

        if val.shape != self.shape[1:]:
            raise Error(
                ShapeError(
                    message=String(
                        "Shape mismatch for slice assignment: expected {} but"
                        " got {}."
                    ).format(self.shape[1:], val.shape),
                    suggestion=String(
                        "Provide RHS slice with exact shape {}; broadcasting"
                        " not yet supported."
                    ).format(self.shape[1:]),
                    location=String(
                        "ComplexNDArray.__setitem__(idx: Int, val: Self)"
                    ),
                )
            )

        if self.flags.C_CONTIGUOUS & val.flags.C_CONTIGUOUS:
            var block = self.size // self.shape[0]
            memcpy(
                dest=self._re._buf.ptr + norm * block,
                src=val._re._buf.ptr,
                count=block,
            )
            memcpy(
                dest=self._im._buf.ptr + norm * block,
                src=val._im._buf.ptr,
                count=block,
            )
            return

        # F order
        self[Self.dtype]._re._write_first_axis_slice(self._re, norm, val._re)
        self[Self.dtype]._im._write_first_axis_slice(self._im, norm, val._im)

    fn __setitem__(mut self, var index: Item, val: ComplexSIMD[cdtype]) raises:
        """
        Sets the value at the index list.

        Args:
            index: Index list.
            val: Value to set.

        Raises:
            Error: If the length of index does not match the number of dimensions.
            Error: If any of the indices is out of bound.

        Examples:

        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var A = nm.full[cf32](Shape(2, 2), CScalar[cf32](1.0))
        A[Item(0, 1)] = CScalar[cf32](3.0, 4.0)
        ```
        """
        if index.__len__() != self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Invalid index length: expected {} but got {}."
                    ).format(self.ndim, index.__len__()),
                    suggestion=String(
                        "Pass exactly {} indices (one per dimension)."
                    ).format(self.ndim),
                    location=String(
                        "ComplexNDArray.__setitem__(index: Item, val:"
                        " Scalar[dtype])"
                    ),
                )
            )
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                raise Error(
                    IndexError(
                        message=String(
                            "Index out of range at dim {}: got {}; valid range"
                            " is [0, {})."
                        ).format(i, index[i], self.shape[i]),
                        suggestion=String(
                            "Clamp or validate indices against the dimension"
                            " size ({})."
                        ).format(self.shape[i]),
                        location=String(
                            "NDArray.__setitem__(index: Item, val:"
                            " Scalar[dtype])"
                        ),
                    )
                )
            index[i] = self.normalize(index[i], self.shape[i])

        var idx: Int = _get_offset(index, self.strides)
        self._re._buf.ptr.store(idx, val.re)
        self._im._buf.ptr.store(idx, val.im)

    fn __setitem__(
        mut self,
        mask: ComplexNDArray[cdtype],
        value: ComplexSIMD[cdtype],
    ) raises:
        """
        Set the value of the array at the indices where the mask is true.
        """
        if (
            mask.shape != self.shape
        ):  # this behaviour could be removed potentially
            raise Error("Mask and array must have the same shape")

        for i in range(mask.size):
            if mask._re._buf.ptr.load[width=1](i):
                self._re._buf.ptr.store(i, value.re)
            if mask._im._buf.ptr.load[width=1](i):
                self._im._buf.ptr.store(i, value.im)

    fn __setitem__(
        mut self, var *slices: Slice, val: ComplexNDArray[cdtype]
    ) raises:
        """
        Retreive slices of an ComplexNDArray from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(slices.__len__()):
            slice_list.append(slices[i])
        # self.__setitem__(slices=slice_list, val=val)
        self[slice_list^] = val

    fn __setitem__(
        mut self, slices: List[Slice], val: ComplexNDArray[cdtype]
    ) raises:
        """
        Sets the slices of an ComplexNDArray from list of slices and ComplexNDArray.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """
        var n_slices: Int = len(slices)
        var ndims: Int = 0
        var count: Int = 0
        var spec: List[Int] = List[Int]()
        var slice_list: List[Slice] = self._adjust_slice(slices)
        for i in range(n_slices):
            # TODO: these conditions can be removed since _adjust_slice takes care of them. But verify it once before removing.
            if (
                slice_list[i].start.value() >= self.shape[i]
                or slice_list[i].end.value() > self.shape[i]
            ):
                var message = String(
                    "Error: Slice value exceeds the array shape!\n"
                    "The {}-th dimension is of size {}.\n"
                    "The slice goes from {} to {}"
                ).format(
                    i,
                    self.shape[i],
                    slice_list[i].start.value(),
                    slice_list[i].end.value(),
                )
                raise Error(message)
            # if slice_list[i].step is None:
            #     raise Error(String("Step of slice is None."))
            var slice_len: Int = (
                (slice_list[i].end.value() - slice_list[i].start.value())
                / slice_list[i].step.or_else(1)
            ).__int__()
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1
            else:
                count += 1
        if count == slice_list.__len__():
            ndims = 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        count = 0
        for _ in range(ndims):
            while spec[j] == 1:
                count += 1
                j += 1
            if j >= self.ndim:
                break
            var slice_len: Int = (
                (slice_list[j].end.value() - slice_list[j].start.value())
                / slice_list[j].step.or_else(1)
            ).__int__()
            nshape.append(slice_len)
            nnum_elements *= slice_len
            ncoefficients.append(
                self.strides[j] * slice_list[j].step.or_else(1)
            )
            j += 1

        # TODO: We can remove this check after we have support for broadcasting
        for i in range(ndims):
            if nshape[i] != val.shape[i]:
                var message = String(
                    "Error: Shape mismatch!\n"
                    "For {}-th dimension: \n"
                    "The size of the array is {}.\n"
                    "The size of the input value is {}."
                ).format(i, nshape[i], val.shape[i])
                raise Error(message)

        var noffset: Int = 0
        if self.flags["C_CONTIGUOUS"]:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.flags["F_CONTIGUOUS"]:
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]

        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative_setter[Self.dtype](
            val._re, self._re, nshape, ncoefficients, nstrides, noffset, index
        )
        _traverse_iterative_setter[Self.dtype](
            val._im, self._im, nshape, ncoefficients, nstrides, noffset, index
        )

    ## compiler doesn't accept this.
    fn __setitem__(
        self, var *slices: Variant[Slice, Int], val: ComplexNDArray[cdtype]
    ) raises:
        """
        Get items by a series of either slices or integers.
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Too many indices or slices: received {} but array has"
                        " only {} dimensions."
                    ).format(n_slices, self.ndim),
                    suggestion=String(
                        "Pass at most {} indices/slices (one per dimension)."
                    ).format(self.ndim),
                    location=String(
                        "NDArray.__setitem__(*slices: Variant[Slice, Int], val:"
                        " Self)"
                    ),
                )
            )
        var slice_list: List[Slice] = List[Slice]()

        var count_int = 0
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                count_int += 1
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int + 1, 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        # self.__setitem__(slices=slice_list, val=val)
        self[slice_list^] = val

    fn __setitem__(self, index: NDArray[DType.int], val: Self) raises:
        """
        Returns the items of the ComplexNDArray from an array of indices.

        Refer to `__getitem__(self, index: List[Int])`.
        """

        for i in range(len(index)):
            self._re.store(
                Int(index.load(i)), rebind[Scalar[Self.dtype]](val._re.load(i))
            )
            self._im.store(
                Int(index.load(i)), rebind[Scalar[Self.dtype]](val._im.load(i))
            )

    # TODO: implement itemset().
    fn __setitem__(
        mut self,
        mask: ComplexNDArray[cdtype],
        val: ComplexNDArray[cdtype],
    ) raises:
        """
        Set the value of the ComplexNDArray at the indices where the mask is true.
        """
        if (
            mask.shape != self.shape
        ):  # this behavious could be removed potentially
            var message = String(
                "Shape of mask does not match the shape of array."
            )
            raise Error(message)

        for i in range(mask.size):
            if mask._re._buf.ptr.load(i):
                self._re._buf.ptr.store(i, val._re._buf.ptr.load(i))
            if mask._im._buf.ptr.load(i):
                self._im._buf.ptr.store(i, val._im._buf.ptr.load(i))

    fn __pos__(self) raises -> Self:
        """
        Unary positive returns self unless boolean type.
        """
        if self.dtype is DType.bool:
            raise Error(
                "complex_ndarray:ComplexNDArray:__pos__: pos does not accept"
                " bool type arrays"
            )
        return self.copy()

    fn __neg__(self) raises -> Self:
        """
        Unary negative returns self unless boolean type.

        For bolean use `__invert__`(~)
        """
        if self.dtype is DType.bool:
            raise Error(
                "complex_ndarray:ComplexNDArray:__neg__: neg does not accept"
                " bool type arrays"
            )
        return self * ComplexSIMD[cdtype](-1.0, -1.0)

    fn __bool__(self) raises -> Bool:
        """
        Check if the complex array is non-zero.

        For a 0-D or length-1 complex array, returns True if the complex number
        is non-zero (i.e., either real or imaginary part is non-zero).

        Returns:
            True if the complex number is non-zero, False otherwise.

        Raises:
            Error: If the array is not 0-D or length-1.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape())  # 0-D array
        A._re._buf.ptr[] = 1.0
        A._im._buf.ptr[] = 0.0
        var result = A.__bool__()  # True
        ```
        """
        if (self.size == 1) or (self.ndim == 0):
            var re_val = self._re._buf.ptr[]
            var im_val = self._im._buf.ptr[]
            return Bool((re_val != 0.0) or (im_val != 0.0))
        else:
            raise Error(
                "\nError in `ComplexNDArray.__bool__(self)`: "
                "Only 0-D arrays (numojo scalar) or length-1 arrays "
                "can be converted to Bool. "
                "The truth value of an array with more than one element is "
                "ambiguous. Use a.any() or a.all()."
            )

    fn __int__(self) raises -> Int:
        """
        Gets `Int` representation of the complex array's real part.

        Only 0-D arrays or length-1 arrays can be converted to scalars.
        The imaginary part is discarded.

        Returns:
            Int representation of the real part of the array.

        Raises:
            Error: If the array is not 0-D or length-1.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape())  # 0-D array
        A._re._buf.ptr[] = 42.7
        A._im._buf.ptr[] = 3.14
        print(A.__int__())  # 42 (only real part)
        ```
        """
        if (self.size == 1) or (self.ndim == 0):
            return Int(self._re._buf.ptr[])
        else:
            raise Error(
                "\nError in `ComplexNDArray.__int__(self)`: "
                "Only 0-D arrays (numojo scalar) or length-1 arrays "
                "can be converted to scalars."
            )

    fn __float__(self) raises -> Float64:
        """
        Gets `Float64` representation of the complex array's magnitude.

        Only 0-D arrays or length-1 arrays can be converted to scalars.
        Returns the magnitude (absolute value) of the complex number.

        Returns:
            Float64 representation of the magnitude of the complex number.

        Raises:
            Error: If the array is not 0-D or length-1.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape())  # 0-D array
        A._re._buf.ptr[] = 3.0
        A._im._buf.ptr[] = 4.0
        print(A.__float__())  # 5.0 (magnitude)
        ```
        """
        if (self.size == 1) or (self.ndim == 0):
            var re_val = self._re._buf.ptr[]
            var im_val = self._im._buf.ptr[]
            var magnitude_sq = Float64(re_val * re_val + im_val * im_val)
            return sqrt(magnitude_sq)
        else:
            raise Error(
                "\nError in `ComplexNDArray.__float__(self)`: "
                "Only 0-D arrays (numojo scalar) or length-1 arrays "
                "can be converted to scalars."
            )

    fn __abs__(self) raises -> NDArray[Self.dtype]:
        """
        Compute the magnitude (absolute value) of each complex element.

        Returns an NDArray of real values containing the magnitude of each
        complex number: sqrt(re^2 + im^2).

        Returns:
            NDArray containing the magnitude of each complex element.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        # Fill with some values
        var mag = A.__abs__()  # Returns NDArray[f64] with magnitudes
        ```
        """
        var re_sq = self._re * self._re
        var im_sq = self._im * self._im
        var sum_sq = re_sq + im_sq
        return misc.sqrt[Self.dtype](sum_sq)

    fn __pow__(self, p: Int) raises -> Self:
        """
        Raise complex array to integer power element-wise.

        Uses De Moivre's formula for complex exponentiation:
        (r * e^(i*theta))^n = r^n * e^(i*n*theta)

        Args:
            p: Integer exponent.

        Returns:
            ComplexNDArray with each element raised to power p.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var B = A ** 3  # Cube each element
        ```
        """
        if p == 0:
            var ones_re = creation.ones[Self.dtype](self.shape)
            var zeros_im = creation.zeros[Self.dtype](self.shape)
            return Self(ones_re^, zeros_im^)
        elif p == 1:
            return self.copy()
        elif p < 0:
            var pos_pow = self.__pow__(-p)
            var denominator = (
                pos_pow._re * pos_pow._re + pos_pow._im * pos_pow._im
            )
            var result_re = pos_pow._re / denominator
            var result_im = -pos_pow._im / denominator
            return Self(result_re^, result_im^)
        else:
            var result = self.copy()
            for _ in range(p - 1):
                var temp = result * self
                result = temp^
            return result^

    fn __pow__(self, rhs: Scalar[Self.dtype]) raises -> Self:
        """
        Raise complex array to real scalar power element-wise.

        Args:
            rhs: Real scalar exponent.

        Returns:
            ComplexNDArray with each element raised to power rhs.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var B = A ** 2.5  # Raise to power 2.5
        ```
        """
        var r = misc.sqrt[Self.dtype](self._re * self._re + self._im * self._im)
        var theta = trig.atan2[Self.dtype](self._im, self._re)

        var r_pow = r.__pow__(rhs)
        var theta_p = theta * rhs

        var result_re = r_pow * trig.cos[Self.dtype](theta_p)
        var result_im = r_pow * trig.sin[Self.dtype](theta_p)

        return Self(result_re^, result_im^)

    fn __pow__(self, p: Self) raises -> Self:
        """
        Raise complex array to complex array power element-wise.

        Args:
            p: ComplexNDArray exponent.

        Returns:
            ComplexNDArray with each element raised to corresponding power.

        Raises:
            Error: If arrays have different sizes.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var B = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var C = A ** B  # Element-wise complex power
        ```
        """
        if self.size != p.size:
            raise Error(
                String(
                    "\nError in `ComplexNDArray.__pow__(self, p)`: "
                    "Both arrays must have same number of elements! "
                    "Self array has {} elements. "
                    "Other array has {} elements"
                ).format(self.size, p.size)
            )

        var mag = misc.sqrt[Self.dtype](
            self._re * self._re + self._im * self._im
        )
        var arg = trig.atan2[Self.dtype](self._im, self._re)

        var log_re = exponents.log[Self.dtype](mag)
        var log_im = arg^

        var exponent_re_temp1 = p._re * log_re
        var exponent_re_temp2 = p._im * log_im
        var exponent_re = exponent_re_temp1 - exponent_re_temp2
        var exponent_im_temp1 = p._re * log_im
        var exponent_im_temp2 = p._im * log_re
        var exponent_im = exponent_im_temp1 + exponent_im_temp2

        var exp_re = exponents.exp[Self.dtype](exponent_re)
        var result_re = exp_re * trig.cos[Self.dtype](exponent_im)
        var result_im = exp_re * trig.sin[Self.dtype](exponent_im)

        return Self(result_re^, result_im^)

    fn __ipow__(mut self, p: Int) raises:
        """
        In-place raise to integer power.

        Args:
            p: Integer exponent.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        A **= 3  # Cube in place
        ```
        """
        self = self.__pow__(p)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.
        """
        return comparison.equal[Self.dtype](
            self._re, other._re
        ) and comparison.equal[Self.dtype](self._im, other._im)

    @always_inline("nodebug")
    fn __eq__(self, other: ComplexSIMD[cdtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence between scalar and ComplexNDArray.
        """
        return comparison.equal[Self.dtype](
            self._re, other.re
        ) and comparison.equal[Self.dtype](self._im, other.im)

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence.
        """
        return comparison.not_equal[Self.dtype](
            self._re, other._re
        ) and comparison.not_equal[Self.dtype](self._im, other._im)

    @always_inline("nodebug")
    fn __ne__(self, other: ComplexSIMD[cdtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence between scalar and ComplexNDArray.
        """
        return comparison.not_equal[Self.dtype](
            self._re, other.re
        ) and comparison.not_equal[Self.dtype](self._im, other.im)

    @always_inline("nodebug")
    fn __lt__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise less than comparison by magnitude.

        For complex numbers, compares the magnitudes: |self| < |other|.
        This provides a natural ordering for complex numbers.

        Args:
            other: The other ComplexNDArray to compare with.

        Returns:
            An array of boolean values indicating where |self| < |other|.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var B = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var result = A < B  # Compare by magnitude
        ```

        Notes:
            Complex number ordering is not naturally defined. This implementation
            compares by magnitude (absolute value) to provide a consistent ordering.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other._re * other._re + other._im * other._im
        return comparison.less[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __lt__(self, other: ComplexSIMD[cdtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than comparison with scalar by magnitude.

        Args:
            other: The ComplexSIMD scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| < |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other.re * other.re + other.im * other.im
        return comparison.less[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __lt__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than comparison with real scalar by magnitude.

        Args:
            other: The real scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| < |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other * other
        return comparison.less[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __le__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal comparison by magnitude.

        For complex numbers, compares the magnitudes: |self| <= |other|.

        Args:
            other: The other ComplexNDArray to compare with.

        Returns:
            An array of boolean values indicating where |self| <= |other|.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var B = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var result = A <= B  # Compare by magnitude
        ```
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other._re * other._re + other._im * other._im
        return comparison.less_equal[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __le__(self, other: ComplexSIMD[cdtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal comparison with scalar by magnitude.

        Args:
            other: The ComplexSIMD scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| <= |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other.re * other.re + other.im * other.im
        return comparison.less_equal[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __le__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal comparison with real scalar by magnitude.

        Args:
            other: The real scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| <= |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other * other
        return comparison.less_equal[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __gt__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than comparison by magnitude.

        For complex numbers, compares the magnitudes: |self| > |other|.

        Args:
            other: The other ComplexNDArray to compare with.

        Returns:
            An array of boolean values indicating where |self| > |other|.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var B = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var result = A > B  # Compare by magnitude
        ```

        Notes:
            Complex number ordering is not naturally defined. This implementation
            compares by magnitude (absolute value) to provide a consistent ordering.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other._re * other._re + other._im * other._im
        return comparison.greater[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __gt__(self, other: ComplexSIMD[cdtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than comparison with scalar by magnitude.

        Args:
            other: The ComplexSIMD scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| > |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other.re * other.re + other.im * other.im
        return comparison.greater[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __gt__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than comparison with real scalar by magnitude.

        Args:
            other: The real scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| > |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other * other
        return comparison.greater[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __ge__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal comparison by magnitude.

        For complex numbers, compares the magnitudes: |self| >= |other|.

        Args:
            other: The other ComplexNDArray to compare with.

        Returns:
            An array of boolean values indicating where |self| >= |other|.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var B = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 2))
        var result = A >= B  # Compare by magnitude
        ```
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other._re * other._re + other._im * other._im
        return comparison.greater_equal[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __ge__(self, other: ComplexSIMD[cdtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal comparison with scalar by magnitude.

        Args:
            other: The ComplexSIMD scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| >= |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other.re * other.re + other.im * other.im
        return comparison.greater_equal[Self.dtype](self_mag, other_mag)

    @always_inline("nodebug")
    fn __ge__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal comparison with real scalar by magnitude.

        Args:
            other: The real scalar to compare with.

        Returns:
            An array of boolean values indicating where |self| >= |other|.
        """
        var self_mag = self._re * self._re + self._im * self._im
        var other_mag = other * other
        return comparison.greater_equal[Self.dtype](self_mag, other_mag)

    # ===------------------------------------------------------------------=== #
    # ARITHMETIC OPERATIONS
    # ===------------------------------------------------------------------=== #

    fn __add__(self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray + ComplexSIMD`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other.im)
        return Self(real^, imag^)

    fn __add__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + Scalar`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __add__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray + ComplexNDArray`.
        """
        print("add complex arrays")
        var real: NDArray[Self.dtype] = math.add[Self.dtype](
            self._re, other._re
        )
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](
            self._im, other._im
        )
        return Self(real^, imag^)

    fn __add__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + NDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __radd__(mut self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD + ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other.im)
        return Self(real^, imag^)

    fn __radd__(mut self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar + ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __radd__(mut self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray + ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __iadd__(mut self, other: ComplexSIMD[cdtype]) raises:
        """
        Enables `ComplexNDArray += ComplexSIMD`.
        """
        self._re += other.re
        self._im += other.im

    fn __iadd__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray += Scalar`.
        """
        self._re += other
        self._im += other

    fn __iadd__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray += ComplexNDArray`.
        """
        self._re += other._re
        self._im += other._im

    fn __iadd__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray += NDArray`.
        """
        self._re += other
        self._im += other

    fn __sub__(self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray - ComplexSIMD`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](self._im, other.im)
        return Self(real^, imag^)

    fn __sub__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - Scalar`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](
            self._re, other.cast[Self.dtype]()
        )
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](
            self._im, other.cast[Self.dtype]()
        )
        return Self(real^, imag^)

    fn __sub__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray - ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](
            self._re, other._re
        )
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](
            self._im, other._im
        )
        return Self(real^, imag^)

    fn __sub__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - NDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __rsub__(mut self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD - ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](other.re, self._re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](other.im, self._im)
        return Self(real^, imag^)

    fn __rsub__(mut self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar - ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._im)
        return Self(real^, imag^)

    fn __rsub__(mut self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray - ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._im)
        return Self(real^, imag^)

    fn __isub__(mut self, other: ComplexSIMD[cdtype]) raises:
        """
        Enables `ComplexNDArray -= ComplexSIMD`.
        """
        self._re -= other.re
        self._im -= other.im

    fn __isub__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray -= Scalar`.
        """
        self._re -= other
        self._im -= other

    fn __isub__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray -= ComplexNDArray`.
        """
        self._re -= other._re
        self._im -= other._im

    fn __isub__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray -= NDArray`.
        """
        self._re -= other
        self._im -= other

    fn __matmul__(self, other: Self) raises -> Self:
        var re_re: NDArray[Self.dtype] = linalg.matmul[Self.dtype](
            self._re, other._re
        )
        var im_im: NDArray[Self.dtype] = linalg.matmul[Self.dtype](
            self._im, other._im
        )
        var re_im: NDArray[Self.dtype] = linalg.matmul[Self.dtype](
            self._re, other._im
        )
        var im_re: NDArray[Self.dtype] = linalg.matmul[Self.dtype](
            self._im, other._re
        )
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray * ComplexSIMD`.
        """
        var re_re: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._re, other.re
        )
        var im_im: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._im, other.re
        )
        var re_im: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._re, other.im
        )
        var im_re: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._im, other.im
        )
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * Scalar`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __mul__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray * ComplexNDArray`.
        """
        var re_re: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._re, other._re
        )
        var im_im: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._im, other._im
        )
        var re_im: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._re, other._im
        )
        var im_re: NDArray[Self.dtype] = math.mul[Self.dtype](
            self._im, other._re
        )
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * NDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __rmul__(self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD * ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other.re)
        return Self(real^, imag^)

    fn __rmul__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar * ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __rmul__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray * ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __imul__(mut self, other: ComplexSIMD[cdtype]) raises:
        """
        Enables `ComplexNDArray *= ComplexSIMD`.
        """
        self._re *= other.re
        self._im *= other.im

    fn __imul__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray *= Scalar`.
        """
        self._re *= other
        self._im *= other

    fn __imul__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray *= ComplexNDArray`.
        """
        self._re *= other._re
        self._im *= other._im

    fn __imul__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray *= NDArray`.
        """
        self._re *= other
        self._im *= other

    fn __truediv__(self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var other_square = other * other.conj()
        var result = self * other.conj() * (1.0 / other_square.re)
        return result^

    fn __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var real: NDArray[Self.dtype] = math.div[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.div[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __truediv__(self, other: ComplexNDArray[cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real^, imag^)

    fn __truediv__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / NDArray`.
        """
        var real: NDArray[Self.dtype] = math.div[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.div[Self.dtype](self._im, other)
        return Self(real^, imag^)

    fn __rtruediv__(mut self, other: ComplexSIMD[cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom.re
        var imag = numer._im / denom.re
        return Self(real^, imag^)

    fn __rtruediv__(mut self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real^, imag^)

    fn __rtruediv__(mut self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real^, imag^)

    fn __itruediv__(mut self, other: ComplexSIMD[cdtype]) raises:
        """
        Enables `ComplexNDArray /= ComplexSIMD`.
        """
        self._re /= other.re
        self._im /= other.im

    fn __itruediv__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray /= Scalar`.
        """
        self._re /= other
        self._im /= other

    fn __itruediv__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray /= ComplexNDArray`.
        """
        self._re /= other._re
        self._im /= other._im

    fn __itruediv__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray /= NDArray`.
        """
        self._re /= other
        self._im /= other

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#
    fn __str__(self) -> String:
        """
        Enables String(array).
        """
        var res: String
        try:
            res = self._array_to_string(0, 0)
        except e:
            res = String("Cannot convert array to string") + String(e)

        return res

    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the array to a writer.

        Args:
            writer: The writer to write the array to.
        """
        if self.ndim == 0:
            # For 0-D array (numojo scalar), we can directly write the value
            writer.write(
                String(
                    ComplexScalar[cdtype](
                        self._re._buf.ptr[], self._im._buf.ptr[]
                    )
                )
                + String(
                    "  (0darray["
                    + _concise_dtype_str(cdtype)
                    + "], use `[]` or `.item()` to unpack)"
                )
            )
        else:
            try:
                writer.write(
                    self._array_to_string(0, 0)
                    + "\n"
                    + String(self.ndim)
                    + "D-array  Shape"
                    + String(self.shape)
                    + "  Strides"
                    + String(self.strides)
                    + "  DType: "
                    + _concise_dtype_str(cdtype)
                    + "  C-cont: "
                    + String(self.flags.C_CONTIGUOUS)
                    + "  F-cont: "
                    + String(self.flags.F_CONTIGUOUS)
                    + "  own data: "
                    + String(self.flags.OWNDATA)
                )
            except e:
                writer.write("Cannot convert array to string.\n" + String(e))

    fn __repr__(self) -> String:
        """
        Compute the "official" string representation of ComplexNDArray.
        An example is:
        ```
        fn main() raises:
            var A = ComplexNDArray[f32](List[ComplexSIMD[f32]](14,97,-59,-4,112,), shape=List[Int](5,))
            print(repr(A))
        ```
        It prints what can be used to construct the array itself:
        ```console
            ComplexNDArray[f32](List[ComplexSIMD[f32]](14,97,-59,-4,112,), shape=List[Int](5,))
        ```.
        """
        try:
            var result: String = (
                String("ComplexNDArray[CDType.")
                + String(self.dtype)
                + String("](List[ComplexSIMD[CDType.c")
                + String(self._re.dtype)
                + String("]](")
            )
            if self._re.size > 6:
                for i in range(6):
                    result = result + String(self.item(i)) + String(",")
                result = result + " ... "
            else:
                for i in range(self._re.size):
                    result = result + String(self.item(i)) + String(",")
            result = result + String("), shape=List[Int](")
            for i in range(self._re.shape.ndim):
                result = result + String(self._re.shape._buf[i]) + ","
            result = result + String("))")
            return result^
        except e:
            print("Cannot convert array to string", e)
            return ""

    fn _array_to_string(
        self,
        dimension: Int,
        offset: Int,
        var summarize: Bool = False,
    ) raises -> String:
        """
        Convert the array to a string.

        Args:
            dimension: The current dimension.
            offset: The offset of the current dimension.
            summarize: Internal flag indicating summarization already chosen.
        """
        var options: PrintOptions = self.print_options
        var separator = options.separator
        var padding = options.padding
        var edge_items = options.edge_items

        # Root-level summarize decision
        if dimension == 0 and (not summarize) and self.size > options.threshold:
            summarize = True

        # Last dimension: actual elements
        if dimension == self.ndim - 1:
            var n_items = self.shape[dimension]
            var edge = edge_items
            if edge * 2 >= n_items:
                edge = n_items

            var out: String = String("[") + padding
            if (not summarize) or (n_items == edge):
                for i in range(n_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    out += format_value(value, options)
                    if i < n_items - 1:
                        out += separator
                out += padding + "]"
            else:
                for i in range(edge):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    out += format_value(value, options)
                    if i < edge - 1:
                        out += separator
                out += separator + String("...") + separator
                for i in range(n_items - edge, n_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    out += format_value(value, options)
                    if i < n_items - 1:
                        out += separator
                out += padding + "]"

            # Greedy line wrapping
            if len(out) > options.line_width:
                var wrapped: String = String("")
                var line_len: Int = 0
                for c in out.codepoint_slices():
                    if c == String("\n"):
                        wrapped += c
                        line_len = 0
                    else:
                        if line_len >= options.line_width and c != String(" "):
                            wrapped += "\n"
                            line_len = 0
                        wrapped += c
                        line_len += 1
                out = wrapped
            return out

        # Higher dimensions
        var n_items_outer = self.shape[dimension]
        var edge_outer = edge_items
        if edge_outer * 2 >= n_items_outer:
            edge_outer = n_items_outer

        var result: String = String("[")
        if (not summarize) or (n_items_outer == edge_outer):
            for i in range(n_items_outer):
                if i > 0:
                    result += "\n" + String(" ") * (dimension)
                result += self._array_to_string(
                    dimension + 1,
                    offset + i * self.strides[dimension].__int__(),
                    summarize=summarize,
                )
        else:
            for i in range(edge_outer):
                if i > 0:
                    result += "\n" + String(" ") * (dimension)
                result += self._array_to_string(
                    dimension + 1,
                    offset + i * self.strides[dimension].__int__(),
                    summarize=summarize,
                )
            result += "\n" + String(" ") * (dimension) + "..."
            for i in range(n_items_outer - edge_outer, n_items_outer):
                result += "\n" + String(" ") * (dimension)
                result += self._array_to_string(
                    dimension + 1,
                    offset + i * self.strides[dimension].__int__(),
                    summarize=summarize,
                )
        result += "]"
        return result^

    fn __len__(self) -> Int:
        return Int(self._re.size)

    fn store[
        width: Int = 1
    ](mut self, index: Int, val: ComplexSIMD[cdtype]) raises:
        """
        Safely stores SIMD element of size `width` at `index`
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                IndexError(
                    message=String(
                        "Index {} out of range for array size {}."
                    ).format(index, self.size),
                    suggestion=String(
                        "Use 0 <= i < {} when storing; adjust index or reshape"
                        " array."
                    ).format(self.size),
                    location=String("ComplexNDArray.store(index: Int)"),
                )
            )

        self._re._buf.ptr.store(index, val.re)
        self._im._buf.ptr.store(index, val.im)

    fn store[
        width: Int = 1
    ](mut self, *indices: Int, val: ComplexSIMD[cdtype]) raises:
        """
        Safely stores SIMD element of size `width` at given variadic indices
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if len(indices) != self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Expected {} indices (ndim) but received {}."
                    ).format(self.ndim, len(indices)),
                    suggestion=String(
                        "Provide one index per dimension for shape {}."
                    ).format(self.shape),
                    location=String("ComplexNDArray.store(*indices)"),
                )
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    IndexError(
                        message=String(
                            "Index {} out of range for dim {} (size {})."
                        ).format(indices[i], i, self.shape[i]),
                        suggestion=String(
                            "Valid range for dim {} is [0, {})."
                        ).format(i, self.shape[i]),
                        location=String("ComplexNDArray.store(*indices)"),
                    )
                )

        var idx: Int = _get_offset(indices, self.strides)
        self._re._buf.ptr.store(idx, val.re)
        self._im._buf.ptr.store(idx, val.im)

    fn reshape(self, shape: NDArrayShape, order: String = "C") raises -> Self:
        """
        Returns an array of the same data with a new shape.

        Args:
            shape: Shape of returned array.
            order: Order of the array - Row major `C` or Column major `F`.

        Returns:
            Array of the same data with a new shape.
        """
        var result: Self = ComplexNDArray[cdtype](
            re=numojo.reshape(self._re.copy(), shape=shape, order=order),
            im=numojo.reshape(self._im.copy(), shape=shape, order=order),
        )
        result._re.flags = self._re.flags
        result._im.flags = self._im.flags
        return result^

    fn __iter__(
        self,
    ) raises -> _ComplexNDArrayIter[origin_of(self._re), cdtype]:
        """
        Iterates over elements of the ComplexNDArray and return sub-arrays as view.

        Returns:
            An iterator of ComplexNDArray elements.
        """

        return _ComplexNDArrayIter[origin_of(self._re), cdtype](
            self,
            dimension=0,
        )

    fn __reversed__(
        self,
    ) raises -> _ComplexNDArrayIter[origin_of(self._re), cdtype, forward=False]:
        """
        Iterates backwards over elements of the ComplexNDArray, returning
        copied value.

        Returns:
            A reversed iterator of NDArray elements.
        """

        return _ComplexNDArrayIter[origin_of(self._re), cdtype, forward=False](
            self,
            dimension=0,
        )

    fn itemset(
        mut self,
        index: Variant[Int, List[Int]],
        item: ComplexSIMD[cdtype],
    ) raises:
        """Set the scalar at the coordinates.

        Args:
            index: The coordinates of the item.
                Can either be `Int` or `List[Int]`.
                If `Int` is passed, it is the index of i-th item of the whole array.
                If `List[Int]` is passed, it is the coordinate of the item.
            item: The scalar to be set.
        """

        # If one index is given
        if index.isa[Int]():
            var idx: Int = index[Int]
            if idx < self.size:
                if self.flags[
                    "F_CONTIGUOUS"
                ]:  # column-major should be converted to row-major
                    # The following code can be taken out as a function that
                    # convert any index to coordinates according to the order
                    var c_stride = NDArrayStrides(shape=self.shape)
                    var c_coordinates = List[Int]()
                    for i in range(c_stride.ndim):
                        var coordinate = idx // c_stride[i]
                        idx = idx - c_stride[i] * coordinate
                        c_coordinates.append(coordinate)
                    self._re._buf.ptr.store(
                        _get_offset(c_coordinates, self.strides), item.re
                    )
                    self._im._buf.ptr.store(
                        _get_offset(c_coordinates, self.strides), item.im
                    )
                else:
                    self._re._buf.ptr.store(idx, item.re)
                    self._im._buf.ptr.store(idx, item.im)
            else:
                raise Error(
                    IndexError(
                        message=String(
                            "Linear index {} out of range for size {}."
                        ).format(idx, self.size),
                        suggestion=String(
                            "Valid linear indices: 0..{}."
                        ).format(self.size - 1),
                        location=String("ComplexNDArray.itemset(Int)"),
                    )
                )

        elif index.isa[List[Int]]():
            var indices: List[Int] = index[List[Int]].copy()
            if indices.__len__() != self.ndim:
                raise Error(
                    IndexError(
                        message=String(
                            "Expected {} indices (ndim) but received {}."
                        ).format(self.ndim, indices.__len__()),
                        suggestion=String(
                            "Provide one index per dimension; shape {} has {}"
                            " dimensions."
                        ).format(self.shape, self.ndim),
                        location=String("ComplexNDArray.itemset(List[Int])"),
                    )
                )
            for i in range(indices.__len__()):
                if indices[i] >= self.shape[i]:
                    raise Error(
                        IndexError(
                            message=String(
                                "Index {} out of range for dim {} (size {})."
                            ).format(indices[i], i, self.shape[i]),
                            suggestion=String("Valid range: [0, {}).").format(
                                self.shape[i]
                            ),
                            location=String(
                                "ComplexNDArray.itemset(List[Int])"
                            ),
                        )
                    )
            self._re._buf.ptr.store(_get_offset(indices, self.strides), item.re)
            self._im._buf.ptr.store(_get_offset(indices, self.strides), item.im)

    fn conj(self) raises -> Self:
        """
        Return the complex conjugate of the ComplexNDArray.
        """
        return Self(self._re.copy(), -self._im.copy())

    fn to_ndarray(
        self, type: String = "re"
    ) raises -> NDArray[dtype = Self.dtype]:
        if type == "re":
            var result: NDArray[dtype = Self.dtype] = NDArray[
                dtype = Self.dtype
            ](self.shape)
            memcpy(dest=result._buf.ptr, src=self._re._buf.ptr, count=self.size)
            return result^
        elif type == "im":
            var result: NDArray[dtype = Self.dtype] = NDArray[
                dtype = Self.dtype
            ](self.shape)
            memcpy(dest=result._buf.ptr, src=self._im._buf.ptr, count=self.size)
            return result^
        else:
            raise Error(
                ValueError(
                    message=String(
                        "Invalid component selector '{}' (expected 're' or"
                        " 'im')."
                    ).format(type),
                    suggestion=String(
                        "Call to_ndarray('re') for real part or"
                        " to_ndarray('im') for imaginary part."
                    ),
                    location=String("ComplexNDArray.to_ndarray"),
                )
            )

    fn squeeze(mut self, axis: Int) raises:
        """
        Remove (squeeze) a single dimension of size 1 from the array shape.

        Args:
            axis: The axis to squeeze. Supports negative indices.

        Raises:
            IndexError: If the axis is out of range.
            ShapeError: If the dimension at the given axis is not of size 1.
        """
        var normalized_axis: Int = axis
        if normalized_axis < 0:
            normalized_axis += self.ndim
        if (normalized_axis < 0) or (normalized_axis >= self.ndim):
            raise Error(
                IndexError(
                    message=String(
                        "Axis {} is out of range for array with {} dimensions."
                    ).format(axis, self.ndim),
                    suggestion=String(
                        "Use an axis value in the range [-{}, {})."
                    ).format(self.ndim, self.ndim),
                    location=String("NDArray.squeeze(axis: Int)"),
                )
            )

        if self.shape[normalized_axis] != 1:
            raise Error(
                ShapeError(
                    message=String(
                        "Cannot squeeze axis {} with size {}."
                    ).format(normalized_axis, self.shape[normalized_axis]),
                    suggestion=String(
                        "Only axes with length 1 can be removed."
                    ),
                    location=String("NDArray.squeeze(axis: Int)"),
                )
            )
        self.shape = self.shape._pop(normalized_axis)
        self.strides = self.strides._pop(normalized_axis)
        self.ndim -= 1

    # ===-------------------------------------------------------------------===#
    # Statistical and Reduction Methods
    # ===-------------------------------------------------------------------===#

    fn all(self) raises -> Bool:
        """
        Check if all complex elements are non-zero.

        A complex number is considered "true" if either its real or imaginary
        part is non-zero.

        Returns:
            True if all elements are non-zero, False otherwise.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        # Fill with non-zero values
        var result = A.all()  # True if all non-zero
        ```
        """
        var re_nonzero = True
        var im_nonzero = True

        for i in range(self.size):
            var re_val = self._re._buf.ptr.load(i)
            var im_val = self._im._buf.ptr.load(i)
            if (re_val == 0.0) and (im_val == 0.0):
                return False
        return True

    fn any(self) raises -> Bool:
        """
        Check if any complex element is non-zero.

        A complex number is considered "true" if either its real or imaginary
        part is non-zero.

        Returns:
            True if any element is non-zero, False otherwise.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        # Fill with some values
        var result = A.any()  # True if any non-zero
        ```
        """
        for i in range(self.size):
            var re_val = self._re._buf.ptr.load(i)
            var im_val = self._im._buf.ptr.load(i)
            if (re_val != 0.0) or (im_val != 0.0):
                return True
        return False

    fn sum(self) raises -> ComplexSIMD[cdtype]:
        """
        Sum of all complex array elements.

        Returns:
            Complex scalar containing the sum of all elements.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var total = A.sum()  # Sum of all elements
        ```
        """
        var sum_re = Scalar[Self.dtype](0)
        var sum_im = Scalar[Self.dtype](0)

        for i in range(self.size):
            sum_re += self._re._buf.ptr.load(i)
            sum_im += self._im._buf.ptr.load(i)

        return ComplexSIMD[cdtype](sum_re, sum_im)

    fn prod(self) raises -> ComplexSIMD[cdtype]:
        """
        Product of all complex array elements.

        Returns:
            Complex scalar containing the product of all elements.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var product = A.prod()  # Product of all elements
        ```
        """
        var prod_re = Scalar[Self.dtype](1)
        var prod_im = Scalar[Self.dtype](0)

        for i in range(self.size):
            var a_re = self._re._buf.ptr.load(i)
            var a_im = self._im._buf.ptr.load(i)
            var new_re = prod_re * a_re - prod_im * a_im
            var new_im = prod_re * a_im + prod_im * a_re
            prod_re = new_re
            prod_im = new_im

        return ComplexSIMD[cdtype](prod_re, prod_im)

    fn mean(self) raises -> ComplexSIMD[cdtype]:
        """
        Mean (average) of all complex array elements.

        Returns:
            Complex scalar containing the mean of all elements.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var average = A.mean()  # Mean of all elements
        ```
        """
        var total = self.sum()
        var n = Scalar[Self.dtype](self.size)
        return ComplexSIMD[cdtype](total.re / n, total.im / n)

    fn max(self) raises -> ComplexSIMD[cdtype]:
        """
        Find the complex element with maximum magnitude.

        Returns:
            The complex element with the largest magnitude.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var max_elem = A.max()  # Element with largest magnitude
        ```

        Notes:
            Returns the element with maximum |z| = sqrt(re^2 + im^2).
        """
        if self.size == 0:
            raise Error("Cannot find max of empty array")

        var max_mag_sq = self._re._buf.ptr.load(0) * self._re._buf.ptr.load(
            0
        ) + self._im._buf.ptr.load(0) * self._im._buf.ptr.load(0)
        var max_idx = 0

        for i in range(1, self.size):
            var re_val = self._re._buf.ptr.load(i)
            var im_val = self._im._buf.ptr.load(i)
            var mag_sq = re_val * re_val + im_val * im_val
            if mag_sq > max_mag_sq:
                max_mag_sq = mag_sq
                max_idx = i

        return ComplexSIMD[cdtype](
            self._re._buf.ptr.load(max_idx), self._im._buf.ptr.load(max_idx)
        )

    fn min(self) raises -> ComplexSIMD[cdtype]:
        """
        Find the complex element with minimum magnitude.

        Returns:
            The complex element with the smallest magnitude.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var min_elem = A.min()  # Element with smallest magnitude
        ```

        Notes:
            Returns the element with minimum |z| = sqrt(re^2 + im^2).
        """
        if self.size == 0:
            raise Error("Cannot find min of empty array")

        var min_mag_sq = self._re._buf.ptr.load(0) * self._re._buf.ptr.load(
            0
        ) + self._im._buf.ptr.load(0) * self._im._buf.ptr.load(0)
        var min_idx = 0

        for i in range(1, self.size):
            var re_val = self._re._buf.ptr.load(i)
            var im_val = self._im._buf.ptr.load(i)
            var mag_sq = re_val * re_val + im_val * im_val
            if mag_sq < min_mag_sq:
                min_mag_sq = mag_sq
                min_idx = i

        return ComplexSIMD[cdtype](
            self._re._buf.ptr.load(min_idx), self._im._buf.ptr.load(min_idx)
        )

    fn argmax(self) raises -> Int:
        """
        Return the index of the element with maximum magnitude.

        Returns:
            Index (flattened) of the element with largest magnitude.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var idx = A.argmax()  # Index of element with largest magnitude
        ```

        Notes:
            Compares by magnitude: |z| = sqrt(re^2 + im^2).
        """
        if self.size == 0:
            raise Error("Cannot find argmax of empty array")

        var max_mag_sq = self._re._buf.ptr.load(0) * self._re._buf.ptr.load(
            0
        ) + self._im._buf.ptr.load(0) * self._im._buf.ptr.load(0)
        var max_idx = 0

        for i in range(1, self.size):
            var re_val = self._re._buf.ptr.load(i)
            var im_val = self._im._buf.ptr.load(i)
            var mag_sq = re_val * re_val + im_val * im_val
            if mag_sq > max_mag_sq:
                max_mag_sq = mag_sq
                max_idx = i

        return max_idx

    fn argmin(self) raises -> Int:
        """
        Return the index of the element with minimum magnitude.

        Returns:
            Index (flattened) of the element with smallest magnitude.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var idx = A.argmin()  # Index of element with smallest magnitude
        ```

        Notes:
            Compares by magnitude: |z| = sqrt(re^2 + im^2).
        """
        if self.size == 0:
            raise Error("Cannot find argmin of empty array")

        var min_mag_sq = self._re._buf.ptr.load(0) * self._re._buf.ptr.load(
            0
        ) + self._im._buf.ptr.load(0) * self._im._buf.ptr.load(0)
        var min_idx = 0

        for i in range(1, self.size):
            var re_val = self._re._buf.ptr.load(i)
            var im_val = self._im._buf.ptr.load(i)
            var mag_sq = re_val * re_val + im_val * im_val
            if mag_sq < min_mag_sq:
                min_mag_sq = mag_sq
                min_idx = i

        return min_idx

    fn cumsum(self) raises -> Self:
        """
        Cumulative sum of complex array elements.

        Returns:
            ComplexNDArray with cumulative sums.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(5))
        var cumulative = A.cumsum()
        ```

        Notes:
            For array [a, b, c, d], returns [a, a+b, a+b+c, a+b+c+d].
        """
        var result = Self(self.shape)
        var cum_re = Scalar[Self.dtype](0)
        var cum_im = Scalar[Self.dtype](0)

        for i in range(self.size):
            cum_re += self._re._buf.ptr.load(i)
            cum_im += self._im._buf.ptr.load(i)
            result._re._buf.ptr.store(i, cum_re)
            result._im._buf.ptr.store(i, cum_im)

        return result^

    fn cumprod(self) raises -> Self:
        """
        Cumulative product of complex array elements.

        Returns:
            ComplexNDArray with cumulative products.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(5))
        var cumulative = A.cumprod()
        ```

        Notes:
            For array [a, b, c, d], returns [a, a*b, a*b*c, a*b*c*d].
        """
        var result = Self(self.shape)
        var cum_re = Scalar[Self.dtype](1)
        var cum_im = Scalar[Self.dtype](0)

        for i in range(self.size):
            var a_re = self._re._buf.ptr.load(i)
            var a_im = self._im._buf.ptr.load(i)
            var new_re = cum_re * a_re - cum_im * a_im
            var new_im = cum_re * a_im + cum_im * a_re
            cum_re = new_re
            cum_im = new_im
            result._re._buf.ptr.store(i, cum_re)
            result._im._buf.ptr.store(i, cum_im)

        return result^

    # ===-------------------------------------------------------------------===#
    # Array Manipulation Methods
    # ===-------------------------------------------------------------------===#

    fn flatten(self, order: String = "C") raises -> Self:
        """
        Return a copy of the array collapsed into one dimension.

        Args:
            order: Order of flattening - 'C' for row-major or 'F' for column-major.

        Returns:
            A 1D ComplexNDArray containing all elements.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 4))
        var flat = A.flatten()  # Shape(12)
        ```
        """
        var flat_re = self._re.flatten(order)
        var flat_im = self._im.flatten(order)
        return Self(flat_re^, flat_im^)

    fn fill(mut self, val: ComplexSIMD[cdtype]):
        """
        Fill all items of array with a complex value.

        Args:
            val: Complex value to fill the array with.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        A.fill(nm.ComplexSIMD[nm.cf64](1.0, 2.0))  # Fill with 1+2i
        ```
        """
        self._re.fill(val.re)
        self._im.fill(val.im)

    fn row(self, id: Int) raises -> Self:
        """
        Get the ith row of the matrix.

        Args:
            id: The row index.

        Returns:
            The ith row as a ComplexNDArray.

        Raises:
            Error: If ndim is greater than 2.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 4))
        var first_row = A.row(0)  # Get first row
        ```
        """
        if self.ndim > 2:
            raise Error(
                ShapeError(
                    message=String(
                        "Cannot extract row from array with {} dimensions."
                    ).format(self.ndim),
                    suggestion=String(
                        "The row() method only works with 1D or 2D arrays."
                    ),
                    location=String("ComplexNDArray.row(id: Int)"),
                )
            )

        var width: Int = self.shape[1]
        var result = Self(Shape(width))
        for i in range(width):
            var idx = i + id * width
            result._re._buf.ptr.store(i, self._re._buf.ptr.load(idx))
            result._im._buf.ptr.store(i, self._im._buf.ptr.load(idx))
        return result^

    fn col(self, id: Int) raises -> Self:
        """
        Get the ith column of the matrix.

        Args:
            id: The column index.

        Returns:
            The ith column as a ComplexNDArray.

        Raises:
            Error: If ndim is greater than 2.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 4))
        var first_col = A.col(0)  # Get first column
        ```
        """
        if self.ndim > 2:
            raise Error(
                ShapeError(
                    message=String(
                        "Cannot extract column from array with {} dimensions."
                    ).format(self.ndim),
                    suggestion=String(
                        "The col() method only works with 1D or 2D arrays."
                    ),
                    location=String("ComplexNDArray.col(id: Int)"),
                )
            )

        var width: Int = self.shape[1]
        var height: Int = self.shape[0]
        var result = Self(Shape(height))
        for i in range(height):
            var idx = id + i * width
            result._re._buf.ptr.store(i, self._re._buf.ptr.load(idx))
            result._im._buf.ptr.store(i, self._im._buf.ptr.load(idx))
        return result^

    fn clip(
        self, a_min: Scalar[Self.dtype], a_max: Scalar[Self.dtype]
    ) raises -> Self:
        """
        Limit the magnitudes of complex values between [a_min, a_max].

        Elements with magnitude less than a_min are scaled to have magnitude a_min.
        Elements with magnitude greater than a_max are scaled to have magnitude a_max.
        The phase (angle) of each complex number is preserved.

        Args:
            a_min: The minimum magnitude.
            a_max: The maximum magnitude.

        Returns:
            A ComplexNDArray with clipped magnitudes.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(10))
        var clipped = A.clip(1.0, 5.0)  # Clip magnitudes to [1, 5]
        ```

        Notes:
            Clips by magnitude while preserving phase angle.
        """
        var result = Self(self.shape)

        for i in range(self.size):
            var re = self._re._buf.ptr.load(i)
            var im = self._im._buf.ptr.load(i)
            var mag_sq = re * re + im * im
            var mag_val = sqrt(mag_sq)

            if mag_val < a_min:
                if mag_val > 0:
                    var scale = a_min / mag_val
                    result._re._buf.ptr.store(i, re * scale)
                    result._im._buf.ptr.store(i, im * scale)
                else:
                    result._re._buf.ptr.store(i, a_min)
                    result._im._buf.ptr.store(i, 0.0)
            elif mag_val > a_max:
                var scale = a_max / mag_val
                result._re._buf.ptr.store(i, re * scale)
                result._im._buf.ptr.store(i, im * scale)
            else:
                result._re._buf.ptr.store(i, re)
                result._im._buf.ptr.store(i, im)

        return result^

    fn round(self) raises -> Self:
        """
        Round the real and imaginary parts of each element to the nearest integer.

        Returns:
            A ComplexNDArray with rounded components.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(10))
        # A contains e.g. 1.7+2.3i
        var rounded = A.round()  # Returns 2.0+2.0i
        ```
        """
        var rounded_re = rounding.tround[Self.dtype](self._re)
        var rounded_im = rounding.tround[Self.dtype](self._im)
        return Self(rounded_re^, rounded_im^)

    fn T(self) raises -> Self:
        """
        Transpose the complex array (reverse all axes).

        Returns:
            Transposed ComplexNDArray.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 4))
        var A_T = A.T()  # Shape(4, 3)
        ```
        """
        var transposed_re = self._re.T()
        var transposed_im = self._im.T()
        return Self(transposed_re^, transposed_im^)

    fn T(self, axes: List[Int]) raises -> Self:
        """
        Transpose the complex array according to the given axes permutation.

        Args:
            axes: Permutation of axes (e.g., [1, 0, 2]).

        Returns:
            Transposed ComplexNDArray.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 3, 4))
        var A_T = A.T(List[Int](2, 0, 1))  # Shape(4, 2, 3)
        ```
        """
        var transposed_re = self._re.T(axes)
        var transposed_im = self._im.T(axes)
        return Self(transposed_re^, transposed_im^)

    fn diagonal(self, offset: Int = 0) raises -> Self:
        """
        Extract the diagonal from a 2D complex array.

        Args:
            offset: Offset from the main diagonal (0 for main diagonal).

        Returns:
            1D ComplexNDArray containing the diagonal elements.

        Raises:
            Error: If array is not 2D.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(4, 4))
        var diag = A.diagonal()      # Main diagonal
        var upper = A.diagonal(1)    # First upper diagonal
        ```
        """
        if self.ndim != 2:
            raise Error(
                ShapeError(
                    message=String(
                        "diagonal() requires a 2D array, got {} dimensions."
                    ).format(self.ndim),
                    suggestion=String(
                        "Use a 2D ComplexNDArray for diagonal extraction."
                    ),
                    location=String("ComplexNDArray.diagonal()"),
                )
            )

        var diag_re = self[Self.dtype]._re.diagonal(offset)
        var diag_im = self[Self.dtype]._im.diagonal(offset)
        return Self(diag_re^, diag_im^)

    fn trace(self) raises -> ComplexSIMD[cdtype]:
        """
        Return the sum of the diagonal elements (trace of the matrix).

        Returns:
            Complex scalar containing the trace.

        Raises:
            Error: If array is not 2D.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 3))
        var tr = A.trace()  # Sum of diagonal elements
        ```
        """
        var diag = self.diagonal()
        return diag.sum()

    fn tolist(self) -> List[ComplexSIMD[cdtype]]:
        """
        Convert the complex array to a List of complex scalars.

        Returns:
            A List containing all complex elements in row-major order.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 3))
        var elements = A.tolist()  # List of 6 complex numbers
        ```
        """
        var result = List[ComplexSIMD[cdtype]](capacity=self.size)
        for i in range(self.size):
            result.append(
                ComplexSIMD[cdtype](
                    self._re._buf.ptr.load(i), self._im._buf.ptr.load(i)
                )
            )
        return result^

    fn num_elements(self) -> Int:
        """
        Return the total number of elements in the array.

        Returns:
            The size of the array (same as self.size).

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(3, 4, 5))
        print(A.num_elements())  # 60
        ```
        """
        return self.size

    fn resize(mut self, shape: NDArrayShape) raises:
        """
        Change shape and size of array in-place.

        If the new shape requires more elements, they are filled with zero.
        If the new shape requires fewer elements, the array is truncated.

        Args:
            shape: The new shape for the array.

        Examples:
        ```mojo
        import numojo as nm
        var A = nm.ComplexNDArray[nm.cf64](nm.Shape(2, 3))
        A.resize(nm.Shape(3, 4))  # Now 3x4, filled with zeros as needed
        ```

        Notes:
            This modifies the array in-place. To get a reshaped copy, use reshape().
        """
        self._re.resize(shape)
        self._im.resize(shape)
        self.shape = shape
        self.ndim = shape.ndim
        self.size = shape.size_of_array()
        var order = "C" if self.flags.C_CONTIGUOUS else "F"
        self.strides = NDArrayStrides(shape, order=order)


struct _ComplexNDArrayIter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
    cdtype: ComplexDType,
    forward: Bool = True,
](Copyable, Movable):
    # TODO:
    # Return a view instead of copy where possible
    # (when Bufferable is supported).
    """
    An iterator yielding `ndim-1` array slices over the given dimension.
    It is the default iterator of the `ComplexNDArray.__iter__() method and for loops.
    It can also be constructed using the `ComplexNDArray.iter_over_dimension()` method.
    It trys to create a view where possible.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        origin: The lifetime of the underlying NDArray data.
        cdtype: The complex data type of the item.
        forward: The iteration direction. `False` is backwards.
    """
    # The equivalent DType of the ComplexDType
    alias dtype: DType = cdtype._dtype

    # FIELDS
    var index: Int
    var re_ptr: LegacyUnsafePointer[Scalar[Self.dtype]]
    var im_ptr: LegacyUnsafePointer[Scalar[Self.dtype]]
    var dimension: Int
    var length: Int
    var shape: NDArrayShape
    var strides: NDArrayStrides
    """Strides of array or view. It is not necessarily compatible with shape."""
    var ndim: Int
    var size_of_item: Int

    fn __init__(
        out self, read a: ComplexNDArray[cdtype], read dimension: Int
    ) raises:
        """
        Initialize the iterator.

        Args:
            a: The array
            dimension: Dimension to iterate over.
        """

        if dimension < 0 or dimension >= a.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Axis {} out of valid range [0, {})."
                    ).format(dimension, a.ndim),
                    suggestion=String(
                        "Valid axes: 0..{}. Use {} for last axis of shape {}."
                    ).format(a.ndim - 1, a.ndim - 1, a.shape),
                    location=String("_ComplexNDArrayIter.__init__"),
                )
            )

        self.re_ptr = a._re._buf.ptr
        self.im_ptr = a._im._buf.ptr
        self.dimension = dimension
        self.shape = a.shape
        self.strides = a.strides
        self.ndim = a.ndim
        self.length = a.shape[dimension]
        self.size_of_item = a.size // a.shape[dimension]
        # Status of the iterator
        self.index = 0 if forward else a.shape[dimension] - 1

    fn __iter__(self) -> Self:
        return self.copy()

    fn __next__(mut self) raises -> ComplexNDArray[cdtype]:
        var result = ComplexNDArray[cdtype](self.shape._pop(self.dimension))
        var current_index = self.index

        @parameter
        if forward:
            self.index += 1
        else:
            self.index -= 1

        for offset in range(self.size_of_item):
            var remainder = offset
            var item: Item = Item(ndim=self.ndim)

            for i in range(self.ndim - 1, -1, -1):
                if i != self.dimension:
                    (item._buf + i).init_pointee_copy(remainder % self.shape[i])
                    remainder = remainder // self.shape[i]
                else:
                    (item._buf + self.dimension).init_pointee_copy(
                        current_index
                    )

            (result._re._buf.ptr + offset).init_pointee_copy(
                self.re_ptr[_get_offset(item, self.strides)]
            )
            (result._im._buf.ptr + offset).init_pointee_copy(
                self.im_ptr[_get_offset(item, self.strides)]
            )
        return result^

    @always_inline
    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index >= 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index

    fn ith(self, index: Int) raises -> ComplexNDArray[cdtype]:
        """
        Gets the i-th array of the iterator.

        Args:
            index: The index of the item. It must be non-negative.

        Returns:
            The i-th `ndim-1`-D array of the iterator.
        """

        if (index >= self.length) or (index < 0):
            raise Error(
                IndexError(
                    message=String(
                        "Iterator index {} out of range [0, {})."
                    ).format(index, self.length),
                    suggestion=String(
                        "Use ith(i) with 0 <= i < {} or iterate via for-loop."
                    ).format(self.length),
                    location=String("_ComplexNDArrayIter.ith"),
                )
            )

        if self.ndim > 1:
            var result = ComplexNDArray[cdtype](self.shape._pop(self.dimension))

            for offset in range(self.size_of_item):
                var remainder = offset
                var item: Item = Item(ndim=self.ndim)

                for i in range(self.ndim - 1, -1, -1):
                    if i != self.dimension:
                        (item._buf + i).init_pointee_copy(
                            remainder % self.shape[i]
                        )
                        remainder = remainder // self.shape[i]
                    else:
                        (item._buf + self.dimension).init_pointee_copy(index)

                (result._re._buf.ptr + offset).init_pointee_copy(
                    self.re_ptr[_get_offset(item, self.strides)]
                )
                (result._im._buf.ptr + offset).init_pointee_copy(
                    self.im_ptr[_get_offset(item, self.strides)]
                )
            return result^

        else:  # 0-D array
            var result = numojo.creation._0darray[cdtype](
                ComplexSIMD[cdtype](self.re_ptr[index], self.im_ptr[index])
            )
            return result^
