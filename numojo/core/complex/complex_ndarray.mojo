# ===----------------------------------------------------------------------=== #
# NuMojo: Complex NDArray
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
""""ComplexNDArray (numojo.core.complex.complex_ndarray)
----------------------------------------------------
Complex NDArray support for NuMojo.

This module provides the `ComplexNDArray` type, which represents N-dimensional arrays
of complex numbers. It includes lifecycle methods, indexing and slicing, operator
overloads, IO, trait, and iterator methods, as well as other utility functions.
"""
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
# === Stdlib ===
# ===----------------------------------------------------------------------===#
from std.algorithm import vectorize
from max.algorithm import parallelize
import std.builtin.bool as builtin_bool
import std.math as builtin_math
from std.collections.optional import Optional
from std.math import log10, sqrt
from std.memory import unsafe_memset_zero, unsafe_memcpy, UnsafePointer
from std.python import Python, PythonObject
from std.sys import simd_width_of
from std.utils import Variant

# ===----------------------------------------------------------------------===#
# === numojo core ===
# ===----------------------------------------------------------------------===#
from numojo.core.dtype.complex_dtype import ComplexDType, _concise_dtype_str
from numojo.core.layout.flags import Flags
from numojo.core.indexing.item import Item
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.layout.ndstrides import NDArrayStrides
from numojo.core.dtype.complex_dtype import ComplexDType, _concise_dtype_str
from numojo.core.layout.flags import Flags
from numojo.core.indexing.item import Item
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.layout.ndstrides import NDArrayStrides
from numojo.core.complex.complex_simd import ComplexSIMD
from numojo.core.type_aliases import ComplexScalar, CScalar
from numojo.core.memory.data_container import DataContainer
from numojo.core.indexing import (
    IndexMethods,
    TraverseMethods,
    to_numpy,
    bool_to_numeric,
)
from numojo.core.error import NumojoError

# ===----------------------------------------------------------------------===#
# === numojo routines (creation / io / logic) ===
# ===----------------------------------------------------------------------===#
import numojo.routines.creation as creation
from numojo.routines.io.formatting import (
    format_value,
    PrintOptions,
)
import numojo.routines.logic.comparison as comparison
import numojo.routines.logic.logical_ops as logical_ops

# ===----------------------------------------------------------------------===#
# === numojo routines (math / bitwise / searching) ===
# ===----------------------------------------------------------------------===#
import numojo.routines.bitwise as bitwise
import numojo.routines.math.arithmetic as arithmetic
import numojo.routines.math.rounding as rounding
import numojo.routines.math.trig as trig
import numojo.routines.math.exponents as exponents
import numojo.routines.math.misc as misc
import numojo.routines.searching as searching
from numojo.routines.manipulation import reshape
from numojo.core.ndarray import NDArray
from numojo.routines import math
from numojo.routines import linalg
from numojo.core.type_aliases import Shape


# ===----------------------------------------------------------------------=== #
# Implements N-Dimensional Complex Array
# ===----------------------------------------------------------------------=== #
struct ComplexNDArray[cdtype: ComplexDType = ComplexDType.float64](
    Copyable,
    FloatableRaising,
    IntableRaising,
    Movable,
    Sized,
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
    comptime dtype: DType = Self.cdtype.dtype
    """Corresponding real data type."""

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
    def __init__(
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
                NumojoError(
                    category="shape",
                    message=String(
                        "Real and imaginary array parts must have identical"
                        " shapes; got re={} vs im={}. Ensure both NDArray"
                        " arguments are created with the same shape before"
                        " constructing ComplexNDArray."
                    ).format(re.shape, im.shape),
                    location="ComplexNDArray.__init__(re, im)",
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
    def __init__(
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
    def __init__(
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
            var A = nm.ComplexNDArray[cf32]([2,3,4])
            ```

        Notes:
            This constructor should not be used by users directly. Use factory functions in `numojo.routines.creation` module instead.
        """
        self._re = NDArray[Self.dtype](NDArrayShape(shape), order)
        self._im = NDArray[Self.dtype](NDArrayShape(shape), order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    # TODO: Remove VariadicList versions.
    @always_inline("nodebug")
    def __init__(
        out self,
        shape: VariadicList[Int, _],
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
            var A = nm.ComplexNDArray[cf32](2,3,4)
            ```

        Notes:
            This constructor should not be used by users directly. Use factory functions in `numojo.routines.creation` module instead.
        """
        self._re = NDArray[Self.dtype](NDArrayShape(shape), order)
        self._im = NDArray[Self.dtype](NDArrayShape(shape), order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    def __init__(
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
            var shape = [2, 3]
            var offset = 0
            var strides = [3, 1]
            var arr = ComplexNDArray[cf32](shape, offset, strides)
            ```

        Notes:
            - This constructor is intended for advanced use cases requiring precise control over memory layout.
            - The resulting array is uninitialized and should be filled before use.
            - Both real and imaginary buffers are created with the same shape, offset, and strides.
        """
        self._re = NDArray[Self.dtype](
            shape=shape, offset=offset, strides=strides
        )
        self._im = NDArray[Self.dtype](
            shape=shape, offset=offset, strides=strides
        )
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    def __init__(
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
        self._re = NDArray[Self.dtype](
            shape=shape,
            strides=strides,
            offset=0,
            ndim=ndim,
            size=size,
            flags=flags,
        )
        self._im = NDArray[Self.dtype](
            shape=shape,
            strides=strides,
            offset=0,
            ndim=ndim,
            size=size,
            flags=flags,
        )
        self.print_options = PrintOptions(
            precision=2, edge_items=2, line_width=100, formatted_width=6
        )

    # FIXME: temporarily disabled this constructor until we setup views for NDArray.
    # def __init__(
    #     out self,
    #     shape: NDArrayShape,
    #     ref buffer_re: UnsafePointer[Scalar[Self.dtype]],
    #     ref buffer_im: UnsafePointer[Scalar[Self.dtype]],
    #     offset: Int,
    #     strides: NDArrayStrides,
    # ) raises:
    #     """
    #     Initialize a ComplexNDArray view with explicit shape, raw buffers, offset, and strides.

    #     This constructor creates a view over existing memory buffers for the real and imaginary parts,
    #     using the provided shape, offset, and stride information. It is intended for advanced or internal
    #     use cases where direct control over memory layout is required.

    #     ***Unsafe!*** This function is unsafe and should only be used internally. The caller is responsible
    #     for ensuring that the buffers are valid and that the shape, offset, and strides are consistent.

    #     Args:
    #         shape: NDArrayShape specifying the dimensions of the array.
    #         buffer_re: Unsafe pointer to the buffer containing the real part data.
    #         buffer_im: Unsafe pointer to the buffer containing the imaginary part data.
    #         offset: Integer offset into the buffers.
    #         strides: NDArrayStrides specifying the stride for each dimension.

    #     Notes:
    #         - No validation is performed on the buffers or metadata.
    #         - The resulting ComplexNDArray shares memory with the provided buffers.
    #         - Incorrect usage may lead to undefined behavior.
    #     """
    #     self._re = NDArray(shape, buffer_re, offset, strides)
    #     self._im = NDArray(shape, buffer_im, offset, strides)
    #     self.ndim = self._re.ndim
    #     self.shape = self._re.shape
    #     self.size = self._re.size
    #     self.strides = self._re.strides
    #     self.flags = self._re.flags
    #     self.print_options = PrintOptions(
    #         precision=2, edge_items=2, line_width=100, formatted_width=6
    #     )

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """
        Copy copy into self.
        """
        self._re = copy._re.copy()
        self._im = copy._im.copy()
        self.ndim = copy.ndim
        self.shape = copy.shape
        self.size = copy.size
        self.strides = copy.strides
        self.flags = copy.flags
        self.print_options = copy.print_options

    @always_inline("nodebug")
    def __init__(out self, *, deinit move: Self):
        """
        Move other into self.
        """
        self._re = move._re^
        self._im = move._im^
        self.ndim = move.ndim
        self.shape = move.shape
        self.size = move.size
        self.strides = move.strides
        self.flags = move.flags
        self.print_options = move.print_options

    @always_inline("nodebug")
    def __deinit__(deinit self):
        """Destroys array buffers."""
        _ = self._re^
        _ = self._im^

    def view(mut self) raises -> Self:
        """
        Create a non-owning view of the current ComplexNDArray.

        Returns:
            A new ComplexNDArray instance that shares the data buffers with
            `self` and does not allocate new memory.

        Examples:
            ```mojo
            import numojo as nm
            var arr = nm.ComplexNDArray[nm.cf32](nm.Shape(3, 4))
            var v = arr.view()  # Create a view into arr.
            ```
        """
        return ComplexNDArray[Self.cdtype](
            re=self._re.view(),
            im=self._im.view(),
        )

    @always_inline("nodebug")
    def _flat_offset(self, flat_index: Int) -> Int:
        """
        Return backing-buffer offset for logical C-order flat index.
        """
        var remainder = flat_index
        var offset = self._re.offset
        for dim in range(self.ndim - 1, -1, -1):
            var dim_size = Int(self.shape.unsafe_load(dim))
            var coord = remainder % dim_size
            remainder = remainder // dim_size
            offset += coord * Int(self.strides.unsafe_load(dim))
        return offset

    @always_inline("nodebug")
    def _flat_load(self, flat_index: Int) -> ComplexSIMD[Self.cdtype]:
        """Stride-safe logical flat load."""
        var off = self._flat_offset(flat_index)
        return ComplexSIMD[Self.cdtype](
            self._re._buf[off],
            self._im._buf[off],
        )

    @always_inline("nodebug")
    def _flat_store(mut self, flat_index: Int, value: ComplexSIMD[Self.cdtype]):
        """Stride-safe logical flat store."""
        var off = self._flat_offset(flat_index)
        self._re._buf[off] = value.re
        self._im._buf[off] = value.im

    @always_inline("nodebug")
    def _lex_less(
        self, a: ComplexSIMD[Self.cdtype], b: ComplexSIMD[Self.cdtype]
    ) -> Bool:
        return (a.re < b.re) or ((a.re == b.re) and (a.im < b.im))

    @always_inline("nodebug")
    def _lex_greater(
        self, a: ComplexSIMD[Self.cdtype], b: ComplexSIMD[Self.cdtype]
    ) -> Bool:
        return (a.re > b.re) or ((a.re == b.re) and (a.im > b.im))

    @always_inline("nodebug")
    def _normalize_axis(self, axis: Int) raises -> Int:
        var normalized_axis = axis
        if normalized_axis < 0:
            normalized_axis += self.ndim
        if (normalized_axis < 0) or (normalized_axis >= self.ndim):
            raise Error(
                String("Axis {} is out of bounds for ndim {}.").format(
                    axis, self.ndim
                )
            )
        return normalized_axis

    def _permute_axis_to_last(self, axis: Int) raises -> List[Int]:
        var normalized_axis = self._normalize_axis(axis)
        var axes = List[Int](capacity=self.ndim)
        for i in range(self.ndim):
            if i != normalized_axis:
                axes.append(i)
        axes.append(normalized_axis)
        return axes^

    def _inverse_permutation(self, axes: List[Int]) raises -> List[Int]:
        if len(axes) != self.ndim:
            raise Error("Invalid permutation length.")
        var inv = List[Int](capacity=self.ndim)
        for _ in range(self.ndim):
            inv.append(0)
        for i in range(self.ndim):
            inv[axes[i]] = i
        return inv^

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing
    # Getter dunders and other getter methods
    # FIXME: currently most of the getitem and setitem methods don't match exactly between NDArray and ComplexNDArray in it's implementation, docstring, argument mutability etc. Fix this.

    # 1. Basic Indexing Operations
    # def _getitem(self, *indices: Int) -> ComplexSIMD[Self.cdtype]                         # Direct unsafe getter
    # def _getitem(self, indices: List[Int]) -> ComplexSIMD[Self.cdtype]                         # Direct unsafe getter
    # def __getitem__(self) raises -> ComplexSIMD[Self.cdtype]                             # Get 0d array value
    # def __getitem__(self, index: Item) raises -> ComplexSIMD[Self.cdtype]                # Get by coordinate list
    #
    # 2. Single Index Slicing
    # def __getitem__(self, idx: Int) raises -> Self                             # Get by single index
    #
    # 3. Multi-dimensional Slicing
    # def __getitem__(self, *slices: Slice) raises -> Self                       # Get by variable slices
    # def __getitem__(self, slice_list: List[Slice]) raises -> Self              # Get by list of slices
    # def __getitem__(self, *slices: Variant[Slice, Int]) raises -> Self         # Get by mix of slices/ints
    #
    # 4. Advanced Indexing
    # def __getitem__(self, indices: NDArray[DType.int]) raises -> Self        # Get by index array
    # def __getitem__(self, indices: List[Int]) raises -> Self                   # Get by list of indices
    # def __getitem__(self, mask: NDArray[DType.bool]) raises -> Self            # Get by boolean mask
    # def __getitem__(self, mask: List[Bool]) raises -> Self                     # Get by boolean list
    #
    # 5. Low-level Access
    # def item(self, var index: Int) raises -> ComplexSIMD[Self.dtype]                   # Get item by linear index
    # def item(self, *index: Int) raises -> ComplexSIMD[Self.dtype]                        # Get item by coordinates
    # def load(self, var index: Int) raises -> ComplexSIMD[Self.dtype]                   # Load with bounds check
    # def load[width: Int](self, index: Int) raises -> ComplexSIMD[Self.dtype, width]        # Load SIMD value
    # def load[width: Int](self, *indices: Int) raises -> ComplexSIMD[Self.dtype, width]     # Load SIMD at coordinates
    # ===-------------------------------------------------------------------===#

    @always_inline
    def normalize(self, idx: Int, dim: Int) -> Int:
        """
        Normalize a potentially negative index to its positive equivalent
        within the bounds of the given dimension.

        Args:
            idx: The index to normalize. Can be negative to indicate indexing
                 from the end (e.g., -1 refers to the last element).
            dim: The size of the dimension to normalize against.

        Returns:
            The normalized index as a non-negative integer.
        """
        var idx_norm = idx
        if idx_norm < 0:
            idx_norm = dim + idx_norm
        return idx_norm

    def _getitem(self, *indices: Int) -> ComplexSIMD[Self.cdtype]:
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
            index_of_buffer += indices[i] * Int(self.strides.unsafe_load(i))
        return ComplexSIMD[Self.cdtype](
            re=self._re._buf[index_of_buffer],
            im=self._im._buf[index_of_buffer],
        )

    def _getitem(self, indices: List[Int]) -> ComplexScalar[Self.cdtype]:
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
            print(A._getitem([1,2,3]))
            ```

        Notes:
            This function is unsafe and should be used only on internal use.
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * Int(self.strides.unsafe_load(i))
        return ComplexSIMD[Self.cdtype](
            re=self._re._buf[index_of_buffer],
            im=self._im._buf[index_of_buffer],
        )

    def __getitem__(self) raises -> ComplexSIMD[Self.cdtype, 1]:
        """
        Gets the value of the 0-D Complex array.

        Returns:
            The value of the 0-D Complex array.

        Raises:
            Error: If the array is not 0-d.

        Examples:
            ```mojo
            import numojo as nm
            from numojo.prelude import *
            var a = nm.arange[cf32](CScalar[cf32](1))[0]
            print(a[]) # gets values of the 0-D complex array.
            ```
        """
        if self.ndim != 0:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Cannot read a scalar value from a non-0D"
                        " ComplexNDArray without indices. Use `A[]` only for 0D"
                        " arrays (scalars). For higher dimensions supply"
                        " indices, e.g. `A[i,j]`."
                    ),
                    location="ComplexNDArray.__getitem__()",
                )
            )
        return ComplexSIMD[Self.cdtype](
            re=self._re._buf.ptr[],
            im=self._im._buf.ptr[],
        )

    def __getitem__(self, index: Item) raises -> ComplexSIMD[Self.cdtype, 1]:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices (ndim) but received {}. Provide"
                        " one index per dimension for shape {}."
                    ).format(self.ndim, index.__len__(), self.shape),
                    location="ComplexNDArray.__getitem__(index: Item)",
                )
            )

        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for dimension {} (size {})."
                            " Valid indices for this dimension are in [0, {})."
                        ).format(index[i], i, self.shape[i], self.shape[i]),
                        location="ComplexNDArray.__getitem__(index: Item)",
                    )
                )

        var idx: Int = IndexMethods.get_1d_index(index, self.strides)
        return ComplexSIMD[Self.cdtype](
            re=self._re._buf.load[width=1](idx),
            im=self._im._buf.load[width=1](idx),
        )

    def __getitem__(self, idx: Int) raises -> Self:
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
                NumojoError(
                    category="index",
                    message=(
                        "Cannot slice a 0D ComplexNDArray (scalar). Use `A[]`"
                        " or `A.item(0)` to read its value."
                    ),
                    location="ComplexNDArray.__getitem__(idx: Int)",
                )
            )

        var norm = idx
        if norm < 0:
            norm += self.shape[0]
        if (norm < 0) or (norm >= self.shape[0]):
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} out of bounds for axis 0 (size {}). Valid"
                        " indices: 0 <= i < {} or -{} <= i < 0 (negative wrap)."
                    ).format(idx, self.shape[0], self.shape[0], self.shape[0]),
                    location="ComplexNDArray.__getitem__(idx: Int)",
                )
            )

        # 1-D -> complex scalar (0-D ComplexNDArray wrapper)
        if self.ndim == 1:
            return creation._0darray[Self.cdtype](
                ComplexSIMD[Self.cdtype](
                    re=self._re._buf[norm],
                    im=self._im._buf[norm],
                )
            )

        var out_shape: NDArrayShape = self.shape[1:]
        var alloc_order: String = String("C")
        if self.flags.F_CONTIGUOUS:
            alloc_order = String("F")
        var result: ComplexNDArray[Self.cdtype] = ComplexNDArray[Self.cdtype](
            shape=out_shape, order=alloc_order
        )

        # Fast path for C-contiguous
        if self.flags.C_CONTIGUOUS:
            var block: Int = self.size // self.shape[0]
            unsafe_memcpy(
                dest=result._re._buf.ptr,
                src=self._re._buf.ptr.unsafe_offset(norm * block),
                count=block,
            )
            unsafe_memcpy(
                dest=result._im._buf.ptr,
                src=self._im._buf.ptr.unsafe_offset(norm * block),
                count=block,
            )
            return result^
        else:
            # F layout
            self._re._copy_first_axis_slice(self._re, norm, result._re)
            self._im._copy_first_axis_slice(self._im, norm, result._im)
            return result^

    def __getitem__(self, var *slices: Slice) raises -> Self:
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

            var a = nm.arange(10).reshape(nm.Shape(2, 5))
            var b = a[:, 2:4]
            print(b) # Output: 2x2 sliced array corresponding to columns 2 and 3 of each row.
            ```
        """
        var n_slices: Int = len(slices)
        if n_slices > self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Too many slices provided: expected at most {} but got"
                        " {}. Provide at most {} slices for an array with {}"
                        " dimensions."
                    ).format(self.ndim, n_slices, self.ndim, self.ndim),
                    location="ComplexNDArray.__getitem__(slices: Slice)",
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

    def _calculate_strides(self, shape: List[Int]) -> List[Int]:
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

    def __getitem__(self, var slice_list: List[Slice]) raises -> Self:
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
            var b = a[[Slice(0, 2, 1), Slice(2, 4, 1)]]  # Equivalent to arr[:, 2:4], returns a 2x2 sliced array.
            print(b)
            ```
        """
        var n_slices: Int = slice_list.__len__()
        # Check error cases
        # I think we can remove this since it seems redundant.
        if n_slices == 0:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Empty slice list provided to"
                        " ComplexNDArray.__getitem__. Provide a List with at"
                        " least one slice to index the array."
                    ),
                    location=(
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
        var narr = ComplexNDArray[Self.cdtype](
            offset=noffset, shape=nshape, strides=nstrides
        )
        # TODO: combine the two traverses into one.
        var index_re: List[Int] = List[Int](length=ndims, fill=0)
        TraverseMethods.traverse_iterative[Self.dtype](
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
        TraverseMethods.traverse_iterative[Self.dtype](
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

    def __getitem__(self, var *slices: Variant[Slice, Int]) raises -> Self:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Too many indices or slices: received {} but array has"
                        " only {} dimensions. Pass at most {} indices/slices"
                        " (one per dimension)."
                    ).format(n_slices, self.ndim, self.ndim),
                    location=(
                        "ComplexNDArray.__getitem__(*slices: Variant[Slice,"
                        " Int])"
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
                        NumojoError(
                            category="index",
                            message=String(
                                "Integer index {} out of bounds for axis {}"
                                " (size {}). Valid indices: 0 <= i < {} or"
                                " negative -{} <= i < 0 (negative indices wrap"
                                " from the end)."
                            ).format(
                                slices[i][Int],
                                i,
                                self.shape[i],
                                self.shape[i],
                                self.shape[i],
                            ),
                            location=(
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
            narr = creation._0darray[Self.cdtype](self._getitem(indices))
            return narr^

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        narr = self.__getitem__(slice_list^)
        return narr^

    def __getitem__(self, indices: NDArray[DType.int]) raises -> Self:
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
        var shape = indices.shape.join(self.shape.pop(0))

        var result: ComplexNDArray[Self.cdtype] = ComplexNDArray[Self.cdtype](
            shape
        )
        var size_per_item: Int = self.size // self.shape[0]

        # Fill in the values
        for i in range(indices.size):
            if indices.item(i) >= Scalar[DType.int](self.shape[0]):
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} (value {}) out of range for first"
                            " dimension size {}. Ensure each index < {}."
                            " Consider clipping or validating indices before"
                            " indexing."
                        ).format(
                            i, indices.item(i), self.shape[0], self.shape[0]
                        ),
                        location=(
                            "ComplexNDArray.__getitem__(indices:"
                            " NDArray[DType.int])"
                        ),
                    )
                )
            unsafe_memcpy(
                dest=result._re._buf.ptr.unsafe_offset(i * size_per_item),
                src=self._re._buf.ptr.unsafe_offset(
                    indices.item(i) * Scalar[DType.int](size_per_item)
                ),
                count=size_per_item,
            )
            unsafe_memcpy(
                dest=result._im._buf.ptr.unsafe_offset(i * size_per_item),
                src=self._im._buf.ptr.unsafe_offset(
                    indices.item(i) * Scalar[DType.int](size_per_item)
                ),
                count=size_per_item,
            )

        return result^

    def __getitem__(self, indices: List[Int]) raises -> Self:
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
            (indices_array._buf.ptr.unsafe_offset(i)).unsafe_write(
                Scalar[DType.int](indices[i])
            )

        return self[indices_array]

    def __getitem__(self, mask: NDArray[DType.bool]) raises -> Self:
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
            var result = ComplexNDArray[Self.cdtype](
                shape=NDArrayShape(len_of_result)
            )

            # Fill in the values
            var offset = 0
            for i in range(mask.size):
                if mask.item(i):
                    (result._re._buf.ptr.unsafe_offset(offset)).unsafe_write(
                        self._re._buf[i]
                    )
                    (result._im._buf.ptr.unsafe_offset(offset)).unsafe_write(
                        self._im._buf[i]
                    )
                    offset += 1

            return result^

        # CASE 2:
        # if array shape is not equal to mask shape,
        # return items from the 0-th dimension of the array where mask is True
        if mask.ndim > 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Boolean mask must be 1-D or match full array shape;"
                        " got ndim={} for mask shape {}. Use a 1-D mask of"
                        " length {} for first-dimension filtering or a"
                        " full-shape mask {} for element-wise selection."
                    ).format(mask.ndim, mask.shape, self.shape[0], self.shape),
                    location="ComplexNDArray.__getitem__(mask: NDArray[bool])",
                )
            )

        if mask.shape[0] != self.shape[0]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Mask length {} does not match first dimension size {}."
                        " Provide mask of length {} to filter along first"
                        " dimension."
                    ).format(mask.shape[0], self.shape[0], self.shape[0]),
                    location="ComplexNDArray.__getitem__(mask: NDArray[bool])",
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

        var result = ComplexNDArray[Self.cdtype](shape)
        var size_per_item = self.size // self.shape[0]

        # Fill in the values
        var offset = 0
        for i in range(mask.size):
            if mask.item(i):
                unsafe_memcpy(
                    dest=result._re._buf.ptr.unsafe_offset(
                        offset * size_per_item
                    ),
                    src=self._re._buf.ptr.unsafe_offset(i * size_per_item),
                    count=size_per_item,
                )
                unsafe_memcpy(
                    dest=result._im._buf.ptr.unsafe_offset(
                        offset * size_per_item
                    ),
                    src=self._im._buf.ptr.unsafe_offset(i * size_per_item),
                    count=size_per_item,
                )
                offset += 1

        return result^

    def __getitem__(self, mask: List[Bool]) raises -> Self:
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
            (mask_array._buf.ptr.unsafe_offset(i)).unsafe_write(mask[i])

        return self[mask_array]

    def item(self, var index: Int) raises -> ComplexSIMD[Self.cdtype]:
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
                NumojoError(
                    category="index",
                    message=(
                        "Cannot index into a 0D ComplexNDArray with a linear"
                        " position. Call item() with no arguments or use A[] to"
                        " read scalar."
                    ),
                    location="ComplexNDArray.item(index: Int)",
                )
            )

        index = self.normalize(index, self.size)

        if (index < 0) or (index >= self.size):
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Linear index {} out of range for array size {}. Valid"
                        " linear indices: 0..{} (inclusive). Use negative"
                        " indices only where supported."
                    ).format(index, self.size, self.size - 1),
                    location="ComplexNDArray.item(index: Int)",
                )
            )

        if self.flags.F_CONTIGUOUS:
            return ComplexSIMD[Self.cdtype](
                re=(
                    self._re._buf.ptr.unsafe_offset(
                        IndexMethods.transfer_offset(index, self.strides)
                    )
                )[],
                im=(
                    self._im._buf.ptr.unsafe_offset(
                        IndexMethods.transfer_offset(index, self.strides)
                    )
                )[],
            )

        else:
            return ComplexSIMD[Self.cdtype](
                re=(self._re._buf.ptr.unsafe_offset(index))[],
                im=(self._im._buf.ptr.unsafe_offset(index))[],
            )

    def item(self, *index: Int) raises -> ComplexSIMD[Self.cdtype]:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices (ndim) but got {}. Provide one"
                        " coordinate per dimension for shape {}."
                    ).format(self.ndim, len(index), self.shape),
                    location="ComplexNDArray.item(*index: Int)",
                )
            )

        if self.ndim == 0:
            return ComplexSIMD[Self.cdtype](
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
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for dimension {} (size {})."
                            " Valid range is [0, {}). Consider adjusting or"
                            " clipping."
                        ).format(
                            list_index[i], i, self.shape[i], self.shape[i]
                        ),
                        location="ComplexNDArray.item(*index: Int)",
                    )
                )
        return ComplexSIMD[Self.cdtype](
            re=(
                self._re._buf.ptr.unsafe_offset(
                    IndexMethods.get_1d_index(index, self.strides)
                )
            )[],
            im=(
                self._im._buf.ptr.unsafe_offset(
                    IndexMethods.get_1d_index(index, self.strides)
                )
            )[],
        )

    def load(self, var index: Int) raises -> ComplexSIMD[Self.cdtype]:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} out of range for size {}. Use 0 <= i < {}."
                        " Adjust negatives manually; negative indices are not"
                        " supported here."
                    ).format(index, self.size, self.size),
                    location="ComplexNDArray.load(index: Int)",
                )
            )

        return ComplexSIMD[Self.cdtype](
            re=self._re._buf[index],
            im=self._im._buf[index],
        )

    def load[
        width: Int = 1
    ](self, index: Int) raises -> ComplexSIMD[Self.cdtype, width]:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} out of range for size {}. Use 0 <= i < {}"
                        " when loading elements."
                    ).format(index, self.size, self.size),
                    location="ComplexNDArray.load[width](index: Int)",
                )
            )

        return ComplexSIMD[Self.cdtype, width](
            re=self._re._buf.load[width=1](index),
            im=self._im._buf.load[width=1](index),
        )

    def load[
        width: Int = 1
    ](self, *indices: Int) raises -> ComplexSIMD[Self.cdtype, width=width]:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices (ndim) but received {}. Provide"
                        " one index per dimension: shape {} needs {}"
                        " coordinates."
                    ).format(self.ndim, len(indices), self.shape, self.ndim),
                    location="ComplexNDArray.load[width](*indices: Int)",
                )
            )

        # NOTE: if we take in an owned instances of indices, we can modify it in place.
        var indices_list: List[Int] = List[Int](capacity=self.ndim)
        for i in range(self.ndim):
            var idx_i = indices[i]
            if idx_i < 0 or idx_i >= self.shape[i]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index out of range at dim {}: got {}; valid range"
                            " is [0, {}). Clamp or validate indices against the"
                            " dimension size ({})."
                        ).format(i, idx_i, self.shape[i], self.shape[i]),
                        location="ComplexNDArray.load[width](*indices: Int)",
                    )
                )
            idx_i = self.normalize(idx_i, self.shape[i])
            indices_list.append(idx_i)

        var idx: Int = IndexMethods.get_1d_index(indices_list, self.strides)
        return ComplexSIMD[Self.cdtype, width=width](
            re=self._re._buf.load[width=width](idx),
            im=self._im._buf.load[width=width](idx),
        )

    def _adjust_slice(self, slice_list: List[Slice]) raises -> List[Slice]:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Too many slice dimensions: got {} but array has {}"
                        " dims. Provide at most {} slices for this array."
                    ).format(n_slices, self.ndim, self.ndim),
                    location="ComplexNDArray._adjust_slice",
                )
            )

        var slices = List[Slice](capacity=self.ndim)
        for i in range(n_slices):
            var dim_size = self.shape[i]
            var step = slice_list[i].step.or_else(1)

            if step == 0:
                raise Error(
                    NumojoError(
                        category="value",
                        message=String(
                            "Slice step cannot be zero (dimension {}). Use"
                            " positive or negative non-zero step."
                        ).format(i),
                        location="ComplexNDArray._adjust_slice",
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

    def _setitem(self, *indices: Int, val: ComplexSIMD[Self.cdtype]):
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
            index_of_buffer += indices[i] * Int(self.strides.unsafe_load(i))
        self._re._buf.ptr[unsafe_offset=index_of_buffer] = val.re
        self._im._buf.ptr[unsafe_offset=index_of_buffer] = val.im

    def __setitem__(mut self, idx: Int, val: Self) raises:
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
                NumojoError(
                    category="index",
                    message=(
                        "Cannot assign slice on 0D ComplexNDArray. Assign to"
                        " its scalar value with `A[] = ...` once supported."
                    ),
                    location="ComplexNDArray.__setitem__(idx: Int, val: Self)",
                )
            )

        var norm = idx
        norm = self.normalize(norm, self.shape[0])
        if (norm < 0) or (norm >= self.shape[0]):
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} out of bounds for axis 0 (size {}). Valid"
                        " indices: 0 <= i < {} or -{} <= i < 0."
                    ).format(idx, self.shape[0], self.shape[0], self.shape[0]),
                    location="ComplexNDArray.__setitem__(idx: Int, val: Self)",
                )
            )

        # 1-D target: expect 0-D complex scalar wrapper (val.ndim == 0)
        if self.ndim == 1:
            if val.ndim != 0:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=(
                            "Shape mismatch: expected 0D value for 1D target"
                            " slice. Provide a 0D ComplexNDArray (scalar"
                            " wrapper)."
                        ),
                        location=(
                            "ComplexNDArray.__setitem__(idx: Int, val: Self)"
                        ),
                    )
                )
            self._re._buf.store[width=1](
                norm, val._re._buf.load[width=1](0)
            )
            self._im._buf.store[width=1](
                norm, val._im._buf.load[width=1](0)
            )
            return

        if val.shape != self.shape[1:]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Shape mismatch for slice assignment: expected {} but"
                        " got {}. Provide RHS slice with exact shape {};"
                        " broadcasting not yet supported."
                    ).format(self.shape[1:], val.shape, self.shape[1:]),
                    location="ComplexNDArray.__setitem__(idx: Int, val: Self)",
                )
            )

        if self.flags.C_CONTIGUOUS & val.flags.C_CONTIGUOUS:
            var block = self.size // self.shape[0]
            unsafe_memcpy(
                dest=self._re._buf.ptr.unsafe_offset(norm * block),
                src=val._re._buf.ptr,
                count=block,
            )
            unsafe_memcpy(
                dest=self._im._buf.ptr.unsafe_offset(norm * block),
                src=val._im._buf.ptr,
                count=block,
            )
            return

        # F order
        self._re._write_first_axis_slice(self._re, norm, val._re)
        self._im._write_first_axis_slice(self._im, norm, val._im)

    def __setitem__(
        mut self, var index: Item, val: ComplexSIMD[Self.cdtype]
    ) raises:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Invalid index length: expected {} but got {}. Pass"
                        " exactly {} indices (one per dimension)."
                    ).format(self.ndim, index.__len__(), self.ndim),
                    location=(
                        "ComplexNDArray.__setitem__(index: Item, val:"
                        " Scalar[dtype])"
                    ),
                )
            )
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index out of range at dim {}: got {}; valid range"
                            " is [0, {}). Clamp or validate indices against the"
                            " dimension size ({})."
                        ).format(i, index[i], self.shape[i], self.shape[i]),
                        location=(
                            "ComplexNDArray.__setitem__(index: Item, val:"
                            " Scalar[dtype])"
                        ),
                    )
                )
            index[i] = self.normalize(index[i], self.shape[i])

        var idx: Int = IndexMethods.get_1d_index(index, self.strides)
        self._re._buf.store[width=1](idx, val.re)
        self._im._buf.store[width=1](idx, val.im)

    def __setitem__(
        mut self, mask: NDArray[DType.bool], value: ComplexSIMD[Self.cdtype]
    ) raises:
        """
        Set the value of the array at the indices where the mask is true.
        """
        if (
            mask.shape != self.shape
        ):  # this behaviour could be removed potentially
            raise Error(
                NumojoError(
                    category="shape",
                    message="Mask and array must have the same shape.",
                    location=(
                        "ComplexNDArray.__setitem__(mask: NDArray[DType.bool],"
                        " val: Scalar[dtype])"
                    ),
                )
            )

        var mask_c = mask.contiguous()
        for i in range(mask_c.size):
            if mask_c._buf.load[width=1](i):
                self.itemset(i, value)

    def __setitem__(
        mut self, var *slices: Slice, val: ComplexNDArray[Self.cdtype]
    ) raises:
        """
        Retreive slices of an ComplexNDArray from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(slices.__len__()):
            slice_list.append(slices[i])
        self[slice_list^] = val

    def __setitem__(
        mut self, slices: List[Slice], val: ComplexNDArray[Self.cdtype]
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

        TraverseMethods.traverse_iterative_setter[Self.dtype](
            val._re, self._re, nshape, ncoefficients, nstrides, noffset, index
        )
        TraverseMethods.traverse_iterative_setter[Self.dtype](
            val._im, self._im, nshape, ncoefficients, nstrides, noffset, index
        )

    ## compiler doesn't accept this.
    def __setitem__(
        mut self,
        var *slices: Variant[Slice, Int],
        val: ComplexNDArray[Self.cdtype],
    ) raises:
        """
        Get items by a series of either slices or integers.
        """
        var n_slices: Int = len(slices)
        if n_slices > self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Too many indices or slices: received {} but array has"
                        " only {} dimensions. Pass at most {} indices/slices"
                        " (one per dimension)."
                    ).format(n_slices, self.ndim, self.ndim),
                    location=(
                        "ComplexNDArray.__setitem__(*slices: Variant[Slice,"
                        " Int], val: Self)"
                    ),
                )
            )
        var slice_list: List[Slice] = List[Slice]()

        var count_int = 0
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i][Slice])
            elif slices[i].isa[Int]():
                count_int += 1
                var int: Int = slices[i][Int]
                slice_list.append(Slice(int, int + 1, 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        self[slice_list^] = val

    def __setitem__(mut self, index: NDArray[DType.int], val: Self) raises:
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
    def __setitem__(
        mut self, mask: NDArray[DType.bool], val: ComplexNDArray[Self.cdtype]
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

        var mask_c = mask.contiguous()
        for i in range(mask_c.size):
            if mask_c._buf.load[width=1](i):
                self.itemset(i, val.item(i))

    def __pos__(self) raises -> Self:
        """
        Unary positive returns self unless boolean type.
        """
        if Self.dtype == DType.bool:
            raise Error(
                "complex_ndarray:ComplexNDArray:__pos__: pos does not accept"
                " bool type arrays"
            )
        return self.copy()

    def __neg__(self) raises -> Self:
        """
        Unary negative returns self unless boolean type.

        For bolean use `__invert__`(~)
        """
        if Self.dtype == DType.bool:
            raise Error(
                "complex_ndarray:ComplexNDArray:__neg__: neg does not accept"
                " bool type arrays"
            )
        return self * ComplexSIMD[Self.cdtype](-1.0, -1.0)

    def __bool__(self) raises -> Bool:
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

    def __int__(self) raises -> Int:
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

    def __float__(self) raises -> Float64:
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

    def __abs__(self) raises -> NDArray[Self.dtype]:
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

    def __pow__(self, p: Int) raises -> Self:
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

    def __pow__(self, rhs: Scalar[Self.dtype]) raises -> Self:
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

    def __pow__(
        self, p: Self
    ) raises -> Self where Self.dtype.is_floating_point():
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

    def __ipow__(mut self, p: Int) raises:
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
    def __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.
        """
        return logical_ops.logical_and(
            comparison.equal[Self.dtype](self._re, other._re),
            comparison.equal[Self.dtype](self._im, other._im),
        )

    @always_inline("nodebug")
    def __eq__(
        self, other: ComplexSIMD[Self.cdtype]
    ) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence between scalar and ComplexNDArray.
        """
        return logical_ops.logical_and(
            comparison.equal[Self.dtype](self._re, other.re),
            comparison.equal[Self.dtype](self._im, other.im),
        )

    @always_inline("nodebug")
    def __ne__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence.
        """
        return logical_ops.logical_or(
            comparison.not_equal[Self.dtype](self._re, other._re),
            comparison.not_equal[Self.dtype](self._im, other._im),
        )

    @always_inline("nodebug")
    def __ne__(
        self, other: ComplexSIMD[Self.cdtype]
    ) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence between scalar and ComplexNDArray.
        """
        return logical_ops.logical_or(
            comparison.not_equal[Self.dtype](self._re, other.re),
            comparison.not_equal[Self.dtype](self._im, other.im),
        )

    @always_inline("nodebug")
    def __lt__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        NumPy-style lexicographic ordering: compare real part first, then imaginary part.
        """
        var re_lt = comparison.less[Self.dtype](self._re, other._re)
        var re_eq = comparison.equal[Self.dtype](self._re, other._re)
        var im_lt = comparison.less[Self.dtype](self._im, other._im)
        var result = logical_ops.logical_or(
            re_lt^, logical_ops.logical_and(re_eq^, im_lt^)
        )
        return result^

    @always_inline("nodebug")
    def __lt__(
        self, other: ComplexSIMD[Self.cdtype]
    ) raises -> NDArray[DType.bool]:
        var re_lt = comparison.less[Self.dtype](self._re, other.re)
        var re_eq = comparison.equal[Self.dtype](self._re, other.re)
        var im_lt = comparison.less[Self.dtype](self._im, other.im)
        var result = logical_ops.logical_or(
            re_lt^, logical_ops.logical_and(re_eq^, im_lt^)
        )
        return result^

    @always_inline("nodebug")
    def __lt__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        var re_lt = comparison.less[Self.dtype](self._re, other)
        var re_eq = comparison.equal[Self.dtype](self._re, other)
        var im_lt = comparison.less[Self.dtype](self._im, 0)
        var result = logical_ops.logical_or(
            re_lt^, logical_ops.logical_and(re_eq^, im_lt^)
        )
        return result^

    @always_inline("nodebug")
    def __le__(self, other: Self) raises -> NDArray[DType.bool]:
        var re_lt = comparison.less[Self.dtype](self._re, other._re)
        var re_eq = comparison.equal[Self.dtype](self._re, other._re)
        var im_le = comparison.less_equal[Self.dtype](self._im, other._im)
        var result = logical_ops.logical_or(
            re_lt^, logical_ops.logical_and(re_eq^, im_le^)
        )
        return result^

    @always_inline("nodebug")
    def __le__(
        self, other: ComplexSIMD[Self.cdtype]
    ) raises -> NDArray[DType.bool]:
        var re_lt = comparison.less[Self.dtype](self._re, other.re)
        var re_eq = comparison.equal[Self.dtype](self._re, other.re)
        var im_le = comparison.less_equal[Self.dtype](self._im, other.im)
        var result = logical_ops.logical_or(
            re_lt^, logical_ops.logical_and(re_eq^, im_le^)
        )
        return result^

    @always_inline("nodebug")
    def __le__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        var re_lt = comparison.less[Self.dtype](self._re, other)
        var re_eq = comparison.equal[Self.dtype](self._re, other)
        var im_le = comparison.less_equal[Self.dtype](self._im, 0)
        var result = logical_ops.logical_or(
            re_lt^, logical_ops.logical_and(re_eq^, im_le^)
        )
        return result^

    @always_inline("nodebug")
    def __gt__(self, other: Self) raises -> NDArray[DType.bool]:
        var re_gt = comparison.greater[Self.dtype](self._re, other._re)
        var re_eq = comparison.equal[Self.dtype](self._re, other._re)
        var im_gt = comparison.greater[Self.dtype](self._im, other._im)
        var result = logical_ops.logical_or(
            re_gt^, logical_ops.logical_and(re_eq^, im_gt^)
        )
        return result^

    @always_inline("nodebug")
    def __gt__(
        self, other: ComplexSIMD[Self.cdtype]
    ) raises -> NDArray[DType.bool]:
        var re_gt = comparison.greater[Self.dtype](self._re, other.re)
        var re_eq = comparison.equal[Self.dtype](self._re, other.re)
        var im_gt = comparison.greater[Self.dtype](self._im, other.im)
        var result = logical_ops.logical_or(
            re_gt^, logical_ops.logical_and(re_eq^, im_gt^)
        )
        return result^

    @always_inline("nodebug")
    def __gt__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        var re_gt = comparison.greater[Self.dtype](self._re, other)
        var re_eq = comparison.equal[Self.dtype](self._re, other)
        var im_gt = comparison.greater[Self.dtype](self._im, 0)
        var result = logical_ops.logical_or(
            re_gt^, logical_ops.logical_and(re_eq^, im_gt^)
        )
        return result^

    @always_inline("nodebug")
    def __ge__(self, other: Self) raises -> NDArray[DType.bool]:
        var re_gt = comparison.greater[Self.dtype](self._re, other._re)
        var re_eq = comparison.equal[Self.dtype](self._re, other._re)
        var im_ge = comparison.greater_equal[Self.dtype](self._im, other._im)
        var result = logical_ops.logical_or(
            re_gt^, logical_ops.logical_and(re_eq^, im_ge^)
        )
        return result^

    @always_inline("nodebug")
    def __ge__(
        self, other: ComplexSIMD[Self.cdtype]
    ) raises -> NDArray[DType.bool]:
        var re_gt = comparison.greater[Self.dtype](self._re, other.re)
        var re_eq = comparison.equal[Self.dtype](self._re, other.re)
        var im_ge = comparison.greater_equal[Self.dtype](self._im, other.im)
        var result = logical_ops.logical_or(
            re_gt^, logical_ops.logical_and(re_eq^, im_ge^)
        )
        return result^

    @always_inline("nodebug")
    def __ge__(self, other: Scalar[Self.dtype]) raises -> NDArray[DType.bool]:
        var re_gt = comparison.greater[Self.dtype](self._re, other)
        var re_eq = comparison.equal[Self.dtype](self._re, other)
        var im_ge = comparison.greater_equal[Self.dtype](self._im, 0)
        var result = logical_ops.logical_or(
            re_gt^, logical_ops.logical_and(re_eq^, im_ge^)
        )
        return result^

    # ===------------------------------------------------------------------=== #
    # ARITHMETIC OPERATIONS
    # ===------------------------------------------------------------------=== #

    def __add__(self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray + ComplexSIMD`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other.im)
        return Self(real^, imag^)

    def __add__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + Scalar`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __add__(self, other: Self) raises -> Self:
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

    def __add__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + NDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __radd__(mut self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD + ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other.im)
        return Self(real^, imag^)

    def __radd__(mut self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar + ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __radd__(mut self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray + ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.add[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.add[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __iadd__(mut self, other: ComplexSIMD[Self.cdtype]) raises:
        """
        Enables `ComplexNDArray += ComplexSIMD`.
        """
        self._re += other.re
        self._im += other.im

    def __iadd__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray += Scalar`.
        """
        self._re += other
        self._im += other

    def __iadd__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray += ComplexNDArray`.
        """
        self._re += other._re
        self._im += other._im

    def __iadd__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray += NDArray`.
        """
        self._re += other
        self._im += other

    def __sub__(self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray - ComplexSIMD`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](self._im, other.im)
        return Self(real^, imag^)

    def __sub__(self, other: Scalar[Self.dtype]) raises -> Self:
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

    def __sub__(self, other: Self) raises -> Self:
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

    def __sub__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - NDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __rsub__(mut self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD - ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](other.re, self._re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](other.im, self._im)
        return Self(real^, imag^)

    def __rsub__(mut self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar - ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._im)
        return Self(real^, imag^)

    def __rsub__(mut self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray - ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._re)
        var imag: NDArray[Self.dtype] = math.sub[Self.dtype](other, self._im)
        return Self(real^, imag^)

    def __isub__(mut self, other: ComplexSIMD[Self.cdtype]) raises:
        """
        Enables `ComplexNDArray -= ComplexSIMD`.
        """
        self._re -= other.re
        self._im -= other.im

    def __isub__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray -= Scalar`.
        """
        self._re -= other
        self._im -= other

    def __isub__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray -= ComplexNDArray`.
        """
        self._re -= other._re
        self._im -= other._im

    def __isub__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray -= NDArray`.
        """
        self._re -= other
        self._im -= other

    def __matmul__(self, other: Self) raises -> Self:
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

    def __mul__(self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
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

    def __mul__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * Scalar`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __mul__(self, other: Self) raises -> Self:
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

    def __mul__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * NDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __rmul__(self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD * ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other.re)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other.re)
        return Self(real^, imag^)

    def __rmul__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar * ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __rmul__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray * ComplexNDArray`.
        """
        var real: NDArray[Self.dtype] = math.mul[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.mul[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __imul__(mut self, other: ComplexSIMD[Self.cdtype]) raises:
        """
        Enables `ComplexNDArray *= ComplexSIMD`.
        """
        self._re *= other.re
        self._im *= other.im

    def __imul__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray *= Scalar`.
        """
        self._re *= other
        self._im *= other

    def __imul__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray *= ComplexNDArray`.
        """
        self._re *= other._re
        self._im *= other._im

    def __imul__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray *= NDArray`.
        """
        self._re *= other
        self._im *= other

    def __truediv__(self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var other_square = other * other.conj()
        var result = self * other.conj() * (1.0 / other_square.re)
        return result^

    def __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var real: NDArray[Self.dtype] = math.div[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.div[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __truediv__(self, other: ComplexNDArray[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real^, imag^)

    def __truediv__(self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / NDArray`.
        """
        var real: NDArray[Self.dtype] = math.div[Self.dtype](self._re, other)
        var imag: NDArray[Self.dtype] = math.div[Self.dtype](self._im, other)
        return Self(real^, imag^)

    def __rtruediv__(mut self, other: ComplexSIMD[Self.cdtype]) raises -> Self:
        """
        Enables `ComplexSIMD / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom.re
        var imag = numer._im / denom.re
        return Self(real^, imag^)

    def __rtruediv__(mut self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Enables `Scalar / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real^, imag^)

    def __rtruediv__(mut self, other: NDArray[Self.dtype]) raises -> Self:
        """
        Enables `NDArray / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real^, imag^)

    def __itruediv__(mut self, other: ComplexSIMD[Self.cdtype]) raises:
        """
        Enables `ComplexNDArray /= ComplexSIMD`.
        """
        self._re /= other.re
        self._im /= other.im

    def __itruediv__(mut self, other: Scalar[Self.dtype]) raises:
        """
        Enables `ComplexNDArray /= Scalar`.
        """
        self._re /= other
        self._im /= other

    def __itruediv__(mut self, other: Self) raises:
        """
        Enables `ComplexNDArray /= ComplexNDArray`.
        """
        self._re /= other._re
        self._im /= other._im

    def __itruediv__(mut self, other: NDArray[Self.dtype]) raises:
        """
        Enables `ComplexNDArray /= NDArray`.
        """
        self._re /= other
        self._im /= other

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#
    def __str__(self) -> String:
        """
        Enables String(array).
        """
        var res: String
        try:
            res = self._array_to_string(0, 0)
        except e:
            res = String("Cannot convert array to string") + String(e)

        return res

    def write_to[W: Writer](self, mut writer: W):
        """
        Writes the array to a writer.

        Args:
            writer: The writer to write the array to.
        """
        if self.ndim == 0:
            # For 0-D array (numojo scalar), we can directly write the value
            writer.write(
                String(
                    ComplexScalar[Self.cdtype](
                        self._re._buf.ptr[], self._im._buf.ptr[]
                    )
                )
                + String(
                    "  (0darray["
                    + _concise_dtype_str(Self.cdtype)
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
                    + _concise_dtype_str(Self.cdtype)
                    + "  C-cont: "
                    + String(self.flags.C_CONTIGUOUS)
                    + "  F-cont: "
                    + String(self.flags.F_CONTIGUOUS)
                    + "  own data: "
                    + String(self.flags.OWNDATA)
                )
            except e:
                writer.write("Cannot convert array to string.\n" + String(e))

    def write_repr_to[W: Writer](self, mut writer: W):
        """Write the string representation to a writer.

        Parameters:
            W: The writer type.

        Args:
            writer: The writer to write to.
        """
        # TODO: Deprecate `__repr__` and move its body directly into this method.
        writer.write(self.__repr__())

    def __repr__(self) -> String:
        """
        Compute the "official" string representation of ComplexNDArray.
        An example is:
        ```
        def main() raises:
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

    def _array_to_string(
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
            if out.byte_length() > options.line_width:
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

    def __len__(self) -> Int:
        return Int(self._re.size)

    def store[
        width: Int = 1
    ](mut self, index: Int, val: ComplexSIMD[Self.cdtype]) raises:
        """
        Safely stores SIMD element of size `width` at `index`
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} out of range for array size {}. Use 0 <= i <"
                        " {} when storing; adjust index or reshape array."
                    ).format(index, self.size, self.size),
                    location="ComplexNDArray.store(index: Int)",
                )
            )

        self._re._buf.store[width=1](index, val.re)
        self._im._buf.store[width=1](index, val.im)

    def store[
        width: Int = 1
    ](mut self, *indices: Int, val: ComplexSIMD[Self.cdtype]) raises:
        """
        Safely stores SIMD element of size `width` at given variadic indices
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if len(indices) != self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Expected {} indices (ndim) but received {}. Provide"
                        " one index per dimension for shape {}."
                    ).format(self.ndim, len(indices), self.shape),
                    location="ComplexNDArray.store(*indices)",
                )
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index {} out of range for dim {} (size {}). Valid"
                            " range for dim {} is [0, {})."
                        ).format(
                            indices[i], i, self.shape[i], i, self.shape[i]
                        ),
                        location="ComplexNDArray.store(*indices)",
                    )
                )

        var idx: Int = IndexMethods.get_1d_index(indices, self.strides)
        self._re._buf.store[width=1](idx, val.re)
        self._im._buf.store[width=1](idx, val.im)

    def reshape(self, shape: NDArrayShape, order: String = "C") raises -> Self:
        """
        Returns an array of the same data with a new shape.

        Args:
            shape: Shape of returned array.
            order: Order of the array - Row major `C` or Column major `F`.

        Returns:
            Array of the same data with a new shape.
        """
        var result: Self = ComplexNDArray[Self.cdtype](
            re=reshape(self._re, shape=shape, order=order),
            im=reshape(self._im, shape=shape, order=order),
        )
        result._re.flags = self._re.flags
        result._im.flags = self._im.flags
        return result^

    # def __iter__(
    #     mut self,
    # ) raises -> _ComplexNDArrayIter[origin_of(self), Self.cdtype]:
    #     """
    #     Iterates over elements of the ComplexNDArray and return sub-arrays as view.

    #     Returns:
    #         An iterator of ComplexNDArray elements.
    #     """

    #     return _ComplexNDArrayIter[origin_of(self), Self.cdtype](
    #         Pointer(to=self),
    #         dimension=0,
    #     )

    # def __reversed__(
    #     mut self,
    # ) raises -> _ComplexNDArrayIter[
    #     origin_of(self), Self.cdtype, forward=False
    # ]:
    #     """
    #     Iterates backwards over elements of the ComplexNDArray, returning
    #     copied value.

    #     Returns:
    #         A reversed iterator of NDArray elements.
    #     """

    #     return _ComplexNDArrayIter[origin_of(self), Self.cdtype, forward=False](
    #         Pointer(to=self),
    #         dimension=0,
    #     )

    def itemset(mut self, index: Int, item: ComplexSIMD[Self.cdtype]) raises:
        """Sets the scalar at the given coordinate.

        Args:
            index: The linear index of the i-th item of the whole array.
            item: The complex scalar to be set.

        Raises:
            Error: If the index is out of bounds.
            Error: If the length of index does not match the number of
                dimensions.

        Examples:

        ```
        import numojo as nm
        def main() raises:
            var A = nm.zeros[nm.cf16](nm.Shape(3, 3))
            print(A)
            A.itemset(5, nm.ComplexSIMD[nm.f16](1.0, 2.0))
            print(A)
        ```
        ```console
        [[      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]
        [      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]
        [      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]]
        2-D array  Shape: [3, 3]  DType: complex16
        [[      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]
        [      (0.0, 0.0)    (0.0, 0.0)    (1.0, 2.0)    ]
        [      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]]
        2-D array  Shape: [3, 3]  DType: complex16
        ```.
        """
        var norm_idx = self.normalize(index, self.size)
        if norm_idx < self.size:
            if self.flags.F_CONTIGUOUS:
                var c_stride = NDArrayStrides(shape=self.shape)
                var c_coordinates = List[Int]()
                for i in range(c_stride.ndim):
                    var coordinate = norm_idx // c_stride[i]
                    norm_idx = norm_idx - c_stride[i] * coordinate
                    c_coordinates.append(coordinate)
                self._re._buf.store[width=1](
                    self._re.offset
                    + IndexMethods.get_1d_index(c_coordinates, self.strides),
                    item.re,
                )
                self._im._buf.store[width=1](
                    self._im.offset
                    + IndexMethods.get_1d_index(c_coordinates, self.strides),
                    item.im,
                )
            else:
                self._re._buf.store[width=1](
                    self._re.offset + norm_idx, item.re
                )
                self._im._buf.store[width=1](
                    self._im.offset + norm_idx, item.im
                )
        else:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Index {} is out of bounds for array of size {}. Use an"
                        " index in [0, {})."
                    ).format(index, self.size, self.size),
                    location=(
                        "ComplexNDArray.itemset(index: Int, item: ComplexSIMD)"
                    ),
                )
            )

    def itemset(
        mut self, var indices: List[Int], item: ComplexSIMD[Self.cdtype]
    ) raises:
        """Sets the scalar at the given coordinates.

        Args:
            indices: The coordinates of the item.
            item: The complex scalar to be set.

        Raises:
            Error: If the index is out of bounds.
            Error: If the length of index does not match the number of
                dimensions.

        Notes:
            This is similar to `numpy.ndarray.itemset`. The difference is that
            we take `List[Int]`, but NumPy takes a tuple.

        Examples:

        ```
        import numojo as nm
        def main() raises:
            var A = nm.zeros[nm.cf16](nm.Shape(3, 3))
            print(A)
            A.itemset(nm.List(1, 1), nm.ComplexSIMD[nm.f16](1.0, 2.0))
            print(A)
        ```
        ```console
        [[      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]
        [      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]
        [      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]]
        2-D array  Shape: [3, 3]  DType: complex16
        [[      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]
        [      (0.0, 0.0)    (1.0, 2.0)    (0.0, 0.0)    ]
        [      (0.0, 0.0)    (0.0, 0.0)    (0.0, 0.0)    ]]
        2-D array  Shape: [3, 3]  DType: complex16
        ```.
        """
        if len(indices) != self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Invalid index length: expected {} but got {}. Pass"
                        " exactly {} indices (one per dimension)."
                    ).format(self.ndim, indices.__len__(), self.ndim),
                    location=(
                        "ComplexNDArray.itemset(indices: List[Int], item:"
                        " ComplexSIMD)"
                    ),
                )
            )
        for i in range(len(indices)):
            var norm_idx = self.normalize(indices[i], self.shape[i])
            if norm_idx >= self.shape[i]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=String(
                            "Index out of range at dim {}: got {}; valid"
                            " range is (-{}..{})."
                        ).format(i, indices[i], self.shape[i], self.shape[i]),
                        location=(
                            "ComplexNDArray.itemset(indices: List[Int],"
                            " item: ComplexSIMD)"
                        ),
                    )
                )
            indices[i] = norm_idx
        self._re._buf.store[width=1](
            self._re.offset + IndexMethods.get_1d_index(indices, self.strides),
            item.re,
        )
        self._im._buf.store[width=1](
            self._im.offset + IndexMethods.get_1d_index(indices, self.strides),
            item.im,
        )

    def conj(self) raises -> Self:
        """
        Return the complex conjugate of the ComplexNDArray.
        """
        return Self(self._re.copy(), -self._im.copy())

    def to_ndarray(
        self, type: String = "re"
    ) raises -> NDArray[dtype=Self.dtype]:
        if type == "re":
            var result: NDArray[dtype=Self.dtype] = NDArray[dtype=Self.dtype](
                self.shape
            )
            unsafe_memcpy(
                dest=result._buf.ptr, src=self._re._buf.ptr, count=self.size
            )
            return result^
        elif type == "im":
            var result: NDArray[dtype=Self.dtype] = NDArray[dtype=Self.dtype](
                self.shape
            )
            unsafe_memcpy(
                dest=result._buf.ptr, src=self._im._buf.ptr, count=self.size
            )
            return result^
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=String(
                        "Invalid component selector '{}' (expected 're' or"
                        " 'im'). Call to_ndarray('re') for real part or"
                        " to_ndarray('im') for imaginary part."
                    ).format(type),
                    location="ComplexNDArray.to_ndarray",
                )
            )

    def squeeze(mut self, axis: Int) raises:
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
                NumojoError(
                    category="index",
                    message=String(
                        "Axis {} is out of range for array with {} dimensions."
                        " Use an axis value in the range [-{}, {})."
                    ).format(axis, self.ndim, self.ndim, self.ndim),
                    location="ComplexNDArray.squeeze(axis: Int)",
                )
            )

        if self.shape[normalized_axis] != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "Cannot squeeze axis {} with size {}. Only axes with"
                        " length 1 can be removed."
                    ).format(normalized_axis, self.shape[normalized_axis]),
                    location="ComplexNDArray.squeeze(axis: Int)",
                )
            )
        self.shape = self.shape.pop(normalized_axis)
        self.strides = self.strides.pop(normalized_axis)
        self.ndim -= 1

    # ===-------------------------------------------------------------------===#
    # Statistical and Reduction Methods
    # ===-------------------------------------------------------------------===#

    def all(self) raises -> Bool:
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
        for i in range(self.size):
            var z = self._flat_load(i)
            var re_val = z.re
            var im_val = z.im
            if (re_val == 0.0) and (im_val == 0.0):
                return False
        return True

    def any(self) raises -> Bool:
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
            var z = self._flat_load(i)
            var re_val = z.re
            var im_val = z.im
            if (re_val != 0.0) or (im_val != 0.0):
                return True
        return False

    def sum(self) raises -> ComplexSIMD[Self.cdtype]:
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

        # TODO: could vectorize this!
        for i in range(self.size):
            var z = self._flat_load(i)
            sum_re += z.re
            sum_im += z.im

        return ComplexSIMD[Self.cdtype](sum_re, sum_im)

    def sum(self, axis: Int) raises -> Self:
        return Self(self._re.sum(axis), self._im.sum(axis))

    def prod(self) raises -> ComplexSIMD[Self.cdtype]:
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
            var a = self._flat_load(i)
            var new_re = prod_re * a.re - prod_im * a.im
            var new_im = prod_re * a.im + prod_im * a.re
            prod_re = new_re
            prod_im = new_im

        return ComplexSIMD[Self.cdtype](prod_re, prod_im)

    def prod(self, axis: Int) raises -> Self:
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var out_shape = self.shape.pop(normalized_axis)
        var result = Self(out_shape)
        for o in range(outer):
            var acc_re = Scalar[Self.dtype](1)
            var acc_im = Scalar[Self.dtype](0)
            var base = o * axis_len
            for k in range(axis_len):
                var a = transposed._flat_load(base + k)
                var new_re = acc_re * a.re - acc_im * a.im
                var new_im = acc_re * a.im + acc_im * a.re
                acc_re = new_re
                acc_im = new_im
            result._flat_store(o, ComplexSIMD[Self.cdtype](acc_re, acc_im))
        return result^

    def mean(self) raises -> ComplexSIMD[Self.cdtype]:
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
        return ComplexSIMD[Self.cdtype](total.re / n, total.im / n)

    def mean(self, axis: Int) raises -> Self:
        var s = self.sum(axis)
        var normalized_axis = axis
        if normalized_axis < 0:
            normalized_axis += self.ndim
        if (normalized_axis < 0) or (normalized_axis >= self.ndim):
            raise Error("Axis out of range in ComplexNDArray.mean(axis)")
        var n = Scalar[Self.dtype](self.shape[normalized_axis])
        return Self(s._re / n, s._im / n)

    def max(self) raises -> ComplexSIMD[Self.cdtype]:
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

        var best = self._flat_load(0)
        for i in range(1, self.size):
            var z = self._flat_load(i)
            if self._lex_greater(z, best):
                best = z
        return best

    def max(self, axis: Int) raises -> Self:
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var out_shape = self.shape.pop(normalized_axis)
        var result = Self(out_shape)
        for o in range(outer):
            var base = o * axis_len
            var best = transposed._flat_load(base)
            for k in range(1, axis_len):
                var z = transposed._flat_load(base + k)
                if self._lex_greater(z, best):
                    best = z
            result._flat_store(o, best)
        return result^

    def min(self) raises -> ComplexSIMD[Self.cdtype]:
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

        var best = self._flat_load(0)
        for i in range(1, self.size):
            var z = self._flat_load(i)
            if self._lex_less(z, best):
                best = z
        return best

    def min(self, axis: Int) raises -> Self:
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var out_shape = self.shape.pop(normalized_axis)
        var result = Self(out_shape)
        for o in range(outer):
            var base = o * axis_len
            var best = transposed._flat_load(base)
            for k in range(1, axis_len):
                var z = transposed._flat_load(base + k)
                if self._lex_less(z, best):
                    best = z
            result._flat_store(o, best)
        return result^

    def argmax(self) raises -> Int:
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

        var max_idx = 0

        for i in range(1, self.size):
            if self._lex_greater(self._flat_load(i), self._flat_load(max_idx)):
                max_idx = i

        return max_idx

    def argmax(self, axis: Int) raises -> NDArray[DType.int]:
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var out_shape = self.shape.pop(normalized_axis)
        var result = NDArray[DType.int](out_shape)
        for o in range(outer):
            var base = o * axis_len
            var best_rel = 0
            var best = transposed._flat_load(base)
            for k in range(1, axis_len):
                var z = transposed._flat_load(base + k)
                if self._lex_greater(z, best):
                    best = z
                    best_rel = k
            result._buf[o] = Scalar[DType.int](best_rel)
            # result.itemset(o, Scalar[DType.int](best_rel))
        return result^

    def argmin(self) raises -> Int:
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

        var min_idx = 0

        for i in range(1, self.size):
            if self._lex_less(self._flat_load(i), self._flat_load(min_idx)):
                min_idx = i

        return min_idx

    def argmin(self, axis: Int) raises -> NDArray[DType.int]:
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var out_shape = self.shape.pop(normalized_axis)
        var result = NDArray[DType.int](out_shape)
        for o in range(outer):
            var base = o * axis_len
            var best_rel = 0
            var best = transposed._flat_load(base)
            for k in range(1, axis_len):
                var z = transposed._flat_load(base + k)
                if self._lex_less(z, best):
                    best = z
                    best_rel = k
            result._buf[o] = Scalar[DType.int](best_rel)
            # result.itemset(o, Scalar[DType.int](best_rel))
        return result^

    def cumsum(self) raises -> Self:
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
        return Self(self._re.cumsum(), self._im.cumsum())

    def cumsum(self, axis: Int) raises -> Self:
        return Self(self._re.cumsum(axis), self._im.cumsum(axis))

    def cumprod(self) raises -> Self:
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
        var cum = ComplexSIMD[Self.cdtype](1, 0)
        for i in range(self.size):
            var a = self._flat_load(i)
            cum = ComplexSIMD[Self.cdtype](
                cum.re * a.re - cum.im * a.im, cum.re * a.im + cum.im * a.re
            )
            result._flat_store(i, cum)
        return result^

    def cumprod(self, axis: Int) raises -> Self:
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var inv_axes = self._inverse_permutation(axes)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var transposed_result = Self(transposed.shape)
        for o in range(outer):
            var base = o * axis_len
            var acc_re = Scalar[Self.dtype](1)
            var acc_im = Scalar[Self.dtype](0)
            for k in range(axis_len):
                var a = transposed._flat_load(base + k)
                var new_re = acc_re * a.re - acc_im * a.im
                var new_im = acc_re * a.im + acc_im * a.re
                acc_re = new_re
                acc_im = new_im
                transposed_result._flat_store(
                    base + k, ComplexSIMD[Self.cdtype](acc_re, acc_im)
                )
        return transposed_result.T(inv_axes)

    # ===-------------------------------------------------------------------===#
    # Array Manipulation Methods
    # ===-------------------------------------------------------------------===#

    def flatten(self, order: String = "C") raises -> Self:
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

    def fill(mut self, val: ComplexSIMD[Self.cdtype]):
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

    def row(self, id: Int) raises -> Self:
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
                NumojoError(
                    category="shape",
                    message=String(
                        "Cannot extract row from array with {} dimensions. The"
                        " row() method only works with 1D or 2D arrays."
                    ).format(self.ndim),
                    location="ComplexNDArray.row(id: Int)",
                )
            )

        return Self(self._re.row(id), self._im.row(id))

    def col(self, id: Int) raises -> Self:
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
                NumojoError(
                    category="shape",
                    message=String(
                        "Cannot extract column from array with {} dimensions."
                        " The col() method only works with 1D or 2D arrays."
                    ).format(self.ndim),
                    location="ComplexNDArray.col(id: Int)",
                )
            )

        return Self(self._re.col(id), self._im.col(id))

    def clip(
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
            var re = self._re._buf.load[width=1](i)
            var im = self._im._buf.load[width=1](i)
            var mag_sq = re * re + im * im
            var mag_val = sqrt(mag_sq)

            if mag_val < a_min:
                if mag_val > 0:
                    var scale = a_min / mag_val
                    result._re._buf.store[width=1](i, re * scale)
                    result._im._buf.store[width=1](i, im * scale)
                else:
                    result._re._buf.store[width=1](i, a_min)
                    result._im._buf.store[width=1](i, 0.0)
            elif mag_val > a_max:
                var scale = a_max / mag_val
                result._re._buf.store[width=1](i, re * scale)
                result._im._buf.store[width=1](i, im * scale)
            else:
                result._re._buf.store[width=1](i, re)
                result._im._buf.store[width=1](i, im)

        return result^

    def round(self) raises -> Self:
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

    def T(self) raises -> Self:
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

    def T(self, axes: List[Int]) raises -> Self:
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
        var A_T = A.T([2, 0, 1])  # Shape(4, 2, 3)
        ```
        """
        var transposed_re = self._re.T(axes)
        var transposed_im = self._im.T(axes)
        return Self(transposed_re^, transposed_im^)

    def diagonal(self, offset: Int = 0) raises -> Self:
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
                NumojoError(
                    category="shape",
                    message=String(
                        "diagonal() requires a 2D array, got {} dimensions. Use"
                        " a 2D ComplexNDArray for diagonal extraction."
                    ).format(self.ndim),
                    location="ComplexNDArray.diagonal()",
                )
            )

        var diag_re = self._re.diagonal(offset)
        var diag_im = self._im.diagonal(offset)
        return Self(diag_re^, diag_im^)

    def trace(self) raises -> ComplexSIMD[Self.cdtype]:
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

    def tolist(self) -> List[ComplexSIMD[Self.cdtype]]:
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
        var result = List[ComplexSIMD[Self.cdtype]](capacity=self.size)
        for i in range(self.size):
            result.append(self._flat_load(i))
        return result^

    def astype[target: ComplexDType](self) raises -> ComplexNDArray[target]:
        """Casts this complex array to another complex dtype."""
        return creation.astype[target](self)

    def compress(
        self, condition: NDArray[DType.bool], axis: Int
    ) raises -> Self:
        return Self(
            self._re.compress(condition, axis),
            self._im.compress(condition, axis),
        )

    def compress(self, condition: NDArray[DType.bool]) raises -> Self:
        return Self(self._re.compress(condition), self._im.compress(condition))

    def contiguous(self) raises -> Self:
        return Self(self._re.contiguous(), self._im.contiguous())

    def is_c_contiguous(self) -> Bool:
        return self._re.is_c_contiguous()

    def is_f_contiguous(self) -> Bool:
        return self._re.is_f_contiguous()

    def is_row_contiguous(self) -> Bool:
        return self._re.is_row_contiguous()

    def is_col_contiguous(self) -> Bool:
        return self._re.is_col_contiguous()

    def unsafe_load[
        width: Int = 1
    ](self, index: Int) -> ComplexSIMD[Self.cdtype, width]:
        return ComplexSIMD[Self.cdtype, width](
            self._re.unsafe_load[width=width](index),
            self._im.unsafe_load[width=width](index),
        )

    def unsafe_store[
        width: Int = 1
    ](mut self, index: Int, val: ComplexSIMD[Self.cdtype, width]):
        self._re.unsafe_store[width=width](index, val.re)
        self._im.unsafe_store[width=width](index, val.im)

    # def unsafe_ptr(
    #     ref self, part: String = "re"
    # ) raises -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
    #     if part == "re":
    #         return self._re.unsafe_ptr()
    #     elif part == "im":
    #         return self._im.unsafe_ptr()
    #     else:
    #         raise Error("part must be either 're' or 'im' in unsafe_ptr")

    def to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var builtins = Python.import_module("builtins")
        var re_np = self._re.to_numpy()
        var im_np = self._im.to_numpy()
        var imag_unit = builtins.complex(0, 1)
        return re_np + im_np * imag_unit

    # def iter_over_dimension(
    #     read self, dimension: Int = 0
    # ) raises -> _ComplexNDArrayIter[origin_of(self), Self.cdtype]:
    #     var normalized_dim = dimension
    #     if normalized_dim < 0:
    #         normalized_dim += self.ndim
    #     if (normalized_dim >= self.ndim) or (normalized_dim < 0):
    #         raise Error(
    #             String(
    #                 "\nError in `ComplexNDArray.iter_over_dimension()`: "
    #                 "Axis ({}) is not in valid range [{}, {})."
    #             ).format(dimension, -self.ndim, self.ndim)
    #         )
    #     return _ComplexNDArrayIter[origin_of(self), Self.cdtype](
    #         a=Pointer(to=self), dimension=normalized_dim
    #     )

    # def iter_along_axis(
    #     read self, axis: Int = 0
    # ) raises -> _ComplexNDArrayIter[origin_of(self), Self.cdtype]:
    #     return self.iter_over_dimension(axis)

    # def nditer(
    #     read self,
    # ) raises -> _ComplexNDArrayIter[origin_of(self), Self.cdtype]:
    #     if self.ndim == 0:
    #         raise Error("nditer is undefined for 0D ComplexNDArray.")
    #     return self.iter_over_dimension(0)

    def argsort(self) raises -> NDArray[DType.int]:
        return self.argsort(axis=-1)

    def argsort(self, axis: Int) raises -> NDArray[DType.int]:
        if self.ndim == 0:
            var out = NDArray[DType.int](Shape())
            out._buf.ptr[] = 0
            return out^
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var inv_axes = self._inverse_permutation(axes)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var idx_t = NDArray[DType.int](transposed.shape)

        for o in range(outer):
            var base = o * axis_len
            var idxs = List[Int](capacity=axis_len)
            for k in range(axis_len):
                idxs.append(k)
            for i in range(1, axis_len):
                var key = idxs[i]
                var j = i - 1
                while j >= 0 and self._lex_greater(
                    transposed._flat_load(base + idxs[j]),
                    transposed._flat_load(base + key),
                ):
                    idxs[j + 1] = idxs[j]
                    j -= 1
                idxs[j + 1] = key
            for k in range(axis_len):
                # idx_t._buf.ptr[base + k] = idxs[k]
                idx_t.itemset(base + k, Scalar[DType.int](idxs[k]))

        return idx_t.T(inv_axes)

    def sort(mut self, axis: Int = -1, stable: Bool = False) raises:
        if self.ndim == 0:
            return
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var inv_axes = self._inverse_permutation(axes)
        var transposed = self.T(axes)
        var idx_t = self.argsort(normalized_axis).T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var sorted_t = Self(transposed.shape)

        for o in range(outer):
            var base = o * axis_len
            for k in range(axis_len):
                var src_rel = Int(idx_t.load(base + k))
                sorted_t._flat_store(
                    base + k, transposed._flat_load(base + src_rel)
                )

        var sorted = sorted_t.T(inv_axes)
        self = sorted^

    def median(self) raises -> ComplexSIMD[Self.cdtype]:
        if self.size == 0:
            raise Error("Cannot compute median of empty ComplexNDArray.")
        var a = self.flatten("C")
        a.sort(axis=0)
        if self.size % 2 == 1:
            return a._flat_load(self.size // 2)
        var left = a._flat_load(self.size // 2 - 1)
        var right = a._flat_load(self.size // 2)
        return ComplexSIMD[Self.cdtype](
            (left.re + right.re) / 2, (left.im + right.im) / 2
        )

    def median(self, axis: Int) raises -> Self:
        var normalized_axis = self._normalize_axis(axis)
        var axes = self._permute_axis_to_last(normalized_axis)
        var transposed = self.T(axes)
        var axis_len = self.shape[normalized_axis]
        var outer = self.size // axis_len
        var out_shape = self.shape.pop(normalized_axis)
        var result = Self(out_shape)

        for o in range(outer):
            var base = o * axis_len
            var vals = List[ComplexSIMD[Self.cdtype]](capacity=axis_len)
            for k in range(axis_len):
                vals.append(transposed._flat_load(base + k))
            for i in range(1, axis_len):
                var key = vals[i]
                var j = i - 1
                while j >= 0 and self._lex_greater(vals[j], key):
                    vals[j + 1] = vals[j]
                    j -= 1
                vals[j + 1] = key
            if axis_len % 2 == 1:
                result._flat_store(o, vals[axis_len // 2])
            else:
                var left = vals[axis_len // 2 - 1]
                var right = vals[axis_len // 2]
                result._flat_store(
                    o,
                    ComplexSIMD[Self.cdtype](
                        (left.re + right.re) / 2, (left.im + right.im) / 2
                    ),
                )
        return result^

    # def std[
    #     returned_dtype: DType = DType.float64
    # ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
    #     var v = self.variance[returned_dtype](ddof=ddof)
    #     return sqrt(Scalar[returned_dtype](v))

    # def std[
    #     returned_dtype: DType = DType.float64
    # ](self, axis: Int, ddof: Int = 0) raises -> NDArray[returned_dtype]:
    #     return misc.sqrt[returned_dtype](
    #         self.variance[returned_dtype](axis, ddof)
    #     )

    # def variance[
    #     returned_dtype: DType = DType.float64
    # ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
    #     if self.size == 0:
    #         raise Error("variance is undefined for an empty ComplexNDArray.")
    #     if ddof < 0:
    #         raise Error("ddof must be non-negative in ComplexNDArray.variance.")
    #     var denom = self.size - ddof
    #     if denom <= 0:
    #         raise Error(
    #             String(
    #                 "ddof={} is too large for size {}. Need size - ddof > 0."
    #             ).format(ddof, self.size)
    #         )

    #     var sum_re = Scalar[returned_dtype](0)
    #     var sum_im = Scalar[returned_dtype](0)
    #     for i in range(self.size):
    #         var z = self._flat_load(i)
    #         sum_re += Scalar[returned_dtype](z.re)
    #         sum_im += Scalar[returned_dtype](z.im)

    #     var n = Scalar[returned_dtype](self.size)
    #     var mean_re = sum_re / n
    #     var mean_im = sum_im / n

    #     var acc = Scalar[returned_dtype](0)
    #     for i in range(self.size):
    #         var z = self._flat_load(i)
    #         var dr = Scalar[returned_dtype](z.re) - mean_re
    #         var di = Scalar[returned_dtype](z.im) - mean_im
    #         acc += dr * dr + di * di

    #     return acc / Scalar[returned_dtype](denom)

    # def variance[
    #     returned_dtype: DType = DType.float64
    # ](self, axis: Int, ddof: Int = 0) raises -> NDArray[returned_dtype]:
    #     if ddof < 0:
    #         raise Error("ddof must be non-negative in ComplexNDArray.variance.")

    #     var normalized_axis = self._normalize_axis(axis)
    #     var axes = self._permute_axis_to_last(normalized_axis)
    #     var transposed = self.T(axes)
    #     var axis_len = self.shape[normalized_axis]
    #     var denom = axis_len - ddof
    #     if denom <= 0:
    #         raise Error(
    #             String(
    #                 "ddof={} is too large for axis length {}. Need n - ddof"
    #                 " > 0."
    #             ).format(ddof, axis_len)
    #         )

    #     var outer = self.size // axis_len
    #     var out_shape = self.shape.pop(normalized_axis)
    #     var result = NDArray[returned_dtype](out_shape)

    #     for o in range(outer):
    #         var base = o * axis_len
    #         var sum_re = Scalar[returned_dtype](0)
    #         var sum_im = Scalar[returned_dtype](0)

    #         for k in range(axis_len):
    #             var z = transposed._flat_load(base + k)
    #             sum_re += Scalar[returned_dtype](z.re)
    #             sum_im += Scalar[returned_dtype](z.im)

    #         var n = Scalar[returned_dtype](axis_len)
    #         var mean_re = sum_re / n
    #         var mean_im = sum_im / n

    #         var acc = Scalar[returned_dtype](0)
    #         for k in range(axis_len):
    #             var z = transposed._flat_load(base + k)
    #             var dr = Scalar[returned_dtype](z.re) - mean_re
    #             var di = Scalar[returned_dtype](z.im) - mean_im
    #             acc += dr * dr + di * di

    #         result._buf.ptr[o] = acc / Scalar[returned_dtype](denom)

    #     return result^

    def num_elements(self) -> Int:
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

    def resize(mut self, shape: NDArrayShape) raises:
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
        self.size = shape.size()
        var order = "C" if self.flags.C_CONTIGUOUS else "F"
        self.strides = NDArrayStrides(shape, order=order)


# struct _ComplexNDArrayIter[
#     is_mutable: Bool,
#     //,
#     origin: Origin[mut=is_mutable],
#     cdtype: ComplexDType,
#     forward: Bool = True,
# ](Copyable, Movable):
#     # TODO:
#     # Return a view instead of copy where possible
#     # (when Bufferable is supported).
#     """
#     An iterator yielding `ndim-1` array slices over the given dimension.
#     It is the default iterator of the `ComplexNDArray.__iter__() method and for loops.
#     It can also be constructed using the `ComplexNDArray.iter_over_dimension()` method.
#     It trys to create a view where possible.
#
#     Parameters:
#         is_mutable: Whether the iterator allows mutation of the underlying data.
#         origin: The lifetime of the underlying NDArray data.
#         cdtype: The complex data type of the item.
#         forward: The iteration direction. `False` is backwards.
#     """
#     comptime dtype: DType = Self.cdtype.dtype
#     """The equivalent DType of the ComplexDType."""
#
#     # FIELDS
#     var index: Int
#     var _re_buf: DataContainer[Self.dtype]
#     var _im_buf: DataContainer[Self.dtype]
#     var offset: Int
#     """Offset of the first element in the data buffer."""
#     var dimension: Int
#     var length: Int
#     var shape: NDArrayShape
#     var strides: NDArrayStrides
#     """Strides of array or view. It is not necessarily compatible with shape."""
#     var ndim: Int
#     var size_of_item: Int
#
#     def __init__(
#         out self,
#         a: Pointer[ComplexNDArray[Self.cdtype], Self.origin],
#         dimension: Int,
#     ) raises:
#         """
#         Initialize the iterator.
#
#         Args:
#             a: The array
#             dimension: Dimension to iterate over.
#         """
#
#         if dimension < 0 or dimension >= a[].ndim:
#             raise Error(
#                 NumojoError(
#                     category="index",
#                     message=String(
#                         "Axis {} out of valid range [0, {}). Valid axes: 0..{}."
#                         " Use {} for last axis of shape {}."
#                     ).format(
#                         dimension,
#                         a[].ndim,
#                         a[].ndim - 1,
#                         a[].ndim - 1,
#                         a[].shape,
#                     ),
#                     location="_ComplexNDArrayIter.__init__",
#                 )
#             )
#
#         self._re_buf = a[]._re._buf.copy()
#         self._im_buf = a[]._im._buf.copy()
#         self.offset = a[]._re.offset
#         self.dimension = dimension
#         self.shape = a[].shape
#         self.strides = a[].strides
#         self.ndim = a[].ndim
#         self.length = a[].shape[dimension]
#         self.size_of_item = a[].size // a[].shape[dimension]
#         # Status of the iterator
#         self.index = 0 if Self.forward else a[].shape[dimension] - 1
#
#     def __iter__(self) -> Self:
#         return self.copy()
#
#     def __next__(mut self) raises -> ComplexNDArray[Self.cdtype]:
#         var result = ComplexNDArray[Self.cdtype](self.shape.pop(self.dimension))
#         var current_index = self.index
#
#         comptime if Self.forward:
#             self.index += 1
#         else:
#             self.index -= 1
#
#         for offset in range(self.size_of_item):
#             var remainder = offset
#             var item: Item = Item(ndim=self.ndim)
#
#             for i in range(self.ndim - 1, -1, -1):
#                 if i != self.dimension:
#                     (item._buf.ptr + i).init_pointee_copy(
#                         Scalar[DType.int](remainder % self.shape[i])
#                     )
#                     remainder = remainder // self.shape[i]
#                 else:
#                     (item._buf.ptr + self.dimension).init_pointee_copy(
#                         Scalar[DType.int](current_index)
#                     )
#
#             var idx = self.offset + IndexMethods.get_1d_index(
#                 item, self.strides
#             )
#             result._re._buf.ptr[offset] = self._re_buf[idx]
#             result._im._buf.ptr[offset] = self._im_buf[idx]
#         return result^
#
#     @always_inline
#     def __has_next__(self) -> Bool:
#         comptime if Self.forward:
#             return self.index < self.length
#         else:
#             return self.index >= 0
#
#     def __len__(self) -> Int:
#         comptime if Self.forward:
#             return self.length - self.index
#         else:
#             return self.index
#
#     def ith(self, index: Int) raises -> ComplexNDArray[Self.cdtype]:
#         """
#         Gets the i-th array of the iterator.
#
#         Args:
#             index: The index of the item. It must be non-negative.
#
#         Returns:
#             The i-th `ndim-1`-D array of the iterator.
#         """
#
#         if (index >= self.length) or (index < 0):
#             raise Error(
#                 NumojoError(
#                     category="index",
#                     message=String(
#                         "Iterator index {} out of range [0, {}). Use ith(i)"
#                         " with 0 <= i < {} or iterate via for-loop."
#                     ).format(index, self.length, self.length),
#                     location="_ComplexNDArrayIter.ith",
#                 )
#             )
#
#         if self.ndim > 1:
#             var result = ComplexNDArray[Self.cdtype](
#                 self.shape.pop(self.dimension)
#             )
#
#             for offset in range(self.size_of_item):
#                 var remainder = offset
#                 var item: Item = Item(ndim=self.ndim)
#
#                 for i in range(self.ndim - 1, -1, -1):
#                     if i != self.dimension:
#                         (item._buf.ptr + i).init_pointee_copy(
#                             Scalar[DType.int](remainder % self.shape[i])
#                         )
#                         remainder = remainder // self.shape[i]
#                     else:
#                         (item._buf.ptr + self.dimension).init_pointee_copy(
#                             Scalar[DType.int](index)
#                         )
#
#                 var idx = self.offset + IndexMethods.get_1d_index(
#                     item, self.strides
#                 )
#                 result._re._buf.ptr[offset] = self._re_buf[idx]
#                 result._im._buf.ptr[offset] = self._im_buf[idx]
#             return result^
#
#         else:  # 0-D array
#             var result = numojo.creation._0darray[Self.cdtype](
#                 ComplexSIMD[Self.cdtype](
#                     self._re_buf.ptr[self.offset + index],
#                     self._im_buf.ptr[self.offset + index],
#                 )
#             )
#             return result^
