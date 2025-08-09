# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements N-Dimensional Complex Array
Last updated: 2025-03-10
"""
# ===----------------------------------------------------------------------===#
# SECTIONS OF THE FILE:
#
# `ComplexNDArray` type
# 1. Life cycle methods.
# 2. Indexing and slicing (get and set dunders and relevant methods).
# 3. Operator dunders.
# 4. IO, trait, and iterator dunders.
# 5. Other methods (Sorted alphabetically).

#
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
#
# ===----------------------------------------------------------------------===#
from algorithm import parallelize, vectorize
import builtin.bool as builtin_bool
import builtin.math as builtin_math
from builtin.type_aliases import Origin
from collections import Dict
from collections.optional import Optional
from memory import UnsafePointer, memset_zero, memcpy
from python import Python, PythonObject
from sys import simdwidthof
from utils import Variant

from numojo.core.complex.complex_simd import ComplexSIMD
from numojo.core.datatypes import _concise_dtype_str
from numojo.core.flags import Flags
from numojo.core.item import Item
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides
from numojo.core.utility import (
    _get_offset,
    _transfer_offset,
    _traverse_iterative,
    _traverse_iterative_setter,
    to_numpy,
    bool_to_numeric,
)
from numojo.routines.math._math_funcs import Vectorized
import numojo.routines.bitwise as bitwise
from numojo.routines.io.formatting import (
    format_floating_precision,
    format_floating_scientific,
    format_value,
    PrintOptions,
    GLOBAL_PRINT_OPTIONS,
)
import numojo.routines.linalg as linalg
from numojo.routines.linalg.products import matmul
import numojo.routines.logic.comparison as comparison
from numojo.routines.logic.truth import any
from numojo.routines.manipulation import reshape, ravel
import numojo.routines.math.rounding as rounding
import numojo.routines.math.arithmetic as arithmetic
from numojo.routines.math.extrema import max, min
from numojo.routines.math.products import prod, cumprod
from numojo.routines.math.sums import sum, cumsum
import numojo.routines.sorting as sorting
from numojo.routines.statistics.averages import mean
from numojo.core.error import (
    IndexError,
    ShapeError,
    BroadcastError,
    MemoryError,
    ValueError,
    ArithmeticError,
)


# ===----------------------------------------------------------------------===#
# ComplexNDArray
# ===----------------------------------------------------------------------===#
# TODO: Add SIMD width as a parameter.
struct ComplexNDArray[dtype: DType = DType.float64](
    Copyable, Movable, Representable, Sized, Stringable, Writable
):
    """
    Represents a Complex N-Dimensional Array.

    Parameters:
        dtype: Complex data type.
    """

    # FIELDS
    var _re: NDArray[Self.dtype]
    var _im: NDArray[Self.dtype]

    # It's redundant, but better to have it as fields.
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

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(
        out self, owned re: NDArray[Self.dtype], owned im: NDArray[Self.dtype]
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
        self._re = re
        self._im = im
        self.ndim = re.ndim
        self.shape = re.shape
        self.size = re.size
        self.strides = re.strides
        self.flags = re.flags

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: NDArrayShape,
        order: String = "C",
    ) raises:
        """
        Initialize a ComplexNDArray with given shape.

        The memory is not filled with values.

        Args:
            shape: Variadic shape.
            order: Memory order C or F.

        Example:
        ```mojo
        from numojo.prelude import *
        var A = nm.ComplexNDArray[f32](Shape(2,3,4))
        ```
        """
        self._re = NDArray[Self.dtype](shape, order)
        self._im = NDArray[Self.dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

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
        """
        self._re = NDArray[Self.dtype](shape, order)
        self._im = NDArray[Self.dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

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
        """
        self._re = NDArray[Self.dtype](shape, order)
        self._im = NDArray[Self.dtype](shape, order)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    fn __init__(
        out self,
        shape: List[Int],
        offset: Int,
        strides: List[Int],
    ) raises:
        """
        Extremely specific ComplexNDArray initializer.
        """
        self._re = NDArray[Self.dtype](shape, offset, strides)
        self._im = NDArray[Self.dtype](shape, offset, strides)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    fn __init__(
        out self,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        ndim: Int,
        size: Int,
        flags: Flags,
    ):
        """
        Constructs an extremely specific ComplexNDArray, with value uninitialized.
        The properties do not need to be compatible and are not checked.
        For example, it can construct a 0-D array (numojo scalar).

        Args:
            shape: Shape of array.
            strides: Strides of array.
            ndim: Number of dimensions.
            size: Size of array.
            flags: Flags of array.
        """

        self.shape = shape
        self.strides = strides
        self.ndim = ndim
        self.size = size
        self.flags = flags
        self._re = NDArray[Self.dtype](shape, strides, ndim, size, flags)
        self._im = NDArray[Self.dtype](shape, strides, ndim, size, flags)

    fn __init__(
        out self,
        shape: NDArrayShape,
        ref buffer_re: UnsafePointer[Scalar[Self.dtype]],
        ref buffer_im: UnsafePointer[Scalar[Self.dtype]],
        offset: Int,
        strides: NDArrayStrides,
    ) raises:
        """
        Initialize an ComplexNDArray view with given shape, buffer, offset, and strides.
        ***Unsafe!*** This function is currently unsafe. Only for internal use.

        Args:
            shape: Shape of the array.
            buffer_re: Unsafe pointer to the real part of the buffer.
            buffer_im: Unsafe pointer to the imaginary part of the buffer.
            offset: Offset value.
            strides: Strides of the array.
        """
        self._re = NDArray(shape, buffer_re, offset, strides)
        self._im = NDArray(shape, buffer_im, offset, strides)
        self.ndim = self._re.ndim
        self.shape = self._re.shape
        self.size = self._re.size
        self.strides = self._re.strides
        self.flags = self._re.flags

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Copy other into self.
        """
        self._re = other._re
        self._im = other._im
        self.ndim = other.ndim
        self.shape = other.shape
        self.size = other.size
        self.strides = other.strides
        self.flags = other.flags

    @always_inline("nodebug")
    fn __moveinit__(out self, owned existing: Self):
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

    # Explicit deallocation
    # @always_inline("nodebug")
    # fn __del__(owned self):
    #     """
    #     Deallocate memory.
    #     """
    #     self._re.__del__()
    #     self._im.__del__()

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing
    # Getter and setter dunders and other methods
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing
    # Getter and setter dunders and other methods
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Getter dunders and other getter methods
    #
    # 1. Basic Indexing Operations
    # fn _getitem(self, *indices: Int) -> ComplexSIMD[Self.dtype]                         # Direct unsafe getter
    # fn __getitem__(self) raises -> ComplexSIMD[Self.dtype]                             # Get 0d array value
    # fn __getitem__(self, index: Item) raises -> ComplexSIMD[Self.dtype]                # Get by coordinate list
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
    # fn __getitem__(self, indices: NDArray[DType.index]) raises -> Self        # Get by index array
    # fn __getitem__(self, indices: List[Int]) raises -> Self                   # Get by list of indices
    # fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self            # Get by boolean mask
    # fn __getitem__(self, mask: List[Bool]) raises -> Self                     # Get by boolean list
    #
    # 5. Low-level Access
    # fn item(self, owned index: Int) raises -> ComplexSIMD[Self.dtype]                   # Get item by linear index
    # fn item(self, *index: Int) raises -> ComplexSIMD[Self.dtype]                        # Get item by coordinates
    # fn load(self, owned index: Int) raises -> ComplexSIMD[Self.dtype]                   # Load with bounds check
    # fn load[width: Int](self, index: Int) raises -> ComplexSIMD[Self.dtype, width]        # Load SIMD value
    # fn load[width: Int](self, *indices: Int) raises -> ComplexSIMD[Self.dtype, width]     # Load SIMD at coordinates
    # ===-------------------------------------------------------------------===#

    fn _getitem(self, *indices: Int) -> ComplexSIMD[Self.dtype]:
        """
        Get item at indices and bypass all boundary checks.
        ***UNSAFE!*** No boundary checks made, for internal use only.

        Args:
            indices: Indices to get the value.

        Returns:
            The element of the array at the indices.

        Notes:
            This function is unsafe and should be used only on internal use.

        Examples:

        ```mojo
        import numojo as nm
        var A = nm.ones[nm.f32](nm.Shape(2,3,4))
        print(A._getitem(1,2,3))
        ```
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
        return ComplexSIMD[Self.dtype](
            re=self._re._buf.ptr.load[width=1](index_of_buffer),
            im=self._im._buf.ptr.load[width=1](index_of_buffer),
        )

    fn __getitem__(self) raises -> ComplexSIMD[Self.dtype]:
        """
        Gets the value of the 0-D Complex array.

        Returns:
            The value of the 0-D Complex array.

        Raises:
            Error: If the array is not 0-d.

        Examples:

        ```console
        >>> import numojo as nm
        >>> var A = nm.ones[nm.f32](nm.Shape(2,3,4))
        >>> print(A[]) # gets values of the 0-D array.
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
        return ComplexSIMD[Self.dtype](
            re=self._re._buf.ptr[],
            im=self._im._buf.ptr[],
        )

    fn __getitem__(self, index: Item) raises -> ComplexSIMD[Self.dtype]:
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
        return ComplexSIMD[Self.dtype](
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
            return creation._0darray[Self.dtype](
                ComplexSIMD[Self.dtype](
                    re=self._re._buf.ptr[norm],
                    im=self._im._buf.ptr[norm],
                )
            )

        var out_shape = self.shape[1:]
        var alloc_order = String("C")
        if self.flags.F_CONTIGUOUS:
            alloc_order = String("F")
        var result = ComplexNDArray[Self.dtype](
            shape=out_shape, order=alloc_order
        )

        # Fast path for C-contiguous
        if self.flags.C_CONTIGUOUS:
            var block = self.size // self.shape[0]
            memcpy(result._re._buf.ptr, self._re._buf.ptr + norm * block, block)
            memcpy(result._im._buf.ptr, self._im._buf.ptr + norm * block, block)
            return result^

        # F layout
        self._re._copy_first_axis_slice[Self.dtype](self._re, norm, result._re)
        self._im._copy_first_axis_slice[Self.dtype](self._im, norm, result._im)
        return result^

    fn __getitem__(self, owned *slices: Slice) raises -> Self:
        """
        Retreive slices of a ComplexNDArray from variadic slices.

        Args:
            slices: Variadic slices.

        Returns:
            A slice of the array.

        Examples:

        ```console
        >>>import numojo as nm
        >>>var a = nm.full[nm.f32](nm.Shape(2, 5), ComplexSIMD[nm.f32](1.0, 1.0))
        >>>var b = a[:, 2:4]
        >>>print(b) # `arr[:, 2:4]` returns the corresponding sliced array (2 x 2).
        ```.
        """

        var n_slices: Int = slices.__len__()
        var slice_list: List[Slice] = List[Slice]()
        for i in range(len(slices)):
            slice_list.append(slices[i])

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        var narr: Self = self[slice_list]
        return narr

    fn __getitem__(self, owned slice_list: List[Slice]) raises -> Self:
        """
        Retrieve slices of a ComplexNDArray from a list of slices.

        Args:
            slice_list: List of slices.

        Returns:
            A slice of the array.

        Raises:
            Error: If the slice list is empty.

        Examples:

        ```console
        >>>import numojo as nm
        >>>var a = nm.full[nm.f32](nm.Shape(2, 5), ComplexSIMD[nm.f32](1.0, 1.0))
        >>>var b = a[List[Slice](Slice(0, 2, 1), Slice(2, 4, 1))] # `arr[:, 2:4]` returns the corresponding sliced array (2 x 2).
        >>>print(b)
        ```.
        """

        # Check error cases
        if slice_list.__len__() == 0:
            raise Error(
                IndexError(
                    message=String("Empty slice list provided."),
                    suggestion=String(
                        "Provide at least one Slice; e.g. use [:] or Slice(0,"
                        " n, 1)."
                    ),
                    location=String(
                        "ComplexNDArray.__getitem__(slice_list: List[Slice])"
                    ),
                )
            )

        if slice_list.__len__() < self.ndim:
            for i in range(slice_list.__len__(), self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        # Adjust slice
        var slices = self._adjust_slice(slice_list)
        var spec = List[Int]()
        var ndims = 0

        # Calculate output shape and validate slices in one pass
        for i in range(self.ndim):
            var start: Int = slices[i].start.value()
            var end: Int = slices[i].end.value()
            var step: Int = slices[i].step.or_else(1)

            var slice_len: Int = len(range(start, end, step))
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1

        ndims = 1 if ndims == 0 else ndims

        # Calculate new slices array shape, coefficients, and offset
        var nshape = List[Int]()
        var ncoefficients = List[Int]()
        var noffset = 0
        var nnum_elements: Int = 1

        for i in range(self.ndim):
            if spec[i] != 1:
                nshape.append(spec[i])
                nnum_elements *= spec[i]
                ncoefficients.append(self.strides[i] * slices[i].step.value())
            noffset += slices[i].start.value() * self.strides[i]

        if nshape.__len__() == 0:
            nshape.append(1)
            # nnum_elements = 1
            ncoefficients.append(1)

        # Calculate strides based on memory layout: only C & F order are supported
        var nstrides = List[Int]()
        if self.flags.C_CONTIGUOUS:
            var temp_stride = 1
            for i in range(nshape.__len__() - 1, -1, -1):
                nstrides.insert(0, temp_stride)
                temp_stride *= nshape[i]
        else:  # F_CONTIGUOUS
            var temp_stride = 1
            for i in range(nshape.__len__()):
                nstrides.append(temp_stride)
                temp_stride *= nshape[i]

        # Create and iteratively set values in the new array
        var narr = ComplexNDArray[Self.dtype](
            offset=noffset, shape=nshape, strides=nstrides
        )
        var index_re = List[Int]()
        for _ in range(ndims):
            index_re.append(0)

        _traverse_iterative[dtype](
            self._re,
            narr._re,
            nshape,
            ncoefficients,
            nstrides,
            noffset,
            index_re,
            0,
        )

        var index_im = List[Int]()
        for _ in range(ndims):
            index_im.append(0)

        _traverse_iterative[dtype](
            self._im,
            narr._im,
            nshape,
            ncoefficients,
            nstrides,
            noffset,
            index_im,
            0,
        )

        return narr

    fn __getitem__(self, owned *slices: Variant[Slice, Int]) raises -> Self:
        """
        Get items of ComplexNDArray with a series of either slices or integers.

        Args:
            slices: A series of either Slice or Int.

        Returns:
            A slice of the ndarray with a smaller or equal dimension of the original one.

        Raises:
            Error: If the number of slices is greater than the number of dimensions of the array.

        Examples:

        ```console
        >>>import numojo as nm
        >>>var a = nm.full[nm.f32](nm.Shape(2, 5), ComplexSIMD[nm.f32](1.0, 1.0))
        >>>var b = a[1, 2:4]
        >>>print(b)
        ```.
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Too many indices/slices: received {} but array has {}"
                        " dimensions."
                    ).format(n_slices, self.ndim),
                    suggestion=String(
                        "Use at most {} indices/slices (one per dimension)."
                    ).format(self.ndim),
                    location=String(
                        "ComplexNDArray.__getitem__(*slices: Variant[Slice,"
                        " Int])"
                    ),
                )
            )
        var slice_list: List[Slice] = List[Slice]()

        var count_int: Int = 0  # Count the number of Int in the argument
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                count_int += 1
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int + 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim))

        var narr: Self
        if count_int == self.ndim:
            narr = creation._0darray[Self.dtype](
                ComplexSIMD[Self.dtype](
                    re=self._re._buf.ptr[],
                    im=self._im._buf.ptr[],
                ),
            )
        else:
            narr = self[slice_list]

        return narr

    fn __getitem__(self, indices: NDArray[DType.index]) raises -> Self:
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

        var result: ComplexNDArray[Self.dtype] = ComplexNDArray[Self.dtype](
            shape
        )
        var size_per_item = self.size // self.shape[0]

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
                result._re._buf.ptr + i * size_per_item,
                self._re._buf.ptr + indices.item(i) * size_per_item,
                size_per_item,
            )
            memcpy(
                result._im._buf.ptr + i * size_per_item,
                self._im._buf.ptr + indices.item(i) * size_per_item,
                size_per_item,
            )

        return result

    fn __getitem__(self, indices: List[Int]) raises -> Self:
        """
        Get items from 0-th dimension of a ComplexNDArray of indices.
        It is an overload of
        `__getitem__(self, indices: NDArray[DType.index]) raises -> Self`.

        Args:
            indices: A list of Int.

        Returns:
            ComplexNDArray with items from the list of indices.

        Raises:
            Error: If the elements of indices are greater than size of the corresponding dimension of the array.

        """

        var indices_array = NDArray[DType.index](shape=Shape(len(indices)))
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
            var result = ComplexNDArray[Self.dtype](
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

            return result

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

        var result = ComplexNDArray[Self.dtype](shape)
        var size_per_item = self.size // self.shape[0]

        # Fill in the values
        var offset = 0
        for i in range(mask.size):
            if mask.item(i):
                memcpy(
                    result._re._buf.ptr + offset * size_per_item,
                    self._re._buf.ptr + i * size_per_item,
                    size_per_item,
                )
                memcpy(
                    result._im._buf.ptr + offset * size_per_item,
                    self._im._buf.ptr + i * size_per_item,
                    size_per_item,
                )
                offset += 1

        return result

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

    fn item(self, owned index: Int) raises -> ComplexSIMD[Self.dtype]:
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

        if index < 0:
            index += self.size

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
            return ComplexSIMD[Self.dtype](
                re=(
                    self._re._buf.ptr + _transfer_offset(index, self.strides)
                )[],
                im=(
                    self._im._buf.ptr + _transfer_offset(index, self.strides)
                )[],
            )

        else:
            return ComplexSIMD[Self.dtype](
                re=(self._re._buf.ptr + index)[],
                im=(self._im._buf.ptr + index)[],
            )

    fn item(self, *index: Int) raises -> ComplexSIMD[Self.dtype]:
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
            return ComplexSIMD[Self.dtype](
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
        return ComplexSIMD[Self.dtype](
            re=(self._re._buf.ptr + _get_offset(index, self.strides))[],
            im=(self._im._buf.ptr + _get_offset(index, self.strides))[],
        )

    fn load(self, owned index: Int) raises -> ComplexSIMD[Self.dtype]:
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

        if index < 0:
            index += self.size

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

        return ComplexSIMD[Self.dtype](
            re=self._re._buf.ptr[index],
            im=self._im._buf.ptr[index],
        )

    fn load[width: Int = 1](self, index: Int) raises -> ComplexSIMD[Self.dtype]:
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

        return ComplexSIMD[Self.dtype](
            re=self._re._buf.ptr.load[width=1](index),
            im=self._im._buf.ptr.load[width=1](index),
        )

    fn load[
        width: Int = 1
    ](self, *indices: Int) raises -> ComplexSIMD[Self.dtype, width=width]:
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
                        location=String(
                            "ComplexNDArray.load[width](*indices: Int)"
                        ),
                    )
                )

        var idx: Int = _get_offset(indices, self.strides)
        return ComplexSIMD[Self.dtype, width=width](
            re=self._re._buf.ptr.load[width=width](idx),
            im=self._im._buf.ptr.load[width=width](idx),
        )

    fn _adjust_slice(self, slice_list: List[Slice]) raises -> List[Slice]:
        """
        Adjusts the slice values to lie within 0 and dim.
        """
        var n_slices: Int = slice_list.__len__()
        var slices = List[Slice]()
        for i in range(n_slices):
            if i >= self.ndim:
                raise Error("Error: Number of slices exceeds array dimensions")
                # Could consider ShapeError, but keep generic until slice API stabilized.

            var start: Int = 0
            var end: Int = self.shape[i]
            var step: Int
            if slice_list[i].start is not None:
                start = slice_list[i].start.value()
                if start < 0:
                    # start += self.shape[i]
                    raise Error(
                        IndexError(
                            message=String(
                                "Negative slice start not supported (dimension"
                                " {} start {})."
                            ).format(i, start),
                            suggestion=String(
                                "Use non-negative starts; add self.shape[dim]"
                                " if you intended python-style negative"
                                " indexing."
                            ),
                            location=String("ComplexNDArray._adjust_slice"),
                        )
                    )

            if slice_list[i].end is not None:
                end = slice_list[i].end.value()
                if end < 0:
                    # end += self.shape[i] + 1
                    raise Error(
                        IndexError(
                            message=String(
                                "Negative slice end not supported (dimension {}"
                                " end {})."
                            ).format(i, end),
                            suggestion=String(
                                "Use non-negative ends; add self.shape[dim] if"
                                " you intended python-style negative indexing."
                            ),
                            location=String("ComplexNDArray._adjust_slice"),
                        )
                    )
            step = slice_list[i].step.or_else(1)
            if step == 0:
                raise Error(
                    ValueError(
                        message=String(
                            "Slice step cannot be zero (dimension {})."
                        ).format(i),
                        suggestion=String(
                            "Use positive or negative non-zero step to define"
                            " slice progression."
                        ),
                        location=String("ComplexNDArray._adjust_slice"),
                    )
                )

            slices.append(
                Slice(
                    start=Optional(start),
                    end=Optional(end),
                    step=Optional(step),
                )
            )

        return slices^

    fn _setitem(self, *indices: Int, val: ComplexSIMD[Self.dtype]):
        """
        (UNSAFE! for internal use only.)
        Get item at indices and bypass all boundary checks.
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
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
        if norm < 0:
            norm += self.shape[0]
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

        if val.ndim != self.ndim - 1:
            raise Error(
                ShapeError(
                    message=String(
                        "Shape mismatch: expected {} dims in source but got {}."
                    ).format(self.ndim - 1, val.ndim),
                    suggestion=String("Ensure RHS has shape {}.").format(
                        self.shape[1:]
                    ),
                    location=String(
                        "ComplexNDArray.__setitem__(idx: Int, val: Self)"
                    ),
                )
            )

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
            if val.size != block:
                raise Error(
                    ShapeError(
                        message=String(
                            "Internal mismatch: computed block {} but"
                            " val.size {}."
                        ).format(block, val.size),
                        suggestion=String(
                            "Report this issue; unexpected size mismatch."
                        ),
                        location=String(
                            "ComplexNDArray.__setitem__(idx: Int, val: Self)"
                        ),
                    )
                )
            memcpy(self._re._buf.ptr + norm * block, val._re._buf.ptr, block)
            memcpy(self._im._buf.ptr + norm * block, val._im._buf.ptr, block)
            return

        # F order
        self._re._write_first_axis_slice[Self.dtype](self._re, norm, val._re)
        self._im._write_first_axis_slice[Self.dtype](self._im, norm, val._im)

    fn __setitem__(mut self, index: Item, val: ComplexSIMD[Self.dtype]) raises:
        """
        Set the value at the index list.
        """
        if index.__len__() != self.ndim:
            var message = String(
                "Error: Length of `index` does not match the number of"
                " dimensions!\n"
                "Length of indices is {}.\n"
                "The array dimension is {}."
            ).format(index.__len__(), self.ndim)
            raise Error(message)
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                var message = String(
                    "Error: `index` exceeds the size!\n"
                    "For {}-th dimension:\n"
                    "The index value is {}.\n"
                    "The size of the corresponding dimension is {}"
                ).format(i, index[i], self.shape[i])
                raise Error(message)
        var idx: Int = _get_offset(index, self.strides)
        self._re._buf.ptr.store(idx, val.re)
        self._im._buf.ptr.store(idx, val.im)

    fn __setitem__(
        mut self,
        mask: ComplexNDArray[Self.dtype],
        value: ComplexSIMD[Self.dtype],
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

    fn __setitem__(mut self, owned *slices: Slice, val: Self) raises:
        """
        Retreive slices of an ComplexNDArray from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced ComplexNDArray (2 x 2).
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(slices.__len__()):
            slice_list.append(slices[i])
        # self.__setitem__(slices=slice_list, val=val)
        self[slice_list] = val

    fn __setitem__(mut self, owned slices: List[Slice], val: Self) raises:
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

        _traverse_iterative_setter[dtype](
            val._re, self._re, nshape, ncoefficients, nstrides, noffset, index
        )
        _traverse_iterative_setter[dtype](
            val._im, self._im, nshape, ncoefficients, nstrides, noffset, index
        )

    ### compiler doesn't accept this.
    # fn __setitem__(self, owned *slices: Variant[Slice, Int], val: NDArray[dtype]) raises:
    #     """
    #     Get items by a series of either slices or integers.
    #     """
    #     var n_slices: Int = slices.__len__()
    #     if n_slices > self.ndim:
    #         raise Error("Error: No of slices greater than rank of array")
    #     var slice_list: List[Slice] = List[Slice]()

    #     var count_int = 0
    #     for i in range(len(slices)):
    #         if slices[i].isa[Slice]():
    #             slice_list.append(slices[i]._get_ptr[Slice]()[0])
    #         elif slices[i].isa[Int]():
    #             count_int += 1
    #             var int: Int = slices[i]._get_ptr[Int]()[0]
    #             slice_list.append(Slice(int, int + 1))

    #     if n_slices < self.ndim:
    #         for i in range(n_slices, self.ndim):
    #             var size_at_dim: Int = self.shape[i]
    #             slice_list.append(Slice(0, size_at_dim))

    #     self.__setitem__(slices=slice_list, val=val)

    fn __setitem__(self, index: NDArray[DType.index], val: Self) raises:
        """
        Returns the items of the ComplexNDArray from an array of indices.

        Refer to `__getitem__(self, index: List[Int])`.
        """

        for i in range(len(index)):
            self._re.store(
                Int(index.load(i)), rebind[Scalar[dtype]](val._re.load(i))
            )
            self._im.store(
                Int(index.load(i)), rebind[Scalar[dtype]](val._im.load(i))
            )

    fn __setitem__(
        mut self,
        mask: ComplexNDArray[Self.dtype],
        val: ComplexNDArray[Self.dtype],
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
        return self

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
        return self * ComplexSIMD[Self.dtype](-1.0, -1.0)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.
        """
        return comparison.equal[dtype](
            self._re, other._re
        ) and comparison.equal[dtype](self._im, other._im)

    @always_inline("nodebug")
    fn __eq__(
        self, other: ComplexSIMD[Self.dtype]
    ) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence between scalar and ComplexNDArray.
        """
        return comparison.equal[dtype](self._re, other.re) and comparison.equal[
            dtype
        ](self._im, other.im)

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence.
        """
        return comparison.not_equal[dtype](
            self._re, other._re
        ) and comparison.not_equal[dtype](self._im, other._im)

    @always_inline("nodebug")
    fn __ne__(
        self, other: ComplexSIMD[Self.dtype]
    ) raises -> NDArray[DType.bool]:
        """
        Itemwise non-equivalence between scalar and ComplexNDArray.
        """
        return comparison.not_equal[dtype](
            self._re, other.re
        ) and comparison.not_equal[dtype](self._im, other.im)

    # ===------------------------------------------------------------------=== #
    # ARITHMETIC OPERATIONS
    # ===------------------------------------------------------------------=== #

    fn __add__(self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + ComplexSIMD`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other.im)
        return Self(real, imag)

    fn __add__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + Scalar`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other)
        return Self(real, imag)

    fn __add__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray + ComplexNDArray`.
        """
        print("add complex arrays")
        var real: NDArray[dtype] = math.add[dtype](self._re, other._re)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other._im)
        return Self(real, imag)

    fn __add__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray + NDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other)
        return Self(real, imag)

    fn __radd__(mut self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexSIMD + ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other.im)
        return Self(real, imag)

    fn __radd__(mut self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar + ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other)
        return Self(real, imag)

    fn __radd__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray + ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.add[dtype](self._re, other)
        var imag: NDArray[dtype] = math.add[dtype](self._im, other)
        return Self(real, imag)

    fn __iadd__(mut self, other: ComplexSIMD[Self.dtype]) raises:
        """
        Enables `ComplexNDArray += ComplexSIMD`.
        """
        self._re += other.re
        self._im += other.im

    fn __iadd__(mut self, other: Scalar[dtype]) raises:
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

    fn __iadd__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `ComplexNDArray += NDArray`.
        """
        self._re += other
        self._im += other

    fn __sub__(self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - ComplexSIMD`.
        """
        var real: NDArray[dtype] = math.sub[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.sub[dtype](self._im, other.im)
        return Self(real, imag)

    fn __sub__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - Scalar`.
        """
        var real: NDArray[dtype] = math.sub[dtype](
            self._re, other.cast[dtype]()
        )
        var imag: NDArray[dtype] = math.sub[dtype](
            self._im, other.cast[dtype]()
        )
        return Self(real, imag)

    fn __sub__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](self._re, other._re)
        var imag: NDArray[dtype] = math.sub[dtype](self._im, other._im)
        return Self(real, imag)

    fn __sub__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray - NDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](self._re, other)
        var imag: NDArray[dtype] = math.sub[dtype](self._im, other)
        return Self(real, imag)

    fn __rsub__(mut self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexSIMD - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](other.re, self._re)
        var imag: NDArray[dtype] = math.sub[dtype](other.im, self._im)
        return Self(real, imag)

    fn __rsub__(mut self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](other, self._re)
        var imag: NDArray[dtype] = math.sub[dtype](other, self._im)
        return Self(real, imag)

    fn __rsub__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray - ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.sub[dtype](other, self._re)
        var imag: NDArray[dtype] = math.sub[dtype](other, self._im)
        return Self(real, imag)

    fn __isub__(mut self, other: ComplexSIMD[Self.dtype]) raises:
        """
        Enables `ComplexNDArray -= ComplexSIMD`.
        """
        self._re -= other.re
        self._im -= other.im

    fn __isub__(mut self, other: Scalar[dtype]) raises:
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

    fn __isub__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `ComplexNDArray -= NDArray`.
        """
        self._re -= other
        self._im -= other

    fn __matmul__(self, other: Self) raises -> Self:
        var re_re: NDArray[dtype] = linalg.matmul[dtype](self._re, other._re)
        var im_im: NDArray[dtype] = linalg.matmul[dtype](self._im, other._im)
        var re_im: NDArray[dtype] = linalg.matmul[dtype](self._re, other._im)
        var im_re: NDArray[dtype] = linalg.matmul[dtype](self._im, other._re)
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * ComplexSIMD`.
        """
        var re_re: NDArray[dtype] = math.mul[dtype](self._re, other.re)
        var im_im: NDArray[dtype] = math.mul[dtype](self._im, other.re)
        var re_im: NDArray[dtype] = math.mul[dtype](self._re, other.im)
        var im_re: NDArray[dtype] = math.mul[dtype](self._im, other.im)
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * Scalar`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __mul__(self, other: Self) raises -> Self:
        """
        Enables `ComplexNDArray * ComplexNDArray`.
        """
        var re_re: NDArray[dtype] = math.mul[dtype](self._re, other._re)
        var im_im: NDArray[dtype] = math.mul[dtype](self._im, other._im)
        var re_im: NDArray[dtype] = math.mul[dtype](self._re, other._im)
        var im_re: NDArray[dtype] = math.mul[dtype](self._im, other._re)
        return Self(re_re - im_im, re_im + im_re)

    fn __mul__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray * NDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __rmul__(self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexSIMD * ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other.re)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other.re)
        return Self(real, imag)

    fn __rmul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar * ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __rmul__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray * ComplexNDArray`.
        """
        var real: NDArray[dtype] = math.mul[dtype](self._re, other)
        var imag: NDArray[dtype] = math.mul[dtype](self._im, other)
        return Self(real, imag)

    fn __imul__(mut self, other: ComplexSIMD[Self.dtype]) raises:
        """
        Enables `ComplexNDArray *= ComplexSIMD`.
        """
        self._re *= other.re
        self._im *= other.im

    fn __imul__(mut self, other: Scalar[dtype]) raises:
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

    fn __imul__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `ComplexNDArray *= NDArray`.
        """
        self._re *= other
        self._im *= other

    fn __truediv__(self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var other_square = other * other.conj()
        var result = self * other.conj() * (1.0 / other_square.re)
        return result^

    fn __truediv__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexSIMD`.
        """
        var real: NDArray[dtype] = math.div[dtype](self._re, other)
        var imag: NDArray[dtype] = math.div[dtype](self._im, other)
        return Self(real, imag)

    fn __truediv__(self, other: ComplexNDArray[Self.dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real, imag)

    fn __truediv__(self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `ComplexNDArray / NDArray`.
        """
        var real: NDArray[dtype] = math.div[dtype](self._re, other)
        var imag: NDArray[dtype] = math.div[dtype](self._im, other)
        return Self(real, imag)

    fn __rtruediv__(mut self, other: ComplexSIMD[Self.dtype]) raises -> Self:
        """
        Enables `ComplexSIMD / ComplexNDArray`.
        """
        var denom = other * other.conj()
        var numer = self * other.conj()
        var real = numer._re / denom.re
        var imag = numer._im / denom.re
        return Self(real, imag)

    fn __rtruediv__(mut self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `Scalar / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real, imag)

    fn __rtruediv__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `NDArray / ComplexNDArray`.
        """
        var denom = self * self.conj()
        var numer = self.conj() * other
        var real = numer._re / denom._re
        var imag = numer._im / denom._re
        return Self(real, imag)

    fn __itruediv__(mut self, other: ComplexSIMD[Self.dtype]) raises:
        """
        Enables `ComplexNDArray /= ComplexSIMD`.
        """
        self._re /= other.re
        self._im /= other.im

    fn __itruediv__(mut self, other: Scalar[dtype]) raises:
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

    fn __itruediv__(mut self, other: NDArray[dtype]) raises:
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
            res = self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
        except e:
            res = String("Cannot convert array to string") + String(e)

        return res

    fn write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(
                self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
                + "\n"
                + String(self.ndim)
                + "D-array  Shape"
                + String(self.shape)
                + "  Strides"
                + String(self.strides)
                + "  DType: "
                + _concise_dtype_str(self.dtype)
                + "  C-cont: "
                + String(self.flags["C_CONTIGUOUS"])
                + "  F-cont: "
                + String(self.flags["F_CONTIGUOUS"])
                + "  own data: "
                + String(self.flags["OWNDATA"])
            )
        except e:
            writer.write("Cannot convert array to string" + String(e))

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
            return result
        except e:
            print("Cannot convert array to string", e)
            return ""

    fn _array_to_string(
        self,
        dimension: Int,
        offset: Int,
        print_options: PrintOptions,
    ) raises -> String:
        """
        Convert the array to a string.

        Args:
            dimension: The current dimension.
            offset: The offset of the current dimension.
            print_options: The print options.
        """
        var seperator = print_options.separator
        var padding = print_options.padding
        var edge_items = print_options.edge_items

        if self.ndim == 0:
            return String(self.item(0))
        if dimension == self.ndim - 1:
            var result: String = String("[") + padding
            var number_of_items = self.shape[dimension]
            if number_of_items <= edge_items:  # Print all items
                for i in range(number_of_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = format_value(value, print_options)
                    result = result + formatted_value
                    if i < (number_of_items - 1):
                        result = result + seperator
                result = result + padding
            else:  # Print first 3 and last 3 items
                for i in range(edge_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = format_value(value, print_options)
                    result = result + formatted_value
                    if i < (edge_items - 1):
                        result = result + seperator
                result = result + seperator + "..." + seperator
                for i in range(number_of_items - edge_items, number_of_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = format_value(value, print_options)
                    result = result + formatted_value
                    if i < (number_of_items - 1):
                        result = result + seperator
                result = result + padding
            result = result + "]"
            return result
        else:
            var result: String = String("[")
            var number_of_items = self.shape[dimension]
            if number_of_items <= edge_items:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    if i > 0:
                        result = (
                            result
                            + String(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                                print_options,
                            )
                        )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                for i in range(edge_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    if i > 0:
                        result = (
                            result
                            + String(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                                print_options,
                            )
                        )
                    if i < (number_of_items - 1):
                        result += "\n"
                result = result + "...\n"
                for i in range(number_of_items - edge_items, number_of_items):
                    result = (
                        result
                        + String(" ") * (dimension + 1)
                        + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            result = result + "]"
            return result

    fn __len__(self) -> Int:
        return Int(self._re.size)

    fn store[
        width: Int = 1
    ](mut self, index: Int, val: ComplexSIMD[Self.dtype]) raises:
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
    ](mut self, *indices: Int, val: ComplexSIMD[Self.dtype]) raises:
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
        var result: Self = ComplexNDArray[dtype](
            re=numojo.reshape(self._re, shape=shape, order=order),
            im=numojo.reshape(self._im, shape=shape, order=order),
        )
        result._re.flags = self._re.flags
        result._im.flags = self._im.flags
        return result^

    fn __iter__(
        self,
    ) raises -> _ComplexNDArrayIter[__origin_of(self._re), Self.dtype]:
        """
        Iterates over elements of the ComplexNDArray and return sub-arrays as view.

        Returns:
            An iterator of ComplexNDArray elements.
        """

        return _ComplexNDArrayIter[__origin_of(self._re), Self.dtype](
            self,
            dimension=0,
        )

    fn __reversed__(
        self,
    ) raises -> _ComplexNDArrayIter[
        __origin_of(self._re), Self.dtype, forward=False
    ]:
        """
        Iterates backwards over elements of the ComplexNDArray, returning
        copied value.

        Returns:
            A reversed iterator of NDArray elements.
        """

        return _ComplexNDArrayIter[
            __origin_of(self._re), Self.dtype, forward=False
        ](
            self,
            dimension=0,
        )

    fn itemset(
        mut self,
        index: Variant[Int, List[Int]],
        item: ComplexSIMD[Self.dtype],
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
            var idx = index._get_ptr[Int]()[]
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

        else:
            var indices = index._get_ptr[List[Int]]()[]
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
        return Self(self._re, -self._im)

    fn to_ndarray(self, type: String = "re") raises -> NDArray[dtype=dtype]:
        if type == "re":
            var result: NDArray[dtype=dtype] = NDArray[dtype=dtype](self.shape)
            memcpy(result._buf.ptr, self._re._buf.ptr, self.size)
            return result^
        elif type == "im":
            var result: NDArray[dtype=dtype] = NDArray[dtype=dtype](self.shape)
            memcpy(result._buf.ptr, self._im._buf.ptr, self.size)
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


struct _ComplexNDArrayIter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
    dtype: DType,
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
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var re_ptr: UnsafePointer[Scalar[dtype]]
    var im_ptr: UnsafePointer[Scalar[dtype]]
    var dimension: Int
    var length: Int
    var shape: NDArrayShape
    var strides: NDArrayStrides
    """Strides of array or view. It is not necessarily compatible with shape."""
    var ndim: Int
    var size_of_item: Int

    fn __init__(
        out self, read a: ComplexNDArray[dtype], read dimension: Int
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
        return self

    fn __next__(mut self) raises -> ComplexNDArray[dtype]:
        var res = ComplexNDArray[dtype](self.shape._pop(self.dimension))
        var current_index = self.index

        @parameter
        if forward:
            self.index += 1
        else:
            self.index -= 1

        for offset in range(self.size_of_item):
            var remainder = offset
            var item = Item(ndim=self.ndim, initialized=False)

            for i in range(self.ndim - 1, -1, -1):
                if i != self.dimension:
                    (item._buf + i).init_pointee_copy(remainder % self.shape[i])
                    remainder = remainder // self.shape[i]
                else:
                    (item._buf + self.dimension).init_pointee_copy(
                        current_index
                    )

            (res._re._buf.ptr + offset).init_pointee_copy(
                self.re_ptr[_get_offset(item, self.strides)]
            )
            (res._im._buf.ptr + offset).init_pointee_copy(
                self.im_ptr[_get_offset(item, self.strides)]
            )
        return res

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

    fn ith(self, index: Int) raises -> ComplexNDArray[dtype]:
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
            var res = ComplexNDArray[dtype](self.shape._pop(self.dimension))

            for offset in range(self.size_of_item):
                var remainder = offset
                var item = Item(ndim=self.ndim, initialized=False)

                for i in range(self.ndim - 1, -1, -1):
                    if i != self.dimension:
                        (item._buf + i).init_pointee_copy(
                            remainder % self.shape[i]
                        )
                        remainder = remainder // self.shape[i]
                    else:
                        (item._buf + self.dimension).init_pointee_copy(index)

                (res._re._buf.ptr + offset).init_pointee_copy(
                    self.re_ptr[_get_offset(item, self.strides)]
                )
                (res._im._buf.ptr + offset).init_pointee_copy(
                    self.im_ptr[_get_offset(item, self.strides)]
                )
            return res

        else:  # 0-D array
            var res = numojo.creation._0darray[dtype](
                ComplexSIMD[dtype](self.re_ptr[index], self.im_ptr[index])
            )
            return res
