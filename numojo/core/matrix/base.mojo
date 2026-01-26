"""
NuMojo Matrix Module

This file implements the core 2D matrix type for the NuMojo numerical computing library. It provides efficient, flexible, and memory-safe matrix operations for scientific and engineering applications.

Features:
- `Matrix`: The primary 2D array type for owning matrix data.
- `Matrix`: Lightweight, non-owning views for fast slicing and submatrix access.
- Iterators for traversing matrix elements.
- Comprehensive dunder methods for initialization, indexing, slicing, and arithmetic.
- Utility functions for broadcasting, memory layout, and linear algebra routines.

Use this module to create, manipulate, and analyze matrices with high performance and safety guarantees.
"""

from algorithm import parallelize, vectorize
from memory import UnsafePointer, memcpy, memset_zero
from random import random_float64
from sys import simd_width_of
from python import PythonObject, Python
from math import ceil

from numojo.core.layout.flags import Flags
from numojo.core.ndarray import NDArray
from numojo.core.indexing import Validator, InternalSlice
from numojo.core.memory.data_container import DataContainer
from numojo.core.traits.buffered import Buffered
from numojo.routines.manipulation import broadcast_to, reorder_layout
from numojo.routines.linalg.misc import issymmetric


# TODO: Currently the copyinit creates a ref counted view instead of a deep copy.
# TODO: currently a lot of the __getitem__ and __setitem__ methods raises if the index is out of bounds. An alternative is to clamp the indices to be within bounds, this will remove a lot of if conditions and improve performance I guess. Need to decide which behavior is preferred.
# ===----------------------------------------------------------------------===#
# Matrix struct
# ===----------------------------------------------------------------------===#


struct Matrix[
    dtype: DType = DType.float64,
](Copyable, Movable, Sized, Stringable, Writable):
    """
    Core implementation struct for 2D matrix operations.

    `Matrix` is the single implementation backing the `Matrix` alias. It provides
    the full API for 2D array manipulation with fixed dimensionality, enabling
    efficient `(row, col)` access patterns and optimizations not possible with a
    generic N-dimensional array.

    This struct represents a specialized case of `NDArray` optimized for 2D operations.
    The fixed dimensionality allows for simpler, more efficient indexing using direct
    `(row, col)` access patterns rather than generic coordinate tuples. This makes it
    particularly suitable for linear algebra, image processing, and other applications
    where 2D structure is fundamental.

    Users should typically use `Matrix[dtype]` (i.e., the `Matrix` comptime alias)
    rather than instantiating `Matrix` directly.

    Type Parameters:
        dtype: The data type of matrix elements (e.g., DType.float32, DType.float64).
               Default is DType.float64. This is a compile-time parameter that determines
               the size and interpretation of stored values.

    Memory Layout:
        Matrices can be stored in either:
        - Row-major (C-style) layout: consecutive elements in a row are adjacent in memory
        - Column-major (Fortran-style) layout: consecutive elements in a column are adjacent

        The layout affects cache efficiency for different access patterns and is tracked
        via the `strides` and `flags` attributes.

    Indexing and Slicing:
        - `mat[i, j]` - Returns scalar element at row i, column j
        - `mat[i]` - Returns a copy of row i as a new Matrix
        - `mat.get(i)` - Returns a lightweight non-owning view of row i (no copy)
        - `mat[row_slice, col_slice]` - Returns a copy of the submatrix
        - `mat.get(row_slice, col_slice)` - Returns a lightweight non-owning view of the submatrix (no copy)

        Negative indices are supported and follow Python conventions (wrap from end).

    The matrix can be uniquely defined by the following features:
        1. The data buffer of all items.
        2. The shape of the matrix.
        3. The data type of the elements (compile-time known).

    Attributes:
        - _buf (saved as row-majored, C-type)
        - shape
        - size (shape[0] * shape[1])
        - strides

    Default constructor:
    - [dtype], shape
    - [dtype], data

    [checklist] CORE METHODS that have been implemented:
    - [x] `Matrix.any` and `mat.logic.all`
    - [x] `Matrix.any` and `mat.logic.any`
    - [x] `Matrix.argmax` and `mat.sorting.argmax`
    - [x] `Matrix.argmin` and `mat.sorting.argmin`
    - [x] `Matrix.argsort` and `mat.sorting.argsort`
    - [x] `Matrix.astype`
    - [x] `Matrix.cumprod` and `mat.mathematics.cumprod`
    - [x] `Matrix.cumsum` and `mat.mathematics.cumsum`
    - [x] `Matrix.fill` and `mat.creation.full`
    - [x] `Matrix.flatten`
    - [x] `Matrix.inv` and `mat.linalg.inv`
    - [x] `Matrix.max` and `mat.sorting.max`
    - [x] `Matrix.mean` and `mat.statistics.mean`
    - [x] `Matrix.min` and `mat.sorting.min`
    - [x] `Matrix.prod` and `mat.mathematics.prod`
    - [x] `Matrix.reshape`
    - [x] `Matrix.resize`
    - [x] `Matrix.round` and `mat.mathematics.round` (TODO: Check this after next Mojo update)
    - [x] `Matrix.std` and `mat.statistics.std`
    - [x] `Matrix.sum` and `mat.mathematics.sum`
    - [x] `Matrix.trace` and `mat.linalg.trace`
    - [x] `Matrix.transpose` and `mat.linalg.transpose` (also `Matrix.T`)
    - [x] `Matrix.variance` and `mat.statistics.variance` (`var` is primitive)
    """

    comptime origin: MutOrigin = MutExternalOrigin
    """Origin of the Matrix."""

    comptime IteratorType[
        is_mutable: Bool,
        forward: Bool,
    ] = _MatrixIter[Self.dtype, is_mutable, forward]
    """Iterator type for the Matrix."""

    comptime width: Int = simd_width_of[Self.dtype]()  #
    """Vector size of the data type."""

    var _buf: DataContainer[Self.dtype]
    """Data buffer of the items in the Matrix."""

    var shape: Tuple[Int, Int]
    """Shape of Matrix."""

    var size: Int
    """Size of Matrix."""

    var strides: Tuple[Int, Int]
    """Strides of matrix."""

    var flags: Flags
    "Information about the memory layout of the array."

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Tuple[Int, Int],
        order: String = "C",
    ):
        """
        Initialize a new matrix with the specified shape and memory layout.

        This constructor creates a matrix of the given shape without initializing
        its data. The memory layout can be specified as either row-major ("C") or
        column-major ("F").

        Args:
            shape: A tuple representing the dimensions of the matrix as (rows, columns).
            order: A string specifying the memory layout. Use "C" for row-major
                   (C-style) layout or "F" for column-major (Fortran-style) layout. Defaults to "C".

        Example:
            ```mojo
            from numojo.prelude import *
            var mat_c = Matrix[f32](shape=(3, 4), order="C")  # Row-major
            var mat_f = Matrix[f32](shape=(3, 4), order="F")  # Column-major
            ```
        """
        self.shape = (shape[0], shape[1])
        if order == "C":
            self.strides = (shape[1], 1)
        else:
            self.strides = (1, shape[0])
        self.size = shape[0] * shape[1]
        self._buf = DataContainer[Self.dtype](size=self.size)
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

    # * Should we take var ref and transfer ownership or take a read ref and copy the data?
    @always_inline("nodebug")
    fn __init__(
        out self,
        var data: Self,
    ):
        """
        Initialize a new matrix by transferring ownership from another matrix.

        This constructor creates a new matrix instance by taking ownership of the
        data from an existing matrix. The source matrix (`data`) will no longer
        own its data after this operation.

        Args:
            data: The source matrix from which ownership of the data will be transferred.

        Notes:
            - This operation is efficient as it avoids copying the data buffer.
            - The source matrix (`data`) becomes invalid after the transfer and should not be used.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat1 = Matrix[f32](shape=(2, 3))
            # ... (initialize mat1 with data) ...
            var mat2 = Matrix[f32](mat1^)  # Transfer ownership from mat1 to mat2
            ```
        """
        self = data^

    @always_inline("nodebug")
    fn __init__(
        out self,
        data: Self,
    ):
        """
        Construct a new matrix by copying from another matrix.

        This initializer creates a new matrix instance by copying the data, shape and order from an existing matrix. The new matrix will have its own independent copy of the data.

        Args:
            data: The source matrix to copy from.
        """
        self = Self(data.shape, data.order())
        memcpy(dest=self._buf.ptr, src=data._buf.ptr, count=data.size)

    @always_inline("nodebug")
    fn __init__(
        out self,
        data: NDArray[Self.dtype],
    ) raises:
        """
        Initialize a new matrix by copying data from an existing NDArray.

        This constructor creates a matrix instance with the same shape, data, and
        memory layout as the provided NDArray. The data is copied into a new memory buffer owned by the matrix.

        Args:
            data: An NDArray instance containing the data to initialize the matrix.

        Raises:
            Error: If the provided NDArray has more than 2 dimensions, as it cannot be represented as a matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var arr = NDArray[f32](Shape(2, 3))
            # ... (initialize arr with data) ...
            var mat = Matrix[f32](arr)  # Create a matrix from the NDArray
            ```
        """
        if data.ndim == 1:
            self.shape = (1, data.shape[0])
            self.strides = (data.shape[0], 1)
            self.size = data.shape[0]
        elif data.ndim == 2:
            self.shape = (data.shape[0], data.shape[1])
            if data.flags["C_CONTIGUOUS"]:
                self.strides = (data.shape[1], 1)
            else:
                self.strides = (1, data.shape[0])
            self.size = data.shape[0] * data.shape[1]
        else:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape too large to be a matrix. Matrix only supports"
                        " 1D or 2D arrays."
                    ),
                    location="Matrix.__init__(from NDArray)",
                )
            )

        self._buf = DataContainer[Self.dtype](self.size)
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )
        memcpy(
            dest=self._buf.ptr,
            src=data._buf.ptr,
            count=self.size,
        )

    # to construct views
    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Tuple[Int, Int],
        strides: Tuple[Int, Int],
        data: DataContainer[Self.dtype],
    ):
        """
        Initialize a non-owning `Matrix`.

        This constructor creates a Matrix instance that acts as a view into an
        existing data buffer. The view does not allocate or manage memory; it
        references data owned by another Matrix. It is an unsafe operation and should not be called by users directly.

        Args:
            shape: A tuple representing the dimensions of the view as (rows, columns).
            strides: A tuple representing the memory strides for accessing elements in the view. Strides determine how to traverse the data buffer to access elements in the matrix.
            data: A DataContainer instance that holds the data buffer being referenced.

        Notes:
            - This constructor is intended for internal use to create views into existing matrices! Users should not call this directly.
            - The view does not own the data and relies on the lifetime of the
              original data owner.
            - Modifications to the view affect the original data by default.
        """
        self.shape = shape
        self.strides = strides
        self.size = shape[0] * shape[1]
        self._buf = data
        self.flags = Flags(
            self.shape, self.strides, owndata=False, writeable=False
        )

    # TODO: prevent copying from views to views or views to owning matrices right now.`where` clause isn't working here either for now, So we use constrained. Move to 'where` clause when it's stable.
    # TODO: Current copyinit creates an instance with same origin. This should be external origin. fix this so that we can use default `.copy()` method and remove `create_copy()` method.
    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Initialize a new matrix by copying data from another matrix.

        This method creates a deep copy of the `other` matrix into `self`. It ensures that the copied matrix is independent of the source matrix, with its own memory allocation.

        Args:
            other: The source matrix to copy from. Must be an owning matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat1 = Matrix[f32](shape=(2, 3))
            # ... (initialize mat1 with data) ...
            var mat2 = mat1.copy() # Calls __copyinit__ to create a copy of mat1
            ```
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self._buf = other._buf.copy()
        memcpy(
            dest=self._buf.ptr, src=other._buf.ptr, count=other.size
        )  # TODO: I think we should remove this.
        self.flags = Flags(
            other.shape, other.strides, owndata=True, writeable=True
        )

    # NOTE: perhaps remove this?
    fn deep_copy(self) -> Self:
        """
        Create a deep copy of the current matrix.

        This method returns a new matrix instance that is a deep copy of the current matrix (`self`). The new matrix will have its own independent copy of the data, shape, and strides.

        Returns:
            A new `Matrix` instance that is a deep copy of `self`.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat1 = Matrix[f32](shape=(2, 3))
            # ... (initialize mat1 with data) ...
            var mat2 = mat1.deep_copy()  # Create a deep copy of mat1
            ```
        """
        var copy_mat: Self
        copy_mat = Self(self.shape)
        memcpy(dest=copy_mat._buf.ptr, src=self._buf.ptr, count=self.size)
        return copy_mat^

    fn view(mut self) -> Self:
        """
        Create a non-owning view of the current matrix.

        This method returns a new `Matrix` instance that acts as a view into the data of the current matrix (`self`). The view does not allocate new memory and directly references the existing data buffer of the matrix.

        Returns:
            A `Matrix` representing a view of `self`.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix[f32](shape=(3, 4))
            # ... (initialize mat with data) ...
            var mat_view = mat.view()  # Create a view of mat
            ```
        """
        return Matrix[Self.dtype](
            shape=(self.shape[0], self.shape[1]),
            strides=(self.strides[0], self.strides[1]),
            data=self._buf.share_with_offset(0),
        )

    @always_inline("nodebug")
    fn __moveinit__(out self, deinit other: Self):
        """
        Transfer ownership of resources from `other` to `self`.

        This method moves the data and metadata from the `other` matrix instance
        into the current instance (`self`). After the move, the `other` instance
        is left in an invalid state and should not be used.

        Args:
            other: The source matrix instance whose resources will be moved.

        Notes:
            - This operation is efficient as it avoids copying data.
            - The `other` instance is deinitialized as part of this operation.
        """
        self.shape = other.shape^
        self.strides = other.strides^
        self.size = other.size
        self._buf = other._buf^
        self.flags = other.flags^

    @always_inline("nodebug")
    fn __del__(deinit self):
        """
        Destructor for the matrix instance.

        This method is called when the matrix instance is deinitialized. It ensures that resources owned by the matrix, such as its memory buffer, are properly released.

        Notes:
            - This method only frees resources if the matrix owns its data.
            - The `own_data` flag determines whether the memory buffer is freed.
        """
        _ = self._buf^

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing methods
    # ===-------------------------------------------------------------------===#
    fn validate_and_normalize(self, x: Int, y: Int) raises -> Tuple[Int, Int]:
        """
        Validate and normalize the provided row and column indices.

        This method checks if the given indices are within the bounds of the matrix shape.
        If an index is negative, it is normalized to its positive equivalent.

        Args:
            x: The row index. Can be negative to index from the end.
            y: The column index. Can be negative to index from the end.

        Returns:
            A tuple containing the normalized (x, y) indices.

        Raises:
            Error: If any of the provided indices are out of bounds for the matrix.
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Row index {} out of bounds for matrix with {} rows."
                        " Use indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.validate_and_normalize",
                )
            )
        if y >= self.shape[1] or y < -self.shape[1]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Column index {} out of bounds for matrix with {}"
                        " columns. Use indices in [{}, {}).".format(
                            y, self.shape[1], -self.shape[1], self.shape[1]
                        )
                    ),
                    location="Matrix.validate_and_normalize",
                )
            )

        var x_norm = Validator.normalize(x, self.shape[0])
        var y_norm = Validator.normalize(y, self.shape[1])
        return (x_norm, y_norm)

    fn __getitem__(self, x: Int, y: Int) raises -> Scalar[Self.dtype]:
        """
        Retrieve the scalar value at the specified row and column indices.

        Args:
            x: The row index. Can be negative to index from the end.
            y: The column index. Can be negative to index from the end.

        Returns:
            The value at the specified (x, y) position in the matrix.

        Raises:
            Error: If the provided indices are out of bounds for the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(3, 4))
            var value = mat[1, 2]  # Retrieve value at row 1, column 2
            ```
        """
        var x_norm: Int
        var y_norm: Int
        x_norm, y_norm = self.validate_and_normalize(x, y)
        return self._buf[
            IndexMethods.get_1d_index((x_norm, y_norm), self.strides)
        ]

    # TODO: temporarily renaming all view returning functions to be `get` or `set` due to a Mojo bug with overloading `__getitem__` and `__setitem__` with different argument types. Created an issue in Mojo GitHub
    fn get(mut self, var x: Int) raises -> Matrix[Self.dtype]:
        """
        Retrieve a view of the specified row in the matrix. This method returns a non-owning `Matrix` that references the data of the specified row in the original matrix. The view does not allocate new memory and directly points to the existing data buffer of the matrix.

        Args:
            x: The row index to retrieve. Negative indices are supported and follow Python conventions (e.g., -1 refers to the last row).

        Returns:
            A `Matrix` representing the specified row as a row vector.

        Raises:
            Error: If the provided row index is out of bounds.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(3, 4))
            var row_view = mat.get(1)  # Get a view of the second row
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.get_row(x: Int)",
                )
            )

        var x_norm = Validator.normalize(x, self.shape[0])
        var offset = x_norm * self.strides[0]
        return Matrix[Self.dtype](
            shape=(1, self.shape[1]),
            strides=(self.strides[0], self.strides[1]),
            data=self._buf.share_with_offset(offset),
        )

    # for creating a copy of the row.
    fn __getitem__(self, var x: Int) raises -> Matrix[Self.dtype]:
        """
        Retrieve a copy of the specified row in the matrix. This method creates and returns a new `Matrix` instance that contains a copy of the data from the specified row of the original matrix. The returned matrix is a row vector with a shape of (1, number_of_columns).

        Args:
            x: The row index to retrieve. Negative indices are supported and follow Python conventions (e.g., -1 refers to the last row).

        Returns:
            A `Matrix` instance representing the specified row as a row vector.

        Raises:
            Error: If the provided row index is out of bounds.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(3, 4))
            var row_copy = mat[1]  # Get a copy of the second row
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.get_row(x: Slice)",
                )
            )
        var x_norm = Validator.normalize(x, self.shape[0])
        var result = Matrix[Self.dtype](
            shape=(1, self.shape[1]), order=self.order()
        )
        if self.flags.C_CONTIGUOUS:
            var ptr = self._buf.offset(x_norm * self.strides[0])
            memcpy(dest=result._buf.ptr, src=ptr, count=self.shape[1])
        else:
            for j in range(self.shape[1]):
                result[0, j] = self[x_norm, j]

        return result^

    fn get(mut self, x: Slice, y: Slice) -> Matrix[Self.dtype]:
        """
        Retrieve a view of the specified slice in the matrix.

        This method returns a non-owning `Matrix` that references the data of the specified row in the original matrix. The view does not allocate new memory and directly points to the existing data buffer of the matrix.

        Args:
            x: The row slice to retrieve.
            y: The column slice to retrieve.

        Returns:
            A `Matrix` representing the specified slice of the matrix.

        Notes:
            - Out of bounds indices are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var slice_view = mat.get(Slice(1, 3), Slice(0, 2))  # Get a view of the submatrix
            ```
        """
        start_x, end_x, step_x = x.indices(self.shape[0])
        start_y, end_y, step_y = y.indices(self.shape[1])

        var offset = start_x * self.strides[0] + start_y * self.strides[1]
        return Matrix[Self.dtype](
            shape=(
                Int(ceil((end_x - start_x) / step_x)),
                Int(ceil((end_y - start_y) / step_y)),
            ),
            strides=(self.strides[0] * step_x, self.strides[1] * step_y),
            data=self._buf.share_with_offset(offset),
        )

    # for creating a copy of the slice.
    fn __getitem__(self, x: Slice, y: Slice) -> Matrix[Self.dtype]:
        """
        Retrieve a copy of the specified slice in the matrix. This method creates and returns a new `Matrix` instance that contains a copy of the data from the specified slice of the original matrix. The returned matrix will have the shape determined by the slice ranges.

        Args:
            x: The row slice to retrieve. Supports Python slice syntax.
            y: The column slice to retrieve. Supports Python slice syntax.

        Returns:
            A `Matrix` instance representing the specified slice of the matrix.

        Notes:
            - Out of bounds indices are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var slice_copy = mat[1:3, 0:2]  # Get a copy of the submatrix
            ```
        """
        var start_x: Int
        var end_x: Int
        var step_x: Int
        var len_x: Int
        start_x, end_x, step_x, len_x = InternalSlice.get_slice_info(
            x, self.shape[0]
        )

        var start_y: Int
        var end_y: Int
        var step_y: Int
        var len_y: Int
        start_y, end_y, step_y, len_y = InternalSlice.get_slice_info(
            y, self.shape[1]
        )

        var range_x = range(start_x, end_x, step_x)
        var range_y = range(start_y, end_y, step_y)

        var B = Matrix[Self.dtype](shape=(len_x, len_y), order=self.order())
        var row = 0
        for i in range_x:
            var col = 0
            for j in range_y:
                B._store(row, col, self._load(i, j))
                col += 1
            row += 1

        return B^

    fn get(mut self, x: Slice, var y: Int) raises -> Matrix[Self.dtype]:
        """
        Retrieve a view of a specific column slice in the matrix. This method returns a non-owning `Matrix` that references the data of the specified column slice in the original matrix. The view does not allocate new memory and directly points to the existing data buffer of the matrix.

        Args:
            x: The row slice to retrieve. This defines the range of rows to include in the view.
            y: The column index to retrieve. This specifies the column to include in the view.

        Returns:
            A `Matrix` representing the specified column slice of the matrix.

        Raises:
            Error: If the provided column index `y` is out of bounds.

        Notes:
            - Out-of-bounds indices for `x` are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var column_view = mat.get(Slice(0, 4), 2)  # Get a view of the third column
            ```
        """
        if y >= self.shape[1] or y < -self.shape[1]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} columns. Use"
                        " indices in [{}, {}).".format(
                            y, self.shape[1], -self.shape[1], self.shape[1]
                        )
                    ),
                    location="Matrix.get_col(y: Int)",
                )
            )
        y = Validator.normalize(y, self.shape[1])
        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])

        var offset = start_x * self.strides[0] + y * self.strides[1]
        return Matrix[Self.dtype](
            shape=(
                Int(ceil((end_x - start_x) / step_x)),
                1,
            ),
            strides=(self.strides[0] * step_x, self.strides[1]),
            data=self._buf.share_with_offset(offset),
        )

    fn __getitem__(self, x: Slice, var y: Int) -> Matrix[Self.dtype]:
        """
        Retrieve a copy of a specific column slice in the matrix. This method creates and returns a new `Matrix` instance that contains a copy
        of the data from the specified and column slice of the original matrix. The returned matrix will have a shape determined by the row slice and a single column.

        Args:
            x: The row slice to retrieve. This defines the range of rows to include in the copy.
            y: The column index to retrieve. This specifies the column to include in the copy.

        Returns:
            A `Matrix` instance representing the specified column slice of the matrix.

        Notes:
            - Negative indices for `y` are normalized to their positive equivalent.
            - Out-of-bounds indices for `x` are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var column_copy = mat[0:4, 2]  # Get a copy of the third column
            ```
        """
        if y < 0:
            y = self.shape[1] + y

        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        var range_x = range(start_x, end_x, step_x)
        var res = Matrix[Self.dtype](
            shape=(
                len(range_x),
                1,
            ),
            order=self.order(),
        )
        var row = 0
        for i in range_x:
            res._store(row, 0, self._load(i, y))
            row += 1
        return res^

    fn get(mut self, var x: Int, y: Slice) raises -> Matrix[Self.dtype]:
        """
        Retrieve a view of a specific row slice in the matrix. This method returns a non-owning `Matrix` that references the data of the specified row slice in the original matrix. The view does not allocate new memory and directly points to the existing data buffer of the matrix.

        Args:
            x: The row index to retrieve. This specifies the row to include in the view. Negative indices are supported and follow Python conventions (e.g., -1 refers to the last row).
            y: The column slice to retrieve. This defines the range of columns to include in the view.

        Returns:
            A `Matrix` representing the specified row slice of the matrix.

        Raises:
            Error: If the provided row index `x` is out of bounds.

        Notes:
            - Out-of-bounds indices for `y` are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var row_view = mat.get(1, Slice(0, 3))  # Get a view of the second row, columns 0 to 2
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.get(x: Int, y: Slice)",
                )
            )
        x = Validator.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])

        var offset = x * self.strides[0] + start_y * self.strides[1]
        return Matrix[Self.dtype](
            shape=(
                1,
                Int(ceil((end_y - start_y) / step_y)),
            ),
            strides=(self.strides[0], self.strides[1] * step_y),
            data=self._buf.share_with_offset(offset),
        )

    fn __getitem__(self, var x: Int, y: Slice) raises -> Matrix[Self.dtype]:
        """
        Retrieve a copy of a specific row slice in the matrix. This method creates and returns a new `Matrix` instance that contains a copy
        of the data from the specified row and column slice of the original matrix. The returned matrix will have a shape of (1, number_of_columns_in_slice).

        Args:
            x: The row index to retrieve. This specifies the row to include in the copy. Negative indices are supported and follow Python conventions (e.g., -1 refers to the last row).
            y: The column slice to retrieve. This defines the range of columns to include in the copy.

        Returns:
            A `Matrix` instance representing the specified row slice of the matrix.

        Raises:
            Error: If the provided row index `x` is out of bounds.

        Notes:
            - Out-of-bounds indices for `y` are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var row_copy = mat[1, 0:3]  # Get a copy of the second row, columns 0 to 2
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.get(x: Slice, y: Int)",
                )
            )
        x = Validator.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_y = range(start_y, end_y, step_y)

        var B = Matrix[Self.dtype](shape=(1, len(range_y)), order=self.order())
        var col = 0
        for j in range_y:
            B._store(0, col, self._load(x, j))
            col += 1

        return B^

    fn __getitem__(self, indices: List[Int]) raises -> Matrix[Self.dtype]:
        """
        Retrieve a copy of specific rows in the matrix based on the provided indices. This method creates and returns a new `Matrix` instance that contains a copy of the data from the specified rows of the original matrix. The returned matrix will have a shape of (number_of_indices, number_of_columns).

        Args:
            indices: A list of row indices to retrieve. Each index specifies a row to include in the resulting matrix. Negative indices are supported and follow Python conventions (e.g., -1 refers to the last row).

        Returns:
            A `Matrix` instance containing the selected rows as a new matrix.

        Raises:
            Error: If any of the provided indices are out of bounds.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var selected_rows = mat[[0, 1, 0]]  # Get a copy of the
            # first and second and first rows in a new matrix with shape (3, 4)
            ```
        """
        var num_cols = self.shape[1]
        var num_rows = len(indices)
        var selected_rows = Matrix.zeros[Self.dtype](shape=(num_rows, num_cols))
        for i in range(num_rows):
            if indices[i] >= self.shape[0] or indices[i] < -self.shape[0]:
                raise Error(
                    NumojoError(
                        category="index",
                        message=(
                            "Index {} out of bounds for matrix with {} rows."
                            " Use indices in [{}, {}).".format(
                                indices[i],
                                self.shape[0],
                                -self.shape[0],
                                self.shape[0],
                            )
                        ),
                        location="Matrix.get(indices: List[Int])",
                    )
                )
            selected_rows[i] = self[indices[i]]
        return selected_rows^

    fn load[width: Int = 1](self, idx: Int) raises -> SIMD[Self.dtype, width]:
        """
        Load a SIMD element from the matrix at the specified linear index.

        Parameters:
            width: The width of the SIMD element to load. Defaults to 1.

        Args:
            idx: The linear index of the element to load. Negative indices are supported and follow Python conventions.

        Returns:
            A SIMD element of the specified width containing the data at the given index.

        Raises:
            Error: If the provided index is out of bounds.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var simd_element = mat.load[4](2)  # Load a SIMD element of width 4 from index 2
            ```
        """
        if idx >= self.size or idx < -self.size:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix size {}. Use indices"
                        " in [{}, {}).".format(
                            idx, self.size, -self.size, self.size
                        )
                    ),
                    location="Matrix.__getitem__(idx: Int)",
                )
            )
        var idx_norm = Validator.normalize(idx, self.size)
        return self._buf.ptr.load[width=width](idx_norm)

    fn _load[width: Int = 1](self, x: Int, y: Int) -> SIMD[Self.dtype, width]:
        """
        `__getitem__` with width.
        Unsafe: No boundary check!
        """
        return self._buf.ptr.load[width=width](
            x * self.strides[0] + y * self.strides[1]
        )

    fn _load[width: Int = 1](self, idx: Int) -> SIMD[Self.dtype, width]:
        """
        `__getitem__` with width.
        Unsafe: No boundary check!
        """
        return self._buf.ptr.load[width=width](idx)

    fn __setitem__(mut self, x: Int, y: Int, value: Scalar[Self.dtype]) raises:
        """
        Set the value at the specified row and column indices in the matrix.

        Args:
            x: The row index. Can be negative to index from the end.
            y: The column index. Can be negative to index from the end.
            value: The value to set at the specified position.

        Raises:
            Error: If the provided indices are out of bounds for the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(3, 4))
            mat[1, 2] = 5.0  # Set value at row 1, column 2 to 5.0
            ```
        """
        if (
            x >= self.shape[0]
            or x < -self.shape[0]
            or y >= self.shape[1]
            or y < -self.shape[1]
        ):
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index ({}, {}) out of bounds for matrix shape ({},"
                        " {}). Use indices in [{}, {}) x [{}, {}).".format(
                            x,
                            y,
                            self.shape[0],
                            self.shape[1],
                            -self.shape[0],
                            self.shape[0],
                            -self.shape[1],
                            self.shape[1],
                        )
                    ),
                    location="Matrix.__setitem__(x: Int, y: Int, val: Scalar)",
                )
            )
        var x_norm: Int = Validator.normalize(x, self.shape[0])
        var y_norm: Int = Validator.normalize(y, self.shape[1])

        self._buf.store(
            IndexMethods.get_1d_index((x_norm, y_norm), self.strides), value
        )

    # FIXME: Setting with views is currently only supported through `.set()` method of the Matrix. Once Mojo resolve the symmetric getter setter issue, we can remove `.set()` methods.
    fn __setitem__(self, var x: Int, value: Matrix[Self.dtype]) raises:
        """
        Assign a row in the matrix at the specified index with the given matrix. This method replaces the row at the specified index `x` with the data from
        the provided `value` matrix. The `value` matrix must be a row vector with
        the same number of columns as the target matrix.

        Args:
            x: The row index where the data will be assigned. Negative indices are
               supported and follow Python conventions (e.g., -1 refers to the last row).
            value: A `Matrix` instance representing the row vector to assign.
                   The `value` matrix can be in either C-contiguous or F-contiguous order.

        Raises:
            Error: If the row index `x` is out of bounds.
            Error: If the `value` matrix does not have exactly one row.
            Error: If the number of columns in the `value` matrix does not match
                   the number of columns in the target matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(3, 4))
            var row_vector = Matrix.ones(shape=(1, 4))
            mat[1] = row_vector  # Set the second row of mat to row_vector
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.set(x: Int, value: Matrix)",
                )
            )
        if value.shape[0] != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Value must have exactly 1 row, but has {} rows."
                        .format(value.shape[0])
                    ),
                    location="Matrix.set(x: Int, value: Matrix)",
                )
            )
        if self.shape[1] != value.shape[1]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Matrix has {} columns, but value has {} columns."
                        " Column counts must match.".format(
                            self.shape[1], value.shape[1]
                        )
                    ),
                    location="Matrix.set(x: Int, value: Matrix)",
                )
            )

        if self.flags.C_CONTIGUOUS:
            if value.flags.C_CONTIGUOUS:
                var dest_ptr = self._buf.offset(x * self.strides[0])
                memcpy(dest=dest_ptr, src=value._buf.ptr, count=self.shape[1])
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

        # For F-contiguous
        else:
            if value.flags.F_CONTIGUOUS:
                for j in range(self.shape[1]):
                    self._buf.offset(x + j * self.strides[1]).store(
                        value._buf.ptr.load(j * value.strides[1])
                    )
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

    fn set(self, var x: Int, value: Matrix[Self.dtype]) raises:
        """
        Assign a row in the matrix at the specified index with the given matrix. This method replaces the row at the specified index `x` with the data from
        the provided `value` matrix. The `value` matrix must be a row vector with
        the same number of columns as the target matrix.

        Args:
            x: The row index where the data will be assigned. Negative indices are
               supported and follow Python conventions (e.g., -1 refers to the last row).
            value: A `Matrix` instance representing the row vector to assign.
                   The `value` matrix can be in either C-contiguous or F-contiguous order.

        Raises:
            Error: If the row index `x` is out of bounds.
            Error: If the `value` matrix does not have exactly one row.
            Error: If the number of columns in the `value` matrix does not match
                   the number of columns in the target matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(3, 4))
            var row_vector = Matrix.ones(shape=(1, 4))
            mat.set(1, row_vector)  # Set the second row of mat to row_vector

            var view = row_vector.view() # create a view of row_vector
            mat.set(2, view)  # Set the third row of mat to the view
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.__setitem__(x: Int, value: Matrix)",
                )
            )
        if value.shape[0] != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Value must have exactly 1 row, but has {} rows."
                        .format(value.shape[0])
                    ),
                    location="Matrix.__setitem__(x: Int, value: Matrix)",
                )
            )
        if self.shape[1] != value.shape[1]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Matrix has {} columns, but value has {} columns."
                        " Column counts must match.".format(
                            self.shape[1], value.shape[1]
                        )
                    ),
                    location="Matrix.__setitem__(x: Int, value: Matrix)",
                )
            )

        if self.flags.C_CONTIGUOUS:
            if value.flags.C_CONTIGUOUS:
                var dest_ptr = self._buf.offset(x * self.strides[0])
                memcpy(dest=dest_ptr, src=value._buf.ptr, count=self.shape[1])
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

        # For F-contiguous
        else:
            if value.flags.F_CONTIGUOUS:
                for j in range(self.shape[1]):
                    self._buf.offset(x + j * self.strides[1]).store(
                        value._buf.ptr.load(j * value.strides[1])
                    )
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

    fn __setitem__(self, x: Slice, y: Int, value: Matrix[Self.dtype]) raises:
        """
        Assign values to a column in the matrix at the specified column index `y`
        and row slice `x` with the given matrix. This method replaces the values
        in the specified column and row slice with the data from the provided
        `value` matrix.

        Args:
            x: The row slice where the data will be assigned. Supports Python slice syntax (e.g., `start:stop:step`).
            y: The column index where the data will be assigned. Negative indices
               are supported and follow Python conventions (e.g., -1 refers to the
               last column).
            value: A `Matrix` instance representing the column vector to assign.
                   The `value` matrix must have the same number of rows as the
                   specified slice `x` and exactly one column.

        Raises:
            Error: If the column index `y` is out of bounds.
            Error: If the shape of the `value` matrix does not match the target
                   slice dimensions.

        Notes:
            - Out of bound slice `x` is clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(4, 4))
            var col_vector = Matrix.ones(shape=(4, 1))
            mat[0:4, 2] = col_vector  # Set the third column of mat to col_vector
            ```
        """
        if y >= self.shape[1] or y < -self.shape[1]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} columns. Use"
                        " indices in [{}, {}).".format(
                            y, self.shape[1], -self.shape[1], self.shape[1]
                        )
                    ),
                    location="Matrix.set(x: Slice, y: Int, value: Matrix)",
                )
            )
        var y_norm = Validator.normalize(y, self.shape[1])
        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        var range_x = range(start_x, end_x, step_x)
        var len_range_x: Int = len(range_x)

        if len_range_x != value.shape[0] or value.shape[1] != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch when assigning to slice: target shape"
                        " ({}, {}) vs value shape ({}, {}). Shapes must match"
                        " exactly.".format(
                            len_range_x, 1, value.shape[0], value.shape[1]
                        )
                    ),
                    location="Matrix.set(x: Slice, y: Int, value: Matrix)",
                )
            )

        var row = 0
        for i in range_x:
            self._store(i, y_norm, value._load(row, 0))
            row += 1

    fn set(self, x: Slice, y: Int, value: Matrix[Self.dtype]) raises:
        """
        Assign values to a column in the matrix at the specified column index `y`
        and row slice `x` with the given matrix. This method replaces the values
        in the specified column and row slice with the data from the provided
        `value` matrix.

        Args:
            x: The row slice where the data will be assigned. Supports Python slice syntax (e.g., `start:stop:step`).
            y: The column index where the data will be assigned. Negative indices
               are supported and follow Python conventions (e.g., -1 refers to the
               last column).
            value: A `Matrix` instance representing the column vector to assign.
                   The `value` matrix must have the same number of rows as the
                   specified slice `x` and exactly one column.

        Raises:
            Error: If the column index `y` is out of bounds.
            Error: If the shape of the `value` matrix does not match the target
                   slice dimensions.

        Notes:
            - Out of bound slice `x` is clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(4, 4))
            var col_vector = Matrix.ones(shape=(4, 1))
            mat.set(Slice(0, 4), 2, col_vector)  # Set the third column of mat to col_vector

            var view = col_vector.view() # create a view of col_vector
            mat.set(Slice(0, 4), 3, view)  # Set the fourth column of mat to the view
            ```
        """
        if y >= self.shape[1] or y < -self.shape[1]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} columns. Use"
                        " indices in [{}, {}).".format(
                            y, self.shape[1], -self.shape[1], self.shape[1]
                        )
                    ),
                    location=(
                        "Matrix.__setitem__(x: Slice, y: Int, value: Matrix)"
                    ),
                )
            )
        var y_norm = Validator.normalize(y, self.shape[1])
        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        var range_x = range(start_x, end_x, step_x)
        var len_range_x: Int = len(range_x)

        if len_range_x != value.shape[0] or value.shape[1] != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch when assigning to slice: target shape"
                        " ({}, {}) vs value shape ({}, {}). Shapes must match"
                        " exactly.".format(
                            len_range_x, 1, value.shape[0], value.shape[1]
                        )
                    ),
                    location=(
                        "Matrix.__setitem__(x: Slice, y: Int, value: Matrix)"
                    ),
                )
            )

        var row = 0
        for i in range_x:
            self._store(i, y_norm, value._load(row, 0))
            row += 1

    fn __setitem__(self, x: Int, y: Slice, value: Matrix[Self.dtype]) raises:
        """
        Assign values to a row in the matrix at the specified row index `x`
        and column slice `y` with the given matrix. This method replaces the values in the specified row and column slice with the data from the provided `value` matrix.

        Args:
            x: The row index where the data will be assigned. Negative indices
               are supported and follow Python conventions (e.g., -1 refers to the
               last row).
            y: The column slice where the data will be assigned. Supports Python slice syntax (e.g., `start:stop:step`).
            value: A `Matrix` instance representing the row vector to assign.
                   The `value` matrix must have the same number of columns as the
                   specified slice `y` and exactly one row.

        Raises:
            Error: If the row index `x` is out of bounds.
            Error: If the shape of the `value` matrix does not match the target
                   slice dimensions.

        Notes:
            - Out of bound slice `y` is clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(4, 4))
            var row_vector = Matrix.ones(shape=(1, 3))
            mat[1, 0:3] = row_vector  # Set the second row, columns 0 to 2 of mat to row_vector
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location="Matrix.set(x: Int, y: Slice, value: Matrix)",
                )
            )
        var x_norm = Validator.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_y = range(start_y, end_y, step_y)
        var len_range_y: Int = len(range_y)

        if len_range_y != value.shape[1] or value.shape[0] != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch when assigning to slice: target shape"
                        " ({}, {}) vs value shape ({}, {}). Shapes must match"
                        " exactly.".format(
                            1, len_range_y, value.shape[0], value.shape[1]
                        )
                    ),
                    location="Matrix.set(x: Int, y: Slice, value: Matrix)",
                )
            )

        var col = 0
        for j in range_y:
            self._store(x_norm, j, value._load(0, col))
            col += 1

    fn set(self, x: Int, y: Slice, value: Matrix[Self.dtype]) raises:
        """
        Assign values to a row in the matrix at the specified row index `x`
        and column slice `y` with the given matrix. This method replaces the values in the specified row and column slice with the data from the provided `value` matrix.

        Args:
            x: The row index where the data will be assigned. Negative indices
               are supported and follow Python conventions (e.g., -1 refers to the
               last row).
            y: The column slice where the data will be assigned. Supports Python slice syntax (e.g., `start:stop:step`).
            value: A `Matrix` instance representing the row vector to assign.
                   The `value` matrix must have the same number of columns as the
                   specified slice `y` and exactly one row.

        Raises:
            Error: If the row index `x` is out of bounds.
            Error: If the shape of the `value` matrix does not match the target
                   slice dimensions.

        Notes:
            - Out of bound slice `y` is clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(4, 4))
            var row_vector = Matrix.ones(shape=(1, 3))
            mat.set(1, Slice(0, 3), row_vector)  # Set the second row, columns 0 to 2 of mat to row_vector

            var view = row_vector.view() # create a view of row_vector
            mat.set(2, Slice(0, 3), view)  # Set the third row, columns 0 to 2 of mat to the view
            ```
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix with {} rows. Use"
                        " indices in [{}, {}).".format(
                            x, self.shape[0], -self.shape[0], self.shape[0]
                        )
                    ),
                    location=(
                        "Matrix.__setitem__(x: Int, y: Slice, value: Matrix)"
                    ),
                )
            )
        var x_norm = Validator.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_y = range(start_y, end_y, step_y)
        var len_range_y: Int = len(range_y)

        if len_range_y != value.shape[1] or value.shape[0] != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch when assigning to slice: target shape"
                        " ({}, {}) vs value shape ({}, {}). Shapes must match"
                        " exactly.".format(
                            1, len_range_y, value.shape[0], value.shape[1]
                        )
                    ),
                    location=(
                        "Matrix.__setitem__(x: Int, y: Slice, value: Matrix)"
                    ),
                )
            )

        var col = 0
        for j in range_y:
            self._store(x_norm, j, value._load(0, col))
            col += 1

    fn __setitem__(self, x: Slice, y: Slice, value: Matrix[Self.dtype]) raises:
        """
        Assign values to a submatrix of the matrix defined by row slice `x` and column slice `y` using the provided `value` matrix. This method replaces the elements in the specified row and column slices with the corresponding elements from `value`.

        Args:
            x: Row slice specifying which rows to assign to. Supports Python slice syntax (e.g., `start:stop:step`).
            y: Column slice specifying which columns to assign to. Supports Python slice syntax (e.g., `start:stop:step`).
            value: A `Matrix` instance containing the values to assign.

        Raises:
            Error: If the shape of `value` does not match the shape of the target slice.

        Notes:
            - Out of bounds slices are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(4, 4))
            var submatrix = Matrix.ones(shape=(2, 2))
            mat[1:3, 1:3] = submatrix  # Set the 2x2 submatrix starting at (1,1) to ones
            ```
        """
        var start_x: Int
        var end_x: Int
        var step_x: Int
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_x = range(start_x, end_x, step_x)
        var range_y = range(start_y, end_y, step_y)

        if len(range_x) != value.shape[0] or len(range_y) != value.shape[1]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch when assigning to slice: target shape"
                        " ({}, {}) vs value shape ({}, {}). Shapes must match"
                        " exactly.".format(
                            len(range_x),
                            len(range_y),
                            value.shape[0],
                            value.shape[1],
                        )
                    ),
                    location="Matrix.set(x: Slice, y: Slice, value: Matrix)",
                )
            )

        var row = 0
        for i in range_x:
            var col = 0
            for j in range_y:
                self._store(i, j, value._load(row, col))
                col += 1
            row += 1

    fn set(self, x: Slice, y: Slice, value: Matrix[Self.dtype]) raises:
        """
        Assign values to a submatrix of the matrix defined by row slice `x` and column slice `y` using the provided `value` matrix. This method replaces the elements in the specified row and column slices with the corresponding elements from `value`.

        Args:
            x: Row slice specifying which rows to assign to. Supports Python slice syntax (e.g., `start:stop:step`).
            y: Column slice specifying which columns to assign to. Supports Python slice syntax (e.g., `start:stop:step`).
            value: A `Matrix` instance containing the values to assign.

        Raises:
            Error: If the shape of `value` does not match the shape of the target slice.

        Notes:
            - Out of bounds slices are clamped using the shape of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.zeros(shape=(4, 4))
            var submatrix = Matrix.ones(shape=(2, 2))
            mat.set(Slice(1, 3), Slice(1, 3), submatrix)  # Set the 2x2 submatrix starting at (1,1) to ones

            var view = submatrix.view() # create a view of submatrix
            mat.set(Slice(2, 4), Slice(2, 4), view
            )  # Set the 2x2 submatrix starting at (2,2) to the view
            ```
        """
        var start_x: Int
        var end_x: Int
        var step_x: Int
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_x = range(start_x, end_x, step_x)
        var range_y = range(start_y, end_y, step_y)

        if len(range_x) != value.shape[0] or len(range_y) != value.shape[1]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch when assigning to slice: target shape"
                        " ({}, {}) vs value shape ({}, {}). Shapes must match"
                        " exactly.".format(
                            len(range_x),
                            len(range_y),
                            value.shape[0],
                            value.shape[1],
                        )
                    ),
                    location=(
                        "Matrix.__setitem__(x: Slice, y: Slice, value: Matrix)"
                    ),
                )
            )

        var row = 0
        for i in range_x:
            var col = 0
            for j in range_y:
                self._store(i, j, value._load(row, col))
                col += 1
            row += 1

    fn _store[
        width: Int = 1
    ](self, x: Int, y: Int, simd: SIMD[Self.dtype, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.ptr.store(x * self.strides[0] + y * self.strides[1], simd)

    fn _store_idx[width: Int = 1](self, idx: Int, val: SIMD[Self.dtype, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.ptr.store(idx, val)

    fn store[
        width: Int = 1
    ](self, idx: Int, val: SIMD[Self.dtype, width]) raises:
        """
        Store a SIMD element into the matrix at the specified linear index.

        Parameters:
            width: The width of the SIMD element to store. Defaults to 1.

        Args:
            idx: The linear index where the element will be stored. Negative indices are supported and follow Python conventions.
            val: The SIMD element to store at the given index.

        Raises:
            Error: If the provided index is out of bounds.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.ones(shape=(4, 4))
            var simd_element = SIMD[f64, 4](2.0, 2.0, 2.0, 2.0)
            mat.store[4](2, simd_element)  # Store a SIMD element of width 4 at index 2
            ```
        """
        if idx >= self.size or idx < -self.size:
            raise Error(
                NumojoError(
                    category="index",
                    message=(
                        "Index {} out of bounds for matrix size {}. Use indices"
                        " in [{}, {}).".format(
                            idx, self.size, -self.size, self.size
                        )
                    ),
                    location="Matrix.__setitem__(idx: Int, val: Scalar)",
                )
            )
        var idx_norm = Validator.normalize(idx, self.size)
        self._buf.ptr.store[width=width](idx_norm, val)

    # ===-------------------------------------------------------------------===#
    # Other dunders and auxiliary methods
    # ===-------------------------------------------------------------------===#
    # fn view(self) -> Matrix[Self.dtype]:
    #     """
    #     Return a non-owning view of the matrix. This method creates and returns a `Matrix` that references the data of the original matrix. The view does not allocate new memory and directly points to the existing data buffer. Modifications to the view affect the original matrix.

    #     Returns:
    #         A `Matrix` referencing the original matrix data.

    #     Example:
    #         ```mojo
    #         from numojo.prelude import *
    #         var mat = Matrix.rand((4, 4))
    #         var mat_view = mat.view()  # Create a view of the original matrix
    #         ```
    #     """
    #     return Matrix[Self.dtype](
    #         shape=self.shape,
    #         strides=self.strides,
    #         data=self._buf.share(),
    #     )

    fn __iter__(ref self) -> Self.IteratorType[True, True]:
        """
        Returns an iterator over the rows of the Matrix. Each iteration yields a Matrix representing a single row.

        Returns:
            Iterator that yields Matrix objects for each row.

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.rand((4, 4))
            for row in mat:
                print(row)  # Each row is a Matrix
            ```
        """
        return Self.IteratorType[True, True](
            index=0,
            matrix_buf=self._buf,
            shape=self.shape,
            strides=self.strides,
        )

    fn __len__(self) -> Int:
        """
        Return the number of rows in the matrix (length of the first dimension).

        Returns:
            The number of rows (self.shape[0]).

        Example:
            ```mojo
            from numojo.prelude import *
            var mat = Matrix.rand((4, 4))
            print(len(mat))  # Outputs: 4
            ```
        """
        return self.shape[0]

    # TODO: The creation of iterators are a bit convoluted rn, clean it up
    fn __reversed__(
        ref self,
    ) -> Self.IteratorType[True, False]:
        """
        Return an iterator that traverses the matrix rows in reverse order.

        Returns:
            A reversed iterator over the rows of the matrix, yielding copies of each row.
        """
        return Self.IteratorType[True, False](
            index=self.shape[0] - 1,
            matrix_buf=self._buf,
            shape=self.shape,
            strides=self.strides,
        )

    fn __str__(self) -> String:
        """
        Return a string representation of the matrix.

        Returns:
            A string showing the matrix contents, shape, strides, order, and ownership.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Write the string representation of the matrix to a writer.

        Args:
            writer: The writer to output the matrix string to.
        """

        fn print_row(self: Self, i: Int, sep: String) raises -> String:
            var result: String = String("[")
            var number_of_sep: Int = 1
            if self.shape[1] <= 6:
                for j in range(self.shape[1]):
                    if j == self.shape[1] - 1:
                        number_of_sep = 0
                    result += String(self[i, j]) + sep * number_of_sep
            else:
                for j in range(3):
                    result += String(self[i, j]) + sep
                result += String("...") + sep
                for j in range(self.shape[1] - 3, self.shape[1]):
                    if j == self.shape[1] - 1:
                        number_of_sep = 0
                    result += String(self[i, j]) + sep * number_of_sep
            result += String("]")
            return result

        var sep: String = String("\t")
        var newline: String = String("\n ")
        var number_of_newline: Int = 1
        var result: String = "["

        try:
            if self.shape[0] <= 6:
                for i in range(self.shape[0]):
                    if i == self.shape[0] - 1:
                        number_of_newline = 0
                    result += (
                        print_row(self, i, sep) + newline * number_of_newline
                    )
            else:
                for i in range(3):
                    result += print_row(self, i, sep) + newline
                result += String("...") + newline
                for i in range(self.shape[0] - 3, self.shape[0]):
                    if i == self.shape[0] - 1:
                        number_of_newline = 0
                    result += (
                        print_row(self, i, sep) + newline * number_of_newline
                    )
            result += String("]")
            writer.write(
                result
                + "\nDType: "
                + String(self.dtype)
                + "  Shape: "
                + String(self.shape[0])
                + "x"
                + String(self.shape[1])
                + "  Strides: "
                + String(self.strides[0])
                + ","
                + String(self.strides[1])
                + "  order: "
                + String("C" if self.flags["C_CONTIGUOUS"] else "F")
                + "  Own: "
                + String(self.flags["OWNDATA"])
            )
        except e:
            print("Cannot transfer matrix to string!", e)

    # ===-------------------------------------------------------------------===#
    # Arithmetic dunder methods
    # ===-------------------------------------------------------------------===#

    fn __add__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Add two matrices element-wise.

        Args:
            other: Matrix to add to self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix containing the element-wise sum.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            print(A + B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__add__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__add__
            ](broadcast_to[Self.dtype](self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__add__
            ](self, broadcast_to[Self.dtype](other, self.shape, self.order()))

    fn __add__(self, other: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Add a scalar to every element of the matrix.

        Args:
            other: Scalar value to add.

        Returns:
            A new Matrix with the scalar added to each element.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(A + 2)
            ```
        """
        return self + broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __radd__(self, other: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Add a matrix to a scalar (right-hand side).

        Args:
            other: Scalar value to add.

        Returns:
            A new Matrix with the scalar added to each element.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            print(2 + A)
            ```
        """
        return broadcast_to[Self.dtype](other, self.shape, self.order()) + self

    fn __sub__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Subtract two matrices element-wise.

        Args:
            other: Matrix to subtract from self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix containing the element-wise difference.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            print(A - B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__sub__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__sub__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__sub__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __sub__(self, other: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Subtract a scalar from every element of the matrix.

        Args:
            other: Scalar value to subtract.

        Returns:
            A new Matrix with the scalar subtracted from each element.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(A - 2)
            ```
        """
        return self - broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __rsub__(self, other: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Subtract a matrix from a scalar (right-hand side).

        Args:
            other: Scalar value to subtract from.

        Returns:
            A new Matrix with each element being the scalar minus the corresponding matrix element.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(2 - A)
            ```
        """
        return broadcast_to[Self.dtype](other, self.shape, self.order()) - self

    fn __mul__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Multiply two matrices element-wise.

        Args:
            other: Matrix to multiply with self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix containing the element-wise product.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            print(A * B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__mul__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__mul__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__mul__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __mul__(self, other: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Multiply matrix by scalar.

        Args:
            other: Scalar value to multiply.

        Returns:
            A new Matrix with each element multiplied by the scalar.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(A * 2)
            ```
        """
        return self * broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __rmul__(self, other: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Multiply scalar by matrix (right-hand side).

        Args:
            other: Scalar value to multiply.

        Returns:
            A new Matrix with each element multiplied by the scalar.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(2 * A)
            ```
        """
        return broadcast_to[Self.dtype](other, self.shape, self.order()) * self

    fn __truediv__(
        self, other: Matrix[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """
        Divide two matrices element-wise.

        Args:
            other: Matrix to divide self by. Must be broadcastable to self's shape.

        Returns:
            A new Matrix containing the element-wise division result.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            print(A / B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__truediv__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__truediv__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__truediv__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __truediv__(
        self, other: Scalar[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """
        Divide matrix by scalar.

        Args:
            other: Scalar value to divide each element of the matrix by.

        Returns:
            A new Matrix with each element divided by the scalar.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(A / 2)
            ```
        """
        return self / broadcast_to[Self.dtype](
            other, self.shape, order=self.order()
        )

    fn __pow__(self, rhs: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Raise each element of the matrix to the power of `rhs`.

        Args:
            rhs: The scalar exponent to which each element of the matrix will be raised.

        Returns:
            A new Matrix where each element is self[i] ** rhs.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(A ** 2)
            ```
        """
        var result: Matrix[Self.dtype] = Matrix[Self.dtype](
            shape=self.shape, order=self.order()
        )
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_pow[w: Int](i: Int) unified {mut result, read self, read rhs}:
            var vec = self._buf.ptr.load[width=w](i)
            result._buf.ptr.store(i, vec.__pow__(rhs))

        vectorize[width](self.size, vec_pow)
        return result^

    fn __iadd__(mut self, other: Matrix[Self.dtype, **_]) raises:
        """
        Add another matrix to this matrix in-place.

        Args:
            other: Matrix to add. Must be broadcastable to self's shape.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            A += B  # A is modified in-place
            ```
        """
        if (self.shape[0] != other.shape[0]) or (
            self.shape[1] != other.shape[1]
        ):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch: cannot add matrix of shape ({}, {}) to"
                        " ({}, {}). Shapes must match exactly.".format(
                            other.shape[0],
                            other.shape[1],
                            self.shape[0],
                            self.shape[1],
                        )
                    ),
                    location="Matrix.__iadd__",
                )
            )
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_add[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            var b = other._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a + b)

        vectorize[width](self.size, vec_add)

    fn __iadd__(mut self, other: Scalar[Self.dtype]):
        """
        Add a scalar to this matrix in-place.

        Args:
            other: Scalar value to add to each element.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            A += 2  # A is modified in-place
            ```
        """
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_add_scalar[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a + other)

        vectorize[width](self.size, vec_add_scalar)

    fn __isub__(mut self, other: Matrix[Self.dtype, **_]) raises:
        """
        Subtract another matrix from this matrix in-place.

        Args:
            other: Matrix to subtract. Must be broadcastable to self's shape.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            A -= B  # A is modified in-place
            ```
        """
        if (self.shape[0] != other.shape[0]) or (
            self.shape[1] != other.shape[1]
        ):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch: cannot subtract matrix of shape ({},"
                        " {}) from ({}, {}). Shapes must match exactly.".format(
                            other.shape[0],
                            other.shape[1],
                            self.shape[0],
                            self.shape[1],
                        )
                    ),
                    location="Matrix.__isub__",
                )
            )
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_sub[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            var b = other._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a - b)

        vectorize[width](self.size, vec_sub)

    fn __isub__(mut self, other: Scalar[Self.dtype]):
        """
        Subtract a scalar from this matrix in-place.

        Args:
            other: Scalar value to subtract from each element.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            A -= 2  # A is modified in-place
            ```
        """
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_sub_scalar[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a - other)

        vectorize[width](self.size, vec_sub_scalar)

    fn __imul__(mut self, other: Matrix[Self.dtype, **_]) raises:
        """
        Multiply this matrix by another matrix element-wise in-place.

        Args:
            other: Matrix to multiply with. Must be broadcastable to self's shape.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.full(shape=(4, 4), fill_value=2)
            A *= B  # A is modified in-place
            ```
        """
        if (self.shape[0] != other.shape[0]) or (
            self.shape[1] != other.shape[1]
        ):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch: cannot multiply matrix of shape ({},"
                        " {}) with ({}, {}). Shapes must match exactly.".format(
                            other.shape[0],
                            other.shape[1],
                            self.shape[0],
                            self.shape[1],
                        )
                    ),
                    location="Matrix.__imul__",
                )
            )
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_mul[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            var b = other._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a * b)

        vectorize[width](self.size, vec_mul)

    fn __imul__(mut self, other: Scalar[Self.dtype]):
        """
        Multiply this matrix by a scalar in-place.

        Args:
            other: Scalar value to multiply each element by.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            A *= 2  # A is modified in-place
            ```
        """
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_mul_scalar[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a * other)

        vectorize[width](self.size, vec_mul_scalar)

    fn __itruediv__(mut self, other: Matrix[Self.dtype, **_]) raises:
        """
        Divide this matrix by another matrix element-wise in-place.

        Args:
            other: Matrix to divide by. Must be broadcastable to self's shape.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.full(shape=(4, 4), fill_value=2)
            A /= B  # A is modified in-place
            ```
        """
        if (self.shape[0] != other.shape[0]) or (
            self.shape[1] != other.shape[1]
        ):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch: cannot divide matrix of shape ({}, {})"
                        " by ({}, {}). Shapes must match exactly.".format(
                            self.shape[0],
                            self.shape[1],
                            other.shape[0],
                            other.shape[1],
                        )
                    ),
                    location="Matrix.__itruediv__",
                )
            )
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_div[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            var b = other._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a / b)

        vectorize[width](self.size, vec_div)

    fn __itruediv__(mut self, other: Scalar[Self.dtype]):
        """
        Divide this matrix by a scalar in-place.

        Args:
            other: Scalar value to divide each element by.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            A /= 2  # A is modified in-place
            ```
        """
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_div_scalar[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a / other)

        vectorize[width](self.size, vec_div_scalar)

    fn __floordiv__(
        self, other: Matrix[Self.dtype, **_]
    ) raises -> Matrix[Self.dtype]:
        """
        Floor divide two matrices element-wise.

        Args:
            other: Matrix to divide by. Must be broadcastable to self's shape.

        Returns:
            A new Matrix containing the element-wise floor division result.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            var B = Matrix.full[f64](shape=(4, 4), fill_value=2.0)
            print(A // B)  # Element-wise floor division
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__floordiv__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__floordiv__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__floordiv__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __floordiv__(
        self, other: Scalar[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """
        Floor divide matrix by scalar.

        Args:
            other: Scalar value to divide each element by.

        Returns:
            A new Matrix with each element floor divided by the scalar.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            print(A // 2)
            ```
        """
        return self // broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __ifloordiv__(mut self, other: Matrix[Self.dtype, **_]) raises:
        """
        Floor divide this matrix by another matrix element-wise in-place.

        Args:
            other: Matrix to divide by. Must be same shape as self.

        Raises:
            Error: If the shapes do not match.

        Example:
            ```mojo
            from numojo.prelude import *
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            var B = Matrix.full[f64](shape=(4, 4), fill_value=2.0)
            A //= B  # A is modified in-place
            ```
        """
        if (self.shape[0] != other.shape[0]) or (
            self.shape[1] != other.shape[1]
        ):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch: cannot floor divide matrix of shape"
                        " ({}, {}) by ({}, {}). Shapes must match exactly."
                        .format(
                            self.shape[0],
                            self.shape[1],
                            other.shape[0],
                            other.shape[1],
                        )
                    ),
                    location="Matrix.__ifloordiv__",
                )
            )
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_floordiv[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            var b = other._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a // b)

        vectorize[width](self.size, vec_floordiv)

    fn __ifloordiv__(mut self, other: Scalar[Self.dtype]):
        """
        Floor divide this matrix by a scalar in-place.

        Args:
            other: Scalar value to divide each element by.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            A //= 2  # A is modified in-place
            ```
        """
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_floordiv_scalar[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a // other)

        vectorize[width](self.size, vec_floordiv_scalar)

    fn __mod__(
        self, other: Matrix[Self.dtype, **_]
    ) raises -> Matrix[Self.dtype]:
        """
        Compute element-wise modulo of two matrices.

        Args:
            other: Matrix to compute modulo with. Must be broadcastable to self's shape.

        Returns:
            A new Matrix containing the element-wise modulo result.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            var B = Matrix.full[f64](shape=(4, 4), fill_value=3.0)
            print(A % B)  # Element-wise modulo
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__mod__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__mod__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                Self.dtype, SIMD.__mod__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __mod__(self, other: Scalar[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Compute element-wise modulo of matrix with scalar.

        Args:
            other: Scalar value to compute modulo with.

        Returns:
            A new Matrix with each element modulo the scalar.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            print(A % 3)
            ```
        """
        return self % broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __imod__(mut self, other: Matrix[Self.dtype, **_]) raises:
        """
        Compute element-wise modulo with another matrix in-place.

        Args:
            other: Matrix to compute modulo with. Must be same shape as self.

        Raises:
            Error: If the shapes do not match.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            var B = Matrix.full[f64](shape=(4, 4), fill_value=3.0)
            A %= B  # A is modified in-place
            ```
        """
        if (self.shape[0] != other.shape[0]) or (
            self.shape[1] != other.shape[1]
        ):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Shape mismatch: cannot compute modulo of matrix of"
                        " shape ({}, {}) with ({}, {}). Shapes must match"
                        " exactly.".format(
                            self.shape[0],
                            self.shape[1],
                            other.shape[0],
                            other.shape[1],
                        )
                    ),
                    location="Matrix.__imod__",
                )
            )
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_mod[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            var b = other._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a % b)

        vectorize[width](self.size, vec_mod)

    fn __imod__(mut self, other: Scalar[Self.dtype]):
        """
        Compute element-wise modulo with a scalar in-place.

        Args:
            other: Scalar value to compute modulo with.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f64](shape=(4, 4), fill_value=7.0)
            A %= 3  # A is modified in-place
            ```
        """
        comptime width = simd_width_of[Self.dtype]()

        @parameter
        fn vec_mod_scalar[w: Int](i: Int) unified {mut self, read other}:
            var a = self._buf.ptr.load[width=w](i)
            self._buf.ptr.store(i, a % other)

        vectorize[width](self.size, vec_mod_scalar)

    fn __lt__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare two matrices element-wise for less-than.

        Args:
            other: Matrix to compare with self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] < other[i, j], else False.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4)) * 2
            print(A < B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.lt](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.lt](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.lt](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __lt__(self, other: Scalar[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare each element of the matrix to a scalar for less-than.

        Args:
            other: Scalar value to compare.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] < other, else False.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            print(A < 2)
            ```
        """
        return self < broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __le__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare two matrices element-wise for less-than-or-equal.

        Args:
            other: Matrix to compare with self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] <= other[i, j], else False.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4)) * 2
            print(A <= B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.le](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.le](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.le](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __le__(self, other: Scalar[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare each element of the matrix to a scalar for less-than-or-equal.

        Args:
            other: Scalar value to compare.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] <= other, else False.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            print(A <= 2)
            ```
        """
        return self <= broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __gt__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare two matrices element-wise for greater-than.

        Args:
            other: Matrix to compare with self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] > other[i, j], else False.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            B = Matrix.ones(shape=(4, 4)) * 2
            print(A > B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.gt](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.gt](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.gt](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __gt__(self, other: Scalar[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare each element of the matrix to a scalar for greater-than.

        Args:
            other: Scalar value to compare.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] > other, else False.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            print(A > 2)
            ```
        """
        return self > broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __ge__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare two matrices element-wise for greater-than-or-equal.

        Args:
            other: Matrix to compare with self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] >= other[i, j], else False.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            B = Matrix.ones(shape=(4, 4)) * 2
            print(A >= B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.ge](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.ge](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.ge](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __ge__(self, other: Scalar[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare each element of the matrix to a scalar for greater-than-or-equal.

        Args:
            other: Scalar value to compare.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] >= other, else False.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            print(A >= 2)
            ```
        """
        return self >= broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __eq__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare two matrices element-wise for equality.

        Args:
            other: Matrix to compare with self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] == other[i, j], else False.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            print(A == B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.eq](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.eq](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.eq](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __eq__(self, other: Scalar[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare each element of the matrix to a scalar for equality.

        Args:
            other: Scalar value to compare.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] == other, else False.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            print(A == 2)
            ```
        """
        return self == broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __ne__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare two matrices element-wise for inequality.

        Args:
            other: Matrix to compare with self. Must be broadcastable to self's shape.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] != other[i, j], else False.

        Raises:
            Error: If the shapes are not compatible for broadcasting.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 4))
            var B = Matrix.ones(shape=(4, 4))
            print(A != B)
            ```
        """
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.ne](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.ne](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[Self.dtype, SIMD.ne](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __ne__(self, other: Scalar[Self.dtype]) raises -> Matrix[DType.bool]:
        """
        Compare each element of the matrix to a scalar for inequality.

        Args:
            other: Scalar value to compare.

        Returns:
            A new Matrix[bool] where each element is True if self[i, j] != other, else False.

        Example:
            ```mojo
            from numojo.prelude import *
            A = Matrix.ones(shape=(4, 4))
            print(A != 2)
            ```
        """
        return self != broadcast_to[Self.dtype](other, self.shape, self.order())

    fn __matmul__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """
        Matrix multiplication using the @ operator.

        Args:
            other: The matrix to multiply with self.

        Returns:
            A new Matrix containing the result of matrix multiplication.

        Raises:
            Error: If the shapes are not compatible for matrix multiplication.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones(shape=(4, 3))
            var B = Matrix.ones(shape=(3, 2))
            print(A @ B)
            ```
        """
        return numojo.linalg.matmul(self, other)

    # # ===-------------------------------------------------------------------===#
    # # Core methods
    # # ===-------------------------------------------------------------------===#

    # FIXME: These return types be Scalar[DType.bool] or Matrix[DType.bool] instead to match numpy. Fix the docstring examples too.
    fn all(self) -> Scalar[Self.dtype]:
        """
        Returns True if all elements of the matrix evaluate to True.

        Returns:
            Scalar[dtype]: True if all elements are True, otherwise False.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(1), 1, 1, 1, 1], (5, 1))
            print(A.all())  # Outputs: True
            var B = Matrix.fromlist([Float64(1), 0, 2, 3, 4], (5, 1))
            print(B.all())  # Outputs: False
            ```
        """
        return numojo.logic.all(self)

    fn all(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Returns a matrix indicating whether all elements along the specified axis evaluate to True.

        Args:
            axis: The axis along which to perform the test.

        Returns:
            Matrix[dtype]: Matrix of boolean values for each slice along the axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist(
               [Float64(1), 1, 1, 0, 1, 3], (2, 3)
            )
            print(A.all(axis=0))  # Outputs: [[0, 1, 1]]
            print(A.all(axis=1))  # Outputs: [[1], [0]]
            ```
        """
        return numojo.logic.all[Self.dtype](self, axis=axis)

    fn any(self) -> Scalar[Self.dtype]:
        """
        Returns True if any element of the matrix evaluates to True.

        Returns:
            Scalar[dtype]: True if any element is True, otherwise False.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(0), 0, 0, 0, 0], (5, 1))
            print(A.any())  # Outputs: False
            var B = Matrix.fromlist([Float64(0), 2, 0, 0, 0], (5, 1))
            print(B.any())  # Outputs: True
            ```
        """
        return numojo.logic.any(self)

    fn any(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Returns a matrix indicating whether any element along the specified axis evaluates to True.

        Args:
            axis: The axis along which to perform the test.

        Returns:
            Matrix[dtype]: Matrix of boolean values for each slice along the axis.
        """
        return numojo.logic.any(self, axis=axis)

    fn argmax(self) raises -> Scalar[DType.int]:
        """
        Returns the index of the maximum element in the flattened matrix.

        Returns:
            Scalar[DType.int]: Index of the maximum element.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(1), 3, 2, 5, 4], (5, 1))
            print(A.argmax())  # Outputs: 3
            ```
        """
        return numojo.math.argmax(self)

    fn argmax(self, axis: Int) raises -> Matrix[DType.int]:
        """
        Returns the indices of the maximum elements along the specified axis.

        Args:
            axis: The axis along which to find the maximum.

        Returns:
            Matrix[DType.int]: Indices of the maximum elements along the axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(1), 3, 2, 5, 4, 6], (2, 3))
            print(A.argmax(axis=0))  # Outputs: [[1, 1, 1]]
            print(A.argmax(axis=1))  # Outputs: [[1], [2]]
            ```
        """
        return numojo.math.argmax(self, axis=axis)

    fn argmin(self) raises -> Scalar[DType.int]:
        """
        Returns the index of the minimum element in the flattened matrix.

        Returns:
            Scalar[DType.int]: Index of the minimum element.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(3), 1, 4, 2, 5], (5, 1))
            print(A.argmin())  # Outputs: 1
            ```
        """
        return numojo.math.argmin(self)

    fn argmin(self, axis: Int) raises -> Matrix[DType.int]:
        """
        Returns the indices of the minimum elements along the specified axis.

        Args:
            axis: The axis along which to find the minimum.

        Returns:
            Matrix[DType.int]: Indices of the minimum elements along the axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(3), 1, 4, 2, 5, 0], (2, 3))
            print(A.argmin(axis=0))  # Outputs: [[1, 1, 1]]
            print(A.argmin(axis=1))  # Outputs: [[1], [2]]
            ```
        """
        return numojo.math.argmin(self, axis=axis)

    fn argsort(self) raises -> Matrix[DType.int]:
        """
        Returns the indices that would sort the flattened matrix.

        Returns:
            Matrix[DType.int]: Indices that sort the flattened matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(3), 1, 4, 2], (4, 1))
            print(A.argsort())  # Outputs: [[1, 3, 0, 2]]
            ```
        """
        return numojo.math.argsort(self)

    fn argsort(self, axis: Int) raises -> Matrix[DType.int]:
        """
        Returns the indices that would sort the matrix along the specified axis.

        Args:
            axis: The axis along which to sort.

        Returns:
            Matrix[DType.int]: Indices that sort the matrix along the axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(3), 1, 4, 2, 5, 0], (2, 3))
            print(A.argsort(axis=0))  # Outputs: [[1, 1, 1], [0, 0, 0]]
            print(A.argsort(axis=1))  # Outputs: [[1, 3, 0], [2, 0, 1]]
            ```
        """
        return numojo.math.argsort(self, axis=axis)

    fn astype[asdtype: DType](self) -> Matrix[asdtype]:
        """
        Returns a copy of the matrix cast to the specified data type.

        Parameters:
            asdtype: The target data type to cast to.

        Returns:
            Matrix[asdtype]: A new matrix with elements cast to the specified type.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float32(1.5), 2.5, 3.5], (3, 1))
            var B = A.astype[i8]()
            print(B)  # Outputs a Matrix[i8] with values [[1], [2], [3]]
            ```
        """
        var casted_matrix = Matrix[asdtype](
            shape=(self.shape[0], self.shape[1]), order=self.order()
        )
        for i in range(self.size):
            casted_matrix._buf.ptr[i] = self._buf.ptr[i].cast[asdtype]()
        return casted_matrix^

    fn cumprod(self) raises -> Matrix[Self.dtype]:
        """
        Compute the cumulative product of all elements in the matrix, flattened into a single dimension.

        Returns:
            Matrix[dtype]: A matrix containing the cumulative product of the flattened input.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.cumprod())
            ```
        """
        return numojo.math.cumprod(self)

    fn cumprod(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Compute the cumulative product of elements along a specified axis.

        Args:
            axis: The axis along which to compute the cumulative product (0 for rows, 1 for columns).

        Returns:
            Matrix[dtype]: A matrix containing the cumulative product along the specified axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.cumprod(axis=0))
            print(A.cumprod(axis=1))
            ```
        """
        return numojo.math.cumprod(self, axis=axis)

    fn cumsum(self) raises -> Matrix[Self.dtype]:
        """
        Compute the cumulative sum of all elements in the matrix, flattened into a single dimension.

        Returns:
            Matrix[dtype]: A matrix containing the cumulative sum of the flattened input.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.cumsum())
            ```
        """
        return numojo.math.cumsum(self)

    fn cumsum(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Compute the cumulative sum of elements along a specified axis.

        Args:
            axis: The axis along which to compute the cumulative sum (0 for rows, 1 for columns).

        Returns:
            Matrix[dtype]: A matrix containing the cumulative sum along the specified axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.cumsum(axis=0))
            print(A.cumsum(axis=1))
            ```
        """
        return numojo.math.cumsum(self, axis=axis)

    fn fill(self, fill_value: Scalar[Self.dtype]):
        """
        Fill the matrix with the specified value. This method sets every element of the matrix to `fill_value`.

        Args:
            fill_value: The value to assign to every element of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            A.fill(5)
            print(A)
            ```

        See also: `Matrix.full`
        """
        for i in range(self.size):
            self._buf.ptr[i] = fill_value

    # * Make it inplace?
    fn flatten(self) -> Matrix[Self.dtype]:
        """
        Return a flattened copy of the matrix. This method returns a new matrix containing all elements of the original matrix in a single row (shape (1, size)), preserving the order.

        Returns:
            Matrix[dtype]: A new matrix with shape (1, self.size) containing the flattened data.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((2, 3))
            print(A.flatten())
            ```
        """
        var res = Matrix[Self.dtype](shape=(1, self.size), order=self.order())
        memcpy(dest=res._buf.ptr, src=self._buf.ptr, count=res.size)
        return res^

    fn inv(self) raises -> Matrix[Self.dtype]:
        """
        Compute the inverse of the matrix.

        Returns:
            Matrix[dtype]: The inverse of the matrix.

        Raises:
            Error: If the matrix is not square or not invertible.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            print(A.inv())
            ```
        """
        return numojo.linalg.inv(self)

    fn order(self) -> String:
        """
        Return the memory layout order of the matrix.

        Returns:
            String: "C" if the matrix is C-contiguous, "F" if F-contiguous.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3), order="F")
            print(A.order())  # "F"
            ```
        """
        var order: String = "F"
        if self.flags.C_CONTIGUOUS:
            order = "C"
        return order

    fn max(self) raises -> Scalar[Self.dtype]:
        """
        Return the maximum element in the matrix.

        The matrix is flattened before finding the maximum.

        Returns:
            Scalar[dtype]: The maximum value in the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            print(A.max())
            ```
        """
        return numojo.math.extrema.max(self)

    fn max(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Return the maximum values along the specified axis.

        Args:
            axis: The axis along which to compute the maximum (0 for rows, 1 for columns).

        Returns:
            Matrix[dtype]: A matrix containing the maximum values along the given axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            print(A.max(axis=0))  # Max of each column
            print(A.max(axis=1))  # Max of each row
            ```
        """
        return numojo.math.extrema.max(self, axis=axis)

    fn mean[
        returned_dtype: DType = DType.float64
    ](self) raises -> Scalar[returned_dtype]:
        """
        Compute the arithmetic mean of all elements in the matrix.

        Returns:
            Scalar[returned_dtype]: The mean value of all elements.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.mean())
            ```
        """
        return numojo.statistics.mean[returned_dtype](self)

    fn mean[
        returned_dtype: DType = DType.float64
    ](self, axis: Int) raises -> Matrix[returned_dtype]:
        """
        Compute the arithmetic mean along the specified axis.

        Args:
            axis: The axis along which to compute the mean (0 for rows, 1 for columns).

        Returns:
            Matrix[returned_dtype]: The mean values along the given axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.mean(axis=0))
            print(A.mean(axis=1))
            ```
        """
        return numojo.statistics.mean[returned_dtype](self, axis=axis)

    fn min(self) raises -> Scalar[Self.dtype]:
        """
        Return the minimum element in the matrix.

        The matrix is flattened before finding the minimum.

        Returns:
            Scalar[dtype]: The minimum value in the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            print(A.min())
            ```
        """
        return numojo.math.extrema.min(self)

    fn min(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Return the minimum values along the specified axis.

        Args:
            axis: The axis along which to compute the minimum (0 for rows, 1 for columns).

        Returns:
            Matrix[dtype]: The minimum values along the given axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            print(A.min(axis=0))  # Min of each column
            print(A.min(axis=1))  # Min of each row
            ```
        """
        return numojo.math.extrema.min(self, axis=axis)

    fn prod(self) -> Scalar[Self.dtype]:
        """
        Compute the product of all elements in the matrix.

        Returns:
            Scalar[dtype]: The product of all elements.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.prod())
            ```
        """
        return numojo.math.prod(self)

    fn prod(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Compute the product of elements along the specified axis.

        Args:
            axis: The axis along which to compute the product (0 for rows, 1 for columns).

        Returns:
            Matrix[dtype]: The product values along the given axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.prod(axis=0))
            print(A.prod(axis=1))
            ```
        """
        return numojo.math.prod(self, axis=axis)

    fn reshape(
        self, shape: Tuple[Int, Int], order: String = "C"
    ) raises -> Matrix[Self.dtype]:
        """
        Return a new matrix with the specified shape containing the same data.

        Args:
            shape: Tuple of (rows, columns) specifying the new shape.
            order: Memory layout order of the new matrix. "C" for C-contiguous, "F" for F-contiguous. Default is "C".

        Returns:
            Matrix[dtype]: A new matrix with the requested shape.

        Raises:
            Error: If the total number of elements does not match the original matrix size.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(4, 4))
            var B = A.reshape((2, 8))
            print(B)
            ```
        """
        if shape[0] * shape[1] != self.size:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Cannot reshape matrix of size {} into shape ({}, {})."
                        " Total elements must match.".format(
                            self.size, shape[0], shape[1]
                        )
                    ),
                    location="Matrix.reshape",
                )
            )
        var res = Matrix[Self.dtype](shape=shape, order=order)

        if self.flags.C_CONTIGUOUS and order == "F":
            for i in range(shape[0]):
                for j in range(shape[1]):
                    var flat_idx = i * shape[1] + j
                    res._buf[
                        j * res.strides[1] + i * res.strides[0]
                    ] = self._buf[flat_idx]
        elif self.flags.F_CONTIGUOUS and order == "C":
            var k = 0
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    var val = self._buf.ptr[
                        row * self.strides[0] + col * self.strides[1]
                    ]
                    var dest_row = Int(k // shape[1])
                    var dest_col = k % shape[1]
                    res._buf.ptr[
                        dest_row * res.strides[0] + dest_col * res.strides[1]
                    ] = val
                    k += 1
        else:
            memcpy(dest=res._buf.ptr, src=self._buf.ptr, count=res.size)
        return res^

    # NOTE: not sure if `where` clause works correctly here yet.
    fn resize(mut self, shape: Tuple[Int, Int]) raises:
        """
        Change the shape and size of the matrix in-place.

        Args:
            shape: Tuple of (rows, columns) specifying the new shape.

        Raises:
            Error: If the new shape requires more elements than the current matrix can hold and memory allocation fails.

        Notes:
            - If the new shape is larger, the matrix is reallocated and new elements are zero-initialized.
            - If the new shape is smaller, the matrix shape and strides are updated without reallocating memory.
            - Only allowed for matrices with own_data=True.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(2, 3))
            A.resize((4, 5))
            print(A)
            ```
        """
        if shape[0] * shape[1] > self.size:
            var other = Matrix[Self.dtype](shape=shape, order=self.order())
            if self.flags.C_CONTIGUOUS:
                memcpy(dest=other._buf.ptr, src=self._buf.ptr, count=self.size)
                for i in range(self.size, other.size):
                    other._buf.ptr[i] = 0
            else:
                var min_rows = min(self.shape[0], shape[0])
                var min_cols = min(self.shape[1], shape[1])

                for j in range(min_cols):
                    for i in range(min_rows):
                        other._buf.ptr[i + j * shape[0]] = self._buf.ptr[
                            i + j * self.shape[0]
                        ]
                    for i in range(min_rows, shape[0]):
                        other._buf.ptr[i + j * shape[0]] = 0

                # Zero the additional columns
                for j in range(min_cols, shape[1]):
                    for i in range(shape[0]):
                        other._buf.ptr[i + j * shape[0]] = 0

            self = other^
        else:
            self.shape[0] = shape[0]
            self.shape[1] = shape[1]
            self.size = shape[0] * shape[1]

            if self.flags.C_CONTIGUOUS:
                self.strides[0] = shape[1]
            else:
                self.strides[1] = shape[0]

    fn round(self, decimals: Int) raises -> Matrix[Self.dtype]:
        """
        Round each element of the matrix to the specified number of decimals.

        Args:
            decimals: Number of decimal places to round to.

        Returns:
            Matrix[dtype]: A new matrix with rounded values.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(1.12345), 2.67891, 3.14159], (3, 1))
            var B = A.round(2)
            print(B)  # Outputs a Matrix[Float64] with values [[1.12], [2.68], [3.14]]
            ```
        """
        return numojo.math.rounding.round(self, decimals=decimals)

    fn std[
        returned_dtype: DType = DType.float64
    ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
        """
        Compute the standard deviation of all elements in the matrix.

        Args:
            ddof: Delta degrees of freedom. The divisor used in calculations is N - ddof, where N is the number of elements.

        Returns:
            Scalar[returned_dtype]: The standard deviation of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.std())
            ```
        """
        return numojo.statistics.std[returned_dtype](self, ddof=ddof)

    fn std[
        returned_dtype: DType = DType.float64
    ](self, axis: Int, ddof: Int = 0) raises -> Matrix[returned_dtype]:
        """
        Compute the standard deviation along the specified axis.

        Args:
            axis: Axis along which to compute the standard deviation (0 for rows, 1 for columns).
            ddof: Delta degrees of freedom. The divisor used in calculations is N - ddof, where N is the number of elements along the axis.

        Returns:
            Matrix[returned_dtype]: The standard deviation along the given axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.std(axis=0))
            print(A.std(axis=1))
            ```
        """
        return numojo.statistics.std[returned_dtype](self, axis=axis, ddof=ddof)

    fn sum(self) -> Scalar[Self.dtype]:
        """
        Compute the sum of all elements in the matrix.

        Returns:
            Scalar[dtype]: The sum of all elements.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.sum())
            ```
        """
        return numojo.math.sum(self)

    fn sum(self, axis: Int) raises -> Matrix[Self.dtype]:
        """
        Compute the sum of elements along the specified axis.

        Args:
            axis: Axis along which to sum (0 for rows, 1 for columns).

        Returns:
            Matrix[dtype]: The sum along the given axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.sum(axis=0))
            print(A.sum(axis=1))
            ```
        """
        return numojo.math.sum(self, axis=axis)

    fn trace(self) raises -> Scalar[Self.dtype]:
        """
        Compute the trace of the matrix (sum of diagonal elements).

        Returns:
            Scalar[dtype]: The trace value.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist(
                [Float64(1), 2, 3, 4, 5, 6, 7, 8, 9], (3, 3)
            )
            print(A.trace())  # Outputs: 15.0
            ```
        """
        return numojo.linalg.trace(self)

    fn issymmetric(self) -> Bool:
        """
        Check if the matrix is symmetric (equal to its transpose).

        Returns:
            Bool: True if the matrix is symmetric, False otherwise.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(1), 2, 2, 1], (2, 2))
            print(A.issymmetric())  # Outputs: True
            var B = Matrix.fromlist([Float64(1), 2, 3, 4], (2, 2))
            print(B.issymmetric())  # Outputs: False
            ```
        """
        return issymmetric(self)

    fn transpose(self) -> Matrix[Self.dtype]:
        """
        Return the transpose of the matrix.

        Returns:
            Matrix[dtype]: The transposed matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(1), 2, 3, 4], (2, 2))
            print(A.transpose())  # Outputs: [[1, 3], [2, 4]]
            ```
        """
        return transpose(self)

    # TODO: we should only allow this for owndata. not for views, it'll lead to weird origin behaviours.
    fn reorder_layout(self) raises -> Matrix[Self.dtype]:
        """
        Reorder the memory layout of the matrix to match its current order ("C" or "F"). This method returns a new matrix with the same data but stored in the requested memory layout. Only allowed for matrices with own_data=True.

        Returns:
            Matrix[dtype]: A new matrix with reordered memory layout.

        Raises:
            Error: If the matrix does not have its own data.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3), order="F")
            var B = A.reorder_layout()
            print(B.order())  # Outputs: "F"
            ```
        """
        return reorder_layout(self)

    fn T(self) -> Matrix[Self.dtype]:
        """
        Return the transpose of the matrix.

        Returns:
            Matrix[dtype]: The transposed matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromlist([Float64(1), 2, 3, 4], (2, 2))
            print(A.T())  # Outputs: [[1, 3], [2, 4]]
            ```
        """
        return transpose(self)

    fn variance[
        returned_dtype: DType = DType.float64
    ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
        """
        Compute the variance of all elements in the matrix.

        Args:
            ddof: Delta degrees of freedom. The divisor used in calculations is N - ddof, where N is the number of elements.

        Returns:
            Scalar[returned_dtype]: The variance of the matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.variance())
            ```
        """
        return numojo.statistics.variance[returned_dtype](self, ddof=ddof)

    fn variance[
        returned_dtype: DType = DType.float64
    ](self, axis: Int, ddof: Int = 0) raises -> Matrix[returned_dtype]:
        """
        Compute the variance along the specified axis.

        Args:
            axis: Axis along which to compute the variance (0 for rows, 1 for columns).
            ddof: Delta degrees of freedom. The divisor used in calculations is N - ddof, where N is the number of elements along the axis.

        Returns:
            Matrix[returned_dtype]: The variance along the given axis.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand(shape=(100, 100))
            print(A.variance(axis=0))
            print(A.variance(axis=1))
            ```
        """
        return numojo.statistics.variance[returned_dtype](
            self, axis=axis, ddof=ddof
        )

    # # ===-------------------------------------------------------------------===#
    # # To other data types
    # # ===-------------------------------------------------------------------===#

    fn to_ndarray(self) raises -> NDArray[Self.dtype]:
        """Create `NDArray` from `Matrix`.

        Returns a new NDArray with the same shape and data as the Matrix.
        The buffer is copied, so changes to the NDArray do not affect the original Matrix.

        Returns:
            NDArray[dtype]: A new NDArray containing the same data as the Matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            var ndarray_A = A.to_ndarray()
            print(ndarray_A)
            ```
        """

        var ndarray: NDArray[Self.dtype] = NDArray[Self.dtype](
            shape=[self.shape[0], self.shape[1]], order=self.order()
        )
        memcpy(dest=ndarray._buf.ptr, src=self._buf.ptr, count=ndarray.size)

        return ndarray^

    fn to_numpy(self) raises -> PythonObject:
        """
        Convert the Matrix to a NumPy ndarray.

        Returns:
            PythonObject: A NumPy ndarray containing the same data as the Matrix.

        Notes:
            - The returned NumPy array is a copy of the Matrix data.
            - The dtype and memory order are matched as closely as possible.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand((3, 3))
            var np_A = A.to_numpy()
            print(np_A)
            ```
        """
        try:
            var np = Python.import_module("numpy")

            var np_arr_dim = Python.list()
            np_arr_dim.append(self.shape[0])
            np_arr_dim.append(self.shape[1])

            np.set_printoptions(4)

            # Implement a dictionary for this later
            var numpyarray: PythonObject
            var np_dtype = np.float64
            if Self.dtype == DType.float16:
                np_dtype = np.float16
            elif Self.dtype == DType.float32:
                np_dtype = np.float32
            elif Self.dtype == DType.int64:
                np_dtype = np.int64
            elif Self.dtype == DType.int32:
                np_dtype = np.int32
            elif Self.dtype == DType.int16:
                np_dtype = np.int16
            elif Self.dtype == DType.int8:
                np_dtype = np.int8
            elif Self.dtype == DType.uint64:
                np_dtype = np.uint64
            elif Self.dtype == DType.uint32:
                np_dtype = np.uint32
            elif Self.dtype == DType.uint16:
                np_dtype = np.uint16
            elif Self.dtype == DType.uint8:
                np_dtype = np.uint8
            elif Self.dtype == DType.bool:
                np_dtype = np.bool_
            elif Self.dtype == DType.int:
                np_dtype = np.int64

            var order = "C" if self.flags.C_CONTIGUOUS else "F"
            numpyarray = np.empty(
                np_arr_dim, dtype=np_dtype, order=PythonObject(order)
            )
            var pointer_d = numpyarray.__array_interface__[
                PythonObject("data")
            ][0].unsafe_get_as_pointer[Self.dtype]()
            memcpy(dest=pointer_d, src=self._buf.get_ptr(), count=self.size)

            return numpyarray^

        except e:
            print("Error in converting to numpy", e)
            return PythonObject()

    # ===-----------------------------------------------------------------------===#
    # Static methods to construct matrix
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    fn full[
        datatype: DType = DType.float64
    ](
        shape: Tuple[Int, Int],
        fill_value: Scalar[datatype] = 0,
        order: String = "C",
    ) -> Matrix[datatype]:
        """
        Create a matrix of the specified shape, filled with the given value.

        Args:
            shape: Tuple specifying the matrix dimensions (rows, columns).
            fill_value: Value to fill every element of the matrix.
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Matrix filled with `fill_value`.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.full[f32](shape=(10, 10), fill_value=100)
            ```
        """

        var matrix = Matrix[datatype](shape, order)
        comptime width = simd_width_of[datatype]()

        @parameter
        fn vec_fill[w: Int](i: Int) unified {mut matrix, read fill_value}:
            matrix._buf.ptr.store(i, SIMD[datatype, w](fill_value))

        vectorize[width](matrix.size, vec_fill)
        return matrix^

    @staticmethod
    fn zeros[
        datatype: DType = DType.float64
    ](shape: Tuple[Int, Int], order: String = "C") -> Matrix[datatype]:
        """
        Create a matrix of the specified shape, filled with zeros.

        Args:
            shape: Tuple specifying the matrix dimensions (rows, columns).
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Matrix filled with zeros.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.zeros[i32](shape=(10, 10))
            ```
        """

        var res = Matrix[datatype](shape, order)
        memset_zero(res._buf.ptr, res.size)
        return res^

    @staticmethod
    fn ones[
        datatype: DType = DType.float64
    ](shape: Tuple[Int, Int], order: String = "C") -> Matrix[datatype]:
        """
        Create a matrix of the specified shape, filled with ones.

        Args:
            shape: Tuple specifying the matrix dimensions (rows, columns).
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Matrix filled with ones.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.ones[f64](shape=(10, 10))
            ```
        """

        return Matrix.full[datatype](shape=shape, fill_value=1, order=order)

    @staticmethod
    fn identity[
        datatype: DType = DType.float64
    ](len: Int, order: String = "C") -> Matrix[datatype]:
        """
        Create an identity matrix of the given size.

        Args:
            len: Size of the identity matrix (number of rows and columns).
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Identity matrix of shape (len, len).

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.identity[f16](12)
            print(A)
            ```
        """
        var matrix = Matrix.zeros[datatype]((len, len), order)
        for i in range(len):
            matrix._buf.ptr.store(
                i * matrix.strides[0] + i * matrix.strides[1], 1
            )
        return matrix^

    @staticmethod
    fn rand[
        datatype: DType = DType.float64
    ](shape: Tuple[Int, Int], order: String = "C") -> Matrix[datatype]:
        """
        Create a matrix of the specified shape, filled with random values uniformly distributed between 0 and 1.

        Args:
            shape: Tuple specifying the matrix dimensions (rows, columns).
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Matrix filled with random values.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand[f64]((12, 12))
            ```
        """
        var result = Matrix[datatype](shape, order)
        comptime width = simd_width_of[datatype]()

        @parameter
        fn vec_rand[w: Int](i: Int) unified {mut result}:
            var rand_vec = SIMD[datatype, w]()
            for j in range(w):
                rand_vec[j] = random_float64(0, 1).cast[datatype]()
            result._buf.ptr.store(i, rand_vec)

        vectorize[width](result.size, vec_rand)
        return result^

    @staticmethod
    fn empty[
        datatype: DType = DType.float64
    ](shape: Tuple[Int, Int], order: String = "C") -> Matrix[datatype]:
        """
        Create a matrix of the specified shape without initializing its values.
        This is faster than zeros() when you plan to immediately overwrite all values.

        Args:
            shape: Tuple specifying the matrix dimensions (rows, columns).
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Uninitialized matrix (values are undefined).

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.empty[f64]((10, 10))
            # Fill A with your own values
            ```

        Warning:
            The matrix values are undefined. Always initialize before reading.
        """
        return Matrix[datatype](shape, order)

    @staticmethod
    fn arange[
        datatype: DType = DType.float64
    ](
        start: Scalar[datatype],
        stop: Scalar[datatype],
        step: Scalar[datatype] = 1,
        order: String = "C",
    ) raises -> Matrix[datatype]:
        """
        Create a matrix with evenly spaced values within a given interval as a row vector.

        Args:
            start: Start of the interval (inclusive).
            stop: End of the interval (exclusive).
            step: Spacing between values.
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Row vector with evenly spaced values.

        Raises:
            Error: If step is zero or has wrong sign for the interval.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.arange[f64](0, 10, 2)  # [0, 2, 4, 6, 8]
            ```
        """
        if step == 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Step cannot be zero. Provide a non-zero step value."
                    ),
                    location="arange",
                )
            )

        var num_elements: Int
        if step > 0:
            if stop <= start:
                raise Error(
                    NumojoError(
                        category="value",
                        message=(
                            "Stop must be greater than start when step is"
                            " positive."
                        ),
                        location="arange",
                    )
                )
            num_elements = Int(ceil((stop - start) / step))
        else:
            if stop >= start:
                raise Error(
                    NumojoError(
                        category="value",
                        message=(
                            "Stop must be less than start when step is"
                            " negative."
                        ),
                        location="arange",
                    )
                )
            num_elements = Int(ceil((start - stop) / (-step)))

        var result = Matrix[datatype](shape=(1, num_elements), order=order)
        for i in range(num_elements):
            result._buf.ptr[i] = start + i * step

        return result^

    @staticmethod
    fn linspace[
        datatype: DType = DType.float64
    ](
        start: Scalar[datatype],
        stop: Scalar[datatype],
        num: Int = 50,
        order: String = "C",
    ) raises -> Matrix[datatype]:
        """
        Create a matrix with evenly spaced values over a specified interval as a row vector.

        Args:
            start: Start of the interval (inclusive).
            stop: End of the interval (inclusive).
            num: Number of values to generate.
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Row vector with linearly spaced values.

        Raises:
            Error: If num is less than 2.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.linspace[f64](0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
            ```
        """
        if num < 2:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of samples must be at least 2. Provide num"
                        " >= 2."
                    ),
                    location="linspace",
                )
            )

        var result = Matrix[datatype](shape=(1, num), order=order)
        var step = (stop - start) / (num - 1)

        for i in range(num):
            result._buf.ptr[i] = start + i * step

        return result^

    @staticmethod
    fn diag[
        datatype: DType = DType.float64
    ](diagonal: Matrix[datatype], order: String = "C") raises -> Matrix[
        datatype
    ]:
        """
        Create a diagonal matrix from a vector or extract diagonal from a matrix.

        Args:
            diagonal: Row or column vector to use as diagonal, or a matrix to extract diagonal from.
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Square diagonal matrix if input is vector, or row vector if input is matrix.

        Raises:
            Error: If input is not a vector or matrix.

        Example:
            ```mojo
            from numojo.prelude import *
            var vec = Matrix.fromlist[f64]([Float64(1), 2, 3], (1, 3))
            var diag_mat = Matrix.diag(vec)  # 3x3 diagonal matrix
            ```
        """
        # Check if input is a vector
        if diagonal.shape[0] == 1:
            # Row vector: create diagonal matrix
            var size = diagonal.shape[1]
            var result = Matrix.zeros[datatype]((size, size), order)
            for i in range(size):
                result._buf.ptr[
                    i * result.strides[0] + i * result.strides[1]
                ] = diagonal._buf.ptr[i]
            return result^
        elif diagonal.shape[1] == 1:
            # Column vector: create diagonal matrix
            var size = diagonal.shape[0]
            var result = Matrix.zeros[datatype]((size, size), order)
            for i in range(size):
                result._buf.ptr[
                    i * result.strides[0] + i * result.strides[1]
                ] = diagonal._buf.ptr[i]
            return result^
        else:
            # Matrix: extract diagonal
            var size = min(diagonal.shape[0], diagonal.shape[1])
            var result = Matrix[datatype](shape=(1, size), order=order)
            for i in range(size):
                result._buf.ptr[i] = diagonal._buf.ptr[
                    i * diagonal.strides[0] + i * diagonal.strides[1]
                ]
            return result^

    @staticmethod
    fn zeros_like[
        datatype: DType = DType.float64
    ](other: Matrix[datatype]) -> Matrix[datatype]:
        """
        Create a matrix of zeros with the same shape as the given matrix.

        Args:
            other: Matrix whose shape to copy.

        Returns:
            Matrix[datatype]: Matrix of zeros with same shape as other.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand[f64]((3, 4))
            var B = Matrix.zeros_like(A)  # 3x4 matrix of zeros
            ```
        """
        return Matrix.zeros[datatype](other.shape, other.order())

    @staticmethod
    fn ones_like[
        datatype: DType = DType.float64
    ](other: Matrix[datatype]) -> Matrix[datatype]:
        """
        Create a matrix of ones with the same shape as the given matrix.

        Args:
            other: Matrix whose shape to copy.

        Returns:
            Matrix[datatype]: Matrix of ones with same shape as other.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand[f64]((3, 4))
            var B = Matrix.ones_like(A)  # 3x4 matrix of ones
            ```
        """
        return Matrix.ones[datatype](other.shape, other.order())

    @staticmethod
    fn empty_like[
        datatype: DType = DType.float64
    ](other: Matrix[datatype]) -> Matrix[datatype]:
        """
        Create an uninitialized matrix with the same shape as the given matrix.

        Args:
            other: Matrix whose shape to copy.

        Returns:
            Matrix[datatype]: Uninitialized matrix with same shape as other.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.rand[f64]((3, 4))
            var B = Matrix.empty_like(A)  # 3x4 uninitialized matrix
            ```

        Warning:
            The matrix values are undefined. Always initialize before reading.
        """
        return Matrix.empty[datatype](other.shape, other.order())

    @staticmethod
    fn fromlist[
        datatype: DType = DType.float64
    ](
        object: List[Scalar[datatype]],
        shape: Tuple[Int, Int] = (0, 0),
        order: String = "C",
    ) raises -> Matrix[datatype]:
        """
        Create a matrix from a 1-dimensional list and reshape to the given shape.

        Args:
            object: List of values to populate the matrix.
            shape: Tuple specifying the matrix dimensions (rows, columns). If not provided, creates a row vector.
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Matrix containing the values from the list.

        Example:
            ```mojo
            from numojo.prelude import *
            var a = Matrix.fromlist([Float64(1), 2, 3, 4, 5], (5, 1))
            print(a)
            ```
        """

        if (shape[0] == 0) and (shape[1] == 0):
            var M = Matrix[datatype](shape=(1, len(object)))
            memcpy(dest=M._buf.ptr, src=object.unsafe_ptr(), count=M.size)
            return M^

        if shape[0] * shape[1] != len(object):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "The input has {} elements, but the target shape {}x{}"
                        " requires {} elements. Total elements must match."
                        .format(
                            len(object), shape[0], shape[1], shape[0] * shape[1]
                        )
                    ),
                    location="matrix_from_numpy",
                )
            )
        var M = Matrix[datatype](shape=shape, order="C")
        memcpy(dest=M._buf.ptr, src=object.unsafe_ptr(), count=M.size)
        if order == "F":
            M = M.reorder_layout()
        return M^

    @staticmethod
    fn fromstring[
        datatype: DType = DType.float64
    ](
        text: String, shape: Tuple[Int, Int] = (0, 0), order: String = "C"
    ) raises -> Matrix[datatype]:
        """
        Create a Matrix from a string representation of its elements.

        The input string should contain numbers separated by commas, right brackets, or whitespace. Digits, underscores, decimal points, and minus signs are treated as part of numbers. If no shape is provided, the returned matrix will be a row vector.

        Args:
            text: String containing the matrix elements.
            shape: Tuple specifying the matrix dimensions (rows, columns). If not provided, creates a row vector.
            order: Memory layout order, "C" (row-major) or "F" (column-major).

        Returns:
            Matrix[datatype]: Matrix constructed from the string data.

        Example:
            ```mojo
            from numojo.prelude import *
            var A = Matrix.fromstring[f32]("1 2 .3 4 5 6.5 7 1_323.12 9 10, 11.12, 12 13 14 15 16", (4, 4))
            print(A)
            ```

            Output:
            ```
            [[1.0   2.0     0.30000001192092896     4.0]
            [5.0   6.5     7.0     1323.1199951171875]
            [9.0   10.0    11.119999885559082      12.0]
            [13.0  14.0    15.0    16.0]]
            Size: 4x4  datatype: float32
            ```
        """

        var data = List[Scalar[datatype]]()
        var bytes = text.as_bytes()
        var number_as_str: String = ""
        var size = shape[0] * shape[1]

        for i in range(len(bytes)):
            var b = bytes[i]
            if (
                # TODO: Check whether is_ascii_digit works as expected.
                chr(Int(b)).is_ascii_digit()
                or (chr(Int(b)) == ".")
                or (chr(Int(b)) == "-")
            ):
                number_as_str = number_as_str + chr(Int(b))
                if i == len(bytes) - 1:  # Last byte
                    var number = atof(number_as_str).cast[datatype]()
                    data.append(number)  # Add the number to the data buffer
                    number_as_str = ""  # Clean the number cache
            if (
                (chr(Int(b)) == ",")
                or (chr(Int(b)) == "]")
                or (chr(Int(b)) == " ")
            ):
                if number_as_str != "":
                    var number = atof(number_as_str).cast[datatype]()
                    data.append(number)  # Add the number to the data buffer
                    number_as_str = ""  # Clean the number cache

        if (shape[0] == 0) and (shape[1] == 0):
            return Matrix.fromlist(data)

        if size != len(data):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "The number of items in the string is {}, which does"
                        " not match the given shape {}x{} (requires {}"
                        " elements). Total elements must match.".format(
                            len(data), shape[0], shape[1], size
                        )
                    ),
                    location="matrix_from_str",
                )
            )

        var result = Matrix[datatype](shape=shape)
        for i in range(len(data)):
            result._buf.ptr[i] = data[i]
        return result^


# # ===-----------------------------------------------------------------------===#
# # MatrixIter struct
# # ===-----------------------------------------------------------------------===#


struct _MatrixIter[
    dtype: DType,
    is_mutable: Bool,
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """
    Iterator for Matrix that yields row views.

    This struct provides iteration over the rows of a Matrix, returning a Matrix for each row. It supports both forward and backward iteration.

    Parameters:
        dtype: The data type of the matrix elements.
        is_mutable: Whether the iterator allows mutable access to the matrix.
        forward: The iteration direction. If True, iterates forward; if False, iterates backward.
    """

    comptime Element = Matrix[Self.dtype]
    """The type of elements yielded by the iterator (Matrix). """

    var index: Int
    """Current index in the iteration."""

    var _buf: DataContainer[Self.dtype]
    """Shared reference to the matrix data buffer."""

    var shape: Tuple[Int, Int]
    """Shape of the matrix being iterated."""

    var strides: Tuple[Int, Int]
    """Strides of the matrix being iterated."""

    fn __init__(
        out self,
        index: Int,
        matrix_buf: DataContainer[Self.dtype],
        shape: Tuple[Int, Int],
        strides: Tuple[Int, Int],
    ):
        """Initialize the iterator.

        Args:
            index: The starting index for iteration.
            matrix_buf: Shared reference to the matrix data buffer.
            shape: Shape of the matrix.
            strides: Strides of the matrix.
        """
        self.index = index
        self._buf = matrix_buf.copy()
        self.shape = shape
        self.strides = strides

    @always_inline
    fn __iter__(ref self) -> Self:
        """Return a copy of the iterator for iteration protocol."""
        return self.copy()

    @always_inline
    fn __has_next__(self) -> Bool:
        """Check if there are more rows to iterate over.

        Returns:
            Bool: True if there are more rows to iterate, False otherwise.
        """

        @parameter
        if Self.forward:
            return self.index < self.shape[0]
        else:
            return self.index >= 0

    fn __next__(mut self) -> Matrix[Self.dtype]:
        """Return a view of the next row.

        Returns:
            Matrix: A view representing the next row in the iteration.
        """

        @parameter
        if Self.forward:
            var current_index = self.index
            self.index += 1
            var offset = current_index * self.strides[0]
            return Matrix[Self.dtype](
                shape=(1, self.shape[1]),
                strides=(self.strides[0], self.strides[1]),
                data=self._buf.share_with_offset(offset),
            )
        else:
            var current_index = self.index
            self.index -= 1
            var offset = current_index * self.strides[0]
            return Matrix[Self.dtype](
                shape=(1, self.shape[1]),
                strides=(self.strides[0], self.strides[1]),
                data=self._buf.share_with_offset(offset),
            )

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        """Return the iteration bounds.

        Returns:
            Tuple[Int, Optional[Int]]: Number of remaining rows and an optional value of the same.
        """
        var remaining_rows: Int

        @parameter
        if Self.forward:
            remaining_rows = self.shape[0] - self.index
        else:
            remaining_rows = self.index + 1

        return (remaining_rows, {remaining_rows})


# # ===-----------------------------------------------------------------------===#
# # Backend fucntions using SMID functions
# # ===-----------------------------------------------------------------------===#


# TODO: we can move the checks in these functions to the caller functions to avoid redundant checks.
fn _arithmetic_func_matrix_matrix_to_matrix[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
](A: Matrix[dtype, **_], B: Matrix[dtype, **_]) raises -> Matrix[dtype]:
    """
    Perform element-wise arithmetic operation between two matrices using a SIMD function.

    Parameters:
        dtype: The data type of the matrix elements.
        simd_func: A SIMD function that takes two SIMD vectors and returns a SIMD vector, representing the desired arithmetic operation (e.g., addition, subtraction).

    Args:
        A: The first input matrix.
        B: The second input matrix.

    Returns:
        Matrix[dtype]: A new matrix containing the result of applying the SIMD function element-wise to A and B.

    Raises:
        Error: If the matrix orders or shapes do not match.

    Notes:
        - Only for internal purposes.
    """
    comptime simd_width = simd_width_of[dtype]()
    if A.order() != B.order():
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "Matrix order {} does not match {}. Both matrices must have"
                    " the same memory order.".format(A.order(), B.order())
                ),
                location="_apply_op_to_matrix_matrix",
            )
        )

    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Shape {}x{} does not match {}x{}. Matrix shapes must match"
                    " exactly.".format(
                        A.shape[0], A.shape[1], B.shape[0], B.shape[1]
                    )
                ),
                location="_apply_op_to_matrix_matrix",
            )
        )

    var res = Matrix[dtype](shape=A.shape, order=A.order())

    @parameter
    fn vec_func[simd_width: Int](i: Int) unified {mut res, read A, read B}:
        res._buf.ptr.store(
            i,
            simd_func(
                A._buf.ptr.load[width=simd_width](i),
                B._buf.ptr.load[width=simd_width](i),
            ),
        )

    vectorize[simd_width](A.size, vec_func)
    return res^


fn _arithmetic_func_matrix_to_matrix[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply a unary SIMD function element-wise to a matrix.

    Parameters:
        dtype: The data type of the matrix elements.
        simd_func: A SIMD function that takes a SIMD vector and returns a SIMD vector representing

    Args:
        A: Input matrix of type Matrix[dtype].

    Returns:
        Matrix[dtype]: A new matrix containing the result of applying the SIMD function to each element of the input matrix.

    Notes:
        - Only for internal purposes.
    """
    comptime simd_width: Int = simd_width_of[dtype]()

    var C: Matrix[dtype] = Matrix[dtype](shape=A.shape, order=A.order())

    @parameter
    fn vec_func[simd_width: Int](i: Int) unified {mut C, read A}:
        C._buf.ptr.store(i, simd_func(A._buf.ptr.load[width=simd_width](i)))

    vectorize[simd_width](A.size, vec_func)

    return C^


fn _logic_func_matrix_matrix_to_matrix[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[DType.bool, simd_width],
](A: Matrix[dtype, **_], B: Matrix[dtype, **_]) raises -> Matrix[DType.bool]:
    """
    Perform element-wise logical comparison between two matrices using a SIMD function.

    Parameters:
        dtype: The data type of the input matrices.
        simd_func: A SIMD function that takes two SIMD vectors of dtype and returns a SIMD vector of bools.

    Args:
        A: The first input matrix.
        B: The second input matrix.

    Returns:
        Matrix[DType.bool]: A new matrix of bools containing the result of the element-wise logical comparison.

    Raises:
        Error: If the matrix orders or shapes do not match.

    Notes:
        - Only for internal purposes.
        - The output matrix has the same shape and order as the input matrices.
    """
    comptime width = simd_width_of[dtype]()

    if A.order() != B.order():
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "Matrix order {} does not match {}. Both matrices must have"
                    " the same memory order.".format(A.order(), B.order())
                ),
                location="_apply_compop_to_matrix_matrix",
            )
        )

    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Shape {}x{} does not match {}x{}. Matrix shapes must match"
                    " exactly.".format(
                        A.shape[0], A.shape[1], B.shape[0], B.shape[1]
                    )
                ),
                location="_apply_compop_to_matrix_matrix",
            )
        )

    var C = Matrix[DType.bool](shape=A.shape, order=A.order())

    # Use vectorized comparison for better performance
    # Process elements using input dtype width, then store results as bool
    @parameter
    fn vec_compare[w: Int](i: Int) unified {mut C, read A, read B}:
        var a_vec = A._buf.ptr.load[width=w](i)
        var b_vec = B._buf.ptr.load[width=w](i)
        var result = simd_func[dtype, w](a_vec, b_vec)

        # Store bool results element by element to avoid width mismatch
        for j in range(w):
            if i + j < A.size:
                C._buf.ptr[i + j] = result[j]

    vectorize[width](A.size, vec_compare)
    return C^
