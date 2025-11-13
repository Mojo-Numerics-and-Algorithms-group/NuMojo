"""
`numojo.Matrix` provides:

- `Matrix` type (2DArray).
- `_MatrixIter` type (for iteration).
- Dunder methods for initialization, indexing, slicing, and arithmetics.
- Auxiliary functions.
"""

from algorithm import parallelize, vectorize
from memory import UnsafePointer, memcpy, memset_zero
from random import random_float64
from sys import simd_width_of
from python import PythonObject, Python
from math import ceil

from numojo.core.flags import Flags

# from numojo.core.ndarray import NDArray
from numojo.core.data_container import DataContainerNew as DataContainer
from numojo.core.traits.buffered import Buffered
from numojo.core.own_data import OwnData
from numojo.core.ref_data import RefData
from numojo.core.utility import _get_offset
from numojo.routines.manipulation import broadcast_to, reorder_layout
from numojo.routines.linalg.misc import issymmetric


# ===----------------------------------------------------------------------===#
# Matrix struct
# ===----------------------------------------------------------------------===#


struct Matrix[
    dtype: DType = DType.float64,
    BufType: Buffered = OwnData,
    origin: MutOrigin = MutOrigin.external,
](Copyable, Movable, Sized, Stringable, Writable):
    """
    `Matrix` is a special case of `NDArray` (2DArray) but has some targeted
    optimization since the number of dimensions is known at the compile time.
    It has simpler indexing and slicing methods, which is very useful when users
    only want to work with 2-dimensional arrays.

    NuMojo's `Matrix` is `NDArray` with fixed `ndim` known at compile time.
    It may be different in some behaviors compared to `numpy.matrix`.

    - For `__getitem__`, passing in two `Int` returns a scalar,
    and passing in one `Int` or two `Slice` returns a `Matrix`.
    - We do not need auxiliary types `NDArrayShape` and `NDArrayStrides`
    as the shape and strides information is fixed in length `Tuple[Int,Int]`.

    Parameters:
        dtype: Type of item in NDArray. Default type is DType.float64.
        BufType: This is only for internal use! The buffer type of the Matrix, denotes whether the instance owns the data or is a view. Default is `OwnData`. Manipulating it can lead to undefined behaviors.
        origin: This is only for internal use! The mutability origin of the Matrix. Default is `MutOrigin.external`. Manipulating it can lead to undefined behaviors.

    The matrix can be uniquely defined by the following features:
        1. The data buffer of all items.
        2. The shape of the matrix.
        3. The data type of the elements (compile-time known).

    Attributes:
        - _buf (saved as row-majored, C-type)
        - shape
        - size (shape[0] * shape[1])
        - strides (shape[1], 1)

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

    alias width: Int = simd_width_of[dtype]()  #
    """Vector size of the data type."""

    var _buf: DataContainer[dtype, origin]
    """Data buffer of the items in the Matrix."""

    var buf_type: BufType
    """View information of the Matrix."""

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
        Create a new matrix of the given shape, without initializing data.

        Args:
            shape: Tuple representing (rows, columns).
            order: Use "C" for row-major (C-style) layout or "F" for column-major
               (Fortran-style) layout. Defaults to "C".
        """
        constrained[
            BufType.is_own_data(),
            "Buffer type must be OwnData to create matrix that owns data.",
        ]()

        self.shape = (shape[0], shape[1])
        if order == "C":
            self.strides = (shape[1], 1)
        else:
            self.strides = (1, shape[0])
        self.size = shape[0] * shape[1]
        self._buf = DataContainer[dtype, origin](size=self.size)
        self.buf_type = BufType()
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

    # * Should we take var ref and transfer ownership or take a read ref and copy it?
    @always_inline("nodebug")
    fn __init__(
        out self,
        var data: Self,
    ):
        """
        Construct a matrix from matrix.
        """
        constrained[
            BufType.is_own_data(),
            "Buffer type must be OwnData to create matrix that owns data.",
        ]()
        self = data^

    @always_inline("nodebug")
    fn __init__(
        out self,
        data: NDArray[dtype],
    ) raises:
        """
        Construct a matrix from array.
        """
        constrained[
            BufType.is_own_data(),
            "Buffer type must be OwnData to create matrix that owns data.",
        ]()
        if data.ndim == 1:
            self.shape = (1, data.shape[0])
            self.strides = (data.shape[0], 1)
            self.size = data.shape[0]
        elif data.ndim == 2:
            self.shape = (data.shape[0], data.shape[1])
            self.strides = (data.shape[1], 1)
            self.size = data.shape[0] * data.shape[1]
        else:
            raise Error(String("Shape too large to be a matrix."))

        self._buf = DataContainer[dtype, origin](self.size)
        self.buf_type = BufType()
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

        if data.flags["C_CONTIGUOUS"]:
            for i in range(data.shape[0]):
                memcpy(
                    dest=self._buf.ptr.offset(i * self.shape[0]),
                    src=data._buf.ptr.offset(i * data.shape[0]),
                    count=self.shape[0],
                )
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self._store(i, j, data._getitem(i, j))

    # to construct views
    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Tuple[Int, Int],
        strides: Tuple[Int, Int],
        offset: Int,
        ptr: UnsafePointerV2[Scalar[dtype], origin],
    ):
        """
        Initialize Matrix that does not own the data.
        The data is owned by another Matrix.

        Args:
            shape: Shape of the view.
            strides: Strides of the view.
            offset: Offset in pointer of the data buffer.
            ptr: Pointer to the data buffer of the original array.
        """
        constrained[
            BufType.is_ref_data(),
            "Buffer type must be RefData to create matrix view.",
        ]()
        self.shape = shape
        self.strides = strides
        self.size = shape[0] * shape[1]
        self._buf = DataContainer[dtype, origin](ptr=ptr.offset(offset))
        self.buf_type = BufType()
        self.flags = Flags(
            self.shape, self.strides, owndata=False, writeable=False
        )

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self._buf = DataContainer[dtype, origin](other.size)
        memcpy(dest=self._buf.ptr, src=other._buf.ptr, count=other.size)
        self.buf_type = BufType()
        self.flags = Flags(
            other.shape, other.strides, owndata=True, writeable=True
        )

    @always_inline("nodebug")
    fn __moveinit__(out self, deinit other: Self):
        """
        Move other into self.
        """
        self.shape = other.shape^
        self.strides = other.strides^
        self.size = other.size
        self._buf = other._buf^
        self.buf_type = other.buf_type^
        self.flags = other.flags^

    @always_inline("nodebug")
    fn __del__(deinit self):
        var owndata: Bool = self.flags.OWNDATA
        # Free the buffer only if it owns the data, but its redudant rn. move buf type checks into compile time and remove redundant check here.
        if owndata and self.buf_type.is_own_data():
            print(
                "Freeing matrix memory",
                self.buf_type.is_own_data(),
                self.size,
                self.shape[0],
                self.shape[1],
            )
            self._buf.ptr.free()

    fn create_copy(self) raises -> Matrix[dtype, OwnData]:
        """
        Create a copy of the matrix with OwnData buffer type.
        """
        var result = Matrix[dtype, OwnData](
            shape=self.shape, order=self.order()
        )
        if self.flags.C_CONTIGUOUS:
            memcpy(dest=result._buf.ptr, src=self._buf.ptr, count=self.size)
        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self[i, j]

        return result^

    # ===-------------------------------------------------------------------===#
    # Slicing and indexing methods
    # ===-------------------------------------------------------------------===#

    fn normalize(self, idx: Int, dim: Int) -> Int:
        """
        Normalize negative indices.
        """
        var idx_norm = idx
        if idx_norm < 0:
            idx_norm = dim + idx_norm
        return idx_norm

    fn __getitem__(self, x: Int, y: Int) raises -> Scalar[dtype]:
        """
        Return the scalar at the index.

        Args:
            x: The row number.
            y: The column number.

        Returns:
            A scalar matching the dtype of the array.
        """
        if (
            x >= self.shape[0]
            or x < -self.shape[0]
            or y >= self.shape[1]
            or y < -self.shape[1]
        ):
            raise Error(
                String(
                    "Index ({}, {}) exceed the matrix shape ({}, {})"
                ).format(x, y, self.shape[0], self.shape[1])
            )
        var x_norm = self.normalize(x, self.shape[0])
        var y_norm = self.normalize(y, self.shape[1])
        return self._buf.ptr.load(
            x_norm * self.strides[0] + y_norm * self.strides[1]
        )

    # NOTE: temporarily renaming all view returning functions to be `get` or `set` due to a Mojo bug with overloading `__getitem__` and `__setitem__` with different argument types. Created an issue in Mojo GitHub
    fn get(
        ref self, x: Int
    ) raises -> Matrix[
        dtype,
        RefData[MutOrigin.cast_from[origin_of(self)]],
        MutOrigin.cast_from[origin_of(self)],
    ]:
        """
        Return the corresponding row at the index.

        Args:
            x: The row number.
        """
        constrained[
            BufType.is_own_data(),
            "Buffer type must be OwnData to get a reference row.",
        ]()
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String("Index {} exceed the row number {}").format(
                    x, self.shape[0]
                )
            )

        var x_norm = self.normalize(x, self.shape[0])
        var new_ptr = self._buf.ptr
        var res = Matrix[
            dtype,
            RefData[MutOrigin.cast_from[origin_of(self)]],
            MutOrigin.cast_from[origin_of(self)],
        ](
            shape=(1, self.shape[1]),
            strides=(self.strides[0], self.strides[1]),
            offset=x_norm * self.strides[0],
            ptr=new_ptr.unsafe_origin_cast[
                MutOrigin.cast_from[origin_of(self)]
            ](),
        )
        return res^

    # for creating a copy of the row.
    fn __getitem__(self, var x: Int) raises -> Matrix[dtype, OwnData]:
        """
        Return the corresponding row at the index.

        Args:
            x: The row number.

        Returns:
            A new Matrix (row vector) copied from the original matrix.

        Notes:
            This function is for internal use only. Users should use `create_copy` to create a copy of the whole matrix instead.
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String("Index {} exceed the row size {}").format(
                    x, self.shape[0]
                )
            )
        var x_norm = self.normalize(x, self.shape[0])
        var result = Matrix[dtype, OwnData](
            shape=(1, self.shape[1]), order=self.order()
        )
        if self.flags.C_CONTIGUOUS:
            var ptr = self._buf.ptr.offset(x_norm * self.strides[0])
            memcpy(dest=result._buf.ptr, src=ptr, count=self.shape[1])
        else:
            for j in range(self.shape[1]):
                result[0, j] = self[x_norm, j]

        return result^

    fn get(
        ref self, x: Slice, y: Slice
    ) -> Matrix[
        dtype,
        RefData[MutOrigin.cast_from[origin_of(self)]],
        MutOrigin.cast_from[origin_of(self)],
    ]:
        """
        Get item from two slices.
        """
        constrained[
            BufType.is_own_data(),
            "Buffer type must be OwnData to get a reference row.",
        ]()
        start_x, end_x, step_x = x.indices(self.shape[0])
        start_y, end_y, step_y = y.indices(self.shape[1])

        var new_ptr = self._buf.ptr
        var res = Matrix[
            dtype,
            RefData[MutOrigin.cast_from[origin_of(self)]],
            MutOrigin.cast_from[origin_of(self)],
        ](
            shape=(
                Int(ceil((end_x - start_x) / step_x)),
                Int(ceil((end_y - start_y) / step_y)),
            ),
            strides=(step_x * self.strides[0], step_y * self.strides[1]),
            offset=start_x * self.strides[0] + start_y * self.strides[1],
            ptr=new_ptr.unsafe_origin_cast[
                MutOrigin.cast_from[origin_of(self)]
            ](),
        )

        return res^

    # for creating a copy of the slice.
    fn __getitem__(self, x: Slice, y: Slice) -> Matrix[dtype, OwnData]:
        """
        Get item from two slices.
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

        var B = Matrix[dtype, OwnData](
            shape=(len(range_x), len(range_y)), order=self.order()
        )
        var row = 0
        for i in range_x:
            var col = 0
            for j in range_y:
                B._store(row, col, self._load(i, j))
                col += 1
            row += 1

        return B^

    fn get(
        ref self, x: Slice, var y: Int
    ) raises -> Matrix[
        dtype,
        RefData[MutOrigin.cast_from[origin_of(self)]],
        MutOrigin.cast_from[origin_of(self)],
    ]:
        """
        Get item from one slice and one int.
        """
        # we could remove this constraint if we wanna allow users to create views from views. But that may complicate the origin tracking?
        constrained[
            BufType.is_own_data(),
            "Buffer type must be OwnData to get a reference slice.",
        ]()
        if y >= self.shape[1] or y < -self.shape[1]:
            raise Error(
                String("Index {} exceed the column number {}").format(
                    y, self.shape[1]
                )
            )
        y = self.normalize(y, self.shape[1])
        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])

        var new_ptr = self._buf.ptr
        var res = Matrix[
            dtype,
            RefData[MutOrigin.cast_from[origin_of(self)]],
            MutOrigin.cast_from[origin_of(self)],
        ](
            shape=(
                Int(ceil((end_x - start_x) / step_x)),
                1,
            ),
            strides=(step_x * self.strides[0], self.strides[1]),
            offset=start_x * self.strides[0] + y * self.strides[1],
            ptr=new_ptr.unsafe_origin_cast[
                MutOrigin.cast_from[origin_of(self)]
            ](),
        )

        return res^

    # for creating a copy of the slice.
    fn __getitem__(self, x: Slice, var y: Int) -> Matrix[dtype, OwnData]:
        """
        Get item from one slice and one int.
        """
        if y < 0:
            y = self.shape[1] + y

        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        var range_x = range(start_x, end_x, step_x)
        var res = Matrix[dtype, OwnData](
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

    fn get(
        ref self, var x: Int, y: Slice
    ) raises -> Matrix[
        dtype,
        RefData[MutOrigin.cast_from[origin_of(self)]],
        MutOrigin.cast_from[origin_of(self)],
    ]:
        """
        Get item from one int and one slice.
        """
        constrained[
            BufType.is_own_data(),
            "Buffer type must be OwnData to get a reference slice.",
        ]()
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String("Index {} exceed the row size {}").format(
                    x, self.shape[0]
                )
            )
        x = self.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        # var range_y = range(start_y, end_y, step_y)
        var new_ptr = self._buf.ptr
        var res = Matrix[
            dtype,
            RefData[MutOrigin.cast_from[origin_of(self)]],
            MutOrigin.cast_from[origin_of(self)],
        ](
            shape=(
                1,
                Int(ceil((end_y - start_y) / step_y)),
            ),
            strides=(self.strides[0], step_y * self.strides[1]),
            offset=x * self.strides[0] + start_y * self.strides[1],
            ptr=new_ptr.unsafe_origin_cast[
                MutOrigin.cast_from[origin_of(self)]
            ](),
        )

        return res^

    # for creating a copy of the slice.
    fn __getitem__(self, var x: Int, y: Slice) raises -> Matrix[dtype]:
        """
        Get item from one int and one slice.
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String("Index {} exceed the row size {}").format(
                    x, self.shape[0]
                )
            )
        x = self.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_y = range(start_y, end_y, step_y)

        var B = Matrix[dtype](shape=(1, len(range_y)), order=self.order())
        var col = 0
        for j in range_y:
            B._store(0, col, self._load(x, j))
            col += 1

        return B^

    fn __getitem__(self, indices: List[Int]) raises -> Matrix[dtype, OwnData]:
        """
        Get item by a list of integers.
        """
        var ncol = self.shape[1]
        var nrow = len(indices)
        var res = Matrix.zeros[dtype](shape=(nrow, ncol))
        for i in range(nrow):
            res.__setitem__(i, self[indices[i]])
        return res^

    fn load[width: Int = 1](self, idx: Int) raises -> SIMD[dtype, width]:
        """
        Returns a SIMD element with width `width` at the given index.

        Parameters:
            width: The width of the SIMD element.

        Args:
            idx: The linear index.

        Returns:
            A SIMD element with width `width`.
        """
        if idx >= self.size or idx < -self.size:
            raise Error(
                String("Index {} exceed the matrix size {}").format(
                    idx, self.size
                )
            )
        var idx_norm = self.normalize(idx, self.size)
        return self._buf.ptr.load[width=width](idx_norm)

    fn _load[width: Int = 1](self, x: Int, y: Int) -> SIMD[dtype, width]:
        """
        `__getitem__` with width.
        Unsafe: No boundary check!
        """
        return self._buf.ptr.load[width=width](
            x * self.strides[0] + y * self.strides[1]
        )

    fn _load[width: Int = 1](self, idx: Int) -> SIMD[dtype, width]:
        """
        `__getitem__` with width.
        Unsafe: No boundary check!
        """
        return self._buf.ptr.load[width=width](idx)

    fn __setitem__(self, x: Int, y: Int, value: Scalar[dtype]) raises:
        """
        Return the scalar at the index.

        Args:
            x: The row number.
            y: The column number.
            value: The value to be set.
        """
        if (
            x >= self.shape[0]
            or x < -self.shape[0]
            or y >= self.shape[1]
            or y < -self.shape[1]
        ):
            raise Error(
                String(
                    "Index ({}, {}) exceed the matrix shape ({}, {})"
                ).format(x, y, self.shape[0], self.shape[1])
            )
        var x_norm = self.normalize(x, self.shape[0])
        var y_norm = self.normalize(y, self.shape[1])

        self._buf.ptr.store(
            x_norm * self.strides[0] + y_norm * self.strides[1], value
        )

    fn __setitem__(self, var x: Int, value: Matrix[dtype, **_]) raises:
        """
        Set the corresponding row at the index with the given matrix.

        Args:
            x: The row number.
            value: Matrix (row vector). Can be either C or F order.
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String(
                    "Error: Elements of `index` ({}) \n"
                    "exceed the matrix shape ({})."
                ).format(x, self.shape[0])
            )

        if value.shape[0] != 1:
            raise Error(
                String(
                    "Error: The value should have only 1 row, "
                    "but it has {} rows."
                ).format(value.shape[0])
            )

        if self.shape[1] != value.shape[1]:
            raise Error(
                String(
                    "Error: Matrix has {} columns, "
                    "but the value has {} columns."
                ).format(self.shape[1], value.shape[1])
            )

        if self.flags.C_CONTIGUOUS:
            if value.flags.C_CONTIGUOUS:
                var dest_ptr = self._buf.ptr.offset(x * self.strides[0])
                memcpy(dest=dest_ptr, src=value._buf.ptr, count=self.shape[1])
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

        # For F-contiguous
        else:
            if value.flags.F_CONTIGUOUS:
                for j in range(self.shape[1]):
                    self._buf.ptr.offset(x + j * self.strides[1]).store(
                        value._buf.ptr.load(j * value.strides[1])
                    )
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

    fn set(self, var x: Int, value: Matrix[dtype, **_]) raises:
        """
        Set the corresponding row at the index with the given matrix.

        Args:
            x: The row number.
            value: Matrix (row vector). Can be either C or F order.
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String(
                    "Error: Elements of `index` ({}) \n"
                    "exceed the matrix shape ({})."
                ).format(x, self.shape[0])
            )

        if value.shape[0] != 1:
            raise Error(
                String(
                    "Error: The value should have only 1 row, "
                    "but it has {} rows."
                ).format(value.shape[0])
            )

        if self.shape[1] != value.shape[1]:
            raise Error(
                String(
                    "Error: Matrix has {} columns, "
                    "but the value has {} columns."
                ).format(self.shape[1], value.shape[1])
            )

        if self.flags.C_CONTIGUOUS:
            if value.flags.C_CONTIGUOUS:
                var dest_ptr = self._buf.ptr.offset(x * self.strides[0])
                memcpy(dest=dest_ptr, src=value._buf.ptr, count=self.shape[1])
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

        # For F-contiguous
        else:
            if value.flags.F_CONTIGUOUS:
                for j in range(self.shape[1]):
                    self._buf.ptr.offset(x + j * self.strides[1]).store(
                        value._buf.ptr.load(j * value.strides[1])
                    )
            else:
                for j in range(self.shape[1]):
                    self._store(x, j, value._load(0, j))

    fn __setitem__(self, x: Slice, y: Int, value: Matrix[dtype, **_]) raises:
        """
        Set item from one slice and one int.
        """
        if y >= self.shape[1] or y < -self.shape[1]:
            raise Error(
                String("Index {} exceed the column number {}").format(
                    y, self.shape[1]
                )
            )
        var y_norm = self.normalize(y, self.shape[1])
        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        var range_x = range(start_x, end_x, step_x)
        var len_range_x: Int = len(range_x)

        if len_range_x != value.shape[0] or value.shape[1] != 1:
            raise Error(
                String(
                    "Shape mismatch when assigning to slice: "
                    "target shape ({}, {}) vs value shape ({}, {})"
                ).format(len_range_x, 1, value.shape[0], value.shape[1])
            )

        var row = 0
        for i in range_x:
            self._store(i, y_norm, value._load(row, 0))
            row += 1

    fn set(self, x: Slice, y: Int, value: Matrix[dtype, **_]) raises:
        """
        Set item from one slice and one int.
        """
        if y >= self.shape[1] or y < -self.shape[1]:
            raise Error(
                String("Index {} exceed the column number {}").format(
                    y, self.shape[1]
                )
            )
        var y_norm = self.normalize(y, self.shape[1])
        var start_x: Int
        var end_x: Int
        var step_x: Int
        start_x, end_x, step_x = x.indices(self.shape[0])
        var range_x = range(start_x, end_x, step_x)
        var len_range_x: Int = len(range_x)

        if len_range_x != value.shape[0] or value.shape[1] != 1:
            raise Error(
                String(
                    "Shape mismatch when assigning to slice: "
                    "target shape ({}, {}) vs value shape ({}, {})"
                ).format(len_range_x, 1, value.shape[0], value.shape[1])
            )

        var row = 0
        for i in range_x:
            self._store(i, y_norm, value._load(row, 0))
            row += 1

    fn __setitem__(self, x: Int, y: Slice, value: Matrix[dtype, **_]) raises:
        """
        Set item from one int and one slice.
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String("Index {} exceed the row size {}").format(
                    x, self.shape[0]
                )
            )
        var x_norm = self.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_y = range(start_y, end_y, step_y)
        var len_range_y: Int = len(range_y)

        if len_range_y != value.shape[1] or value.shape[0] != 1:
            raise Error(
                String(
                    "Shape mismatch when assigning to slice: "
                    "target shape ({}, {}) vs value shape ({}, {})"
                ).format(1, len_range_y, value.shape[0], value.shape[1])
            )

        var col = 0
        for j in range_y:
            self._store(x_norm, j, value._load(0, col))
            col += 1

    fn set(self, x: Int, y: Slice, value: Matrix[dtype, **_]) raises:
        """
        Set item from one int and one slice.
        """
        if x >= self.shape[0] or x < -self.shape[0]:
            raise Error(
                String("Index {} exceed the row size {}").format(
                    x, self.shape[0]
                )
            )
        var x_norm = self.normalize(x, self.shape[0])
        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_y = range(start_y, end_y, step_y)
        var len_range_y: Int = len(range_y)

        if len_range_y != value.shape[1] or value.shape[0] != 1:
            raise Error(
                String(
                    "Shape mismatch when assigning to slice: "
                    "target shape ({}, {}) vs value shape ({}, {})"
                ).format(1, len_range_y, value.shape[0], value.shape[1])
            )

        var col = 0
        for j in range_y:
            self._store(x_norm, j, value._load(0, col))
            col += 1

    fn __setitem__(self, x: Slice, y: Slice, value: Matrix[dtype, **_]) raises:
        """
        Set item from two slices.
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
                String(
                    "Shape mismatch when assigning to slice: "
                    "target shape ({}, {}) vs value shape ({}, {})"
                ).format(
                    len(range_x), len(range_y), value.shape[0], value.shape[1]
                )
            )

        var row = 0
        for i in range_x:
            var col = 0
            for j in range_y:
                self._store(i, j, value._load(row, col))
                col += 1
            row += 1

    fn set(self, x: Slice, y: Slice, value: Matrix[dtype, **_]) raises:
        """
        Set item from two slices.
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
                String(
                    "Shape mismatch when assigning to slice: "
                    "target shape ({}, {}) vs value shape ({}, {})"
                ).format(
                    len(range_x), len(range_y), value.shape[0], value.shape[1]
                )
            )

        var row = 0
        for i in range_x:
            var col = 0
            for j in range_y:
                self._store(i, j, value._load(row, col))
                col += 1
            row += 1

    fn _store[width: Int = 1](self, x: Int, y: Int, simd: SIMD[dtype, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.ptr.store(x * self.strides[0] + y * self.strides[1], simd)

    fn _store_idx[width: Int = 1](self, idx: Int, val: SIMD[dtype, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.ptr.store(idx, val)

    # ===-------------------------------------------------------------------===#
    # Other dunders and auxiliary methods
    # ===-------------------------------------------------------------------===#

    fn __iter__(mut self) raises -> _MatrixIter[origin, dtype, BufType]:
        """Iterate over elements of the Matrix, returning copied value.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand((4,4))
        for i in A:
            print(i)
        ```

        Returns:
            An iterator of Matrix elements.
        """

        return _MatrixIter[origin, dtype, BufType](
            matrix=self,
            length=self.shape[0],
        )

    fn __len__(self) -> Int:
        """
        Returns length of 0-th dimension.
        """
        return self.shape[0]

    fn __reversed__(
        mut self,
    ) raises -> _MatrixIter[origin, dtype, BufType, forward=False]:
        """Iterate backwards over elements of the Matrix, returning
        copied value.

        Returns:
            A reversed iterator of Matrix elements.
        """

        return _MatrixIter[origin, dtype, BufType, forward=False](
            matrix=self,
            length=self.shape[0],
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
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

    fn __add__(
        read self, read other: Matrix[dtype, *_]
    ) raises -> Matrix[dtype, OwnData]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__add__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__add__
            ](broadcast_to[dtype](self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__add__
            ](self, broadcast_to[dtype](other, self.shape, self.order()))

    fn __add__(self, other: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """Add matrix to scalar.

        ```mojo
        from numojo import Matrix
        var A = Matrix.ones(shape=(4, 4))
        print(A + 2)
        ```
        """
        return self + broadcast_to[dtype](other, self.shape, self.order())

    fn __radd__(self, other: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """
        Right-add.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(2 + A)
        ```
        """
        return broadcast_to[dtype](other, self.shape, self.order()) + self

    fn __sub__(
        read self, read other: Matrix[dtype, *_]
    ) raises -> Matrix[dtype, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__sub__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__sub__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__sub__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __sub__(self, other: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """Subtract matrix by scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix(shape=(4, 4))
        print(A - 2)
        ```
        """
        return self - broadcast_to[dtype](other, self.shape, self.order())

    fn __rsub__(self, other: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """
        Right-sub.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(2 - A)
        ```
        """
        return broadcast_to[dtype](other, self.shape, self.order()) - self

    fn __mul__(self, other: Matrix[dtype, **_]) raises -> Matrix[dtype, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__mul__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__mul__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__mul__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __mul__(self, other: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """Mutiply matrix by scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A * 2)
        ```
        """
        return self * broadcast_to[dtype](other, self.shape, self.order())

    fn __rmul__(self, other: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """
        Right-mul.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(2 * A)
        ```
        """
        return broadcast_to[dtype](other, self.shape, self.order()) * self

    fn __truediv__(
        self, other: Matrix[dtype, **_]
    ) raises -> Matrix[dtype, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__truediv__
            ](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__truediv__
            ](broadcast_to(self, other.shape, self.order()), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__truediv__
            ](self, broadcast_to(other, self.shape, self.order()))

    fn __truediv__(self, other: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """Divide matrix by scalar."""
        return self / broadcast_to[dtype](other, self.shape, order=self.order())

    # Shouldn't we do the operation inplace?
    fn __pow__(self, rhs: Scalar[dtype]) raises -> Matrix[dtype, **_]:
        """Power of items."""
        var result: Matrix[dtype, OwnData] = Matrix[dtype, OwnData](
            shape=self.shape, order=self.order()
        )
        for i in range(self.size):
            result._buf.ptr[i] = self._buf.ptr[i].__pow__(rhs)
        return result^

    fn __lt__(
        self, other: Matrix[dtype, **_]
    ) raises -> Matrix[DType.bool, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.lt](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.lt](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.lt](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __lt__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool, **_]:
        """Matrix less than scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A < 2)
        ```
        """
        return self < broadcast_to[dtype](other, self.shape, self.order())

    fn __le__(
        self, other: Matrix[dtype, **_]
    ) raises -> Matrix[DType.bool, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.le](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.le](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.le](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __le__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool, **_]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A <= 2)
        ```
        """
        return self <= broadcast_to[dtype](other, self.shape, self.order())

    fn __gt__(
        self, other: Matrix[dtype, **_]
    ) raises -> Matrix[DType.bool, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.gt](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.gt](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.gt](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __gt__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool, **_]:
        """Matrix greater than scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A > 2)
        ```
        """
        return self > broadcast_to[dtype](other, self.shape, self.order())

    fn __ge__(
        self, other: Matrix[dtype, **_]
    ) raises -> Matrix[DType.bool, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.ge](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.ge](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.ge](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __ge__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool, **_]:
        """Matrix greater than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A >= 2)
        ```
        """
        return self >= broadcast_to[dtype](other, self.shape, self.order())

    fn __eq__(
        self, other: Matrix[dtype, **_]
    ) raises -> Matrix[DType.bool, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.eq](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.eq](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.eq](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __eq__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool, **_]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A == 2)
        ```
        """
        return self == broadcast_to[dtype](other, self.shape, self.order())

    fn __ne__(
        self, other: Matrix[dtype, **_]
    ) raises -> Matrix[DType.bool, **_]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.ne](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.ne](
                broadcast_to(self, other.shape, self.order()), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.ne](
                self, broadcast_to(other, self.shape, self.order())
            )

    fn __ne__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool, **_]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A != 2)
        ```
        """
        return self != broadcast_to[dtype](other, self.shape, self.order())

    fn __matmul__(self, other: Matrix[dtype, **_]) raises -> Matrix[dtype, **_]:
        return numojo.linalg.matmul(self, other)

    # ===-------------------------------------------------------------------===#
    # Core methods
    # ===-------------------------------------------------------------------===#

    fn all(self) -> Scalar[dtype]:
        """
        Test whether all array elements evaluate to True.
        """
        return numojo.logic.all(self)

    fn all(self, axis: Int) raises -> Matrix[dtype, OwnData]:
        """
        Test whether all array elements evaluate to True along axis.
        """
        return numojo.logic.all[dtype](self, axis=axis)

    fn any(self) -> Scalar[dtype]:
        """
        Test whether any array elements evaluate to True.
        """
        return numojo.logic.any(self)

    fn any(self, axis: Int) raises -> Matrix[dtype, OwnData]:
        """
        Test whether any array elements evaluate to True along axis.
        """
        return numojo.logic.any(self, axis=axis)

    fn argmax(self) raises -> Scalar[DType.int]:
        """
        Index of the max. It is first flattened before sorting.
        """
        return numojo.math.argmax(self)

    fn argmax(self, axis: Int) raises -> Matrix[DType.int]:
        """
        Index of the max along the given axis.
        """
        return numojo.math.argmax(self, axis=axis)

    fn argmin(self) raises -> Scalar[DType.int]:
        """
        Index of the min. It is first flattened before sorting.
        """
        return numojo.math.argmin(self)

    fn argmin(self, axis: Int) raises -> Matrix[DType.int]:
        """
        Index of the min along the given axis.
        """
        return numojo.math.argmin(self, axis=axis)

    fn argsort(self) raises -> Matrix[DType.int]:
        """
        Argsort the Matrix. It is first flattened before sorting.
        """
        return numojo.math.argsort(self)

    fn argsort(self, axis: Int) raises -> Matrix[DType.int]:
        """
        Argsort the Matrix along the given axis.
        """
        return numojo.math.argsort(self, axis=axis)

    fn astype[asdtype: DType](self) -> Matrix[asdtype]:
        """
        Copy of the matrix, cast to a specified type.
        """
        var res = Matrix[asdtype](
            shape=(self.shape[0], self.shape[1]), order=self.order()
        )
        for i in range(self.size):
            res._buf.ptr[i] = self._buf.ptr[i].cast[asdtype]()
        return res^

    fn cumprod(self) raises -> Matrix[dtype, OwnData]:
        """
        Cumprod of flattened matrix.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand(shape=(100, 100))
        print(A.cumprod())
        ```
        """
        return numojo.math.cumprod(self.copy())

    fn cumprod(self, axis: Int) raises -> Matrix[dtype, OwnData]:
        """
        Cumprod of Matrix along the axis.

        Args:
            axis: 0 or 1.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand(shape=(100, 100))
        print(A.cumprod(axis=0))
        print(A.cumprod(axis=1))
        ```
        """
        return numojo.math.cumprod(self.copy(), axis=axis)

    fn cumsum(self) raises -> Matrix[dtype, OwnData]:
        return numojo.math.cumsum(self.copy())

    fn cumsum(self, axis: Int) raises -> Matrix[dtype, OwnData]:
        return numojo.math.cumsum(self.copy(), axis=axis)

    fn fill(self, fill_value: Scalar[dtype]):
        """
        Fill the matrix with value.

        See also function `mat.creation.full`.
        """
        for i in range(self.size):
            self._buf.ptr[i] = fill_value

    # * Make it inplace?
    fn flatten(self) -> Matrix[dtype, OwnData]:
        """
        Return a flattened copy of the matrix.
        """
        var res = Matrix[dtype, OwnData](
            shape=(1, self.size), order=self.order()
        )
        memcpy(dest=res._buf.ptr, src=self._buf.ptr, count=res.size)
        return res^

    fn inv(self) raises -> Matrix[dtype, OwnData]:
        """
        Inverse of matrix.
        """
        return numojo.linalg.inv(self)

    fn order(self) -> String:
        """
        Returns the order.
        """
        var order: String = "F"
        if self.flags.C_CONTIGUOUS:
            order = "C"
        return order

    fn max(self) raises -> Scalar[dtype]:
        """
        Find max item. It is first flattened before sorting.
        """
        return numojo.math.extrema.max(self)

    fn max(self, axis: Int) raises -> Matrix[dtype, OwnData]:
        """
        Find max item along the given axis.
        """
        return numojo.math.extrema.max(self, axis=axis)

    fn mean[
        returned_dtype: DType = DType.float64
    ](self) raises -> Scalar[returned_dtype]:
        """
        Calculate the arithmetic average of all items in the Matrix.
        """
        return numojo.statistics.mean[returned_dtype](self)

    fn mean[
        returned_dtype: DType = DType.float64
    ](self, axis: Int) raises -> Matrix[returned_dtype]:
        """
        Calculate the arithmetic average of a Matrix along the axis.

        Args:
            axis: 0 or 1.
        """
        return numojo.statistics.mean[returned_dtype](self, axis=axis)

    fn min(self) raises -> Scalar[dtype]:
        """
        Find min item. It is first flattened before sorting.
        """
        return numojo.math.extrema.min(self)

    fn min(self, axis: Int) raises -> Matrix[dtype, OwnData]:
        """
        Find min item along the given axis.
        """
        return numojo.math.extrema.min(self, axis=axis)

    fn prod(self) -> Scalar[dtype]:
        """
        Product of all items in the Matrix.
        """
        return numojo.math.prod(self)

    fn prod(self, axis: Int) raises -> Matrix[dtype]:
        """
        Product of items in a Matrix along the axis.

        Args:
            axis: 0 or 1.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand(shape=(100, 100))
        print(A.prod(axis=0))
        print(A.prod(axis=1))
        ```
        """
        return numojo.math.prod(self, axis=axis)

    fn reshape(self, shape: Tuple[Int, Int]) raises -> Matrix[dtype]:
        """
        Change shape and size of matrix and return a new matrix.
        """
        if shape[0] * shape[1] != self.size:
            raise Error(
                String(
                    "Cannot reshape matrix of size {} into shape ({}, {})."
                ).format(self.size, shape[0], shape[1])
            )
        var res = Matrix[dtype](shape=shape, order="C")
        if self.flags.F_CONTIGUOUS:
            var temp = self.reorder_layout()
            memcpy(dest=res._buf.ptr, src=temp._buf.ptr, count=res.size)
            res = res.reorder_layout()
        else:
            memcpy(dest=res._buf.ptr, src=self._buf.ptr, count=res.size)
        return res^

    fn resize(mut self, shape: Tuple[Int, Int]) raises:
        """
        Change shape and size of matrix in-place.
        """
        if shape[0] * shape[1] > self.size:
            var other = Matrix[dtype, Self.BufType, origin](shape=shape, order=self.order())
            if self.flags.C_CONTIGUOUS:
                memcpy(dest=other._buf.ptr, src=self._buf.ptr, count=self.size)
                for i in range(self.size, other.size):
                    other._buf.ptr[i] = 0
            else:
                var min_rows = min(self.shape[0], shape[0])
                var min_cols = min(self.shape[1], shape[1])

                for j in range(min_cols):
                    for i in range(min_rows):
                        other._buf.ptr[i + j * shape[0]] = self._buf.ptr[i + j * self.shape[0]]
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

    fn round(self, decimals: Int) raises -> Matrix[dtype]:
        return numojo.math.rounding.round(self, decimals=decimals)

    fn std[
        returned_dtype: DType = DType.float64
    ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
        """
        Compute the standard deviation.

        Args:
            ddof: Delta degree of freedom.
        """
        return numojo.statistics.std[returned_dtype](self, ddof=ddof)

    fn std[
        returned_dtype: DType = DType.float64
    ](self, axis: Int, ddof: Int = 0) raises -> Matrix[returned_dtype]:
        """
        Compute the standard deviation along axis.

        Args:
            axis: 0 or 1.
            ddof: Delta degree of freedom.
        """
        return numojo.statistics.std[returned_dtype](self, axis=axis, ddof=ddof)

    fn sum(self) -> Scalar[dtype]:
        """
        Sum up all items in the Matrix.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand(shape=(100, 100))
        print(A.sum())
        ```
        """
        return numojo.math.sum(self)

    fn sum(self, axis: Int) raises -> Matrix[dtype, OwnData]:
        """
        Sum up the items in a Matrix along the axis.

        Args:
            axis: 0 or 1.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand(shape=(100, 100))
        print(A.sum(axis=0))
        print(A.sum(axis=1))
        ```
        """
        return numojo.math.sum(self, axis=axis)

    fn trace(self) raises -> Scalar[dtype]:
        """
        Trace of matrix.
        """
        return numojo.linalg.trace(self)

    fn issymmetric(self) -> Bool:
        """
        Transpose of matrix.
        """
        return issymmetric(self)

    fn transpose(self) -> Matrix[dtype, OwnData]:
        """
        Transpose of matrix.
        """
        return transpose(self)

    # TODO: we should only allow this for owndata. not for views, it'll lead to weird origin behaviours.
    fn reorder_layout(self) raises -> Matrix[dtype, **_]:
        """
        Reorder_layout matrix.
        """
        return reorder_layout(self)

    fn T(self) -> Matrix[dtype, OwnData]:
        return transpose(self)

    fn variance[
        returned_dtype: DType = DType.float64
    ](self, ddof: Int = 0) raises -> Scalar[returned_dtype]:
        """
        Compute the variance.

        Args:
            ddof: Delta degree of freedom.
        """
        return numojo.statistics.variance[returned_dtype](self, ddof=ddof)

    fn variance[
        returned_dtype: DType = DType.float64
    ](self, axis: Int, ddof: Int = 0) raises -> Matrix[returned_dtype]:
        """
        Compute the variance along axis.

        Args:
            axis: 0 or 1.
            ddof: Delta degree of freedom.
        """
        return numojo.statistics.variance[returned_dtype](
            self, axis=axis, ddof=ddof
        )

    # ===-------------------------------------------------------------------===#
    # To other data types
    # ===-------------------------------------------------------------------===#

    fn to_ndarray(self) raises -> NDArray[dtype]:
        """Create `NDArray` from `Matrix`.

        It makes a copy of the buffer of the matrix.
        """

        var ndarray: NDArray[dtype] = NDArray[dtype](
            shape=List[Int](self.shape[0], self.shape[1]), order="C"
        )
        memcpy(dest=ndarray._buf.ptr, src=self._buf.ptr, count=ndarray.size)

        return ndarray^

    fn to_numpy(self) raises -> PythonObject:
        """See `numojo.core.utility.to_numpy`."""
        try:
            var np = Python.import_module("numpy")

            var np_arr_dim = Python.list()
            np_arr_dim.append(self.shape[0])
            np_arr_dim.append(self.shape[1])

            np.set_printoptions(4)

            # Implement a dictionary for this later
            var numpyarray: PythonObject
            var np_dtype = np.float64
            if dtype == DType.float16:
                np_dtype = np.float16
            elif dtype == DType.float32:
                np_dtype = np.float32
            elif dtype == DType.int64:
                np_dtype = np.int64
            elif dtype == DType.int32:
                np_dtype = np.int32
            elif dtype == DType.int16:
                np_dtype = np.int16
            elif dtype == DType.int8:
                np_dtype = np.int8
            elif dtype == DType.uint64:
                np_dtype = np.uint64
            elif dtype == DType.uint32:
                np_dtype = np.uint32
            elif dtype == DType.uint16:
                np_dtype = np.uint16
            elif dtype == DType.uint8:
                np_dtype = np.uint8
            elif dtype == DType.bool:
                np_dtype = np.bool_
            elif dtype == DType.int:
                np_dtype = np.int64

            var order = "C" if self.flags.C_CONTIGUOUS else "F"
            numpyarray = np.empty(np_arr_dim, dtype=np_dtype, order=order)
            var pointer_d = numpyarray.__array_interface__["data"][
                0
            ].unsafe_get_as_pointer[dtype]()
            memcpy(dest=pointer_d, src=self._buf.ptr, count=self.size)

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
    ) -> Matrix[datatype, OwnData]:
        """Return a matrix with given shape and filled value.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.full(shape=(10, 10), fill_value=100)
        ```
        """

        var matrix = Matrix[datatype, OwnData](shape, order)
        for i in range(shape[0] * shape[1]):
            matrix._buf.ptr.store(i, fill_value)

        return matrix^

    @staticmethod
    fn zeros[
        datatype: DType = DType.float64
    ](shape: Tuple[Int, Int], order: String = "C") -> Matrix[datatype, OwnData]:
        """Return a matrix with given shape and filled with zeros.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.ones(shape=(10, 10))
        ```
        """

        var res = Matrix[datatype, OwnData](shape, order)
        memset_zero(res._buf.ptr, res.size)
        return res^

    @staticmethod
    fn ones[
        datatype: DType = DType.float64
    ](shape: Tuple[Int, Int], order: String = "C") -> Matrix[datatype, OwnData]:
        """Return a matrix with given shape and filled with ones.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.ones(shape=(10, 10))
        ```
        """

        return Matrix.full[datatype](shape=shape, fill_value=1)

    @staticmethod
    fn identity[
        datatype: DType = DType.float64
    ](len: Int, order: String = "C") -> Matrix[datatype, OwnData]:
        """Return an identity matrix with given size.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.identity(12)
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
    ](shape: Tuple[Int, Int], order: String = "C") -> Matrix[datatype, OwnData]:
        """Return a matrix with random values uniformed distributed between 0 and 1.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand((12, 12))
        ```

        Args:
            shape: The shape of the Matrix.
            order: The order of the Matrix. "C" or "F".
        """
        var result = Matrix[datatype](shape, order)
        for i in range(result.size):
            result._buf.ptr.store(i, random_float64(0, 1).cast[datatype]())
        return result^

    @staticmethod
    fn fromlist[
        datatype: DType = DType.float64
    ](
        object: List[Scalar[datatype]],
        shape: Tuple[Int, Int] = (0, 0),
        order: String = "C",
    ) raises -> Matrix[datatype, OwnData]:
        """Create a matrix from a 1-dimensional list into given shape.

        If no shape is passed, the return matrix will be a row vector.

        Example:
        ```mojo
        from numojo import Matrix
        fn main() raises:
            print(Matrix.fromlist(List[Float64](1, 2, 3, 4, 5), (5, 1)))
        ```
        """

        if (shape[0] == 0) and (shape[1] == 0):
            var M = Matrix[datatype](shape=(1, len(object)))
            memcpy(dest=M._buf.ptr, src=object.unsafe_ptr(), count=M.size)
            return M^

        if shape[0] * shape[1] != len(object):
            var message = String(
                "The input has {} elements, but the target has the shape {}x{}"
            ).format(len(object), shape[0], shape[1])
            raise Error(message)
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
    ) raises -> Matrix[datatype, OwnData]:
        """Matrix initialization from string representation of an matrix.

        Comma, right brackets, and whitespace are treated as seperators of numbers.
        Digits, underscores, and minus signs are treated as a part of the numbers.

        If now shape is passed, the return matrix will be a row vector.

        Example:
        ```mojo
        from numojo.prelude import *
        from numojo import Matrix
        fn main() raises:
            var A = Matrix[f32].fromstring(
            "1 2 .3 4 5 6.5 7 1_323.12 9 10, 11.12, 12 13 14 15 16", (4, 4))
        ```
        ```console
        [[1.0   2.0     0.30000001192092896     4.0]
        [5.0   6.5     7.0     1323.1199951171875]
        [9.0   10.0    11.119999885559082      12.0]
        [13.0  14.0    15.0    16.0]]
        Size: 4x4  datatype: float32
        ```

        Args:
            text: String representation of a matrix.
            shape: Shape of the matrix.
            order: Order of the matrix. "C" or "F".
        """

        var data = List[Scalar[datatype]]()
        var bytes = text.as_bytes()
        var number_as_str: String = ""
        var size = shape[0] * shape[1]

        for i in range(len(bytes)):
            var b = bytes[i]
            if (
                chr(Int(b)).isdigit()
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
            var message = String(
                "The number of items in the string is {}, which does not match"
                " the given shape {}x{}."
            ).format(len(data), shape[0], shape[1])
            raise Error(message)

        var result = Matrix[datatype](shape=shape)
        for i in range(len(data)):
            result._buf.ptr[i] = data[i]
        return result^


# ===-----------------------------------------------------------------------===#
# MatrixIter struct
# ===-----------------------------------------------------------------------===#


# ! Should the iterator be mutable or not?
struct _MatrixIter[
    lifetime: MutOrigin,
    dtype: DType,
    buf_type: Buffered,
    forward: Bool = True,
](Copyable, Movable):
    """Iterator for Matrix.

    Parameters:
        lifetime: The lifetime of the underlying Matrix data.
        dtype: The data type of the item.
        buf_type: The buffer type of the underlying Matrix, OwnData or RefData.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var matrix: Matrix[dtype, buf_type, lifetime]
    var length: Int

    fn __init__(
        out self,
        mut matrix: Matrix[dtype, buf_type, lifetime],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.matrix = matrix

    fn __iter__(self) -> Self:
        return self.copy()

    fn __next__(
        mut self,
    ) raises -> Matrix[
        dtype,
        RefData[origin_of(self.matrix)],
        origin_of(self.matrix)
    ]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.matrix.get(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.matrix.get(current_index)

    @always_inline
    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index


# ===-----------------------------------------------------------------------===#
# Backend fucntions using SMID functions
# ===-----------------------------------------------------------------------===#


fn _arithmetic_func_matrix_matrix_to_matrix[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
](read A: Matrix[dtype, **_], read B: Matrix[dtype, **_]) raises -> Matrix[
    dtype, OwnData
]:
    """
    Matrix[dtype] & Matrix[dtype] -> Matrix[dtype]

    For example: `__add__`, `__sub__`, etc.
    """
    alias simd_width = simd_width_of[dtype]()
    if A.order() != B.order():
        raise Error(
            String("Matrix order {} does not match {}.").format(
                A.order(), B.order()
            )
        )

    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        raise Error(
            String("Shape {}x{} does not match {}x{}.").format(
                A.shape[0], A.shape[1], B.shape[0], B.shape[1]
            )
        )

    var res = Matrix[dtype](shape=A.shape, order=A.order())

    @parameter
    fn vec_func[simd_width: Int](i: Int):
        res._buf.ptr.store(
            i,
            simd_func(
                A._buf.ptr.load[width=simd_width](i),
                B._buf.ptr.load[width=simd_width](i),
            ),
        )

    vectorize[vec_func, simd_width](A.size)
    return res^


fn _arithmetic_func_matrix_to_matrix[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Matrix[dtype] -> Matrix[dtype]

    For example: `sin`, `cos`, etc.
    """
    alias simd_width: Int = simd_width_of[dtype]()

    var C: Matrix[dtype] = Matrix[dtype](shape=A.shape, order=A.order())

    @parameter
    fn vec_func[simd_width: Int](i: Int):
        C._buf.ptr.store(i, simd_func(A._buf.ptr.load[width=simd_width](i)))

    vectorize[vec_func, simd_width](A.size)

    return C^


fn _logic_func_matrix_matrix_to_matrix[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[DType.bool, simd_width],
](A: Matrix[dtype, **_], B: Matrix[dtype, **_]) raises -> Matrix[
    DType.bool, **_
]:
    """
    Matrix[dtype] & Matrix[dtype] -> Matrix[bool]
    """
    alias width = simd_width_of[dtype]()

    if A.order() != B.order():
        raise Error(
            String("Matrix order {} does not match {}.").format(
                A.order(), B.order()
            )
        )

    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        raise Error(
            String("Shape {}x{} does not match {}x{}.").format(
                A.shape[0], A.shape[1], B.shape[0], B.shape[1]
            )
        )

    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var C = Matrix[DType.bool](shape=A.shape, order=A.order())

    @parameter
    fn calculate_CC(m: Int):
        @parameter
        fn vec_func[simd_width: Int](n: Int):
            C._store[simd_width](
                m,
                n,
                simd_func(A._load[simd_width](m, n), B._load[simd_width](m, n)),
            )

        vectorize[vec_func, width](t1)

    parallelize[calculate_CC](t0, t0)

    var _t0 = t0
    var _t1 = t1
    var _A = (
        A.copy()
    )  # ! perhaps remove this explicit copy if we don't need to extend it's lifetime.
    var _B = B.copy()

    return C^
