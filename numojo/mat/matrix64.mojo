"""
`numojo.mat.matrix` provides:

- `Matrix64` type (2DArray).
- `_Matrix64Iter` type (for iteration).
- Dunder methods for initialization, indexing, slicing, and arithmetics.
- Auxiliary functions.
"""

from numojo.core.ndarray import NDArray
from memory import UnsafePointer, memcpy
from sys import simdwidthof
from algorithm import parallelize, vectorize
from python import PythonObject, Python

# ===----------------------------------------------------------------------===#
# Matrix64 struct
# ===----------------------------------------------------------------------===#

alias Matrix64 = Matrix64Base[OwnedData]


trait ArrayBuffered:
    # var ptr: UnsafePointer[Float64]
    # var size: Int

    fn __init__(out self, size: Int):
        ...

    fn __init__(out self, ptr: UnsafePointer[Float64], offset: Int = 0):
        ...

    fn __copyinit__(out self, other: Self):
        ...

    fn __moveinit__(out self, owned other: Self):
        ...

    fn get_ptr(self) -> UnsafePointer[Float64]:
        ...


struct OwnedData(ArrayBuffered):
    var ptr: UnsafePointer[Float64]

    fn __init__(out self, size: Int):
        self.ptr = UnsafePointer[Float64]().alloc(size)

    fn __init__(out self, ptr: UnsafePointer[Float64], offset: Int = 0):
        self.ptr = ptr + offset

    fn __copyinit__(out self, other: Self):
        self.ptr = other.ptr

    fn __moveinit__(out self, owned other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> UnsafePointer[Float64]:
        return self.ptr


struct RefData[is_mutable: Bool, //, origin: Origin[is_mutable]](ArrayBuffered):
    var ptr: UnsafePointer[Float64]

    fn __init__(out self, size: Int):
        self.ptr = UnsafePointer[Float64]().alloc(size)

    fn __init__(out self, ptr: UnsafePointer[Float64], offset: Int = 0):
        self.ptr = ptr + offset

    fn __copyinit__(out self, other: Self):
        self.ptr = other.ptr

    fn __moveinit__(out self, owned other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> UnsafePointer[Float64]:
        return self.ptr


struct Matrix64Base[BType: ArrayBuffered](Stringable, Writable):
    """
    `Matrix64` is a special case of `NDArray` (2DArray) but has some targeted
    optimization since the number of dimensions is known at the compile time.
    It gains some advantages in running speed, which is very useful when users
    only want to work with 2-dimensional arrays.
    The indexing and slicing is also more consistent with `numpy`.

    For certain behaviors, `Matrix64` type is more like `NDArray` with
    fixed `ndim` than `numpy.matrix`.

    - For `__getitem__`, passing in two `Int` returns a scalar,
    and passing in one `Int` or two `Slice` returns a `Matrix64`.
    - We do not need auxiliary types `NDArrayShape` and `NDArrayStrides`
    as the shape and strides information is fixed in length `Tuple[Int,Int]`.

    Parameters:
        BType: Buffer type.

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
    - , shape
    - , data

    [checklist] CORE METHODS that have been implemented:
    - [x] `Matrix64.any` and `mat.logic.all`
    - [x] `Matrix64.any` and `mat.logic.any`
    - [x] `Matrix64.argmax` and `mat.sorting.argmax`
    - [x] `Matrix64.argmin` and `mat.sorting.argmin`
    - [x] `Matrix64.argsort` and `mat.sorting.argsort`
    - [x] `Matrix64.astype`
    - [x] `Matrix64.cumprod` and `mat.mathematics.cumprod`
    - [x] `Matrix64.cumsum` and `mat.mathematics.cumsum`
    - [x] `Matrix64.fill` and `mat.creation.full`
    - [x] `Matrix64.flatten`
    - [x] `Matrix64.inv` and `mat.linalg.inv`
    - [x] `Matrix64.max` and `mat.sorting.max`
    - [x] `Matrix64.mean` and `mat.statistics.mean`
    - [x] `Matrix64.min` and `mat.sorting.min`
    - [x] `Matrix64.prod` and `mat.mathematics.prod`
    - [x] `Matrix64.reshape`
    - [x] `Matrix64.resize`
    - [x] `Matrix64.round` and `mat.mathematics.round` (TODO: Check this after next Mojo update)
    - [x] `Matrix64.std` and `mat.statistics.std`
    - [x] `Matrix64.sum` and `mat.mathematics.sum`
    - [x] `Matrix64.trace` and `mat.linalg.trace`
    - [x] `Matrix64.transpose` and `mat.linalg.transpose` (also `Matrix64.T`)
    - [x] `Matrix64.variance` and `mat.statistics.variance` (`var` is primitive)

    TODO: Introduce `ArrayLike` trait for `NDArray` type and `Matrix64` type.

    """

    var shape: Tuple[Int, Int]
    """Shape of Matrix64."""

    # To be calculated at the initialization.
    var size: Int
    """Size of Matrix64."""
    var strides: Tuple[Int, Int]
    """Strides of matrix."""

    # To be filled by constructing functions.
    var _buf: BType
    """Data buffer of the items in the NDArray."""

    alias width: Int = simdwidthof[DType.float64]()  #
    """Vector size of the data type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Tuple[Int, Int],
    ):
        """
        Matrix64 NDArray initialization.

        Args:
            shape: List of shape.
        """

        self.shape = (shape[0], shape[1])
        self.strides = (shape[1], 1)
        self.size = shape[0] * shape[1]
        self._buf = BType(size=self.size)

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Tuple[Int, Int],
        ptr: UnsafePointer[Float64],
    ):
        """
        Matrix64 NDArray initialization.
        """

        self.shape = (shape[0], shape[1])
        self.strides = (shape[1], 1)
        self.size = shape[0] * shape[1]
        self._buf = BType(ptr=ptr)

    @always_inline("nodebug")
    fn __init__(
        out self,
        data: Self,
    ):
        """Create a matrix from a matrix."""

        self = data

    @always_inline("nodebug")
    fn __init__(
        mut self,
        data: NDArray[DType.float64],
    ) raises:
        """
        Create Matrix64 from NDArray.
        """

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

        self._buf = BType(self.size)

        if (data.order == "C") or (data.ndim == 1):
            memcpy(self._buf.get_ptr(), data._buf, self.size)
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self._store(i, j, data.load(i, j))

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self._buf = BType(other.size)
        memcpy(self._buf.get_ptr(), other._buf.get_ptr(), other.size)

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned other: Self):
        """
        Move other into self.
        """
        self.shape = other.shape^
        self.strides = other.strides^
        self.size = other.size
        self._buf = other._buf^

    @always_inline("nodebug")
    fn __del__(owned self):
        self._buf.get_ptr().free()

    # ===-------------------------------------------------------------------===#
    # Dunder methods
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, owned x: Int, owned y: Int) raises -> Float64:
        """
        Return the scalar at the index.

        Args:
            x: The row number.
            y: The column number.

        Returns:
            A scalar matching the dtype of the array.
        """

        if x < 0:
            x = self.shape[0] + x

        if y < 0:
            y = self.shape[1] + y

        if (x >= self.shape[0]) or (y >= self.shape[1]):
            raise Error(
                String(
                    "Index ({}, {}) exceed the matrix shape ({}, {})"
                ).format(x, y, self.shape[0], self.shape[1])
            )

        return self._buf.get_ptr().load(x * self.strides[0] + y)

    fn __getitem__(
        ref self, owned x: Int
    ) raises -> Matrix64Base[RefData[__origin_of(self)]]:
        """
        Return the corresponding row at the index.

        Args:
            x: The row number.
        """

        var res = Matrix64Base[RefData[__origin_of(self)]](
            shape=(1, self.shape[1]),
            ptr=self._buf.get_ptr().offset(x * self.shape[1]),
        )
        return res

    fn _load[
        width: Int = 1
    ](self, x: Int, y: Int) -> SIMD[DType.float64, width]:
        """
        `__getitem__` with width.
        Unsafe: No boundary check!
        """
        return self._buf.get_ptr().load[width=width](x * self.strides[0] + y)

    fn __setitem__(self, x: Int, y: Int, value: Float64) raises:
        """
        Return the scalar at the index.

        Args:
            x: The row number.
            y: The column number.
            value: The value to be set.
        """

        if (x >= self.shape[0]) or (y >= self.shape[1]):
            raise Error(
                String(
                    "Index ({}, {}) exceed the matrix shape ({}, {})"
                ).format(x, y, self.shape[0], self.shape[1])
            )

        self._buf.get_ptr().store(x * self.strides[0] + y, value)

    fn __setitem__(self, owned x: Int, value: Self) raises:
        """
        Set the corresponding row at the index with the given matrix.

        Args:
            x: The row number.
            value: Matrix64 (row vector).
        """

        if x < 0:
            x = self.shape[0] + x

        if x >= self.shape[0]:
            raise Error(
                String(
                    "Error: Elements of `index` ({}) \n"
                    "exceed the matrix shape ({})."
                ).format(x, self.shape[0])
            )

        if value.shape[0] != 1:
            raise Error(
                String(
                    "Error: The value should has only 1 row, "
                    "but it has {} rows."
                ).format(value.shape[0])
            )

        if self.shape[1] != value.shape[1]:
            raise Error(
                String(
                    "Error: Matrix64 has {} columns, "
                    "but the value has {} columns."
                ).format(self.shape[1], value.shape[1])
            )

        var ptr = self._buf.get_ptr().offset(x * self.shape[1])
        memcpy(ptr, value._buf.get_ptr(), value.size)

    fn _store[
        width: Int = 1
    ](mut self, x: Int, y: Int, simd: SIMD[DType.float64, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.get_ptr().store(x * self.strides[0] + y, simd)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        fn print_row(self: Self, i: Int, sep: String) raises -> String:
            var result: String = str("[")
            var number_of_sep: Int = 1
            if self.shape[1] <= 6:
                for j in range(self.shape[1]):
                    if j == self.shape[1] - 1:
                        number_of_sep = 0
                    result += str(self[i, j]) + sep * number_of_sep
            else:
                for j in range(3):
                    result += str(self[i, j]) + sep
                result += str("...") + sep
                for j in range(self.shape[1] - 3, self.shape[1]):
                    if j == self.shape[1] - 1:
                        number_of_sep = 0
                    result += str(self[i, j]) + sep * number_of_sep
            result += str("]")
            return result

        var sep: String = str("\t")
        var newline: String = str("\n ")
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
                result += str("...") + newline
                for i in range(self.shape[0] - 3, self.shape[0]):
                    if i == self.shape[0] - 1:
                        number_of_newline = 0
                    result += (
                        print_row(self, i, sep) + newline * number_of_newline
                    )
        except e:
            print("Cannot transfer matrix to string!", e)
        result += str("]")
        writer.write(
            result
            + "\nSize: "
            + str(self.shape[0])
            + "x"
            + str(self.shape[1])
            + "  DType: "
            + str("Float64")
        )

    fn __iter__(ref self) raises -> _Matrix64Iter[__origin_of(self)]:
        """Iterate over elements of the Matrix64, returning copied value.

        Returns:
            An iterator of Matrix64 elements.
        """

        return _Matrix64Iter[__origin_of(self)](
            ptr=self._buf.get_ptr(),
            shape1=self.shape[1],
            length=self.shape[0],
        )

    fn __reversed__(
        ref self,
    ) raises -> _Matrix64Iter[__origin_of(self), forward=False]:
        """Iterate backwards over elements of the Matrix64, returning
        copied value.

        Returns:
            A reversed iterator of Matrix64 elements.
        """

        return _Matrix64Iter[__origin_of(self), forward=False](
            ptr=self._buf.get_ptr(),
            shape1=self.shape[1],
            length=self.shape[0],
        )


# ===-----------------------------------------------------------------------===#
# MatrixIter struct
# ===-----------------------------------------------------------------------===#


@value
struct _Matrix64Iter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
    forward: Bool = True,
]:
    """Iterator for Matrix64.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        origin: The lifetime of the underlying Matrix64 data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var ptr: UnsafePointer[Float64]
    var shape1: Int
    var length: Int

    fn __init__(
        out self,
        ptr: UnsafePointer[Float64],
        shape1: Int,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.shape1 = shape1
        self.ptr = ptr

    fn __iter__(ref self) -> Self:
        return self

    fn __next__(mut self) -> Matrix64Base[RefData[origin]]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            print("address of data:", self.ptr)
            return Matrix64Base[RefData[origin]](
                shape=(1, self.shape1),
                ptr=self.ptr.offset(current_index * self.shape1),
            )
        else:
            var current_index = self.index
            self.index -= 1
            return Matrix64Base[RefData[origin]](
                shape=(1, self.shape1),
                ptr=self.ptr.offset(current_index * self.shape1),
            )

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index

    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0
