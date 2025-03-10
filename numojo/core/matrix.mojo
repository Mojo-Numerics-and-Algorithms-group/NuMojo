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
from sys import simdwidthof
from python import PythonObject, Python

from numojo.core.flags import Flags
from numojo.core.ndarray import NDArray
from numojo.core.own_data import OwnData
from numojo.core.utility import _get_offset
from numojo.routines.manipulation import broadcast_to

# ===----------------------------------------------------------------------===#
# Matrix struct
# ===----------------------------------------------------------------------===#


struct Matrix[dtype: DType = DType.float64](
    CollectionElement, Sized, Stringable, Writable
):
    # TODO: Matrix[dtype: DType = DType.float64,
    #               Buffer: Bufferable[dtype] = OwnData[dtype]]
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

    alias width: Int = simdwidthof[dtype]()  #
    """Vector size of the data type."""

    var _buf: OwnData[dtype]
    """Data buffer of the items in the NDArray."""

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
        mut self,
        shape: Tuple[Int, Int],
    ):
        """
        Construct a matrix without initializing data.

        Args:
            shape: List of shape.
        """

        self.shape = (shape[0], shape[1])
        self.strides = (shape[1], 1)
        self.size = shape[0] * shape[1]
        self._buf = OwnData[dtype](size=self.size)
        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

    @always_inline("nodebug")
    fn __init__(
        mut self,
        data: Self,
    ):
        """
        Construct a matrix from matrix.
        """

        self = data

    @always_inline("nodebug")
    fn __init__(
        mut self,
        data: NDArray[dtype],
    ) raises:
        """
        Construct a matrix from array.
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

        self._buf = OwnData[dtype](self.size)

        self.flags = Flags(
            self.shape, self.strides, owndata=True, writeable=True
        )

        if data.flags["C_CONTIGUOUS"]:
            for i in range(data.shape[0]):
                memcpy(
                    self._buf.ptr.offset(i * self.shape[0]),
                    data._buf.ptr.offset(i * data.shape[0]),
                    self.shape[0],
                )
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self._store(i, j, data._getitem(i, j))

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self._buf = OwnData[dtype](other.size)
        memcpy(self._buf.ptr, other._buf.ptr, other.size)
        self.flags = other.flags

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned other: Self):
        """
        Move other into self.
        """
        self.shape = other.shape^
        self.strides = other.strides^
        self.size = other.size
        self._buf = other._buf^
        self.flags = other.flags^

    @always_inline("nodebug")
    fn __del__(owned self):
        var owndata = True
        try:
            owndata = self.flags["OWNDATA"]
        except:
            print("Invalid `OWNDATA` flag. Treat as `True`.")
        if owndata:
            self._buf.ptr.free()

    # ===-------------------------------------------------------------------===#
    # Slicing and indexing methods
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, owned x: Int, owned y: Int) raises -> Scalar[dtype]:
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

        return self._buf.ptr.load(x * self.strides[0] + y * self.strides[1])

    fn __getitem__(self, owned x: Int) raises -> Self:
        """
        Return the corresponding row at the index.

        Args:
            x: The row number.
        """

        if x < 0:
            x = self.shape[0] + x

        if x >= self.shape[0]:
            raise Error(
                String("Index {} exceed the row number {}").format(
                    x, self.shape[0]
                )
            )

        var res = Self(shape=(1, self.shape[1]))
        var ptr = self._buf.ptr.offset(x * self.shape[1])
        memcpy(res._buf.ptr, ptr, res.size)
        return res

    fn __getitem__(self, x: Slice, y: Slice) -> Self:
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

        # The new matrix with the corresponding shape
        var B = Matrix[dtype](shape=(len(range_x), len(range_y)))

        # Fill in the values at the corresponding index
        var c = 0
        for i in range_x:
            for j in range_y:
                B._buf.ptr[c] = self._load(i, j)
                c += 1

        return B

    fn __getitem__(self, x: Slice, owned y: Int) -> Self:
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

        # The new matrix with the corresponding shape
        var B = Matrix[dtype](shape=(len(range_x), 1))

        # Fill in the values at the corresponding index
        var c = 0
        for i in range_x:
            B._buf.ptr[c] = self._load(i, y)
            c += 1

        return B

    fn __getitem__(self, owned x: Int, y: Slice) -> Self:
        """
        Get item from one int and one slice.
        """
        if x < 0:
            x = self.shape[0] + x

        var start_y: Int
        var end_y: Int
        var step_y: Int
        start_y, end_y, step_y = y.indices(self.shape[1])
        var range_y = range(start_y, end_y, step_y)

        # The new matrix with the corresponding shape
        var B = Matrix[dtype](shape=(1, len(range_y)))

        # Fill in the values at the corresponding index
        var c = 0
        for j in range_y:
            B._buf.ptr[c] = self._load(x, j)
            c += 1

        return B

    fn __getitem__(self, indices: List[Int]) raises -> Self:
        """
        Get item by a list of integers.
        """

        var ncol = self.shape[1]
        var nrow = len(indices)
        var res = Matrix.zeros[dtype](shape=(nrow, ncol))
        for i in range(nrow):
            res[i] = self[indices[i]]
        return res^

    fn _load[width: Int = 1](self, x: Int, y: Int) -> SIMD[dtype, width]:
        """
        `__getitem__` with width.
        Unsafe: No boundary check!
        """
        return self._buf.ptr.load[width=width](
            x * self.strides[0] + y * self.strides[1]
        )

    fn __setitem__(self, x: Int, y: Int, value: Scalar[dtype]) raises:
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

        self._buf.ptr.store(x * self.strides[0] + y * self.strides[1], value)

    fn __setitem__(self, owned x: Int, value: Self) raises:
        """
        Set the corresponding row at the index with the given matrix.

        Args:
            x: The row number.
            value: Matrix (row vector).
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
                    "Error: Matrix has {} columns, "
                    "but the value has {} columns."
                ).format(self.shape[1], value.shape[1])
            )

        var ptr = self._buf.ptr.offset(x * self.shape[1])
        memcpy(ptr, value._buf.ptr, value.size)

    fn _store[
        width: Int = 1
    ](mut self, x: Int, y: Int, simd: SIMD[dtype, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.ptr.store(x * self.strides[0] + y * self.strides[1], simd)

    # ===-------------------------------------------------------------------===#
    # Other dunders and auxiliary methods
    # ===-------------------------------------------------------------------===#

    fn __iter__(self) raises -> _MatrixIter[__origin_of(self), dtype]:
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

        return _MatrixIter[__origin_of(self), dtype](
            matrix=self,
            length=self.shape[0],
        )

    fn __len__(self) -> Int:
        """
        Returns length of 0-th dimension.
        """
        return self.shape[0]

    fn __reversed__(
        self,
    ) raises -> _MatrixIter[__origin_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the Matrix, returning
        copied value.

        Returns:
            A reversed iterator of Matrix elements.
        """

        return _MatrixIter[__origin_of(self), dtype, forward=False](
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
                + "  C: "
                + String(self.flags["C_CONTIGUOUS"])
                + "  F: "
                + String(self.flags["F_CONTIGUOUS"])
                + "  Own: "
                + String(self.flags["OWNDATA"])
            )
        except e:
            print("Cannot transfer matrix to string!", e)

    # ===-------------------------------------------------------------------===#
    # Arithmetic dunder methods
    # ===-------------------------------------------------------------------===#

    fn __add__(self, other: Self) raises -> Self:
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
            ](broadcast_to(self, other.shape), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__add__
            ](self, broadcast_to(other, self.shape))

    fn __add__(self, other: Scalar[dtype]) raises -> Self:
        """Add matrix to scalar.

        ```mojo
        from numojo import Matrix
        var A = Matrix.ones(shape=(4, 4))
        print(A + 2)
        ```
        """
        return self + broadcast_to[dtype](other, self.shape)

    fn __radd__(self, other: Scalar[dtype]) raises -> Self:
        """
        Right-add.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(2 + A)
        ```
        """
        return broadcast_to[dtype](other, self.shape) + self

    fn __sub__(self, other: Self) raises -> Self:
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
            ](broadcast_to(self, other.shape), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__sub__
            ](self, broadcast_to(other, self.shape))

    fn __sub__(self, other: Scalar[dtype]) raises -> Self:
        """Subtract matrix by scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix(shape=(4, 4))
        print(A - 2)
        ```
        """
        return self - broadcast_to[dtype](other, self.shape)

    fn __rsub__(self, other: Scalar[dtype]) raises -> Self:
        """
        Right-sub.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(2 - A)
        ```
        """
        return broadcast_to[dtype](other, self.shape) - self

    fn __mul__(self, other: Self) raises -> Self:
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
            ](broadcast_to(self, other.shape), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__mul__
            ](self, broadcast_to(other, self.shape))

    fn __mul__(self, other: Scalar[dtype]) raises -> Self:
        """Mutiply matrix by scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A * 2)
        ```
        """
        return self * broadcast_to[dtype](other, self.shape)

    fn __rmul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Right-mul.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(2 * A)
        ```
        """
        return broadcast_to[dtype](other, self.shape) * self

    fn __truediv__(self, other: Self) raises -> Self:
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
            ](broadcast_to(self, other.shape), other)
        else:
            return _arithmetic_func_matrix_matrix_to_matrix[
                dtype, SIMD.__truediv__
            ](self, broadcast_to(other, self.shape))

    fn __truediv__(self, other: Scalar[dtype]) raises -> Self:
        """Divide matrix by scalar."""
        return self / broadcast_to[dtype](other, self.shape)

    fn __pow__(self, rhs: Scalar[dtype]) raises -> Self:
        """Power of items."""
        var res = self
        for i in range(self.size):
            res._buf.ptr[i] = self._buf.ptr[i].__pow__(rhs)
        return res^

    fn __lt__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__lt__](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__lt__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__lt__](
                self, broadcast_to(other, self.shape)
            )

    fn __lt__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A < 2)
        ```
        """
        return self < broadcast_to[dtype](other, self.shape)

    fn __le__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__le__](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__le__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__le__](
                self, broadcast_to(other, self.shape)
            )

    fn __le__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A <= 2)
        ```
        """
        return self <= broadcast_to[dtype](other, self.shape)

    fn __gt__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__gt__](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__gt__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__gt__](
                self, broadcast_to(other, self.shape)
            )

    fn __gt__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix greater than scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A > 2)
        ```
        """
        return self > broadcast_to[dtype](other, self.shape)

    fn __ge__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__ge__](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__ge__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__ge__](
                self, broadcast_to(other, self.shape)
            )

    fn __ge__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix greater than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A >= 2)
        ```
        """
        return self >= broadcast_to[dtype](other, self.shape)

    fn __eq__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__eq__](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__eq__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__eq__](
                self, broadcast_to(other, self.shape)
            )

    fn __eq__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A == 2)
        ```
        """
        return self == broadcast_to[dtype](other, self.shape)

    fn __ne__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__ne__](
                self, other
            )
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__ne__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func_matrix_matrix_to_matrix[dtype, SIMD.__ne__](
                self, broadcast_to(other, self.shape)
            )

    fn __ne__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo import Matrix
        A = Matrix.ones(shape=(4, 4))
        print(A != 2)
        ```
        """
        return self != broadcast_to[dtype](other, self.shape)

    fn __matmul__(self, other: Self) raises -> Self:
        return numojo.linalg.matmul(self, other)

    # ===-------------------------------------------------------------------===#
    # Core methods
    # ===-------------------------------------------------------------------===#

    fn all(self) -> Scalar[dtype]:
        """
        Test whether all array elements evaluate to True.
        """
        return numojo.logic.all(self)

    fn all(self, axis: Int) raises -> Self:
        """
        Test whether all array elements evaluate to True along axis.
        """
        return numojo.logic.all(self, axis=axis)

    fn any(self) -> Scalar[dtype]:
        """
        Test whether any array elements evaluate to True.
        """
        return numojo.logic.any(self)

    fn any(self, axis: Int) raises -> Self:
        """
        Test whether any array elements evaluate to True along axis.
        """
        return numojo.logic.any(self, axis=axis)

    fn argmax(self) raises -> Scalar[DType.index]:
        """
        Index of the max. It is first flattened before sorting.
        """
        return numojo.math.argmax(self)

    fn argmax(self, axis: Int) raises -> Matrix[DType.index]:
        """
        Index of the max along the given axis.
        """
        return numojo.math.argmax(self, axis=axis)

    fn argmin(self) raises -> Scalar[DType.index]:
        """
        Index of the min. It is first flattened before sorting.
        """
        return numojo.math.argmin(self)

    fn argmin(self, axis: Int) raises -> Matrix[DType.index]:
        """
        Index of the min along the given axis.
        """
        return numojo.math.argmin(self, axis=axis)

    fn argsort(self) raises -> Matrix[DType.index]:
        """
        Argsort the Matrix. It is first flattened before sorting.
        """
        return numojo.math.argsort(self)

    fn argsort(self, axis: Int) raises -> Matrix[DType.index]:
        """
        Argsort the Matrix along the given axis.
        """
        return numojo.math.argsort(self, axis=axis)

    fn astype[asdtype: DType](self) -> Matrix[asdtype]:
        """
        Copy of the matrix, cast to a specified type.
        """
        var res = Matrix[asdtype](shape=(self.shape[0], self.shape[1]))
        for i in range(self.size):
            res._buf.ptr[i] = self._buf.ptr[i].cast[asdtype]()
        return res^

    fn cumprod(self) -> Matrix[dtype]:
        """
        Cumprod of flattened matrix.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand(shape=(100, 100))
        print(A.cumprod())
        ```
        """
        return numojo.math.cumprod(self)

    fn cumprod(self, axis: Int) raises -> Matrix[dtype]:
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
        return numojo.math.cumprod(self, axis=axis)

    fn cumsum(self) -> Matrix[dtype]:
        return numojo.math.cumsum(self)

    fn cumsum(self, axis: Int) raises -> Matrix[dtype]:
        return numojo.math.cumsum(self, axis=axis)

    fn fill(self: Self, fill_value: Scalar[dtype]):
        """
        Fill the matrix with value.

        See also function `mat.creation.full`.
        """
        for i in range(self.size):
            self._buf.ptr[i] = fill_value

    fn flatten(self) -> Self:
        """
        Return a flattened copy of the matrix.
        """
        var res = Self(shape=(1, self.size))
        memcpy(res._buf.ptr, self._buf.ptr, res.size)
        return res^

    fn inv(self) raises -> Self:
        """
        Inverse of matrix.
        """
        return numojo.linalg.inv(self)

    fn max(self) raises -> Scalar[dtype]:
        """
        Find max item. It is first flattened before sorting.
        """
        return numojo.math.extrema.max(self)

    fn max(self, axis: Int) raises -> Self:
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

    fn min(self, axis: Int) raises -> Self:
        """
        Find min item along the given axis.
        """
        return numojo.math.extrema.min(self, axis=axis)

    fn prod(self) -> Scalar[dtype]:
        """
        Product of all items in the Matrix.
        """
        return numojo.math.prod(self)

    fn prod(self, axis: Int) raises -> Self:
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

    fn reshape(self, shape: Tuple[Int, Int]) raises -> Self:
        """
        Change shape and size of matrix and return a new matrix.
        """
        if shape[0] * shape[1] != self.size:
            raise Error(
                String(
                    "Cannot reshape matrix of size {} into shape ({}, {})."
                ).format(self.size, shape[0], shape[1])
            )
        var res = Self(shape=shape)
        memcpy(res._buf.ptr, self._buf.ptr, res.size)
        return res^

    fn resize(mut self, shape: Tuple[Int, Int]):
        """
        Change shape and size of matrix in-place.
        """
        if shape[0] * shape[1] > self.size:
            var other = Self(shape=shape)
            memcpy(other._buf.ptr, self._buf.ptr, self.size)
            for i in range(self.size, other.size):
                other._buf.ptr[i] = 0
            self = other
        else:
            self.shape[0] = shape[0]
            self.shape[1] = shape[1]
            self.size = shape[0] * shape[1]
            self.strides[0] = shape[1]

    fn round(self, decimals: Int) raises -> Self:
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

    fn sum(self, axis: Int) raises -> Self:
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
        Transpose of matrix.
        """
        return numojo.linalg.trace(self)

    fn transpose(self) -> Self:
        """
        Transpose of matrix.
        """
        return transpose(self)

    fn T(self) -> Self:
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

        var ndarray = NDArray[dtype](
            shape=List[Int](self.shape[0], self.shape[1]), order="C"
        )
        memcpy(ndarray._buf.ptr, self._buf.ptr, ndarray.size)

        return ndarray

    fn to_numpy(self) raises -> PythonObject:
        """See `numojo.core.utility.to_numpy`."""
        try:
            var np = Python.import_module("numpy")

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
            elif dtype == DType.index:
                np_dtype = np.int64

            numpyarray = np.empty(self.shape, dtype=np_dtype)
            var pointer_d = numpyarray.__array_interface__["data"][
                0
            ].unsafe_get_as_pointer[dtype]()
            memcpy(pointer_d, self._buf.ptr, self.size)

            return numpyarray^

        except e:
            print("Error in converting to numpy", e)
            return PythonObject()

    # ===-----------------------------------------------------------------------===#
    # Static methods to construct matrix
    # ===-----------------------------------------------------------------------===#

    @staticmethod
    fn full[
        dtype: DType = DType.float64
    ](shape: Tuple[Int, Int], fill_value: Scalar[dtype] = 0) -> Matrix[dtype]:
        """Return a matrix with given shape and filled value.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.full(shape=(10, 10), fill_value=100)
        ```
        """

        var matrix = Matrix[dtype](shape)
        for i in range(shape[0] * shape[1]):
            matrix._buf.ptr.store(i, fill_value)

        return matrix^

    @staticmethod
    fn zeros[
        dtype: DType = DType.float64
    ](shape: Tuple[Int, Int]) -> Matrix[dtype]:
        """Return a matrix with given shape and filled with zeros.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.ones(shape=(10, 10))
        ```
        """

        var M = Matrix[dtype](shape)
        memset_zero(M._buf.ptr, M.size)
        return M^

    @staticmethod
    fn ones[
        dtype: DType = DType.float64
    ](shape: Tuple[Int, Int]) -> Matrix[dtype]:
        """Return a matrix with given shape and filled with ones.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.ones(shape=(10, 10))
        ```
        """

        return Matrix.full[dtype](shape=shape, fill_value=1)

    @staticmethod
    fn identity[dtype: DType = DType.float64](len: Int) -> Matrix[dtype]:
        """Return a matrix with given shape and filled value.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.identity(12)
        ```
        """

        var matrix = Matrix.zeros[dtype]((len, len))
        for i in range(len):
            matrix._buf.ptr.store(i * matrix.strides[0] + i, 1)
        return matrix^

    @staticmethod
    fn rand[
        dtype: DType = DType.float64
    ](shape: Tuple[Int, Int]) -> Matrix[dtype]:
        """Return a matrix with random values uniformed distributed between 0 and 1.

        Example:
        ```mojo
        from numojo import Matrix
        var A = Matrix.rand((12, 12))
        ```

        Parameters:
            dtype: The data type of the NDArray elements.

        Args:
            shape: The shape of the Matrix.
        """
        var result = Matrix[dtype](shape)
        for i in range(result.size):
            result._buf.ptr.store(i, random_float64(0, 1).cast[dtype]())
        return result^

    @staticmethod
    fn fromlist[
        dtype: DType
    ](
        object: List[Scalar[dtype]], shape: Tuple[Int, Int] = (0, 0)
    ) raises -> Matrix[dtype]:
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
            var M = Matrix[dtype](shape=(1, object.size))
            memcpy(M._buf.ptr, object.data, M.size)
            return M^

        if shape[0] * shape[1] != object.size:
            var message = String(
                "The input has {} elements, but the target has the shape {}x{}"
            ).format(object.size, shape[0], shape[1])
            raise Error(message)
        var M = Matrix[dtype](shape=shape)
        memcpy(M._buf.ptr, object.data, M.size)
        return M^

    @staticmethod
    fn fromstring[
        dtype: DType = DType.float64
    ](text: String, shape: Tuple[Int, Int] = (0, 0)) raises -> Matrix[dtype]:
        """Matrix initialization from string representation of an matrix.

        Comma, right brackets, and whitespace are treated as seperators of numbers.
        Digits, underscores, and minus signs are treated as a part of the numbers.

        If now shape is passed, the return matrix will be a row vector.

        Example:
        ```mojo
        from numojo.prelude import *
        from numojo import Matrix
        fn main() raises:
            var A = Matrix.fromstring[f32](
            "1 2 .3 4 5 6.5 7 1_323.12 9 10, 11.12, 12 13 14 15 16", (4, 4))
        ```
        ```console
        [[1.0   2.0     0.30000001192092896     4.0]
        [5.0   6.5     7.0     1323.1199951171875]
        [9.0   10.0    11.119999885559082      12.0]
        [13.0  14.0    15.0    16.0]]
        Size: 4x4  DType: float32
        ```

        Args:
            text: String representation of a matrix.
            shape: Shape of the matrix.
        """

        var data = List[Scalar[dtype]]()
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
                    var number = atof(number_as_str).cast[dtype]()
                    data.append(number)  # Add the number to the data buffer
                    number_as_str = ""  # Clean the number cache
            if (
                (chr(Int(b)) == ",")
                or (chr(Int(b)) == "]")
                or (chr(Int(b)) == " ")
            ):
                if number_as_str != "":
                    var number = atof(number_as_str).cast[dtype]()
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

        var result = Matrix[dtype](shape=shape)
        for i in range(len(data)):
            result._buf.ptr[i] = data[i]
        return result^


# ===-----------------------------------------------------------------------===#
# MatrixIter struct
# ===-----------------------------------------------------------------------===#


@value
struct _MatrixIter[
    is_mutable: Bool, //,
    lifetime: Origin[is_mutable],
    dtype: DType,
    forward: Bool = True,
]:
    """Iterator for Matrix.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying Matrix data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var matrix: Matrix[dtype]
    var length: Int

    fn __init__(
        mut self,
        matrix: Matrix[dtype],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.matrix = matrix

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> Matrix[dtype]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.matrix[current_index]
        else:
            var current_index = self.index
            self.index -= 1
            return self.matrix[current_index]

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
](A: Matrix[dtype], B: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Matrix[dtype] & Matrix[dtype] -> Matrix[dtype]

    For example: `__add__`, `__sub__`, etc.
    """
    alias simd_width = simdwidthof[dtype]()

    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        raise Error(
            String("Shape {}x{} does not match {}x{}.").format(
                A.shape[0], A.shape[1], B.shape[0], B.shape[1]
            )
        )

    var C = Matrix[dtype](shape=A.shape)

    @parameter
    fn vec_func[simd_width: Int](i: Int):
        C._buf.ptr.store(
            i,
            simd_func(
                A._buf.ptr.load[width=simd_width](i),
                B._buf.ptr.load[width=simd_width](i),
            ),
        )

    vectorize[vec_func, simd_width](A.size)

    return C^


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
    alias simd_width = simdwidthof[dtype]()

    var C = Matrix[dtype](shape=A.shape)

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
](A: Matrix[dtype], B: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """
    Matrix[dtype] & Matrix[dtype] -> Matrix[bool]
    """
    alias width = simdwidthof[dtype]()

    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        raise Error(
            String("Shape {}x{} does not match {}x{}.").format(
                A.shape[0], A.shape[1], B.shape[0], B.shape[1]
            )
        )

    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var C = Matrix[DType.bool](shape=A.shape)

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
    var _A = A
    var _B = B

    return C^
