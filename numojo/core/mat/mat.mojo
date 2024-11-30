"""
`numojo.core.mat` module provides:

- Matrix type (2D array) with initialization and basic manipulation.
- Auxilliary types, e.g., MatrixIter.
- Functioon to construct matrix from other data objects,
e.g., List, NDArray, String, and numpy array.

Because the number of dimension is known at the compile time,
the Matrix type gains advantages in the running speed compared to
the NDArray type when the users only want to deal with the matrices
manipulation, and it can also be more consistent with numpy.
For example:

- For `__getitem__`, inputting two `Int` returns a scalar,
inputting one `Int` or no `Int` returns a Matrix.
- We do not need auxillary types `NDArrayShape` and `NDArrayStrides`
as the shape and strides information is fixed in length `Tuple[Int,Int]`.

TODO: In future, we can also make use of the trait `ArrayLike` to align
the behavior of `NDArray` type and the `Matrix` type.

"""

from numojo.prelude import *
from numojo.core.ndarray import NDArray
from memory import memcmp, memcpy
from sys import simdwidthof

# ===----------------------------------------------------------------------===#
# Matrix struct
# ===----------------------------------------------------------------------===#


struct Matrix[dtype: DType = DType.float64](Stringable, Formattable):
    """A marix (2d-array).

    The buffer is saved row-majored (C-type).

    Parameters:
        dtype: Type of item in NDArray. Default type is DType.float64.

    The matrix can be uniquely defined by the following features:
        1. The data buffer of all items.
        2. The shape of the matrix.
        3. The strides (row-major or column-major).
        4. The data type of the elements (compile-time known).

    Attributes:
        - _buf
        - shape
        - size
        - strides

    Default constructor: dtype(parameter), shape, object.
    """

    var shape: Tuple[Int, Int]
    """Shape of Matrix."""

    # To be calculated at the initialization.
    var size: Int
    """Size of Matrix."""
    var strides: Tuple[Int, Int]
    """Strides of matrix."""

    # To be filled by constructing functions.
    var _buf: UnsafePointer[Scalar[dtype]]
    """Data buffer of the items in the NDArray."""

    alias width: Int = simdwidthof[dtype]()  #
    """Vector size of the data type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: Tuple[Int, Int],
    ):
        """
        Matrix NDArray initialization.

        Args:
            shape: List of shape.
        """

        self.shape = (shape[0], shape[1])
        self.strides = (shape[1], 1)
        self.size = shape[0] * shape[1]
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(self.size)

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(other.size)
        memcpy(self._buf, other._buf, other.size)

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Self):
        """
        Move other into self.
        """
        self.shape = other.shape^
        self.strides = other.strides^
        self.size = other.size
        self._buf = other._buf

    @always_inline("nodebug")
    fn __del__(owned self):
        self._buf.free()

    # ===-------------------------------------------------------------------===#
    # Dunder methods
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, owned x: Int, owned y: Int) -> Scalar[dtype]:
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

        try:
            if (x >= self.shape[0]) or (y >= self.shape[1]):
                raise Error(
                    "Error: Elements of `index` exceed the array shape."
                )
        except e:
            print(e)
        return self._buf.load(x * self.strides[0] + y)

    fn __getitem__(self, owned x: Int) -> Self:
        """
        Return the corresponding row at the index.

        Args:
            x: The row number.
        """

        if x < 0:
            x = self.shape[0] + x

        try:
            if x >= self.shape[0]:
                raise Error(
                    "Error: Elements of `index` exceed the array shape."
                )
        except e:
            print(e)

        var res = Self(shape=(1, self.shape[1]))
        var ptr = self._buf.offset(x * self.shape[1])
        memcpy(res._buf, ptr, res.size)
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
        var B = mat.Matrix[dtype](shape=(len(range_x), len(range_y)))

        # Fill in the values at the corresponding index
        var c = 0
        for i in range_x:
            for j in range_y:
                B._buf[c] = self._load(i, j)
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
        var B = mat.Matrix[dtype](shape=(len(range_x), 1))

        # Fill in the values at the corresponding index
        var c = 0
        for i in range_x:
            B._buf[c] = self._load(i, y)
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
        var B = mat.Matrix[dtype](shape=(1, len(range_y)))

        # Fill in the values at the corresponding index
        var c = 0
        for j in range_y:
            B._buf[c] = self._load(x, j)
            c += 1

        return B

    fn _load[width: Int = 1](self, x: Int, y: Int) -> SIMD[dtype, width]:
        """
        `__getitem__` with width.
        Unsafe: No boundary check!
        """
        return self._buf.load[width=width](x * self.strides[0] + y)

    fn __setitem__(self, x: Int, y: Int, value: Scalar[dtype]):
        """
        Return the scalar at the index.

        Args:
            x: The row number.
            y: The column number.
            value: The value to be set.
        """

        try:
            if (x >= self.shape[0]) or (y >= self.shape[1]):
                raise Error("Error: Elements of `index` exceed the array shape")
        except e:
            print(e)
        self._buf.store(x * self.strides[0] + y, value)

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
                    "exceed the array shape ({})."
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

        var ptr = self._buf.offset(x * self.shape[1])
        memcpy(ptr, value._buf, value.size)

    fn _store[
        width: Int = 1
    ](inout self, x: Int, y: Int, simd: SIMD[dtype, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.store[width=width](x * self.strides[0] + y, simd)

    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
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
            print("Cannot tranfer matrix to string!", e)
        result += str("]")
        writer.write(
            result
            + "\nSize: "
            + str(self.shape[0])
            + "x"
            + str(self.shape[1])
            + "  DType: "
            + str(self.dtype)
        )

    fn __iter__(self) raises -> _MatrixIter[__lifetime_of(self), dtype]:
        """Iterate over elements of the Matrix, returning copied value.

        Example:
        ```mojo
        from numojo import mat
        var A = mat.rand((4,4))
        for i in A:
            print(i)
        ```

        Returns:
            An iterator of Matrix elements.
        """

        return _MatrixIter[__lifetime_of(self), dtype](
            matrix=self,
            length=self.shape[0],
        )

    fn __reversed__(
        self,
    ) raises -> _MatrixIter[__lifetime_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the Matrix, returning
        copied value.

        Returns:
            A reversed iterator of Matrix elements.
        """

        return _MatrixIter[__lifetime_of(self), dtype, forward=False](
            matrix=self,
            length=self.shape[0],
        )

    # ===-------------------------------------------------------------------===#
    # Arithmetic dunder methods
    # ===-------------------------------------------------------------------===#

    fn __add__(self, other: Self) raises -> Self:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__add__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__add__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _arithmetic_func[dtype, SIMD.__add__](
                self, broadcast_to(other, self.shape)
            )

    fn __add__(self, other: Scalar[dtype]) raises -> Self:
        """Add matrix to scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A + 2)
        ```
        """
        return self + broadcast_to[dtype](other, self.shape)

    fn __radd__(self, other: Scalar[dtype]) raises -> Self:
        """
        Right-add.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(2 + A)
        ```
        """
        return broadcast_to[dtype](other, self.shape) + self

    fn __sub__(self, other: Self) raises -> Self:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__sub__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__sub__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _arithmetic_func[dtype, SIMD.__sub__](
                self, broadcast_to(other, self.shape)
            )

    fn __sub__(self, other: Scalar[dtype]) raises -> Self:
        """Substract matrix by scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A - 2)
        ```
        """
        return self - broadcast_to[dtype](other, self.shape)

    fn __rsub__(self, other: Scalar[dtype]) raises -> Self:
        """
        Right-sub.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(2 - A)
        ```
        """
        return broadcast_to[dtype](other, self.shape) - self

    fn __mul__(self, other: Self) raises -> Self:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__mul__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__mul__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _arithmetic_func[dtype, SIMD.__mul__](
                self, broadcast_to(other, self.shape)
            )

    fn __mul__(self, other: Scalar[dtype]) raises -> Self:
        """Mutiply matrix by scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A * 2)
        ```
        """
        return self * broadcast_to[dtype](other, self.shape)

    fn __rmul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Right-mul.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(2 * A)
        ```
        """
        return broadcast_to[dtype](other, self.shape) * self

    fn __truediv__(self, other: Self) raises -> Self:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__truediv__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _arithmetic_func[dtype, SIMD.__truediv__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _arithmetic_func[dtype, SIMD.__truediv__](
                self, broadcast_to(other, self.shape)
            )

    fn __truediv__(self, other: Scalar[dtype]) raises -> Self:
        """Divide matrix by scalar."""
        return self / broadcast_to[dtype](other, self.shape)

    fn __lt__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__lt__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__lt__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func[dtype, SIMD.__lt__](
                self, broadcast_to(other, self.shape)
            )

    fn __lt__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A < 2)
        ```
        """
        return self < broadcast_to[dtype](other, self.shape)

    fn __le__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__le__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__le__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func[dtype, SIMD.__le__](
                self, broadcast_to(other, self.shape)
            )

    fn __le__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A <= 2)
        ```
        """
        return self <= broadcast_to[dtype](other, self.shape)

    fn __gt__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__gt__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__gt__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func[dtype, SIMD.__gt__](
                self, broadcast_to(other, self.shape)
            )

    fn __gt__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix greater than scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A > 2)
        ```
        """
        return self > broadcast_to[dtype](other, self.shape)

    fn __ge__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__ge__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__ge__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func[dtype, SIMD.__ge__](
                self, broadcast_to(other, self.shape)
            )

    fn __ge__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix greater than and equal to scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A >= 2)
        ```
        """
        return self >= broadcast_to[dtype](other, self.shape)

    fn __eq__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__eq__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__eq__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func[dtype, SIMD.__eq__](
                self, broadcast_to(other, self.shape)
            )

    fn __eq__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A == 2)
        ```
        """
        return self == broadcast_to[dtype](other, self.shape)

    fn __ne__(self, other: Self) raises -> Matrix[DType.bool]:
        if (self.shape[0] == other.shape[0]) and (
            self.shape[1] == other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__ne__](self, other)
        elif (self.shape[0] < other.shape[0]) or (
            self.shape[1] < other.shape[1]
        ):
            return _logic_func[dtype, SIMD.__ne__](
                broadcast_to(self, other.shape), other
            )
        else:
            return _logic_func[dtype, SIMD.__ne__](
                self, broadcast_to(other, self.shape)
            )

    fn __ne__(self, other: Scalar[dtype]) raises -> Matrix[DType.bool]:
        """Matrix less than and equal to scalar.

        ```mojo
        from numojo.mat import ones
        A = ones(shape=(4, 4))
        print(A != 2)
        ```
        """
        return self != broadcast_to[dtype](other, self.shape)

    fn __matmul__(self, other: Self) -> Self:
        return matmul(self, other)

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    fn transpose(self) -> Self:
        return transpose(self)

    fn T(self) -> Self:
        return transpose(self)

    fn inv(self) raises -> Self:
        return inv(self)

    fn to_ndarray(self) raises -> NDArray[dtype]:
        """Create a ndarray from a matrix.

        It makes a copy of the buffer of the matrix.
        """

        var ndarray = NDArray[dtype](
            shape=List[Int](self.shape[0], self.shape[1]), order="C"
        )
        memcpy(ndarray._buf, self._buf, ndarray.size())

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

            numpyarray = np.empty(self.shape, dtype=np_dtype)
            var pointer_d = numpyarray.__array_interface__["data"][
                0
            ].unsafe_get_as_pointer[dtype]()
            memcpy(pointer_d, self._buf, self.size)

            return numpyarray^

        except e:
            print("Error in converting to numpy", e)
            return PythonObject()


# ===-----------------------------------------------------------------------===#
# MatrixIter struct
# ===-----------------------------------------------------------------------===#


@value
struct _MatrixIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
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
        inout self,
        matrix: Matrix[dtype],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.matrix = matrix

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) raises -> Matrix[dtype]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.matrix[current_index]
        else:
            var current_index = self.index
            self.index -= 1
            return self.matrix[current_index]

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index


# ===-----------------------------------------------------------------------===#
# Fucntions for arithmetic
# ===-----------------------------------------------------------------------===#


fn _arithmetic_func[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]:
    alias width = max(simdwidthof[dtype](), 16)
    try:
        if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
            raise Error("The shapes of matrices do not match!")
    except e:
        print(e)

    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var C = Matrix[dtype](shape=A.shape)

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


fn _logic_func[
    dtype: DType,
    simd_func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[DType.bool, simd_width],
](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[DType.bool]:
    alias width = max(simdwidthof[dtype](), 16)
    try:
        if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
            raise Error("The shapes of matrices do not match!")
    except e:
        print(e)

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


fn broadcast_to[
    dtype: DType
](A: Matrix[dtype], shape: Tuple[Int, Int]) raises -> Matrix[dtype]:
    """
    Broadcase the vector to the given shape.

    Example:

    ```console
    > from numojo import mat
    > a = mat.fromstring("1 2 3", shape=(1, 3))
    > print(mat.broadcast_to(a, (3, 3)))
    [[1.0   2.0     3.0]
     [1.0   2.0     3.0]
     [1.0   2.0     3.0]]
    > a = mat.fromstring("1 2 3", shape=(3, 1))
    > print(mat.broadcast_to(a, (3, 3)))
    [[1.0   1.0     1.0]
     [2.0   2.0     2.0]
     [3.0   3.0     3.0]]
    > a = mat.fromstring("1", shape=(1, 1))
    > print(mat.broadcast_to(a, (3, 3)))
    [[1.0   1.0     1.0]
     [1.0   1.0     1.0]
     [1.0   1.0     1.0]]
    > a = mat.fromstring("1 2", shape=(1, 2))
    > print(mat.broadcast_to(a, (1, 2)))
    [[1.0   2.0]]
    > a = mat.fromstring("1 2 3 4", shape=(2, 2))
    > print(mat.broadcast_to(a, (4, 2)))
    Unhandled exception caught during execution: Cannot broadcast shape 2x2 to shape 4x2!
    ```
    """

    var B = Matrix[dtype](shape)
    if (A.shape[0] == shape[0]) and (A.shape[1] == shape[1]):
        B = A
    elif (A.shape[0] == 1) and (A.shape[1] == 1):
        B = full[dtype](shape, A[0, 0])
    elif (A.shape[0] == 1) and (A.shape[1] == shape[1]):
        for i in range(shape[0]):
            memcpy(dest=B._buf.offset(shape[1] * i), src=A._buf, count=shape[1])
    elif (A.shape[1] == 1) and (A.shape[0] == shape[0]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                B._store(i, j, A._buf[i])
    else:
        var message = String(
            "Cannot broadcast shape {}x{} to shape {}x{}!"
        ).format(A.shape[0], A.shape[1], shape[0], shape[1])
        raise Error(message)
    return B^


fn broadcast_to[
    dtype: DType
](A: Scalar[dtype], shape: Tuple[Int, Int]) raises -> Matrix[dtype]:
    """
    Broadcase the scalar to the given shape.
    """

    var B = Matrix[dtype](shape)
    B = full[dtype](shape, A)
    return B^
