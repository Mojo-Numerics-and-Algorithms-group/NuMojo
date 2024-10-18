"""
`numojo.core.matrix` module provides:

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
inputting one `Int` returns a vector, and inputting no `Int`
returns a Matrix.
- For row-major and column-major matrices, it is easier to get the
values by the indices, as strides are only two numbers.
- We do not need auxillary types `NDArrayShape` and `NDArrayStrides`
as the shape and strides information is fixed in length `Tuple[Int,Int]`.

TODO: In future, we can also make use of the trait `ArrayLike` to align
the behavior of `NDArray` type and the `Matrix` type.

"""

from .ndarray import NDArray
from memory import memcmp
from sys import simdwidthof

# ===----------------------------------------------------------------------===#
# Matrix struct
# ===----------------------------------------------------------------------===#


struct Matrix[dtype: DType = DType.float64](Stringable, Formattable):
    """A marix (2d-array).

    Parameters:
        dtype: Type of item in NDArray. Default type is DType.float64.

    The matrix can be uniquely defined by the following features:
        1. The data buffer of all items.
        2. The shape of the matrix.
        3. The strides (row-major or column-major).
        4. The data type of the elements (compile-time known).

    Attributes:
        - _buffer
        - shape
        - size
        - strides
        - order

    Default constructor: dtype(parameter), shape, order.
    """

    var shape: Tuple[Int, Int]
    """Shape of Matrix."""
    var order: String
    "C (C-type, row-major) or F (Fortran-type col-major)."

    # To be calculated at the initialization.
    var size: Int
    """Size of Matrix."""
    var strides: Tuple[Int, Int]
    """Strides of matrix."""

    # To be filled by constructing functions.
    var _buffer: UnsafePointer[Scalar[dtype]]
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
        order: String = "C",
    ):
        """
        Matrix NDArray initialization.

        Args:
            shape: List of shape.
            order: Memory order C or F.
        """

        self.shape = (shape[0], shape[1])
        self.strides = Tuple(shape[1], 1) if order == "C" else Tuple(
            1, shape[0]
        )
        self.size = shape[0] * shape[1]
        self._buffer = UnsafePointer[Scalar[dtype]]().alloc(self.size)
        self.order = order

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self.order = other.order
        self._buffer = UnsafePointer[Scalar[dtype]]().alloc(other.size)
        memcpy(self._buffer, other._buffer, other.size)

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Self):
        """
        Move other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self.order = other.order
        self._buffer = other._buffer

    @always_inline("nodebug")
    fn __del__(owned self):
        self._buffer.free()

    # ===-------------------------------------------------------------------===#
    # Dunder methods
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, x: Int, y: Int) -> Scalar[dtype]:
        """
        Return the scalar at the coordinates.

        Args:
            x: The row number.
            y: The column number.

        Returns:
            A scalar matching the dtype of the array.
        """

        try:
            if (x >= self.shape[0]) or (y >= self.shape[1]):
                raise Error(
                    "Error: Elements of `index` exceed the array shape."
                )
        except e:
            print(e)
        return self._buffer.load(x * self.strides[0] + y * self.strides[1])

    fn load[width: Int](self, x: Int, y: Int) -> SIMD[dtype, width]:
        """
        `__getitem__` with width.
        """
        try:
            if (x >= self.shape[0]) or (y >= self.shape[1]):
                raise Error(
                    "Error: Elements of `index` exceed the array shape."
                )
        except e:
            print(e)
        return self._buffer.load[width=width](
            x * self.strides[0] + y * self.strides[1]
        )

    fn __setitem__(self, x: Int, y: Int, value: Scalar[dtype]):
        """
        Return the scalar at the coordinates.

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
        self._buffer.store(x * self.strides[0] + y * self.strides[1], value)

    fn store[width: Int](inout self, x: Int, y: Int, simd: SIMD[dtype, width]):
        """
        `__setitem__` with width.
        """
        self._buffer.store[width=width](
            x * self.strides[0] + y * self.strides[1], simd
        )

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

    fn __matmul__(self, other: Self) -> Self:
        return matmul(self, other)

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    fn to_ndarray(self) raises -> NDArray[dtype]:
        """Create a ndarray from a matrix.

        It makes a copy of the buffer of the matrix.
        """

        var ndarray = NDArray[dtype](
            shape=List[Int](self.shape[0], self.shape[1]), order=self.order
        )
        memcpy(ndarray.data, self._buffer, ndarray.size())

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
            memcpy(pointer_d, self._buffer, self.size)

            return numpyarray^

        except e:
            print("Error in converting to numpy", e)
            return PythonObject()


# ===-----------------------------------------------------------------------===#
# Fucntions for constructing Matrix
# ===-----------------------------------------------------------------------===#


fn full[
    dtype: DType = DType.float64
](
    shape: Tuple[Int, Int], fill_value: Scalar[dtype] = 0, order: String = "C"
) -> Matrix[dtype]:
    """Return a matrix with given shape and filled value."""

    var matrix = Matrix[dtype](shape, order)
    for i in range(shape[0] * shape[1]):
        matrix._buffer.store(i, fill_value)

    return matrix


fn zeros[
    dtype: DType = DType.float64
](shape: Tuple[Int, Int], order: String = "C") -> Matrix[dtype]:
    """Return a matrix with given shape and filled with zeros."""

    return full[dtype](shape=shape, fill_value=0, order=order)


fn identity[
    dtype: DType = DType.float64
](len: Int, order: String = "C") -> Matrix[dtype]:
    """Return a matrix with given shape and filled value."""

    var matrix = Matrix[dtype]((len, len), order)
    for i in range(len):
        matrix._buffer.store(i * matrix.strides[0] + i * matrix.strides[1], 1)
    return matrix


fn rand[
    dtype: DType = DType.float64
](shape: Tuple[Int, Int], order: String = "C") -> Matrix[dtype]:
    """Return a matrix with random values uniformed distributed between 0 and 1.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the Matrix.
        order: The order of the Matrix.
    """
    var result = Matrix[dtype](shape, order)
    for i in range(result.size):
        result._buffer.store(i, random.random_float64(0, 1).cast[dtype]())
    return result


# ===-----------------------------------------------------------------------===#
# Fucntions for constructing Matrix from an object
# ===-----------------------------------------------------------------------===#


fn matrix[dtype: DType](object: NDArray[dtype]) raises -> Matrix[dtype]:
    """Create a matrix from a ndarray. It must be 2-dimensional.

    It makes a copy of the buffer of the ndarray.

    It is useful when we want to solve a linear system. In this case, we treat
    ndarray as a matrix. This simplify calculation and avoid too much check.
    """

    if object.ndim != 2:
        raise Error("The original array is not 2-dimensional!")

    var matrix = Matrix[dtype](
        shape=(object.ndshape[0], object.ndshape[1]), order=object.order
    )
    memcpy(matrix._buffer, object.data, matrix.size)

    return matrix


fn matrix[dtype: DType](owned object: Matrix[dtype]) raises -> Matrix[dtype]:
    """Create a matrix from a matrix."""

    return object^


fn _from_2darray[dtype: DType](array: NDArray[dtype]) raises -> Matrix[dtype]:
    """Create a matrix from an 2-darray.

    [Unsafe] It simply uses the buffer of an ndarray.

    It is useful when we want to solve a linear system. In this case, we treat
    ndarray as a matrix. This simplify calculation and avoid too much check.
    """

    var matrix = Matrix[dtype](
        shape=(array.ndshape[0], array.ndshape[1]), order=array.order
    )
    matrix._buffer = array.data

    return matrix


# ===-----------------------------------------------------------------------===#
# Fucntions for linear algebra
# ===-----------------------------------------------------------------------===#


fn lu_decomposition[
    dtype: DType
](A: Matrix[dtype]) raises -> Tuple[Matrix[dtype], Matrix[dtype]]:
    # Check whether the matrix is square
    if A.shape[0] != A.shape[1]:
        raise ("The matrix is not square!")
    var n = A.shape[0]

    # Initiate upper and lower triangular matrices
    var U = full[dtype](shape=(n, n))
    var L = full[dtype](shape=(n, n))

    # Fill in L and U
    for i in range(0, n):
        for j in range(i, n):
            # Fill in L
            if i == j:
                L[i, i] = 1
            else:
                var sum_of_products_for_L: Scalar[dtype] = 0
                for k in range(0, i):
                    sum_of_products_for_L += L[j, k] * U[k, i]
                L[j, i] = (A[j, i] - sum_of_products_for_L) / U[i, i]

            # Fill in U
            var sum_of_products_for_U: Scalar[dtype] = 0
            for k in range(0, i):
                sum_of_products_for_U += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum_of_products_for_U

    return L, U


fn solve[
    dtype: DType
](A: Matrix[dtype], Y: Matrix[dtype]) raises -> Matrix[dtype]:
    var U: Matrix[dtype]
    var L: Matrix[dtype]
    L, U = lu_decomposition[dtype](A)

    var m = A.shape[0]
    var n = Y.shape[1]

    var Z = full[dtype]((m, n))
    var X = full[dtype]((m, n))

    @parameter
    fn calculate_X(col: Int) -> None:
        # Solve `LZ = Y` for `Z` for each col
        for i in range(m):  # row of L
            var _temp = Y[i, col]
            for j in range(i):  # col of L
                _temp = _temp - L[i, j] * Z[j, col]
            _temp = _temp / L[i, i]
            Z[i, col] = _temp

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z[i, col]
            for j in range(i + 1, m):
                _temp2 = _temp2 - U[i, j] * X[j, col]
            _temp2 = _temp2 / U[i, i]
            X[i, col] = _temp2

    parallelize[calculate_X](n, n)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    var _L = L^
    var _U = U^
    var _Z = Z^
    var _m = m
    var _n = n

    return X^


fn matmul[dtype: DType](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]:
    """Matrix multiplication.

    See `numojo.math.linalg.matmul.matmul_parallelized()`.
    """

    alias width = simdwidthof[dtype]()

    try:
        if A.shape[1] != B.shape[0]:
            raise Error("The shapes of matrices do not match!")
    except e:
        print(e)

    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var t2 = B.shape[1]
    var C: Matrix[dtype] = zeros[dtype](shape=(t0, t2))

    @parameter
    fn calculate_A_rows(m: Int):
        # for m in range(t0):
        for k in range(t1):
            if B.order == "C":

                @parameter
                fn dot[simd_width: Int](n: Int):
                    C.store[simd_width](
                        m,
                        n,
                        C.load[simd_width](m, n)
                        + A[m, k] * B.load[simd_width](k, n),
                    )

                vectorize[dot, width](t2)
            else:
                for n in range(t2):
                    C[m, n] = C[m, n] + A[m, k] * B[k, n]

    parallelize[calculate_A_rows](t0, t0)

    var _t0 = t0
    var _t1 = t1
    var _t2 = t2
    var _A = A
    var _B = B

    return C^
