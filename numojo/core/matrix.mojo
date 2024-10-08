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

# ===----------------------------------------------------------------------===#
# Matrix struct
# ===----------------------------------------------------------------------===#


struct Matrix[dtype: DType = DType.float64](Stringable, Formattable):
    """A marix (2d array).

    Parameters:
        dtype: Type of item in NDArray. Default type is DType.float64.

    The matrix can be uniquely defined by the following features:
        1. The data buffer of all items.
        2. The shape of the matrix.
        3. The strides (row-major or column-major).
        4. The data type of the elements (compile-time known).

    Attributes:
        - data
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
    var data: UnsafePointer[Scalar[dtype]]
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
        self.data = UnsafePointer[Scalar[dtype]]().alloc(self.size)
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
        self.data = UnsafePointer[Scalar[dtype]]().alloc(other.size)
        memcpy(self.data, other.data, other.size)

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Self):
        """
        Move other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self.order = other.order
        self.data = other.data

    @always_inline("nodebug")
    fn __del__(owned self):
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Dunder methods
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, index: Tuple[Int, Int]) -> Scalar[dtype]:
        """
        Return the scalar at the coordinates (tuple).

        Args:
            index: The coordinates of the item.

        Returns:
            A scalar matching the dtype of the array.
        """

        # If more than one index is given
        try:
            if index.__len__() != 2:
                raise Error("Error: Length of the index does not match the shape.")
        except e:
            print(e)
        try:
            if (index[0] >= self.shape[0]) or (index[1] >= self.shape[1]):
                raise Error("Error: Elements of `index` exceed the array shape.")
        except e:
            print(e)
        return self.data.load(
            index[0] * self.strides[0] + index[1] * self.strides[1]
        )

    fn _getitem(self, index: Tuple[Int, Int]) -> Scalar[dtype]:
        """
        Return the scalar at the coordinates (tuple).

        [Unsafe] It does not raise error when boundary check fails!

        Args:
            index: The coordinates of the item.

        Returns:
            A scalar matching the dtype of the array.
        """

        # If more than one index is given
        if index.__len__() != 2:
            print("Error: Length of the index does not match the shape.")
        if (index[0] >= self.shape[0]) or (index[1] >= self.shape[1]):
            print("Error: Elements of `index` exceed the array shape.")
        return self.data.load(
            index[0] * self.strides[0] + index[1] * self.strides[1]
        )

    fn __setitem__(self, index: Tuple[Int, Int], value: Scalar[dtype]):
        """
        Return the scalar at the coordinates (tuple).

        Args:
            index: The coordinates of the item.
            value: The value to be set.
        """

        # If more than one index is given
        try:
            if index.__len__() != 2:
                raise Error("Error: Length of the index does not match the shape.")
        except e:
            print(e)
        try:
            if (index[0] >= self.shape[0]) or (index[1] >= self.shape[1]):
                raise Error("Error: Elements of `index` exceed the array shape")
        except e:
            print(e)
        self.data.store(
            index[0] * self.strides[0] + index[1] * self.strides[1], value
        )

    fn _setitem(self, index: Tuple[Int, Int], value: Scalar[dtype]):
        """
        Return the scalar at the coordinates (tuple).

        [Unsafe] It does not raise error when boundary check fails!

        Args:
            index: The coordinates of the item.
            value: The value to be set.
        """

        # If more than one index is given
        if index.__len__() != 2:
            print("Error: Length of the index does not match the shape.")
        if (index[0] >= self.shape[0]) or (index[1] >= self.shape[1]):
            print("Error: Elements of `index` exceed the array shape")
        self.data.store(
            index[0] * self.strides[0] + index[1] * self.strides[1], value
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
                    result += str(self[(i, j)]) + sep * number_of_sep
            else:
                for j in range(3):
                    result += str(self[(i, j)]) + sep
                result += str("...") + sep
                for j in range(self.shape[1] - 3, self.shape[1]):
                    if j == self.shape[1] - 1:
                        number_of_sep = 0
                    result += str(self[(i, j)]) + sep * number_of_sep
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
        memcpy(ndarray.data, self.data, ndarray.size())

        return ndarray

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
        matrix.data.store(i, fill_value)

    return matrix


fn identity[
    dtype: DType = DType.float64
](len: Int, order: String = "C") -> Matrix[dtype]:
    """Return a matrix with given shape and filled value."""

    var matrix = Matrix[dtype]((len, len), order)
    for i in range(len):
        matrix.data.store(i * matrix.strides[0] + i * matrix.strides[1], 1)
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
        result.data.store(i, random.random_float64(0, 1).cast[dtype]())
    return result


fn from_ndarray[dtype: DType](array: NDArray[dtype]) raises -> Matrix[dtype]:
    """Create a matrix from a ndarray. It must be 2-dimensional.

    It makes a copy of the buffer of the ndarray.

    It is useful when we want to solve a linear system. In this case, we treat
    ndarray as a matrix. This simplify calculation and avoid too much check.
    """

    if array.ndim != 2:
        raise Error("The original array is not 2-dimensional!")

    var matrix = Matrix[dtype](
        shape=(array.ndshape[0], array.ndshape[1]), order=array.order
    )
    memcpy(matrix.data, array.data, matrix.size)

    return matrix


fn _from_2darray[dtype: DType](array: NDArray[dtype]) raises -> Matrix[dtype]:
    """Create a matrix from an 2-darray.

    [Unsafe] It simply uses the buffer of an ndarray.

    It is useful when we want to solve a linear system. In this case, we treat
    ndarray as a matrix. This simplify calculation and avoid too much check.
    """

    var matrix = Matrix[dtype](
        shape=(array.ndshape[0], array.ndshape[1]), order=array.order
    )
    matrix.data = array.data

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
    var U = full[dtype](shape=(n,n))
    var L = full[dtype](shape=(n,n))

    # Fill in L and U
    for i in range(0, n):
        for j in range(i, n):
            # Fill in L
            if i == j:
                L[(i, i)] = 1
            else:
                var sum_of_products_for_L: Scalar[dtype] = 0
                for k in range(0, i):
                    sum_of_products_for_L += L[(j, k)] * U[(k, i)]
                L[(j, i)] = (A[(j, i)] - sum_of_products_for_L) / U[(i, i)]

            # Fill in U
            var sum_of_products_for_U: Scalar[dtype] = 0
            for k in range(0, i):
                sum_of_products_for_U += L[(i, k)] * U[(k ,j)]
            U[(i, j)] = A[(i,j)] - sum_of_products_for_U

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
            var _temp = Y[(i, col)]
            for j in range(i):  # col of L
                _temp = _temp - L[(i, j)] * Z[(j, col)]
            _temp = _temp / L[(i ,i)]
            Z[(i, col)] = _temp

        # Solve `UZ = Z` for `X` for each col
        for i in range(m - 1, -1, -1):
            var _temp2 = Z[(i, col)]
            for j in range(i + 1, m):
                _temp2 = _temp2 - U[(i,j)] * X[(j,col)]
            _temp2 = _temp2 / U[(i,i)]
            X[(i, col)] = _temp2

    parallelize[calculate_X](n, n)

    # Force extending the lifetime of the matrices because they are destroyed before `parallelize`
    # This is disadvantage of Mojo's ASAP policy
    var _L = L^
    var _U = U^
    var _Z = Z^
    var _m = m
    var _n = n

    return X^
