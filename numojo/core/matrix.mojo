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
        - stride
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
    var stride: Tuple[Int, Int]
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
        self.stride = Tuple(shape[1], 1) if order == "C" else Tuple(1, shape[0])
        self.size = shape[0] * shape[1]
        self.data = UnsafePointer[Scalar[dtype]]().alloc(self.size)
        self.order = order

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.stride = (other.stride[0], other.stride[1])
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
        self.stride = (other.stride[0], other.stride[1])
        self.size = other.size
        self.order = other.order
        self.data = other.data

    @always_inline("nodebug")
    fn __del__(owned self):
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Dunder methods
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, index: Tuple[Int, Int]) raises -> Scalar[dtype]:
        """
        Return the scalar at the coordinates (tuple).

        Args:
            index: The coordinates of the item.

        Returns:
            A scalar matching the dtype of the array.
        """

        # If more than one index is given
        if index.__len__() != 2:
            raise Error("Error: Length of the index does not match the shape.")
        if (index[0] >= self.shape[0]) or (index[1] >= self.shape[1]):
            raise Error("Error: Elements of `index` exceed the array shape")
        return self.data.load(
            index[0] * self.stride[0] + index[1] * self.stride[1]
        )

    fn __setitem__(self, index: Tuple[Int, Int], value: Scalar[dtype]) raises:
        """
        Return the scalar at the coordinates (tuple).

        Args:
            index: The coordinates of the item.
            value: The value to be set.
        """

        # If more than one index is given
        if index.__len__() != 2:
            raise Error("Error: Length of the index does not match the shape.")
        if (index[0] >= self.shape[0]) or (index[1] >= self.shape[1]):
            raise Error("Error: Elements of `index` exceed the array shape")
        self.data.store(
            index[0] * self.stride[0] + index[1] * self.stride[1], value
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
                            print_row(self, i, sep)
                            + newline * number_of_newline
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


# ===----------------------------------------------------------------------===#
# Fucntions for constructing Matrix
# ===----------------------------------------------------------------------===#


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
