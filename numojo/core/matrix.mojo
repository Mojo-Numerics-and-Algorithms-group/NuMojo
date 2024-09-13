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
# Matrix
# ===----------------------------------------------------------------------===#


struct Matrix[dtype: DType = DType.float64]():
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

    var data: DTypePointer[dtype]
    """Data buffer of the items in the NDArray."""
    var shape: Tuple[Int, Int]
    """Shape of Matrix."""
    var size: Int
    """Size of Matrix."""
    var stride: Tuple[Int, Int]
    """Strides of matrix."""
    var order: String
    "C (C-type, row-major) or F (Fortran-type col-major)."

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
        self.data = DTypePointer[dtype].alloc(self.size)
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
        self.data = DTypePointer[dtype].alloc(other.size)
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
