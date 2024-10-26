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
inputting one `Int` returns a vector, and inputting no `Int`
returns a Matrix.
- For row-major and column-major matrices, it is easier to get the
values by the indices, as strides are only two numbers.
- We do not need auxillary types `NDArrayShape` and `NDArrayStrides`
as the shape and strides information is fixed in length `Tuple[Int,Int]`.

TODO: In future, we can also make use of the trait `ArrayLike` to align
the behavior of `NDArray` type and the `Matrix` type.

"""

from numojo.prelude import *
from .ndarray import NDArray
from memory import memcmp
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

    fn __getitem__(self, x: Int, y: Int) -> Scalar[dtype]:
        """
        Return the scalar at the index.

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
        return self._buf.load(x * self.strides[0] + y)

    fn __getitem__(self, x: Int) -> Self:
        """
        Return the corresponding row at the index.

        Args:
            x: The row number.
        """

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

    fn __add__(self, other: Self) -> Self:
        return _arithmetic_func[dtype, SIMD.__add__](self, other)

    fn __sub__(self, other: Self) -> Self:
        return _arithmetic_func[dtype, SIMD.__sub__](self, other)

    fn __mul__(self, other: Self) -> Self:
        return _arithmetic_func[dtype, SIMD.__mul__](self, other)

    fn __lt__(self, other: Self) -> Matrix[DType.bool]:
        return _logic_func[dtype, SIMD.__lt__](self, other)

    fn __le__(self, other: Self) -> Matrix[DType.bool]:
        return _logic_func[dtype, SIMD.__le__](self, other)

    fn __gt__(self, other: Self) -> Matrix[DType.bool]:
        return _logic_func[dtype, SIMD.__gt__](self, other)

    fn __ge__(self, other: Self) -> Matrix[DType.bool]:
        return _logic_func[dtype, SIMD.__ge__](self, other)

    fn __eq__(self, other: Self) -> Matrix[DType.bool]:
        return _logic_func[dtype, SIMD.__eq__](self, other)

    fn __ne__(self, other: Self) -> Matrix[DType.bool]:
        return _logic_func[dtype, SIMD.__ne__](self, other)

    fn __matmul__(self, other: Self) -> Self:
        return matmul(self, other)

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    fn transpose(self) -> Self:
        return transpose(self)

    fn T(self) -> Self:
        return transpose(self)

    fn inv(self) -> Self:
        return inv(self)

    fn to_ndarray(self) raises -> NDArray[dtype]:
        """Create a ndarray from a matrix.

        It makes a copy of the buffer of the matrix.
        """

        var ndarray = NDArray[dtype](
            shape=List[Int](self.shape[0], self.shape[1]), order="C"
        )
        memcpy(ndarray.data, self._buf, ndarray.size())

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
# Fucntions for constructing Matrix
# ===-----------------------------------------------------------------------===#


fn full[
    dtype: DType = DType.float64
](shape: Tuple[Int, Int], fill_value: Scalar[dtype] = 0) -> Matrix[dtype]:
    """Return a matrix with given shape and filled value.
    
    Example:
    ```mojo
    from numojo import mat
    var A = mat.full(shape=(10, 10), fill_value=100)
    ```
    """

    var matrix = Matrix[dtype](shape)
    for i in range(shape[0] * shape[1]):
        matrix._buf.store(i, fill_value)

    return matrix


fn zeros[dtype: DType = DType.float64](shape: Tuple[Int, Int]) -> Matrix[dtype]:
    """Return a matrix with given shape and filled with zeros.
    
    Example:
    ```mojo
    from numojo import mat
    var A = mat.zeros(shape=(10, 10))
    ```
    """

    return full[dtype](shape=shape, fill_value=0)


fn ones[dtype: DType = DType.float64](shape: Tuple[Int, Int]) -> Matrix[dtype]:
    """Return a matrix with given shape and filled with ones.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.ones(shape=(10, 10))
    ```
    """

    return full[dtype](shape=shape, fill_value=1)


fn identity[dtype: DType = DType.float64](len: Int) -> Matrix[dtype]:
    """Return a matrix with given shape and filled value.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.identity(12)
    ```
    """

    var matrix = zeros[dtype]((len, len))
    for i in range(len):
        matrix._buf.store(i * matrix.strides[0] + i, 1)
    return matrix


fn rand[dtype: DType = DType.float64](shape: Tuple[Int, Int]) -> Matrix[dtype]:
    """Return a matrix with random values uniformed distributed between 0 and 1.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand((12, 12))
    ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the Matrix.
    """
    var result = Matrix[dtype](shape)
    for i in range(result.size):
        result._buf.store(i, random.random_float64(0, 1).cast[dtype]())
    return result


# ===-----------------------------------------------------------------------===#
# Fucntions for constructing Matrix from an object
# ===-----------------------------------------------------------------------===#


fn matrix[dtype: DType](object: NDArray[dtype]) raises -> Matrix[dtype]:
    """Create a matrix from a row-majored ndarray. It must be 2-dimensional.

    It makes a copy of the buffer of the ndarray.

    It is useful when we want to solve a linear system. In this case, we treat
    ndarray as a matrix. This simplify calculation and avoid too much check.
    """

    try:
        if object.ndim != 2:
            raise Error("The original array is not 2-dimensional!")
    except e:
        print(e)

    var matrix = Matrix[dtype](shape=(object.ndshape[0], object.ndshape[1]))
    memcpy(matrix._buf, object.data, matrix.size)

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

    var matrix = Matrix[dtype](shape=(array.ndshape[0], array.ndshape[1]))
    matrix._buf = array.data

    return matrix


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


# ===-----------------------------------------------------------------------===#
# Fucntions for linear algebra
# ===-----------------------------------------------------------------------===#


fn lu_decomposition[
    dtype: DType
](A: Matrix[dtype]) -> Tuple[Matrix[dtype], Matrix[dtype]]:
    # Check whether the matrix is square
    try:
        if A.shape[0] != A.shape[1]:
            raise Error("The matrix is not square!")
    except e:
        print(e)
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


fn solve[dtype: DType](A: Matrix[dtype], Y: Matrix[dtype]) -> Matrix[dtype]:
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


fn inv[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Inverse of matrix."""

    # Check whether the matrix is square
    try:
        if A.shape[0] != A.shape[1]:
            raise Error("The matrix is not square!")
    except e:
        print(e)

    var I = identity[dtype](A.shape[0])
    var B = solve(A, I)

    return B^


fn transpose[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """Transpose."""

    var B = Matrix[dtype](Tuple(A.shape[1], A.shape[0]))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] = A[j, i]
    return B


fn matmul[dtype: DType](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]:
    """Matrix multiplication.

    See `numojo.math.linalg.matmul.matmul_parallelized()`.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.rand(shape=(1000, 1000))
    var B = mat.rand(shape=(1000, 1000))
    var C = mat.matmul(A, B)
    ```
    """

    alias width = max(simdwidthof[dtype](), 16)

    try:
        if A.shape[1] != B.shape[0]:
            raise Error("The shapes of matrices do not match!")
    except e:
        print(e)
        print("`matmul` error.")

    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var t2 = B.shape[1]
    var C: Matrix[dtype] = zeros[dtype](shape=(t0, t2))

    @parameter
    fn calculate_CC(m: Int):
        for k in range(t1):

            @parameter
            fn dot[simd_width: Int](n: Int):
                C._store[simd_width](
                    m,
                    n,
                    C._load[simd_width](m, n)
                    + A._load(m, k) * B._load[simd_width](k, n),
                )

            vectorize[dot, width](t2)

    parallelize[calculate_CC](t0, t0)

    var _t0 = t0
    var _t1 = t1
    var _t2 = t2
    var _A = A
    var _B = B

    return C^


# ===-----------------------------------------------------------------------===#
# Fucntions for statistics
# ===-----------------------------------------------------------------------===#


fn lstsq[dtype: DType](X: Matrix[dtype], y: Matrix[dtype]) -> Matrix[dtype]:
    """Caclulate the OLS estimates.

    Example:
    ```mojo
    from numojo import mat
    X = mat.rand((1000000, 5))
    y = mat.rand((1000000, 1))
    print(mat.lstsq(X, y))
    ```
    ```console
    [[0.18731374756029967]
     [0.18821352688798607]
     [0.18717162200411439]
     [0.1867570378683612]
     [0.18828715376701158]]
    Size: 5x1  DType: float64
    ```
    """

    try:
        if X.shape[0] != y.shape[0]:
            var message = "The row number of `X` {X.shape[0]} does not match the row number of `y` {y.shape[0]}"
            raise Error(message)
    except e:
        print(e)
    var X_prime = X.T()
    var b = (X_prime @ X).inv() @ X_prime @ y
    return b^
