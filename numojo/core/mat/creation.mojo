from .mat import *

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
    memcpy(matrix._buf, object._buf, matrix.size)

    return matrix


fn matrix[dtype: DType](owned object: Matrix[dtype]) raises -> Matrix[dtype]:
    """Create a matrix from a matrix."""

    return object^


fn matrix[
    dtype: DType
](object: Matrix[dtype], shape: Tuple[Int, Int]) raises -> Matrix[dtype]:
    """Create a matrix from a matrix and into certain shape."""

    if shape[0] * shape[1] != object.size:
        var message = String(
            "The input has {} elements, but the target has the shape {}x{}"
        ).format(object.size, shape[0], shape[1])
        raise Error(message)
    var B = Matrix[dtype](shape=shape)
    memcpy(B._buf, object._buf, B.size)
    return B^


fn fromstring[
    dtype: DType = DType.float64
](text: String, shape: Tuple[Int, Int]) raises -> Matrix[dtype]:
    """Matrix initialization from string representation of an matrix.

    Comma, right brackets, and whitespace are treated as seperators of numbers.
    Digits, underscores, and minus signs are treated as a part of the numbers.

    Example:
    ```mojo
    from numojo.prelude import *
    from numojo import mat
    fn main() raises:
        var A = mat.fromstring[f32](
        "1 2 .3 4 5 6.5 7 1_323.12 9 10, 11.12, 12 13 14 15 16", (4, 4)
    )
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
    var result = Matrix[dtype](shape=shape)
    var size = shape[0] * shape[1]

    for i in range(len(bytes)):
        var b = bytes[i]
        if isdigit(b) or (chr(int(b)) == ".") or (chr(int(b)) == "-"):
            number_as_str = number_as_str + chr(int(b))
            if i == len(bytes) - 1:  # Last byte
                var number = atof(number_as_str).cast[dtype]()
                data.append(number)  # Add the number to the data buffer
                number_as_str = ""  # Clean the number cache
        if (chr(int(b)) == ",") or (chr(int(b)) == "]") or (chr(int(b)) == " "):
            if number_as_str != "":
                var number = atof(number_as_str).cast[dtype]()
                data.append(number)  # Add the number to the data buffer
                number_as_str = ""  # Clean the number cache

    if size != len(data):
        var message = String(
            "The number of items in the string is {}, which does not match the"
            " given shape {}x{}."
        ).format(len(data), shape[0], shape[1])
        raise Error(message)

    result = Matrix[dtype](shape=shape)
    for i in range(len(data)):
        result._buf[i] = data[i]
    return result^
