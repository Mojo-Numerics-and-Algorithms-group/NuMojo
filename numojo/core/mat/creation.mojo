"""
`numojo.core.mat.creation` module provides functions for creating matrix.

"""

from numojo.core.ndarray import NDArray
from memory.memory import memset_zero

# ===-----------------------------------------------------------------------===#
# Constructing Matrix
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

    return matrix^


fn zeros[dtype: DType = DType.float64](shape: Tuple[Int, Int]) -> Matrix[dtype]:
    """Return a matrix with given shape and filled with zeros.

    Example:
    ```mojo
    from numojo import mat
    var A = mat.zeros(shape=(10, 10))
    ```
    """

    var M = Matrix[dtype](shape)
    memset_zero(M._buf, M.size)
    return M^


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
    return matrix^


# ===-----------------------------------------------------------------------===#
# Constructing random Matrix
# ===-----------------------------------------------------------------------===#


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
    return result^


# ===-----------------------------------------------------------------------===#
# Constructing Matrix from other type
# ===-----------------------------------------------------------------------===#


fn fromlist[
    dtype: DType
](
    object: List[Scalar[dtype]], shape: Tuple[Int, Int] = (0, 0)
) raises -> Matrix[dtype]:
    """Create a matrix from a 1-dimensional list into given shape.

    If no shape is passed, the return matrix will be a row vector.

    Example:
    ```mojo
    from numojo import mat
    fn main() raises:
        print(mat.fromlist(List[Float64](1, 2, 3, 4, 5), (5, 1)))
    ```
    """

    if (shape[0] == 0) and (shape[1] == 0):
        var M = Matrix[dtype](shape=(1, object.size))
        memcpy(M._buf, object.data, M.size)
        return M^

    if shape[0] * shape[1] != object.size:
        var message = String(
            "The input has {} elements, but the target has the shape {}x{}"
        ).format(object.size, shape[0], shape[1])
        raise Error(message)
    var M = Matrix[dtype](shape=shape)
    memcpy(M._buf, object.data, M.size)
    return M^


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
    from numojo import mat
    fn main() raises:
        var A = mat.fromstring[f32](
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

    if (shape[0] == 0) and (shape[1] == 0):
        return fromlist(data)

    if size != len(data):
        var message = String(
            "The number of items in the string is {}, which does not match the"
            " given shape {}x{}."
        ).format(len(data), shape[0], shape[1])
        raise Error(message)

    var result = Matrix[dtype](shape=shape)
    for i in range(len(data)):
        result._buf[i] = data[i]
    return result^
