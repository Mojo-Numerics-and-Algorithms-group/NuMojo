"""
Implements signal processing.

It is like `scipy.signal` in Python.
"""

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import Shape
from numojo.core.item import Item
from numojo.routines.creation import fromstring, zeros
from numojo.routines.math.sums import sum


fn convolve2d[
    dtype: DType, //,
](in1: NDArray[dtype], in2: NDArray[dtype]) raises -> NDArray[dtype]:
    """Convolve two 2-dimensional arrays.

    Args:
        in1: Input array 1.
        in2: Input array 2. It should be of a smaller size of in1.

    Currently, the mode is "valid".

    TODO: Add more modes.

    Example:
    ```mojo
    import numojo as nm
    fn main() raises:
        var in1 = nm.routines.creation.fromstring("[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]")
        var in2 = nm.routines.creation.fromstring("[[1, 0], [0, -1]]")
        print(nm.science.signal.convolve2d(in1, in2))
    ```
    """

    var in2_mirrored: NDArray[dtype] = in2.copy()
    var length: Int = in2.size
    for i in range(length):
        in2_mirrored._buf.ptr[i] = in2._buf.ptr[length - i - 1]

    var in1_height: Int = in1.shape[0]
    var in1_width: Int = in1.shape[1]
    var in2_height: Int = in2_mirrored.shape[0]
    var in2_width: Int = in2_mirrored.shape[1]

    var output_height: Int = in1_height - in2_height + 1
    var output_width: Int = in1_width - in2_width + 1

    var output: NDArray[dtype] = zeros[dtype](
        Shape(output_height, output_width)
    )

    for i in range(output_height):
        for j in range(output_width):
            output[Item(i, j)] = sum(
                in1[i : i + in2_height, j : j + in2_width] * in2_mirrored
            )

    return output^
