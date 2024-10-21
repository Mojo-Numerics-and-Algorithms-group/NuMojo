"""
Implements more complex numerical functions for arrays.
"""

from numojo.core.array_creation_routines import fromstring, zeros
from numojo.math.statistics.stats import sumall
from numojo.prelude import *


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
        var in1 = nm.fromstring("[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]")
        var in2 = nm.fromstring("[[1, 0], [0, -1]]")
        print(nm.signal.convolve2d(in1, in2))
    ```
    """

    var in2_mirrored = in2
    var length = in2.size()
    for i in range(length):
        in2_mirrored.data[i] = in2.data[length - i - 1]

    var in1_height = in1.shape()[0]
    var in1_width = in1.shape()[1]
    var in2_height = in2_mirrored.shape()[0]
    var in2_width = in2_mirrored.shape()[1]

    var output_height = in1_height - in2_height + 1
    var output_width = in1_width - in2_width + 1

    var output = zeros[dtype](Shape(output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[Idx(i, j)] = sumall(
                in1[i : i + in2_height, j : j + in2_width] * in2_mirrored
            )

    return output
