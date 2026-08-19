# ===----------------------------------------------------------------------=== #
# NuMojo: Summation routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Summation (numojo.routines.math.sums).
======================================
Sum reductions and cumulative sums for arrays.

Computes sum reductions along axes and cumulative sums for NDArrays, with
both flattened and axis-aware variants.

Exports
-------
- `sum`: Sum of all elements or along an axis.
- `cumsum`: Cumulative sum along an axis or flattened.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm import vectorize
from std.sys import simd_width_of

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.indexing import TraverseMethods
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import zeros


def sum[dtype: DType](A: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Returns sum of all items in the array.

    Example:
    ```console
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
     [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
     [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32
    > print(nm.sum(A))
    3.5140917301177979
    ```

    Args:
        A: NDArray.

    Returns:
        Scalar.
    """

    if not A.is_c_contiguous():
        return sum(A.contiguous())
    comptime width: Int = simd_width_of[dtype]()
    var result: Scalar[dtype] = Scalar[dtype](0)

    def cal_vec[width: Int](i: Int) {mut result, A}:
        result += A.unsafe_load[width=width](i).reduce_add()

    vectorize[width](A.size, cal_vec)
    return result


def sum[dtype: DType](A: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Returns sums of array elements over a given axis.

    Example:
    ```mojo
    import numojo as nm
    var A = nm.random.randn(100, 100)
    print(nm.sum(A, axis=0))
    ```

    Raises:
        Error: If the axis is out of bound.
        Error: If the number of dimensions is 1.

    Args:
        A: NDArray.
        axis: The axis along which the sum is performed.

    Returns:
        An NDArray.
    """

    var normalized_axis: Int = axis
    if normalized_axis < 0:
        normalized_axis += A.ndim

    if (normalized_axis < 0) or (normalized_axis >= A.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=(
                    "Axis out of range: got {}, expected 0 <= axis < {}."
                    .format(axis, A.ndim)
                ),
                location=String("routines.math.sums.sum(A, axis)"),
            )
        )
    if A.ndim == 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Cannot use axis with 1D array. Call `sum(A)` without axis,"
                    " or reshape A to 2D or higher."
                ),
                location=String("routines.math.sums.sum(A, axis)"),
            )
        )

    var result_shape: List[Int] = List[Int]()
    var size_of_axis: Int = A.shape[normalized_axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(A.ndim):
        if i != normalized_axis:
            result_shape.append(A.shape[i])
            slices.append(Slice(0, A.shape[i]))
        else:
            slices.append(Slice(0, 0))  # Temp value
    var result: NDArray[dtype] = zeros[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[normalized_axis] = Slice(i, i + 1)
        var arr_slice: NDArray[dtype] = A._getitem_list_slices(slices.copy())
        result += arr_slice

    return result^


def cumsum[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Returns cumsum of all items of an array.
    The array is flattened before cumsum.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.

    Returns:
        Cumsum of all items of an array.
    """

    if A.ndim == 1:
        var B = A.contiguous()
        for i in range(A.size - 1):
            B.unsafe_set(i + 1, B.unsafe_get(i + 1) + B.unsafe_get(i))
        return B^

    else:
        return cumsum(A.flatten(), axis=-1)


# Why do we do in inplace operation here?
def cumsum[
    dtype: DType
](A: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
    """
    Returns cumsum of array by axis.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.
        axis: Axis.

    Returns:
        Cumsum of array by axis.
    """
    # TODO: reduce copies if possible
    var B: NDArray[dtype] = A.contiguous()
    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("Invalid index: index out of bound [0, {}).").format(A.ndim)
        )

    var I = NDArray[DType.int](Shape(A.size))
    var ptr = I.unsafe_ptr()

    var _shape = B.shape.move_axis_to_end(axis)
    var _strides = B.strides.move_axis_to_end(axis)

    TraverseMethods.traverse_buffer_according_to_shape_and_strides(
        ptr, _shape, _strides
    )

    for i in range(0, B.size, B.shape[axis]):
        for j in range(B.shape[axis] - 1):
            var next = Int(I.unsafe_get(i + j + 1))
            var current = Int(I.unsafe_get(i + j))
            B.unsafe_set(next, B.unsafe_get(next) + B.unsafe_get(current))

    return B^
