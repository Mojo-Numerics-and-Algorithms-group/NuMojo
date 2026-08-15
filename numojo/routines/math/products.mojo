# ===----------------------------------------------------------------------=== #
# NuMojo: Product routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Product routines for NuMojo (numojo.routines.math.products).

Implements product and cumulative product reductions for NDArrays and Matrices.
"""
# ===----------------------------------------------------------------------===#
# Stdlib
# ===----------------------------------------------------------------------===#
from std.algorithm.functional import vectorize
from std.memory import unsafe_memcpy
from std.sys import simd_width_of

# ===----------------------------------------------------------------------===#
# External
# ===----------------------------------------------------------------------===#
from max.algorithm import parallelize

# ===----------------------------------------------------------------------===#
# numojo
# ===----------------------------------------------------------------------===#
from numojo.core.indexing import TraverseMethods
from numojo.core.layout.ndshape import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import ones


def prod[dtype: DType](A: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Returns products of all items in the array.

    Example:
    ```console
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
    [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
    [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32

    > print(nm.prod(A))
    6.1377261317829834e-07
    ```

    Args:
        A: NDArray.

    Returns:
        Scalar.
    """

    if not A.is_c_contiguous():
        return prod(A.contiguous())
    comptime width: Int = simd_width_of[dtype]()
    var res = Scalar[dtype](1)

    def cal_vec[width: Int](i: Int) {mut res, A}:
        res *= A.unsafe_load[width=width](i).reduce_mul()

    vectorize[width](A.size, cal_vec)
    return res


def prod[
    dtype: DType
](A: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
    """
    Returns products of array elements over a given axis.

    Args:
        A: NDArray.
        axis: The axis along which the product is performed.

    Returns:
        An NDArray.
    """

    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("Invalid index: index out of bound [0, {}).").format(A.ndim)
        )

    var result_shape: List[Int] = List[Int]()
    var size_of_axis: Int = A.shape[axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(A.ndim):
        if i != axis:
            result_shape.append(A.shape[i])
            slices.append(Slice(0, A.shape[i]))
        else:
            slices.append(Slice(0, 0))  # Temp value
    var result: NDArray[dtype] = ones[dtype](NDArrayShape(result_shape))
    for i in range(size_of_axis):
        slices[axis] = Slice(i, i + 1)
        # TODO: modify slicing getter to avoid copy.
        var arr_slice: NDArray[dtype] = A._getitem_list_slices(slices.copy())
        result *= arr_slice

    return result^
def cumprod[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Returns cumprod of all items of an array.
    The array is flattened before cumprod.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.

    Returns:
        Cumprod of all items of an array.
    """

    if A.ndim == 1:
        var B = A.contiguous()
        for i in range(A.size - 1):
            B.unsafe_set(i + 1, B.unsafe_get(i + 1) * B.unsafe_get(i))
        return B^

    else:
        return cumprod(A.flatten(), axis=-1)


def cumprod[
    dtype: DType
](A: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
    """
    Returns cumprod of array by axis.

    Parameters:
        dtype: The element type.

    Args:
        A: NDArray.
        axis: Axis.

    Returns:
        Cumprod of array by axis.
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
            B.unsafe_set(next, B.unsafe_get(next) * B.unsafe_get(current))

    return B^
