# ===----------------------------------------------------------------------=== #
# NuMojo: Products
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Products (numojo.routines.linalg.products).
===========================================
Array and vector product operations.

Functions for computing products of vectors and arrays (dot product, matrix
multiplication, cross product).

Exports
-------
- `dot`: Dot product of vectors.
- `matmul`: Matrix multiplication.
- `cross`: Cross product.
- `outer`: Outer product of two flattened arrays.
- `kron`: Kronecker product.
- `tensordot`: Sum of products over given axes.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm import (
    Static2DTileUnitFunc as Tile2DFunc,
    vectorize,
)
from std.memory import unsafe_memcpy
from std.sys import simd_width_of

# ===----------------------------------------------------------------------===#
# External
# ===----------------------------------------------------------------------===#
from max.algorithm import parallelize

# ===----------------------------------------------------------------------===#
# numojo
# ===----------------------------------------------------------------------===#
from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import zeros
from numojo.routines.manipulation import reshape, transpose
from numojo.routines.math.sums import sum


def cross[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the cross product of two arrays.

    Parameters
        dtype: The element type.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be of shape (3,).

    Returns:
        The cross product of two arrays.
    """

    if (array1.size == array2.size == 3) and (array1.ndim == array2.ndim == 1):
        var array3: NDArray[dtype] = NDArray[dtype](NDArrayShape(3))
        array3.store(
            0,
            (array1.load(1) * array2.load(2) - array1.load(2) * array2.load(1)),
        )
        array3.store(
            1,
            (array1.load(2) * array2.load(0) - array1.load(0) * array2.load(2)),
        )
        array3.store(
            2,
            (array1.load(0) * array2.load(1) - array1.load(1) * array2.load(0)),
        )
        return array3^
    else:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "resultross product is not supported for arrays of shape "
                )
                + array1.shape.__str__()
                + " and "
                + array2.shape.__str__(),
                location="cross",
            )
        )


# TODO: implement other cases for dot function
def dot[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the dot product of two arrays.

    Parameters
        dtype: The element type.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be 1 dimensional.

    Returns:
        The dot product of two arrays.
    """

    if not array1.is_c_contiguous():
        return dot(array1.contiguous(), array2)
    if not array2.is_c_contiguous():
        return dot(array1, array2.contiguous())

    comptime width = simd_width_of[dtype]()
    if array1.ndim == array2.ndim == 1:
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(array1.size))

        def vectorized_dot[
            simd_width: Int
        ](idx: Int) {mut result, imm array1, imm array2} -> None:
            result.unsafe_store[width=simd_width](
                idx,
                array1.unsafe_load[width=simd_width](idx)
                * array2.unsafe_load[width=simd_width](idx),
            )

        vectorize[width](array1.size, vectorized_dot)
        return result^
    else:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "resultross product is not supported for arrays of shape "
                )
                + array1.shape.__str__()
                + " and "
                + array2.shape.__str__(),
                location="dot",
            )
        )


# Perform 2D tiling on the iteration space defined by end_x and end_y.
def tile[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# https://docs.modular.com/mojo/notebooks/Matmul
def matmul_tiled_unrolled_parallelized[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication vectorized, tiled, unrolled, and parallelized.
    """
    if not A.is_c_contiguous():
        return matmul_tiled_unrolled_parallelized(A.contiguous(), B)
    if not B.is_c_contiguous():
        return matmul_tiled_unrolled_parallelized(A, B.contiguous())

    comptime width = max(simd_width_of[dtype](), 16)
    var result: NDArray[dtype] = zeros[dtype](Shape(A.shape[0], B.shape[1]))
    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var t2 = B.shape[1]

    @parameter
    def calculate_A_rows(m: Int):
        @parameter
        def calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                def dot[
                    simd_width: Int
                ](n: Int) {
                    mut result,
                    imm A,
                    imm B,
                    imm t1,
                    imm t2,
                    imm x,
                    imm m,
                    imm k,
                } -> None:
                    result.unsafe_store[width=simd_width](
                        m * t2 + (n + x),
                        val=result.unsafe_load[width=simd_width](
                            m * t2 + (n + x)
                        )
                        + A.unsafe_load[width=1](m * t1 + k)
                        * B.unsafe_load[width=simd_width](k * t2 + (n + x)),
                    )

                comptime unroll_factor = tile_x // width
                vectorize[width, unroll_factor=unroll_factor](tile_x, dot)

        comptime tile_size = 4
        tile[calc_tile, width * tile_size, tile_size](t1, t2)

    parallelize[calculate_A_rows](t0, t0)
    return result^


def matmul_1darray[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication for 1-d arrays (inner dot).
    """

    var result = NDArray[dtype](Shape(1, 1))

    if A.ndim * B.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message="The dimensions of the arrays should be 1.",
                location="matmul_1darray",
            )
        )
    elif A.size != B.size:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "matmul: a mismatch in core dimension 0: "
                    "size {} is different from {}"
                ).format(A.size, B.size),
                location="matmul_1darray",
            )
        )
    else:
        result.unsafe_set(0, sum(A * B))

    return result^


def matmul_2darray[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication for 2-d arrays (inner dot).

    Parameter:
        dtype: Data type.

    Args:
        A: First array.
        B: Second array.

    Return:
        A multiplied by B.

    Raises:
        When the shape does not match.

    Notes:
        The multiplication is vectorized and parallelized.

    References:
        [1] https://docs.modular.com/mojo/notebooks/Matmul.
        resultompared to the reference, we increases the size of
        the SIMD vector from the default width to 16. The purpose is to
        increase the performance via SIMD.
        This reduces the execution time by ~50 percent compared to
        `matmul_parallelized` and `matmul_tiled_unrolled_parallelized` for large
        matrices.
    """

    if not A.is_c_contiguous():
        return matmul_2darray(A.contiguous(), B)
    if not B.is_c_contiguous():
        return matmul_2darray(A, B.contiguous())

    comptime width = max(simd_width_of[dtype](), 16)

    if A.ndim * B.ndim == 1:
        return matmul_1darray(A, B)

    if (A.ndim == 1) and (A.size == B.shape[0]):
        var A_reshaped = A.reshape(Shape(1, A.shape[0]))
        var res = matmul_2darray(A_reshaped, B)
        return res.reshape(Shape(B.shape[1]))

    if (B.ndim == 1) and (A.shape[1] == B.size):
        var B_reshaped = B.reshape(Shape(B.shape[0], 1))
        var res = matmul_2darray(A, B_reshaped)
        return res.reshape(Shape(A.shape[0]))

    if (A.ndim == 1) or (B.ndim == 1):
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "matmul: a mismatch in shapes: {} is different from {}"
                ).format(A.shape[-1], B.shape[0]),
                location="matmul_2darray",
            )
        )

    if A.shape[1] != B.shape[0]:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "matmul: a mismatch in shapes: {} is different from {}"
                ).format(A.shape[1], B.shape[0]),
                location="matmul_2darray",
            )
        )

    var result: NDArray[dtype] = zeros[dtype](Shape(A.shape[0], B.shape[1]))
    var t0 = A.shape[0]
    var t1 = A.shape[1]
    var t2 = B.shape[1]

    @parameter
    def calculate_A_rows(m: Int):
        for k in range(t1):

            def dot[
                simd_width: Int
            ](n: Int) {
                mut result, imm A, imm B, imm t2, imm t1, imm k, imm m
            } -> None:
                result.unsafe_store[width=simd_width](
                    m * t2 + n,
                    val=result.unsafe_load[width=simd_width](m * t2 + n)
                    + A.unsafe_load[width=1](m * t1 + k)
                    * B.unsafe_load[width=simd_width](k * t2 + n),
                )

            vectorize[width](t2, dot)

    parallelize[calculate_A_rows](t0, t0)

    return result^


def matmul[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication for any dimensions.

    Parameter:
        dtype: Data type.

    Args:
        A: First array.
        B: Second array.

    Return:
        A multiplied by B.

    Raises:
        (1) The shapes of first n-2 dimensions do not match.
        (2) The shape of -2 dimension of first array does not match
        the shape of -1 dimension of the second array.

    Notes:\n
        When A and B are 1darray, it is equal to dot of vectors:
        `(i) @ (i) -> (1)`.\n
        When A and B are 2darray, it is equal to inner products of matrices:
        `(i,j) @ (j,k) -> (i,k)`.\n
        When A and B are more than 2d, it is equal to a stack of 2darrays:
        `(i,j,k) @ (i,k,l) -> (i,j,l)` and
        `(i,j,k,l) @ (i,j,l,m) -> (i,j,k,m)`.
    """

    if not A.is_c_contiguous():
        return matmul(A.contiguous(), B)
    if not B.is_c_contiguous():
        return matmul(A, B.contiguous())

    if (A.ndim <= 2) and (B.ndim <= 2):
        return matmul_2darray(A, B)

    if A.ndim != B.ndim:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "matmul: dimension {} is different from {}"
                ).format(A.ndim, B.ndim),
                location="matmul",
            )
        )

    for i in range(A.ndim - 2):
        if A.shape[i] != B.shape[i]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "matmul: {}-th dimensions mismatch: {} vs {}"
                    ).format(A.shape[i], B.shape[i]),
                    location="matmul",
                )
            )

    if A.shape[-1] != B.shape[-2]:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "matmul: a mismatch in shapes: {} is different from {}"
                ).format(A.shape[-1], B.shape[-2]),
                location="matmul",
            )
        )

    var shape_as_list = List[Int]()
    for i in range(A.ndim - 2):
        shape_as_list.append(A.shape[i])
    shape_as_list.append(A.shape[-2])
    shape_as_list.append(B.shape[-1])

    var result = NDArray[dtype](Shape(shape_as_list))
    var A_sub_matrix = NDArray[dtype](Shape(A.shape[-2], A.shape[-1]))
    var B_sub_matrix = NDArray[dtype](Shape(B.shape[-2], B.shape[-1]))
    var result_sub_matrix = NDArray[dtype](
        Shape(result.shape[-2], result.shape[-1])
    )

    for i in range(result.size // result_sub_matrix.size):
        unsafe_memcpy(
            dest=A_sub_matrix.unsafe_ptr(),
            src=A.unsafe_ptr().unsafe_offset(i * A_sub_matrix.size),
            count=A_sub_matrix.size,
        )
        unsafe_memcpy(
            dest=B_sub_matrix.unsafe_ptr(),
            src=B.unsafe_ptr().unsafe_offset(i * B_sub_matrix.size),
            count=B_sub_matrix.size,
        )
        result_sub_matrix = matmul_2darray(A_sub_matrix, B_sub_matrix)
        unsafe_memcpy(
            dest=result.unsafe_ptr().unsafe_offset(i * result_sub_matrix.size),
            src=result_sub_matrix.unsafe_ptr(),
            count=result_sub_matrix.size,
        )

    return result^


def matmul_naive[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Array multiplication with three nested loops.
    """
    var result: NDArray[dtype]
    if B.ndim == 1:
        result = zeros[dtype](NDArrayShape(A.shape[0]))
        for m in range(result.shape[0]):
            for k in range(A.shape[1]):
                result.store(m, val=result.load(m) + A.load(m, k) * B.load(k))
    elif B.ndim != 1:
        result = zeros[dtype](NDArrayShape(A.shape[0], B.shape[1]))
        for m in range(result.shape[0]):
            for k in range(A.shape[1]):
                for n in range(result.shape[1]):
                    result.store(
                        m,
                        n,
                        val=result.load(m, n) + A.load(m, k) * B.load(k, n),
                    )
    else:
        raise Error(
            NumojoError(
                category="shape",
                message="Invalid shape for B",
                location="matmul_naive",
            )
        )

    return result^


def outer[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the outer product of two arrays.

    Both inputs are flattened (in C order) before the product is taken,
    matching `numpy.outer`.

    Parameters:
        dtype: The element type.

    Args:
        A: First input, treated as a flat vector of length `A.size`.
        B: Second input, treated as a flat vector of length `B.size`.

    Returns:
        A 2-D array of shape `(A.size, B.size)`, where
        `result[i, j] = A_flat[i] * B_flat[j]`.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](3)
        var b = nm.arange[nm.f64](4)
        print(nm.linalg.outer(a, b))
        ```
    """

    var a_flat = A.flatten()
    var b_flat = B.flatten()
    var m = a_flat.size
    var n = b_flat.size
    var result = NDArray[dtype](Shape(m, n))

    for i in range(m):
        var a_val = a_flat.unsafe_get(i)
        for j in range(n):
            result.unsafe_set(i * n + j, a_val * b_flat.unsafe_get(j))

    return result^


def kron[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the Kronecker product of two arrays.

    If `A` and `B` have different numbers of dimensions, the shape of the
    array with fewer dimensions is padded with leading 1s, matching
    `numpy.kron`.

    Parameters:
        dtype: The element type.

    Args:
        A: First input array.
        B: Second input array.

    Returns:
        The Kronecker product. If `A` has (padded) shape
        `(r0, r1, ..., rN)` and `B` has (padded) shape
        `(s0, s1, ..., sN)`, the result has shape
        `(r0 * s0, r1 * s1, ..., rN * sN)`.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](6).reshape(nm.Shape(2, 3))
        var b = nm.eye[nm.f64](2, 2)
        print(nm.linalg.kron(a, b))
        ```
    """

    if not A.is_c_contiguous():
        return kron(A.contiguous(), B)
    if not B.is_c_contiguous():
        return kron(A, B.contiguous())

    var ndim = max(A.ndim, B.ndim)

    var a_shape = List[Int]()
    var b_shape = List[Int]()
    for i in range(ndim):
        var a_dim_index = i - (ndim - A.ndim)
        a_shape.append(A.shape[a_dim_index] if a_dim_index >= 0 else 1)
        var b_dim_index = i - (ndim - B.ndim)
        b_shape.append(B.shape[b_dim_index] if b_dim_index >= 0 else 1)

    var out_shape = List[Int]()
    for i in range(ndim):
        out_shape.append(a_shape[i] * b_shape[i])

    var result = NDArray[dtype](Shape(out_shape))

    # Strides (row-major) for the padded logical shapes of A, B and the
    # result, used to convert a flat result-index into per-axis
    # coordinates and back into flat source offsets.
    var out_strides = List[Int](capacity=ndim)
    var a_strides = List[Int](capacity=ndim)
    var b_strides = List[Int](capacity=ndim)
    for _ in range(ndim):
        out_strides.append(0)
        a_strides.append(0)
        b_strides.append(0)
    var out_acc = 1
    var a_acc = 1
    var b_acc = 1
    for i in range(ndim - 1, -1, -1):
        out_strides[i] = out_acc
        out_acc *= out_shape[i]
        a_strides[i] = a_acc
        a_acc *= a_shape[i]
        b_strides[i] = b_acc
        b_acc *= b_shape[i]

    for flat in range(result.size):
        var rem = flat
        var a_offset = 0
        var b_offset = 0
        for i in range(ndim):
            var coord = rem // out_strides[i]
            rem = rem % out_strides[i]
            var a_coord = coord // b_shape[i]
            var b_coord = coord % b_shape[i]
            a_offset += a_coord * a_strides[i]
            b_offset += b_coord * b_strides[i]
        result.unsafe_set(flat, A.unsafe_get(a_offset) * B.unsafe_get(b_offset))

    return result^


def tensordot[
    dtype: DType
](
    A: NDArray[dtype],
    B: NDArray[dtype],
    axes_a: List[Int],
    axes_b: List[Int],
) raises -> NDArray[dtype]:
    """
    Compute the tensor dot product along specified axes.

    Sums the products of the elements of `A` and `B` over the axes given
    in `axes_a` and `axes_b`. This is a generalization of `dot` and
    `matmul` to arbitrary axes, matching `numpy.tensordot`.

    Parameters:
        dtype: The element type.

    Args:
        A: First input array.
        B: Second input array.
        axes_a: Axes of `A` to sum over. Negative values count from the
            end.
        axes_b: Axes of `B` to sum over, paired positionally with
            `axes_a`. Negative values count from the end.

    Returns:
        The tensor dot product. Its shape is the concatenation of the
        axes of `A` not in `axes_a` followed by the axes of `B` not in
        `axes_b`.

    Raises:
        NumojoError: If `axes_a` and `axes_b` have different lengths, or
            the paired axis lengths of `A` and `B` do not match.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](60).reshape(nm.Shape(3, 4, 5))
        var b = nm.arange[nm.f64](24).reshape(nm.Shape(4, 3, 2))
        print(nm.linalg.tensordot(a, b, axes_a=[1, 0], axes_b=[0, 1]))
        ```
    """

    if len(axes_a) != len(axes_b):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "tensordot: `axes_a` (len {}) and `axes_b` (len {}) must"
                    " have the same length"
                ).format(len(axes_a), len(axes_b)),
                location="tensordot",
            )
        )

    var norm_axes_a = List[Int]()
    for ax in axes_a:
        norm_axes_a.append(ax + A.ndim if ax < 0 else ax)
    var norm_axes_b = List[Int]()
    for ax in axes_b:
        norm_axes_b.append(ax + B.ndim if ax < 0 else ax)

    for i in range(len(norm_axes_a)):
        if A.shape[norm_axes_a[i]] != B.shape[norm_axes_b[i]]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "tensordot: shape mismatch on paired axes {} (size"
                        " {}) and {} (size {})"
                    ).format(
                        norm_axes_a[i],
                        A.shape[norm_axes_a[i]],
                        norm_axes_b[i],
                        B.shape[norm_axes_b[i]],
                    ),
                    location="tensordot",
                )
            )

    var notin_a = List[Int]()
    for i in range(A.ndim):
        if i not in norm_axes_a:
            notin_a.append(i)
    var notin_b = List[Int]()
    for i in range(B.ndim):
        if i not in norm_axes_b:
            notin_b.append(i)

    var newaxes_a = List[Int]()
    for i in notin_a:
        newaxes_a.append(i)
    for i in norm_axes_a:
        newaxes_a.append(i)
    var newaxes_b = List[Int]()
    for i in norm_axes_b:
        newaxes_b.append(i)
    for i in notin_b:
        newaxes_b.append(i)

    var olda = List[Int]()
    for i in notin_a:
        olda.append(A.shape[i])
    var oldb = List[Int]()
    for i in notin_b:
        oldb.append(B.shape[i])

    var n_free_a = 1
    for d in olda:
        n_free_a *= d
    var n_free_b = 1
    for d in oldb:
        n_free_b *= d
    var n_contract = 1
    for i in norm_axes_a:
        n_contract *= A.shape[i]

    var at = transpose(A, newaxes_a).reshape(Shape(n_free_a, n_contract))
    var bt = transpose(B, newaxes_b).reshape(Shape(n_contract, n_free_b))

    var out_shape = List[Int]()
    for d in olda:
        out_shape.append(d)
    for d in oldb:
        out_shape.append(d)
    var result = matmul_2darray(at, bt)
    if len(out_shape) == 0:
        return result.reshape(Shape(1))
    return result.reshape(Shape(out_shape))


def tensordot[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype], axes: Int = 2) raises -> NDArray[dtype]:
    """
    (overload) Tensor dot product summing over the last `axes` axes of
    `A` and the first `axes` axes of `B`.

    `axes=2` reproduces `matmul` for 2-D inputs; `axes=1` reproduces
    `dot`-like contraction over one axis; `axes=0` reproduces the full
    outer product (no summation).

    Parameters:
        dtype: The element type.

    Args:
        A: First input array.
        B: Second input array.
        axes: Number of trailing/leading axes of `A`/`B` to sum over.

    Returns:
        The tensor dot product. See the `axes_a`/`axes_b` overload.
    """

    var axes_a = List[Int]()
    var axes_b = List[Int]()
    for i in range(axes):
        axes_a.append(A.ndim - axes + i)
        axes_b.append(i)
    return tensordot(A, B, axes_a, axes_b)
