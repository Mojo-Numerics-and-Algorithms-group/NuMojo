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

Functions for computing products of vectors, matrices, and arrays.

Exports
-------
- `dot`: Dot product of vectors.
- `inner`: Inner product over the last dimensions.
- `vdot`: Dot product of flattened arrays.
- `vecdot`: Batched vector dot product.
- `matmul`: Matrix multiplication.
- `matvec`: Batched matrix-vector product.
- `vecmat`: Batched vector-matrix product.
- `cross`: Cross product.
- `outer`: Outer product of two flattened arrays.
- `kron`: Kronecker product.
- `tensordot`: Sum of products over given axes.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm import vectorize
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
from numojo.routines.creation import _0darray, zeros
from numojo.routines.manipulation import reshape, transpose
from numojo.routines.math.sums import sum


def _shape_product(shape: NDArrayShape) raises -> Int:
    """Computes a shape's element count, including the 0-D empty product."""
    var result = 1
    for i in range(shape.ndim):
        result *= shape[i]
    return result


def _shape_from_dimensions(dimensions: List[Int]) raises -> NDArrayShape:
    """Constructs a shape while supporting an empty dimension list.

    Args:
        dimensions: The shape dimensions.

    Returns:
        A 0-D shape for an empty list, otherwise the listed dimensions.
    """
    if len(dimensions) == 0:
        return NDArrayShape()
    return NDArrayShape(dimensions)


def _leading_shape(shape: NDArrayShape, core_ndim: Int) raises -> NDArrayShape:
    """Returns the dimensions preceding an operation's core dimensions.

    Args:
        shape: The complete input shape.
        core_ndim: The number of trailing core dimensions to exclude.

    Returns:
        The leading batch dimensions.
    """
    var dimensions = List[Int]()
    for i in range(shape.ndim - core_ndim):
        dimensions.append(shape[i])
    return _shape_from_dimensions(dimensions)


def _shape_without_axis(shape: NDArrayShape, axis: Int) raises -> NDArrayShape:
    """Returns a shape with one normalized axis removed.

    Args:
        shape: The complete input shape.
        axis: The normalized axis to remove.

    Returns:
        The shape containing every dimension except `axis`.
    """
    var dimensions = List[Int]()
    for i in range(shape.ndim):
        if i != axis:
            dimensions.append(shape[i])
    return _shape_from_dimensions(dimensions)


def _row_major_strides(shape: NDArrayShape) raises -> List[Int]:
    """Computes row-major strides for a shape.

    Args:
        shape: The shape whose strides are required.

    Returns:
        Row-major strides, one for each dimension.
    """
    var strides = List[Int](capacity=shape.ndim)
    for _ in range(shape.ndim):
        strides.append(0)
    var stride = 1
    for i in range(shape.ndim - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides^


def _broadcast_flat_offset(
    output_flat: Int,
    output_shape: NDArrayShape,
    output_strides: List[Int],
    input_shape: NDArrayShape,
    input_strides: List[Int],
) raises -> Int:
    """Maps a broadcast batch index to an input batch index.

    Args:
        output_flat: The flat index in the broadcast batch shape.
        output_shape: The broadcast batch shape.
        output_strides: Row-major strides for `output_shape`.
        input_shape: The input's unbroadcast batch shape.
        input_strides: Row-major strides for `input_shape`.

    Returns:
        The corresponding flat index in the input batch dimensions.
    """
    var offset = 0
    var leading = output_shape.ndim - input_shape.ndim
    for input_axis in range(input_shape.ndim):
        if input_shape[input_axis] != 1:
            var output_axis = input_axis + leading
            var coordinate = (
                output_flat // output_strides[output_axis]
            ) % output_shape[output_axis]
            offset += coordinate * input_strides[input_axis]
    return offset


def _zeros_for_shape[
    dtype: DType
](shape: NDArrayShape) raises -> NDArray[dtype]:
    """Creates zeros while preserving the one-element invariant for 0-D arrays.

    Parameters:
        dtype: The element type.

    Args:
        shape: The requested result shape.

    Returns:
        A zero-filled array with the requested shape.
    """
    if shape.ndim == 0:
        return _0darray[dtype](0)
    return zeros[dtype](shape)


def _normalize_vector_axis(
    axis: Int, ndim: Int, location: String
) raises -> Int:
    """Normalizes an axis used as a vector core dimension.

    Args:
        axis: The possibly negative axis.
        ndim: The number of dimensions in the input.
        location: The public function name used in an error.

    Returns:
        The non-negative normalized axis.

    Raises:
        NumojoError: If the input is 0-D or the axis is out of bounds.
    """
    if ndim == 0:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Vector products require arrays with at least one"
                    " dimension."
                ),
                location=location,
            )
        )
    var normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise Error(
            NumojoError(
                category="index",
                message=String("Axis {} is out of bounds for rank {}.").format(
                    axis, ndim
                ),
                location=location,
            )
        )
    return normalized


def _move_axis_to_end[
    dtype: DType
](array: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """Moves one normalized axis to the last dimension.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.
        axis: The normalized axis to move.

    Returns:
        A view with `axis` in the final position.
    """
    var axes = List[Int]()
    for i in range(array.ndim):
        if i != axis:
            axes.append(i)
    axes.append(axis)
    return transpose(array, axes)


def cross[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the cross product of two arrays.

    Parameters:
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

    Returns:
        The dot product of two arrays.

    Raises:
        NumojoError: If the inputs are not equal-length one-dimensional arrays.

    Constraints:
        Each input must have at most two dimensions.

    Examples:
        `dot(a, b)` computes the inner product of vectors `a` and `b`.
    """

    if not array1.is_c_contiguous():
        return dot(array1.contiguous(), array2)
    if not array2.is_c_contiguous():
        return dot(array1, array2.contiguous())

    if array1.ndim == array2.ndim == 1:
        if array1.size != array2.size:
            raise Error(
                NumojoError(
                    category="shape",
                    message="dot requires vectors with the same length.",
                    location="dot",
                )
            )
        comptime width = simd_width_of[dtype]()
        var accumulator = SIMD[dtype, width](0)
        var i = 0
        while i + width <= array1.size:
            accumulator += array1.unsafe_load[width=width](
                i
            ) * array2.unsafe_load[width=width](i)
            i += width

        var result = accumulator.reduce_add()
        while i < array1.size:
            result += (
                array1.unsafe_load[width=1](i) * array2.unsafe_load[width=1](i)
            ).reduce_add()
            i += 1
        return _0darray(val=result)
    if array1.ndim <= 2 and array2.ndim <= 2:
        return matmul(array1, array2)
    raise Error(
        NumojoError(
            category="shape",
            message=(
                "dot currently supports arrays with at most two dimensions."
            ),
            location="dot",
        )
    )


def inner[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """Computes the inner product over the inputs' last dimensions.

    Parameters:
        dtype: The element type.

    Args:
        array1: The first input array.
        array2: The second input array.

    Returns:
        The inner product. For non-scalar inputs, the result shape is
        `array1.shape[:-1] + array2.shape[:-1]`.

    Raises:
        NumojoError: If non-scalar inputs have different final dimensions.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.f64](6).reshape(nm.Shape(2, 3))
        var b = nm.arange[nm.f64](3)
        print(nm.inner(a, b))
        ```
    """
    var output_dimensions = List[Int]()
    if array1.ndim == 0 or array2.ndim == 0:
        if array1.ndim != 0:
            for i in range(array1.ndim):
                output_dimensions.append(array1.shape[i])
        elif array2.ndim != 0:
            for i in range(array2.ndim):
                output_dimensions.append(array2.shape[i])

        var scalar_result = _zeros_for_shape[dtype](
            _shape_from_dimensions(output_dimensions)
        )
        if scalar_result.size == 0:
            return scalar_result^
        if not array1.is_c_contiguous():
            return inner(array1.contiguous(), array2)
        if not array2.is_c_contiguous():
            return inner(array1, array2.contiguous())

        if array1.ndim == 0:
            var scalar = array1.unsafe_get(0)
            for i in range(scalar_result.size):
                scalar_result.unsafe_set(i, scalar * array2.unsafe_get(i))
        else:
            var scalar = array2.unsafe_get(0)
            for i in range(scalar_result.size):
                scalar_result.unsafe_set(i, array1.unsafe_get(i) * scalar)
        return scalar_result^

    var contract_size = array1.shape[array1.ndim - 1]
    if contract_size != array2.shape[array2.ndim - 1]:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "inner requires equal final dimensions, but received {}"
                    " and {}."
                ).format(contract_size, array2.shape[array2.ndim - 1]),
                location="inner",
            )
        )

    for i in range(array1.ndim - 1):
        output_dimensions.append(array1.shape[i])
    for i in range(array2.ndim - 1):
        output_dimensions.append(array2.shape[i])
    var result = _zeros_for_shape[dtype](
        _shape_from_dimensions(output_dimensions)
    )
    if result.size == 0 or contract_size == 0:
        return result^

    if not array1.is_c_contiguous():
        return inner(array1.contiguous(), array2)
    if not array2.is_c_contiguous():
        return inner(array1, array2.contiguous())

    var array1_vectors = array1.size // contract_size
    var array2_vectors = array2.size // contract_size
    for i in range(array1_vectors):
        for j in range(array2_vectors):
            var total = Scalar[dtype](0)
            for k in range(contract_size):
                total += array1.unsafe_get(
                    i * contract_size + k
                ) * array2.unsafe_get(j * contract_size + k)
            result.unsafe_set(i * array2_vectors + j, total)
    return result^


def vdot[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """Computes the dot product of two flattened arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: The first input array.
        array2: The second input array.

    Returns:
        A 0-D array containing the flattened vector dot product.

    Raises:
        NumojoError: If the inputs contain different numbers of elements.

    Notes:
        This overload accepts real-valued `NDArray` inputs. Complex conjugation
        will be provided by future `ComplexNDArray` overloads.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.f64](6).reshape(nm.Shape(2, 3))
        print(nm.vdot(a, a))
        ```
    """
    if array1.size != array2.size:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "vdot requires equal element counts, but received {}"
                    " and {}."
                ).format(array1.size, array2.size),
                location="vdot",
            )
        )
    if array1.size == 0:
        return _0darray[dtype](0)
    if not array1.is_c_contiguous():
        return vdot(array1.contiguous(), array2)
    if not array2.is_c_contiguous():
        return vdot(array1, array2.contiguous())

    var total = Scalar[dtype](0)
    for i in range(array1.size):
        total += array1.unsafe_get(i) * array2.unsafe_get(i)
    return _0darray[dtype](total)


def vecdot[
    dtype: DType
](
    array1: NDArray[dtype],
    array2: NDArray[dtype],
    axis: Int = -1,
) raises -> NDArray[dtype]:
    """Computes broadcast vector dot products along an axis.

    Parameters:
        dtype: The element type.

    Args:
        array1: The first input array.
        array2: The second input array.
        axis: The vector axis in both inputs. Negative axes are supported.

    Returns:
        The vector dot products with the contracted axis removed.

    Raises:
        NumojoError: If an input is 0-D, an axis is invalid, the vector lengths
            differ, or the remaining dimensions cannot be broadcast together.

    Notes:
        This overload accepts real-valued `NDArray` inputs. Complex conjugation
        will be provided by future `ComplexNDArray` overloads.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.f64](6).reshape(nm.Shape(2, 3))
        print(nm.vecdot(a, a))
        ```
    """
    var axis1 = _normalize_vector_axis(axis, array1.ndim, "vecdot")
    var axis2 = _normalize_vector_axis(axis, array2.ndim, "vecdot")
    var contract_size = array1.shape[axis1]
    if contract_size != array2.shape[axis2]:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "vecdot requires equal vector lengths, but received {}"
                    " and {}."
                ).format(contract_size, array2.shape[axis2]),
                location="vecdot",
            )
        )

    var batch1 = _shape_without_axis(array1.shape, axis1)
    var batch2 = _shape_without_axis(array2.shape, axis2)
    var output_shape = batch1.broadcast(batch2)
    var result = _zeros_for_shape[dtype](output_shape)
    var batch_count = _shape_product(output_shape)
    if batch_count == 0 or contract_size == 0:
        return result^

    var moved1 = _move_axis_to_end(array1, axis1).contiguous()
    var moved2 = _move_axis_to_end(array2, axis2).contiguous()
    var output_strides = _row_major_strides(output_shape)
    var batch1_strides = _row_major_strides(batch1)
    var batch2_strides = _row_major_strides(batch2)

    for batch in range(batch_count):
        var batch_offset1 = _broadcast_flat_offset(
            batch, output_shape, output_strides, batch1, batch1_strides
        )
        var batch_offset2 = _broadcast_flat_offset(
            batch, output_shape, output_strides, batch2, batch2_strides
        )
        var total = Scalar[dtype](0)
        for k in range(contract_size):
            total += moved1.unsafe_get(
                batch_offset1 * contract_size + k
            ) * moved2.unsafe_get(batch_offset2 * contract_size + k)
        result.unsafe_set(batch, total)
    return result^


def matvec[
    dtype: DType
](matrix: NDArray[dtype], vector: NDArray[dtype]) raises -> NDArray[dtype]:
    """Computes broadcast matrix-vector products.

    Parameters:
        dtype: The element type.

    Args:
        matrix: Matrices stored in the final two dimensions.
        vector: Vectors stored in the final dimension.

    Returns:
        Matrix-vector products with broadcast leading dimensions.

    Raises:
        NumojoError: If the inputs lack their required core dimensions, their
            contraction dimensions differ, or their batches cannot broadcast.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.f64](6).reshape(nm.Shape(2, 3))
        var v = nm.arange[nm.f64](3)
        print(nm.matvec(a, v))
        ```
    """
    if matrix.ndim < 2 or vector.ndim < 1:
        raise Error(
            NumojoError(
                category="shape",
                message="matvec requires a matrix and a vector.",
                location="matvec",
            )
        )
    var rows = matrix.shape[matrix.ndim - 2]
    var contract_size = matrix.shape[matrix.ndim - 1]
    if contract_size != vector.shape[vector.ndim - 1]:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "matvec requires equal contraction dimensions, but received"
                    " {} and {}."
                ).format(contract_size, vector.shape[vector.ndim - 1]),
                location="matvec",
            )
        )

    var matrix_batch = _leading_shape(matrix.shape, 2)
    var vector_batch = _leading_shape(vector.shape, 1)
    var batch_shape = matrix_batch.broadcast(vector_batch)
    var output_dimensions = List[Int]()
    for i in range(batch_shape.ndim):
        output_dimensions.append(batch_shape[i])
    output_dimensions.append(rows)
    var result = _zeros_for_shape[dtype](NDArrayShape(output_dimensions))
    var batch_count = _shape_product(batch_shape)
    if result.size == 0 or contract_size == 0:
        return result^

    if not matrix.is_c_contiguous():
        return matvec(matrix.contiguous(), vector)
    if not vector.is_c_contiguous():
        return matvec(matrix, vector.contiguous())
    var batch_strides = _row_major_strides(batch_shape)
    var matrix_batch_strides = _row_major_strides(matrix_batch)
    var vector_batch_strides = _row_major_strides(vector_batch)

    for batch in range(batch_count):
        var matrix_batch_offset = _broadcast_flat_offset(
            batch,
            batch_shape,
            batch_strides,
            matrix_batch,
            matrix_batch_strides,
        )
        var vector_batch_offset = _broadcast_flat_offset(
            batch,
            batch_shape,
            batch_strides,
            vector_batch,
            vector_batch_strides,
        )
        for row in range(rows):
            var total = Scalar[dtype](0)
            for k in range(contract_size):
                total += matrix.unsafe_get(
                    matrix_batch_offset * rows * contract_size
                    + row * contract_size
                    + k
                ) * vector.unsafe_get(vector_batch_offset * contract_size + k)
            result.unsafe_set(batch * rows + row, total)
    return result^


def vecmat[
    dtype: DType
](vector: NDArray[dtype], matrix: NDArray[dtype]) raises -> NDArray[dtype]:
    """Computes broadcast vector-matrix products.

    Parameters:
        dtype: The element type.

    Args:
        vector: Vectors stored in the final dimension.
        matrix: Matrices stored in the final two dimensions.

    Returns:
        Vector-matrix products with broadcast leading dimensions.

    Raises:
        NumojoError: If the inputs lack their required core dimensions, their
            contraction dimensions differ, or their batches cannot broadcast.

    Notes:
        This overload accepts real-valued `NDArray` inputs. Complex conjugation
        will be provided by a future `ComplexNDArray` overload.

    Examples:
        ```mojo
        import numojo as nm

        var v = nm.arange[nm.f64](3)
        var a = nm.arange[nm.f64](6).reshape(nm.Shape(3, 2))
        print(nm.vecmat(v, a))
        ```
    """
    if vector.ndim < 1 or matrix.ndim < 2:
        raise Error(
            NumojoError(
                category="shape",
                message="vecmat requires a vector and a matrix.",
                location="vecmat",
            )
        )
    var contract_size = vector.shape[vector.ndim - 1]
    if contract_size != matrix.shape[matrix.ndim - 2]:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "vecmat requires equal contraction dimensions, but received"
                    " {} and {}."
                ).format(contract_size, matrix.shape[matrix.ndim - 2]),
                location="vecmat",
            )
        )
    var columns = matrix.shape[matrix.ndim - 1]

    var vector_batch = _leading_shape(vector.shape, 1)
    var matrix_batch = _leading_shape(matrix.shape, 2)
    var batch_shape = vector_batch.broadcast(matrix_batch)
    var output_dimensions = List[Int]()
    for i in range(batch_shape.ndim):
        output_dimensions.append(batch_shape[i])
    output_dimensions.append(columns)
    var result = _zeros_for_shape[dtype](NDArrayShape(output_dimensions))
    var batch_count = _shape_product(batch_shape)
    if result.size == 0 or contract_size == 0:
        return result^

    if not vector.is_c_contiguous():
        return vecmat(vector.contiguous(), matrix)
    if not matrix.is_c_contiguous():
        return vecmat(vector, matrix.contiguous())
    var batch_strides = _row_major_strides(batch_shape)
    var vector_batch_strides = _row_major_strides(vector_batch)
    var matrix_batch_strides = _row_major_strides(matrix_batch)

    for batch in range(batch_count):
        var vector_batch_offset = _broadcast_flat_offset(
            batch,
            batch_shape,
            batch_strides,
            vector_batch,
            vector_batch_strides,
        )
        var matrix_batch_offset = _broadcast_flat_offset(
            batch,
            batch_shape,
            batch_strides,
            matrix_batch,
            matrix_batch_strides,
        )
        for column in range(columns):
            var total = Scalar[dtype](0)
            for k in range(contract_size):
                total += vector.unsafe_get(
                    vector_batch_offset * contract_size + k
                ) * matrix.unsafe_get(
                    matrix_batch_offset * contract_size * columns
                    + k * columns
                    + column
                )
            result.unsafe_set(batch * columns + column, total)
    return result^


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

    comptime tile_size = 4
    comptime tile_x = width * tile_size
    comptime tile_y = tile_size
    comptime unroll_factor = tile_x // width

    def calculate_A_rows(m: Int) {mut result, A, B, t1, t2}:
        # Note: this assumes that t1 and t2 are multiples of the tiles.
        for y in range(0, t2, tile_y):
            for x in range(0, t1, tile_x):
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

                    vectorize[width, unroll_factor=unroll_factor](tile_x, dot)

    parallelize(calculate_A_rows, t0, t0)
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

    def calculate_A_rows(m: Int) {mut result, A, B, t1, t2}:
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

    parallelize(calculate_A_rows, t0, t0)

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

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](60).reshape(nm.Shape(3, 4, 5))
        var b = nm.arange[nm.f64](120).reshape(nm.Shape(4, 5, 6))
        print(nm.linalg.tensordot(a, b, axes=2))
        ```
    """

    var axes_a = List[Int]()
    var axes_b = List[Int]()
    for i in range(axes):
        axes_a.append(A.ndim - axes + i)
        axes_b.append(i)
    return tensordot(A, B, axes_a, axes_b)
