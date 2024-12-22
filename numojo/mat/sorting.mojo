"""
`numojo.mat.sorting` module provides sorting functions for Matrix type.

- Sorting
- Searching

"""

from .matrix import Matrix
from .creation import full, zeros
from .linalg import transpose

# ===-----------------------------------------------------------------------===#
# Sorting
# ===-----------------------------------------------------------------------===#


fn argsort[dtype: DType](A: Matrix[dtype]) raises -> Matrix[DType.index]:
    """
    Argsort the Matrix. It is first flattened before sorting.
    """
    var I = Matrix[DType.index](shape=(1, A.size))
    for i in range(I.size):
        I._buf[i] = i
    var B = A.flatten()
    _sort_inplace(B, I, 0, A.size - 1)

    return I^


fn argsort[
    dtype: DType
](owned A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Argsort the Matrix along the given axis.
    """
    if axis == 1:
        var I = Matrix[DType.index](shape=A.shape)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                I._store(i, j, j)

        for i in range(A.shape[0]):
            _sort_inplace(
                A, I, left=i * A.strides[0], right=(i + 1) * A.strides[0] - 1
            )
        return I^
    elif axis == 0:
        return transpose(argsort(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn sort[dtype: DType](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Sort the Matrix. It is first flattened before sorting.
    """
    var I = zeros[DType.index](shape=A.shape)
    var B = A.flatten()
    _sort_inplace(B, I, 0, A.size - 1)

    return B^


fn sort[
    dtype: DType
](owned A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Sort the Matrix along the given axis.
    """
    if axis == 1:
        var I = zeros[DType.index](shape=A.shape)
        for i in range(A.shape[0]):
            _sort_inplace(
                A, I, left=i * A.strides[0], right=(i + 1) * A.strides[0] - 1
            )
        return A^
    elif axis == 0:
        return transpose(sort(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn _sort_partition(
    mut A: Matrix, mut I: Matrix, left: Int, right: Int, pivot_index: Int
) raises -> Int:
    """
    Do partition for the data buffer of Matrix.

    Args:
        A: A Matrix.
        I: A Matrix used to store indices.
        left: Left index of the partition.
        right: Right index of the partition.
        pivot_index: Input pivot index

    Returns:
        New pivot index.
    """

    # Boundary check due to use of unsafe way.
    if (left >= A.size) or (right >= A.size) or (pivot_index >= A.size):
        raise Error(
            String(
                "Index out of boundary! "
                "left={}, right={}, pivot_index={}, matrix.size={}"
            ).format(left, right, pivot_index, A.size)
        )

    var pivot_value = A._buf[pivot_index]

    A._buf[pivot_index], A._buf[right] = A._buf[right], A._buf[pivot_index]
    I._buf[pivot_index], I._buf[right] = I._buf[right], I._buf[pivot_index]

    var store_index = left

    for i in range(left, right):
        if A._buf[i] < pivot_value:
            A._buf[store_index], A._buf[i] = A._buf[i], A._buf[store_index]
            I._buf[store_index], I._buf[i] = I._buf[i], I._buf[store_index]
            store_index = store_index + 1

    A._buf[store_index], A._buf[right] = A._buf[right], A._buf[store_index]
    I._buf[store_index], I._buf[right] = I._buf[right], I._buf[store_index]

    return store_index


fn _sort_inplace(mut A: Matrix, mut I: Matrix, left: Int, right: Int) raises:
    """
    Sort in-place of the data buffer (quick-sort).
    It is not guaranteed to be stable.

    Args:
        A: A Matrix.
        I: A Matrix used to store indices.
        left: Left index of the partition.
        right: Right index of the partition.
    """

    if right > left:
        var pivot_index = left + (right - left) // 2
        var pivot_new_index = _sort_partition(A, I, left, right, pivot_index)
        _sort_inplace(A, I, left, pivot_new_index - 1)
        _sort_inplace(A, I, pivot_new_index + 1, right)


# ===-----------------------------------------------------------------------===#
# Searching
# ===-----------------------------------------------------------------------===#


fn max[dtype: DType](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Find max item. It is first flattened before sorting.
    """

    var max_value: Scalar[dtype]
    max_value, _ = _max(A, 0, A.size - 1)

    return max_value


fn max[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Find max item along the given axis.
    """
    if axis == 1:
        var B = mat.Matrix[dtype](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _max(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    0
                ],
            )
        return B^
    elif axis == 0:
        return transpose(max(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn argmax[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.index]:
    """
    Index of the max. It is first flattened before sorting.
    """

    var max_index: Scalar[DType.index]
    _, max_index = _max(A, 0, A.size - 1)

    return max_index


fn argmax[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Index of the max along the given axis.
    """
    if axis == 1:
        var B = mat.Matrix[DType.index](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _max(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    1
                ]
                - i * A.strides[0],
            )
        return B^
    elif axis == 0:
        return transpose(argmax(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn _max[
    dtype: DType
](A: Matrix[dtype], start: Int, end: Int) raises -> Tuple[
    Scalar[dtype], Scalar[DType.index]
]:
    """
    Auxiliary function that find the max value in a range of the buffer.
    Both ends are included.
    """
    if (end >= A.size) or (start >= A.size):
        raise Error(
            String(
                "Index out of boundary! start={}, end={}, matrix.size={}"
            ).format(start, end, A.size)
        )

    var max_index: Scalar[DType.index] = start
    var max_value = A._buf[start]

    for i in range(start, end + 1):
        if A._buf[i] > max_value:
            max_value = A._buf[i]
            max_index = i

    return (max_value, max_index)


fn min[dtype: DType](A: Matrix[dtype]) raises -> Scalar[dtype]:
    """
    Find min item. It is first flattened before sorting.
    """

    var min_value: Scalar[dtype]
    min_value, _ = _min(A, 0, A.size - 1)

    return min_value


fn min[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Find min item along the given axis.
    """
    if axis == 1:
        var B = mat.Matrix[dtype](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _min(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    0
                ],
            )
        return B^
    elif axis == 0:
        return transpose(min(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn argmin[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.index]:
    """
    Index of the min. It is first flattened before sorting.
    """

    var min_index: Scalar[DType.index]
    _, min_index = _min(A, 0, A.size - 1)

    return min_index


fn argmin[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Index of the min along the given axis.
    """
    if axis == 1:
        var B = mat.Matrix[DType.index](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _min(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    1
                ]
                - i * A.strides[0],
            )
        return B^
    elif axis == 0:
        return transpose(argmin(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn _min[
    dtype: DType
](A: Matrix[dtype], start: Int, end: Int) raises -> Tuple[
    Scalar[dtype], Scalar[DType.index]
]:
    """
    Auxiliary function that find the min value in a range of the buffer.
    Both ends are included.
    """
    if (end >= A.size) or (start >= A.size):
        raise Error(
            String(
                "Index out of boundary! start={}, end={}, matrix.size={}"
            ).format(start, end, A.size)
        )

    var min_index: Scalar[DType.index] = start
    var min_value = A._buf[start]

    for i in range(start, end + 1):
        if A._buf[i] < min_value:
            min_value = A._buf[i]
            min_index = i

    return (min_value, min_index)
