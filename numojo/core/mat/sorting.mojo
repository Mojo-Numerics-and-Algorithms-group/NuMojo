"""
`numojo.core.mat.sorting` module provides sorting functions for Matrix type.

- Sorting
- Searching

"""

from .mat import Matrix
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
    inout A: Matrix, inout I: Matrix, left: Int, right: Int, pivot_index: Int
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


fn _sort_inplace(
    inout A: Matrix, inout I: Matrix, left: Int, right: Int
) raises:
    """
    Sort in-place of the data buffer (quick-sort). It is not guaranteed to be stable.

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
