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


fn sort[dtype: DType](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Sort the Matrix. It is first flattened before sorting.
    """
    var B = _quick_sort(A.flatten(), 0, A.size - 1)

    return B^


fn sort[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Sort the Matrix along the given axis.
    """
    if axis == 1:
        var B = A
        for i in range(B.shape[0]):
            _sort_inplace(
                B, left=i * B.strides[0], right=(i + 1) * B.strides[0] - 1
            )
        return B^
    elif axis == 0:
        var B = transpose(sort(transpose(A), axis=1))
        return B^
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn _partition(
    inout A: Matrix, left: Int, right: Int, pivot_index: Int
) raises -> Int:
    """
    Do partition for the data buffer of Matrix.

    Args:
        A: A Matrix.
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

    var store_index = left

    for i in range(left, right):
        if A._buf[i] < pivot_value:
            A._buf[store_index], A._buf[i] = A._buf[i], A._buf[store_index]
            store_index = store_index + 1

    A._buf[store_index], A._buf[right] = A._buf[right], A._buf[store_index]

    return store_index


fn _sort_inplace(inout A: Matrix, left: Int, right: Int) raises:
    """
    Sort in-place of the data buffer (quick-sort). It is not guaranteed to be stable.

    Args:
        A: A Matrix.
        left: Left index of the partition.
        right: Right index of the partition.
    """

    if right > left:
        var pivot_index = left + (right - left) // 2
        var pivot_new_index = _partition(A, left, right, pivot_index)
        _sort_inplace(A, left, pivot_new_index - 1)
        _sort_inplace(A, pivot_new_index + 1, right)


fn _quick_sort[
    dtype: DType
](A: Matrix[dtype], start: Int, end: Int) raises -> Matrix[dtype]:
    """
    Quick sort the selected range of the data buffer of the Matrix.
    Adopt in-place partition.
    Average complexity: O(nlogn).
    Worst-case complexity: O(n^2).
    Worst-case space complexity: O(n).
    Unstable.

    Args:
        A: A Matrix.
        start: The start index.
        end: The end index.

    """

    var B = A
    _sort_inplace(B, start, end)

    return B^
