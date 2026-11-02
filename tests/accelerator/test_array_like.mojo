# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator array-like tests
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Accelerator array-like tests (tests.accelerator.test_array_like).
=================================================================
Tests origin-preserving read and write pointers.

Exports
-------
- `test_cpu_pointers`: Tests low-level CPU pointer behavior.
- `test_owner_and_view_read_access`: Tests generic owner/view read access.
- `test_mutable_view_write_access`: Tests mutable view pointer access.
- `test_owner_copy_is_independent`: Tests owning copy semantics.
- `test_view_layout_access`: Tests transformed and strided view access.
- `test_mutable_transformed_view`: Tests transformed writable views.
- `test_zero_dimensional_view`: Tests scalar borrowed views.
- `test_common_numpy_metadata`: Tests the shared NumPy-like metadata API.
- `test_numpy_view_semantics`: Tests indexing and layout transformations.
- `test_numpy_index_errors`: Tests familiar bounds and rank errors.
- `main`: Runs all discovered tests.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.array_like import NDArrayLike
from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import AcceleratorNDArray, zeros
from numojo.core.layout.ndlayout import Shape


def test_cpu_pointers() raises:
    """Test full pointer and SIMD access through CPU arrays.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = AcceleratorNDArray[DType.float32, Device.CPU](Shape(4))
    for i in range(4):
        array.itemset(i, Float32(i + 1))

    assert_equal(
        array.unsafe_ptr().unsafe_load[width=4](0),
        SIMD[DType.float32, 4](1, 2, 3, 4),
    )

    array.unsafe_mut_ptr().unsafe_store[width=4](
        0, SIMD[DType.float32, 4](5, 6, 7, 8)
    )
    assert_equal(
        array.unsafe_ptr().unsafe_load[width=4](0),
        SIMD[DType.float32, 4](5, 6, 7, 8),
    )


def _load_first_four[
    A: NDArrayLike
](ref array: A) raises -> SIMD[A.ElementType, 4]:
    comptime assert A.DeviceType.type == "cpu", "test helper is CPU-only"
    return array.unsafe_ptr().unsafe_load[width=4](0)


def _assert_read_only[A: NDArrayLike](ref array: A):
    """Assert at compile time that an array-like value is read-only."""
    comptime assert not A.Writable, "expected a read-only array-like value"


def _store_first_four[A: NDArrayLike](mut array: A):
    """Store four lanes through the generic mutable-pointer contract."""
    comptime assert A.DeviceType.type == "cpu", "test helper is CPU-only"
    array.unsafe_mut_ptr().unsafe_store[width=4](
        0, SIMD[A.ElementType, 4](1, 2, 3, 4)
    )


def _assert_common_cpu_api[
    A: NDArrayLike
](
    ref array: A,
    expected_shape: List[Int],
    expected_strides: List[Int],
    expected_offset: Int,
    expected_c_contiguous: Bool,
    expected_f_contiguous: Bool,
    expected_owns_data: Bool,
    expected_writeable: Bool,
    expected_first: Scalar[A.ElementType],
) raises:
    """Check the metadata and scalar API shared by owners and views."""
    comptime assert A.DeviceType.type == "cpu", "test helper is CPU-only"
    assert_equal(array.ndim(), len(expected_shape))
    assert_equal(array.shape().tolist(), expected_shape)
    assert_equal(array.strides().tolist(), expected_strides)
    assert_equal(array.size(), array.num_elements())
    assert_equal(array.offset(), expected_offset)
    assert_equal(array.is_c_contiguous(), expected_c_contiguous)
    assert_equal(array.is_f_contiguous(), expected_f_contiguous)
    assert_equal(array.owns_data(), expected_owns_data)
    assert_equal(array.is_writeable(), expected_writeable)
    assert_true(array.is_cpu())
    assert_false(array.is_gpu())
    assert_equal(array.item(0), expected_first)

    var layout = array.layout()
    assert_equal(layout.shape_list(), expected_shape)
    assert_equal(layout.strides_list(), expected_strides)
    assert_equal(layout.offset, expected_offset)

    if len(expected_shape) == 0:
        assert_equal(len(array), 1)
    else:
        assert_equal(len(array), expected_shape[0])


def _last_matrix_item[
    A: NDArrayLike
](ref array: A,) raises -> Scalar[A.ElementType]:
    """Read a matrix corner through the generic variadic `item` contract."""
    comptime assert A.DeviceType.type == "cpu", "test helper is CPU-only"
    return array.item(-1, -1)


def test_owner_and_view_read_access() raises:
    """Test generic full-pointer reads from an owner and immutable view.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = AcceleratorNDArray[DType.float32, Device.CPU](Shape(4))
    for i in range(4):
        array.itemset(i, Float32(i + 1))
    var view = array.view()
    var expected = SIMD[DType.float32, 4](1, 2, 3, 4)
    _assert_read_only(view)
    assert_false(view.flags.WRITEABLE)
    assert_equal(
        rebind[SIMD[DType.float32, 4]](_load_first_four(array)), expected
    )
    assert_equal(
        rebind[SIMD[DType.float32, 4]](_load_first_four(view)), expected
    )


def test_mutable_view_write_access() raises:
    """Test mutation through a view pointer tied to its owner origin.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = zeros[DType.float32, Device.CPU](Shape(4))
    var view = array.view_mut()
    assert_true(view.flags.WRITEABLE)
    _store_first_four(view)
    assert_equal(view.item(3), 4.0)


def test_owner_copy_is_independent() raises:
    """Test that owner `copy()` duplicates storage rather than borrowing it.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var original = zeros[DType.float32, Device.CPU](Shape(2))
    original.itemset(0, 3.0)
    var copied = original.copy()
    copied.itemset(0, 9.0)
    assert_equal(original.item(0), 3.0)
    assert_equal(copied.item(0), 9.0)


def test_view_layout_access() raises:
    """Test transformed and strided views preserve logical indexing.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = AcceleratorNDArray[DType.float32, Device.CPU](Shape(2, 4))
    for i in range(8):
        array.itemset(i, Float32(i + 1))

    var view = array.view()
    var transposed = view.transpose()
    assert_equal(transposed.shape().tolist(), [4, 2])
    assert_equal(transposed.item(3, 1), 8.0)

    var every_other = view.reshape(Shape(8)).slice(0, 0, 4, 2)
    assert_equal(every_other.item(0), 1.0)
    assert_equal(every_other.item(1), 3.0)
    assert_equal(every_other.item(2), 5.0)
    assert_equal(every_other.item(3), 7.0)


def test_mutable_transformed_view() raises:
    """Test transformed writable views retain write capability.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = zeros[DType.float32, Device.CPU](Shape(2, 2))
    var writable = array.view_mut()
    var row = writable.select(0, 1)
    row.itemset(0, 11.0)
    row.itemset(1, 12.0)
    assert_equal(row.item(0), 11.0)
    assert_equal(row.item(1), 12.0)


def test_zero_dimensional_view() raises:
    """Test that rank-zero owner semantics are preserved by a view.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var scalar = zeros[DType.float32, Device.CPU](Shape())
    scalar.itemset(0, 7.0)
    var view = scalar.view()
    assert_equal(view.ndim(), 0)
    assert_equal(view.size(), 1)
    assert_equal(view[], 7.0)


def test_common_numpy_metadata() raises:
    """Test NumPy-like metadata through the common owner/view trait.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = AcceleratorNDArray[DType.float32, Device.CPU](Shape(2, 3))
    for i in range(array.size()):
        array.itemset(i, Float32(i + 1))

    _assert_common_cpu_api(
        array,
        [2, 3],
        [3, 1],
        0,
        True,
        False,
        True,
        True,
        1.0,
    )

    var transposed = array.view().transpose()
    _assert_common_cpu_api(
        transposed,
        [3, 2],
        [1, 3],
        0,
        False,
        True,
        False,
        False,
        1.0,
    )
    assert_equal(_last_matrix_item(transposed), 6.0)


def test_numpy_view_semantics() raises:
    """Test familiar selection, slicing, reshape, and negative indexing.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = AcceleratorNDArray[DType.float32, Device.CPU](Shape(2, 3))
    for i in range(array.size()):
        array.itemset(i, Float32(i + 1))

    var row = array.select(0, -1)
    _assert_common_cpu_api(row, [3], [1], 3, True, True, False, False, 4.0)
    assert_equal(row.item(-1), 6.0)

    var columns = array.slice(1, 1, 2)
    _assert_common_cpu_api(
        columns, [2, 2], [3, 1], 1, False, False, False, False, 2.0
    )
    assert_equal(columns.item(-1, -1), 6.0)

    var reshaped = array.reshape(Shape(3, -1))
    assert_equal(reshaped.shape().tolist(), [3, 2])
    assert_equal(reshaped.strides().tolist(), [2, 1])
    assert_equal(reshaped.item(-1, -1), 6.0)

    var writable = array.view_mut()
    assert_false(writable.owns_data())
    assert_true(writable.is_writeable())
    writable.itemset(-1, 9.0)
    assert_equal(writable.item(-1), 9.0)


def test_numpy_index_errors() raises:
    """Test that invalid indices and shapes raise instead of accessing memory.

    Raises:
        NumojoError: If test setup unexpectedly fails outside an assertion.

    Examples:
        Run this test through `pixi run test_accelerator`.
    """
    var array = zeros[DType.float32, Device.CPU](Shape(2, 3))
    var view = array.view()
    with assert_raises():
        _ = view.item(6)
    with assert_raises():
        _ = view.item(2, 0)
    with assert_raises():
        _ = view.item(0, 0, 0)
    with assert_raises():
        _ = view.slice(1, 2, 2)
    with assert_raises():
        _ = view.reshape(Shape(4, 2))


def main() raises:
    """Run all tests in this module.

    Examples:
        Run this file directly with `pixi run mojo run`.
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
