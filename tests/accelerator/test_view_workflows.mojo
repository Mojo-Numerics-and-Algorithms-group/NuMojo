# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator owner/view workflow tests
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Accelerator owner/view workflow tests (tests.accelerator.test_view_workflows).
==============================================================================
Tests NumPy-style workflows across owners and borrowed views.

Programs the compiler must reject live in `tests/accelerator/compile_fail/`.

Exports
-------
- `test_return_slice_from_function`: Tests returning a borrowed view.
- `test_store_view_in_struct`: Tests a struct holding a view.
- `test_repeated_slicing_in_loop`: Tests `x = x[::2]`-style reslicing.
- `test_generic_function_accepts_owner_and_views`: Tests `NDArrayLike` callers.
- `test_write_through_mutable_view_reaches_owner`: Tests shared storage writes.
- `test_owner_write_is_visible_through_view`: Tests views observe owner writes.
- `main`: Runs all discovered tests.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.testing import assert_equal, TestSuite

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.array_like import NDArrayLike
from numojo.core.accelerator.device import Device
from numojo.core.accelerator.ndarray_view import AcceleratorNDArrayView
from numojo.core.accelerator_ndarray import AcceleratorNDArray, zeros
from numojo.core.layout.ndlayout import Shape


def _arange_owner(
    shape: Shape,
) raises -> AcceleratorNDArray[DType.float32, Device.CPU]:
    """Return a C-ordered CPU owner holding `0, 1, ..., size - 1`."""
    var array = zeros[DType.float32, Device.CPU](shape)
    for i in range(array.size()):
        array.itemset(i, Float32(i))
    return array^


def _first_row[
    o: Origin
](
    ref[o] array: AcceleratorNDArray[DType.float32, Device.CPU]
) raises -> AcceleratorNDArrayView[DType.float32, Device.CPU, ImmOrigin(o)]:
    """Return the first row of `array` as a view tied to its borrow."""
    return array.select(0, 0)


struct _RowHolder[o: Origin[mut=False]](Movable):
    """A user-defined struct that stores a read-only view."""

    var row: AcceleratorNDArrayView[DType.float32, Device.CPU, Self.o]

    def __init__(
        out self,
        var row: AcceleratorNDArrayView[DType.float32, Device.CPU, Self.o],
    ):
        self.row = row^


def _total[A: NDArrayLike](ref array: A) raises -> Float64:
    """Sum every logical element through the generic `NDArrayLike` API."""
    comptime assert A.DeviceType.type == "cpu", "test helper is CPU-only"
    var total: Float64 = 0
    for i in range(array.size()):
        total += Float64(array.item(i))
    return total


def test_return_slice_from_function() raises:
    """Tests that a function can return a view of its argument."""
    var matrix = _arange_owner(Shape(2, 3))
    var row = _first_row(matrix)
    assert_equal(row.shape().tolist(), [3])
    assert_equal(row.item(2), 2.0)


def test_store_view_in_struct() raises:
    """Tests that a struct parameterized by origin can hold a view."""
    var array = _arange_owner(Shape(4))
    var holder = _RowHolder(array.view())
    assert_equal(holder.row.size(), 4)
    assert_equal(holder.row.item(3), 3.0)


def test_repeated_slicing_in_loop() raises:
    """Tests rebinding a view to a slice of itself, like `x = x[::2]`."""
    var array = _arange_owner(Shape(8))
    var view = array.view()
    for _ in range(2):
        view = view.slice(0, 0, (view.size() + 1) // 2, 2)
    assert_equal(view.size(), 2)
    assert_equal(view.item(0), 0.0)
    assert_equal(view.item(1), 4.0)


def test_generic_function_accepts_owner_and_views() raises:
    """Tests that one generic function accepts owners and views."""
    var matrix = _arange_owner(Shape(2, 2))
    assert_equal(_total(matrix), 6.0)
    assert_equal(_total(matrix.view()), 6.0)
    assert_equal(_total(matrix.view().transpose()), 6.0)
    assert_equal(_total(matrix.select(0, 1)), 5.0)


def test_write_through_mutable_view_reaches_owner() raises:
    """Tests that writes through a transformed mutable view reach the owner."""
    var matrix = zeros[DType.float32, Device.CPU](Shape(2, 3))
    var writable = matrix.view_mut()
    var column = writable.transpose().select(0, 1)
    column.itemset(0, 5.0)
    column.itemset(1, 6.0)
    assert_equal(matrix.item(0, 1), 5.0)
    assert_equal(matrix.item(1, 1), 6.0)


def test_owner_write_is_visible_through_view() raises:
    """Tests that a live read-only view observes later writes to its owner.

    A read-only view forbids writes through itself; it does not freeze the
    storage it borrows.
    """
    var array = zeros[DType.float32, Device.CPU](Shape(2))
    var view = array.view()
    array.itemset(0, 5.0)
    assert_equal(view.item(0), 5.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
