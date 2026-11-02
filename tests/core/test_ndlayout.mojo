# ===----------------------------------------------------------------------=== #
# NuMojo: NDLayout tests
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
NDLayout tests (tests.core.test_ndlayout).
==========================================
Tests compact inline and spilled array layout metadata.

Exports
-------
- `test_scalar_layout`, `test_contiguous_layouts`, `test_positions`.
- `test_inline_boundary_and_spill`, `test_explicit_strided_layout`.
- `test_shape_and_strides_access`: Tests compact metadata access.
- `test_slice_and_select`, `test_transpose`, `test_reshape`.
- `test_mixed_sliced_indices`: Tests integers, slices, ellipsis, and new axes.
- `test_invalid_layout_inputs`: Validates rejected metadata.
- `main`: Runs the discovered tests.
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
from numojo.core.layout.array_methods import newaxis
from numojo.core.layout.ndlayout import (
    INLINE_LAYOUT_RANK,
    LayoutIndex,
    NDLayout,
    Shape,
    Strides,
)


def test_scalar_layout() raises:
    """Tests the size-one, empty-metadata representation of a 0-D array."""
    var unallocated = NDLayout()
    assert_equal(len(unallocated), 0)
    assert_equal(unallocated.size, 0)

    var scalar = NDLayout(Shape())
    assert_equal(len(scalar), 0)
    assert_equal(scalar.size, 1)
    assert_equal(scalar.offset, 0)
    assert_equal(len(scalar.shape()), 0)
    assert_equal(len(scalar.strides()), 0)
    assert_true(scalar.is_inline())
    assert_true(scalar.is_c_contiguous())
    assert_true(scalar.is_f_contiguous())


def test_contiguous_layouts() raises:
    """Tests shape, strides, size, offset, and contiguity."""
    var c_layout = NDLayout(Shape(2, 3, 4))
    assert_equal(len(c_layout), 3)
    assert_equal(c_layout.size, 24)
    assert_equal(c_layout.offset, 0)
    assert_equal(c_layout.shape_at(0), 2)
    assert_equal(c_layout.shape_at(1), 3)
    assert_equal(c_layout.shape_at(2), 4)
    assert_equal(c_layout.stride_at(0), 12)
    assert_equal(c_layout.stride_at(1), 4)
    assert_equal(c_layout.stride_at(2), 1)
    assert_true(c_layout.is_c_contiguous())
    assert_true(c_layout.contiguous())
    assert_false(c_layout.is_f_contiguous())

    var f_layout = NDLayout(Shape(2, 3, 4), order="F")
    assert_equal(f_layout.stride_at(0), 1)
    assert_equal(f_layout.stride_at(1), 2)
    assert_equal(f_layout.stride_at(2), 6)
    assert_false(f_layout.is_c_contiguous())
    assert_true(f_layout.is_f_contiguous())


def test_positions() raises:
    """Tests coordinate and C-order flat index to storage offset mapping."""
    var layout = NDLayout(Shape(2, 3), Strides(10, 2), offset=5)
    assert_equal(layout.position([0, 0]), 5)
    assert_equal(layout.position([1, 2]), 19)
    assert_equal(layout.position([-1, -1]), 19)
    assert_equal(layout.flat_position(4), 17)
    assert_equal(layout.flat_position(-1), 19)

    var scalar = NDLayout(Shape(), Strides(), offset=7)
    assert_equal(scalar.position([]), 7)
    assert_equal(scalar.flat_position(0), 7)

    with assert_raises():
        _ = layout.position([1])
    with assert_raises():
        _ = layout.position([2, 0])
    with assert_raises():
        _ = layout.flat_position(6)
    with assert_raises():
        _ = NDLayout([0, 3]).flat_position(0)


def test_inline_boundary_and_spill() raises:
    """Tests the rank-eight inline boundary and higher-rank spill path."""
    var inline_shape = List[Int](length=INLINE_LAYOUT_RANK, fill=1)
    var inline_layout = NDLayout(inline_shape)
    assert_true(inline_layout.is_inline())

    var spilled_shape = List[Int](length=INLINE_LAYOUT_RANK + 1, fill=1)
    spilled_shape[INLINE_LAYOUT_RANK] = 2
    var spilled_layout = NDLayout(spilled_shape)
    assert_false(spilled_layout.is_inline())
    assert_equal(len(spilled_layout), INLINE_LAYOUT_RANK + 1)
    assert_equal(spilled_layout.size, 2)
    assert_equal(spilled_layout.shape_at(INLINE_LAYOUT_RANK), 2)
    assert_equal(spilled_layout.stride_at(INLINE_LAYOUT_RANK), 1)


def test_explicit_strided_layout() raises:
    """Tests explicit non-contiguous and negative-stride metadata."""
    var layout = NDLayout(Shape(2, 3), Strides(6, -2), offset=5)
    assert_equal(layout.offset, 5)
    assert_equal(layout.size, 6)
    assert_equal(layout.stride_at(0), 6)
    assert_equal(layout.stride_at(1), -2)
    assert_false(layout.is_c_contiguous())
    assert_false(layout.is_f_contiguous())

    var strided_1d = NDLayout(Shape(4), Strides(2), offset=0)
    assert_false(strided_1d.is_c_contiguous())
    assert_false(strided_1d.is_f_contiguous())


def test_slice_and_select() raises:
    """Tests axis slicing, negative steps, empty slices, and selection."""
    var layout = NDLayout([4, 5])
    var sliced = layout.slice(1, 1, 2, 2)
    assert_equal(sliced.shape_list(), [4, 2])
    assert_equal(sliced.strides_list(), [5, 2])
    assert_equal(sliced.offset, 1)
    assert_equal(sliced.size, 8)

    var reversed = layout.slice(-1, 4, 3, -2)
    assert_equal(reversed.shape_list(), [4, 3])
    assert_equal(reversed.strides_list(), [5, -2])
    assert_equal(reversed.offset, 4)

    var empty = layout.slice(0, 0, 0)
    assert_equal(empty.shape_list(), [0, 5])
    assert_equal(empty.strides_list(), [0, 1])
    assert_equal(empty.offset, 0)
    assert_equal(empty.size, 0)

    var selected = NDLayout([2, 3, 4]).select(1, 2)
    assert_equal(selected.shape_list(), [2, 4])
    assert_equal(selected.strides_list(), [12, 1])
    assert_equal(selected.offset, 8)
    assert_equal(selected.size, 8)

    var scalar = NDLayout([3]).select(0, -1)
    assert_equal(len(scalar), 0)
    assert_equal(scalar.size, 1)
    assert_equal(scalar.offset, 2)

    with assert_raises():
        _ = layout.slice(2, 0, 1)
    with assert_raises():
        _ = layout.slice(1, 0, -1)
    with assert_raises():
        _ = layout.slice(1, 0, 1, 0)
    with assert_raises():
        _ = layout.slice(1, 4, 2, 1)
    with assert_raises():
        _ = layout.select(0, 4)


def test_transpose() raises:
    """Tests explicit, negative-axis, and reversed-axis permutations."""
    var layout = NDLayout([2, 3, 4])
    var transposed = layout.transpose([2, 0, 1])
    assert_equal(transposed.shape_list(), [4, 2, 3])
    assert_equal(transposed.strides_list(), [1, 12, 4])
    assert_equal(transposed.size, 24)

    var negative_axes = layout.transpose([-1, 0, 1])
    assert_equal(negative_axes.shape_list(), [4, 2, 3])
    assert_equal(negative_axes.strides_list(), [1, 12, 4])

    var reversed = layout.transpose()
    assert_equal(reversed.shape_list(), [4, 3, 2])
    assert_equal(reversed.strides_list(), [1, 4, 12])

    with assert_raises():
        _ = layout.transpose([0, 1])
    with assert_raises():
        _ = layout.transpose([0, 0, 2])
    with assert_raises():
        _ = layout.transpose([0, 1, 3])


def test_reshape() raises:
    """Tests compatible reshape, inferred dimensions, and scalar shapes."""
    var layout = NDLayout(Shape(2, 3), Strides(3, 1), offset=5)
    var reshaped = layout.reshape(Shape(3, 2))
    assert_equal(reshaped.shape_list(), [3, 2])
    assert_equal(reshaped.strides_list(), [2, 1])
    assert_equal(reshaped.offset, 5)

    var inferred = layout.reshape([-1, 2])
    assert_equal(inferred.shape_list(), [3, 2])

    var scalar = NDLayout(Shape()).reshape([1])
    assert_equal(scalar.shape_list(), [1])
    assert_equal(scalar.size, 1)
    var scalar_again = scalar.reshape(Shape())
    assert_equal(len(scalar_again), 0)
    assert_equal(scalar_again.size, 1)

    with assert_raises():
        _ = layout.transpose().reshape([6])
    with assert_raises():
        _ = layout.reshape([4, 2])
    with assert_raises():
        _ = layout.reshape([-1, -1])
    with assert_raises():
        _ = NDLayout([0, 3]).reshape([-1, 0])


def test_mixed_sliced_indices() raises:
    """Tests mixed layout slicing with omitted axes and axis insertion."""
    var layout = NDLayout([2, 3, 4])
    var mixed = layout.sliced(1, Slice(None, None, -1))
    assert_equal(mixed.shape_list(), [3, 4])
    assert_equal(mixed.strides_list(), [-4, 1])
    assert_equal(mixed.offset, 20)

    var expanded = layout.sliced(
        LayoutIndex(newaxis), LayoutIndex(...), LayoutIndex(2)
    )
    assert_equal(expanded.shape_list(), [1, 2, 3])
    assert_equal(expanded.strides_list(), [0, 12, 4])
    assert_equal(expanded.offset, 2)

    with assert_raises():
        _ = layout.sliced(LayoutIndex(...), LayoutIndex(...))
    with assert_raises():
        _ = layout.sliced(0, 0, 0, 0)
    with assert_raises():
        _ = layout.sliced(Slice(None, None, 0))


def test_shape_and_strides_access() raises:
    """Tests access through the compact shape and stride containers."""
    var layout = NDLayout(Shape(2, 3))
    var shape = layout.shape()
    var strides = layout.strides()
    assert_equal(shape.tolist(), [2, 3])
    assert_equal(strides.tolist(), [3, 1])


def test_invalid_layout_inputs() raises:
    """Tests order and explicit-rank validation."""
    with assert_raises():
        _ = NDLayout(Shape(2, 3), order="A")
    with assert_raises():
        _ = NDLayout(Shape(2, 3), Strides(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
