# ===----------------------------------------------------------------------=== #
# NuMojo: Broadcasting and layout manipulation tests
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Broadcasting tests (tests.routines.test_broadcasting).
======================================================
NumPy differential tests for broadcasting and layout manipulation routines.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.python import Python
from std.testing import TestSuite, assert_equal, assert_raises

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
import numojo as nm
from numojo import Shape
from numojo.routines.creation import _0darray

# ===----------------------------------------------------------------------=== #
# Test utilities
# ===----------------------------------------------------------------------=== #
from utils_for_test import check


def test_broadcast_shapes() raises:
    """Tests broadcast shape calculation against NumPy."""
    var np = Python.import_module("numpy")
    var actual = nm.broadcast_shapes(Shape(5, 1), Shape(1, 6), Shape(6))
    var expected = np.broadcast_shapes(
        Python.tuple(5, 1), Python.tuple(1, 6), Python.tuple(6)
    )

    assert_equal(actual.ndim, Int(expected.__len__()))
    for i in range(actual.ndim):
        assert_equal(actual[i], Int(py=expected[i]))

    var scalar_shape = nm.broadcast_shapes()
    var scalar_expected = np.broadcast_shapes()
    assert_equal(scalar_shape.ndim, Int(scalar_expected.__len__()))

    with assert_raises():
        _ = nm.broadcast_shapes(Shape(2, 3), Shape(4))


def test_broadcast_arrays() raises:
    """Tests array broadcasting for F-order, sliced, and empty inputs."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.i32](0, 3).reshape(Shape(1, 3), order="F")
    var parent = nm.arange[nm.i32](0, 12).reshape(Shape(4, 1, 3))
    var sliced = parent[Slice(0, 4, 2), Slice(0, 1), Slice(0, 3)]
    var actual = nm.broadcast_arrays(a, sliced)
    var expected = np.broadcast_arrays(a.to_numpy(), sliced.to_numpy())

    check(actual[0], expected[0], "`broadcast_arrays` first result fails.")
    check(actual[1], expected[1], "`broadcast_arrays` second result fails.")

    var empty = nm.empty[nm.i32](Shape(0, 1))
    var row = nm.arange[nm.i32](0, 3).reshape(Shape(1, 3))
    var empty_actual = nm.broadcast_arrays(empty, row)
    var empty_expected = np.broadcast_arrays(empty.to_numpy(), row.to_numpy())
    check(
        empty_actual[0],
        empty_expected[0],
        "`broadcast_arrays` empty first result fails.",
    )
    check(
        empty_actual[1],
        empty_expected[1],
        "`broadcast_arrays` empty second result fails.",
    )

    var none = nm.broadcast_arrays[nm.i32]()
    assert_equal(len(none), Int(np.broadcast_arrays().__len__()))

    with assert_raises():
        _ = nm.broadcast_arrays(
            nm.ones[nm.i32](Shape(2, 3)),
            nm.ones[nm.i32](Shape(4)),
        )


def test_squeeze() raises:
    """Tests singleton removal for axes, scalars, and empty arrays."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.i32](0, 6).reshape(Shape(1, 2, 1, 3))
    var anp = a.to_numpy()

    var squeezed = nm.squeeze(a)
    check(squeezed, np.squeeze(anp), "`squeeze` without axes fails.")
    var squeezed_negative = nm.squeeze(a, -2)
    check(
        squeezed_negative,
        np.squeeze(anp, axis=-2),
        "`squeeze` with a negative axis fails.",
    )
    var axes: List[Int] = [0, -2]
    var squeezed_axes = nm.squeeze(a, axes)
    check(
        squeezed_axes,
        np.squeeze(anp, axis=Python.tuple(0, -2)),
        "`squeeze` with multiple axes fails.",
    )

    var scalar = _0darray[nm.i32](7)
    var squeezed_scalar = nm.squeeze(scalar)
    check(
        squeezed_scalar,
        np.squeeze(scalar.to_numpy()),
        "`squeeze` on a 0-D array fails.",
    )

    var empty = nm.empty[nm.i32](Shape(1, 0, 1))
    var squeezed_empty = nm.squeeze(empty)
    check(
        squeezed_empty,
        np.squeeze(empty.to_numpy()),
        "`squeeze` on an empty array fails.",
    )

    with assert_raises():
        _ = nm.squeeze(a, 1)
    var duplicate_axes: List[Int] = [0, 0]
    with assert_raises():
        _ = nm.squeeze(a, duplicate_axes)


def test_permute_dims() raises:
    """Tests axis permutations for C-order, F-order, and sliced arrays."""
    var np = Python.import_module("numpy")
    var c = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var axes: List[Int] = [-1, 0, 1]
    var c_actual = nm.permute_dims(c, axes)
    check(
        c_actual,
        np.permute_dims(c.to_numpy(), Python.tuple(-1, 0, 1)),
        "`permute_dims` on a C-order array fails.",
    )

    var f = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4), order="F")
    var f_actual = nm.permute_dims(f, axes)
    check(
        f_actual,
        np.permute_dims(f.to_numpy(), Python.tuple(-1, 0, 1)),
        "`permute_dims` on an F-order array fails.",
    )

    var sliced = c[Slice(0, 2), Slice(0, 3, 2), Slice(0, 4)]
    var sliced_actual = nm.permute_dims(sliced, axes)
    check(
        sliced_actual,
        np.permute_dims(sliced.to_numpy(), Python.tuple(-1, 0, 1)),
        "`permute_dims` on a sliced view fails.",
    )

    var duplicate_axes: List[Int] = [0, 0, 1]
    with assert_raises():
        _ = nm.permute_dims(c, duplicate_axes)


def test_matrix_transpose() raises:
    """Tests matrix transpose for F-order and empty arrays."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4), order="F")
    var actual = nm.matrix_transpose(a)
    check(
        actual,
        np.matrix_transpose(a.to_numpy()),
        "`matrix_transpose` on an F-order array fails.",
    )

    var empty = nm.empty[nm.i32](Shape(2, 0, 3))
    var empty_actual = nm.matrix_transpose(empty)
    check(
        empty_actual,
        np.matrix_transpose(empty.to_numpy()),
        "`matrix_transpose` on an empty array fails.",
    )

    with assert_raises():
        _ = nm.matrix_transpose(nm.arange[nm.i32](0, 3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
