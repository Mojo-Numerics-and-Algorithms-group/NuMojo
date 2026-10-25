# ===----------------------------------------------------------------------=== #
# NuMojo: Vector product tests
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Vector product tests (tests.routines.test_vector_products).
===========================================================
NumPy differential tests for vector and matrix-vector products.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.python import Python
from std.testing import TestSuite, assert_raises

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
import numojo as nm
from numojo.prelude import *
from numojo.routines.creation import _0darray

# ===----------------------------------------------------------------------=== #
# Test utilities
# ===----------------------------------------------------------------------=== #
from utils_for_test import check_is_close


def test_inner() raises:
    """Tests scalar, multidimensional, strided, and empty inner products."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](24).reshape(Shape(2, 3, 4))
    var b = nm.arange[nm.f64](20).reshape(Shape(5, 4), order="F")
    check_is_close(
        nm.inner(a, b),
        np.inner(a.to_numpy(), b.to_numpy()),
        "`inner` with multidimensional inputs fails",
    )

    var scalar = _0darray[nm.f64](2.0)
    check_is_close(
        nm.inner(scalar, a),
        np.inner(scalar.to_numpy(), a.to_numpy()),
        "`inner` with a scalar fails",
    )
    var vector = nm.arange[nm.f64](4)
    check_is_close(
        nm.inner(vector, vector),
        np.inner(vector.to_numpy(), vector.to_numpy()),
        "`inner` with vectors fails",
    )
    check_is_close(
        nm.inner(scalar, scalar),
        np.inner(scalar.to_numpy(), scalar.to_numpy()),
        "`inner` with two scalars fails",
    )

    var sliced = a[Slice(0, 2), Slice(0, 3), Slice(0, 4, 2)]
    var sliced_rhs = nm.arange[nm.f64](10).reshape(Shape(5, 2))
    check_is_close(
        nm.inner(sliced, sliced_rhs),
        np.inner(sliced.to_numpy(), sliced_rhs.to_numpy()),
        "`inner` with a sliced input fails",
    )

    var empty1 = nm.empty[nm.f64](Shape(2, 0))
    var empty2 = nm.empty[nm.f64](Shape(3, 0))
    check_is_close(
        nm.inner(empty1, empty2),
        np.inner(np.empty(Python.tuple(2, 0)), np.empty(Python.tuple(3, 0))),
        "`inner` with an empty contraction fails",
    )

    with assert_raises():
        _ = nm.inner(nm.ones[nm.f64](Shape(2)), nm.ones[nm.f64](Shape(3)))


def test_vdot() raises:
    """Tests flattened products for F-order, sliced, and empty inputs."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](12).reshape(Shape(3, 4), order="F")
    var parent = nm.arange[nm.f64](24).reshape(Shape(3, 8))
    var b = parent[Slice(0, 3), Slice(0, 8, 2)]
    check_is_close(
        nm.vdot(a, b),
        np.vdot(a.to_numpy(), b.to_numpy()),
        "`vdot` with F-order and sliced inputs fails",
    )

    var empty1 = nm.empty[nm.f64](Shape(2, 0))
    var empty2 = nm.empty[nm.f64](Shape(0, 3))
    check_is_close(
        nm.vdot(empty1, empty2),
        np.vdot(np.empty(Python.tuple(2, 0)), np.empty(Python.tuple(0, 3))),
        "`vdot` with empty inputs fails",
    )

    with assert_raises():
        _ = nm.vdot(nm.ones[nm.f64](Shape(2)), nm.ones[nm.f64](Shape(3)))


def test_vecdot() raises:
    """Tests vector contractions with broadcasting and custom axes."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](6).reshape(Shape(2, 1, 3))
    var b = nm.arange[nm.f64](12).reshape(Shape(1, 4, 3), order="F")
    check_is_close(
        nm.vecdot(a, b),
        np.vecdot(a.to_numpy(), b.to_numpy()),
        "`vecdot` batch broadcasting fails",
    )
    var vector = nm.arange[nm.f64](5)
    check_is_close(
        nm.vecdot(vector, vector),
        np.vecdot(vector.to_numpy(), vector.to_numpy()),
        "`vecdot` with vectors fails",
    )

    var axis_a = nm.arange[nm.f64](24).reshape(Shape(2, 3, 4))
    var axis_b = nm.arange[nm.f64](12).reshape(Shape(1, 3, 4))
    check_is_close(
        nm.vecdot(axis_a, axis_b, axis=1),
        np.vecdot(axis_a.to_numpy(), axis_b.to_numpy(), axis=1),
        "`vecdot` with a custom axis fails",
    )

    var parent = nm.arange[nm.f64](48).reshape(Shape(2, 4, 6))
    var sliced = parent[Slice(0, 2), Slice(0, 4), Slice(0, 6, 2)]
    var sliced_rhs = nm.arange[nm.f64](12).reshape(Shape(1, 4, 3))
    check_is_close(
        nm.vecdot(sliced, sliced_rhs),
        np.vecdot(sliced.to_numpy(), sliced_rhs.to_numpy()),
        "`vecdot` with a sliced input fails",
    )

    var empty1 = nm.empty[nm.f64](Shape(2, 0))
    var empty2 = nm.empty[nm.f64](Shape(1, 0))
    check_is_close(
        nm.vecdot(empty1, empty2),
        np.vecdot(np.empty(Python.tuple(2, 0)), np.empty(Python.tuple(1, 0))),
        "`vecdot` with an empty core dimension fails",
    )

    with assert_raises():
        _ = nm.vecdot(
            nm.ones[nm.f64](Shape(2, 3)),
            nm.ones[nm.f64](Shape(4, 3)),
        )
    with assert_raises():
        _ = nm.vecdot(a, b, axis=3)


def test_matvec() raises:
    """Tests matrix-vector products with broadcast batches and layouts."""
    var np = Python.import_module("numpy")
    var matrix = nm.arange[nm.f64](24).reshape(Shape(2, 1, 3, 4))
    var vector = nm.arange[nm.f64](20).reshape(Shape(1, 5, 4))
    check_is_close(
        nm.matvec(matrix, vector),
        np.matvec(matrix.to_numpy(), vector.to_numpy()),
        "`matvec` batch broadcasting fails",
    )

    var matrix_f = nm.arange[nm.f64](12).reshape(Shape(3, 4), order="F")
    var vector_f = nm.arange[nm.f64](4)
    check_is_close(
        nm.matvec(matrix_f, vector_f),
        np.matvec(matrix_f.to_numpy(), vector_f.to_numpy()),
        "`matvec` with an F-order matrix fails",
    )

    var parent = nm.arange[nm.f64](24).reshape(Shape(3, 8))
    var sliced = parent[Slice(0, 3), Slice(0, 8, 2)]
    check_is_close(
        nm.matvec(sliced, vector_f),
        np.matvec(sliced.to_numpy(), vector_f.to_numpy()),
        "`matvec` with a sliced matrix fails",
    )

    var empty_matrix = nm.empty[nm.f64](Shape(2, 3, 0))
    var empty_vector = nm.empty[nm.f64](Shape(1, 0))
    check_is_close(
        nm.matvec(empty_matrix, empty_vector),
        np.matvec(
            np.empty(Python.tuple(2, 3, 0)), np.empty(Python.tuple(1, 0))
        ),
        "`matvec` with an empty core dimension fails",
    )

    with assert_raises():
        _ = nm.matvec(matrix_f, nm.ones[nm.f64](Shape(5)))


def test_vecmat() raises:
    """Tests vector-matrix products with broadcast batches and layouts."""
    var np = Python.import_module("numpy")
    var vector = nm.arange[nm.f64](6).reshape(Shape(2, 1, 3))
    var matrix = nm.arange[nm.f64](60).reshape(Shape(1, 4, 3, 5))
    check_is_close(
        nm.vecmat(vector, matrix),
        np.vecmat(vector.to_numpy(), matrix.to_numpy()),
        "`vecmat` batch broadcasting fails",
    )

    var vector_f = nm.arange[nm.f64](3)
    var matrix_f = nm.arange[nm.f64](15).reshape(Shape(3, 5), order="F")
    check_is_close(
        nm.vecmat(vector_f, matrix_f),
        np.vecmat(vector_f.to_numpy(), matrix_f.to_numpy()),
        "`vecmat` with an F-order matrix fails",
    )

    var parent = nm.arange[nm.f64](30).reshape(Shape(3, 10))
    var sliced = parent[Slice(0, 3), Slice(0, 10, 2)]
    check_is_close(
        nm.vecmat(vector_f, sliced),
        np.vecmat(vector_f.to_numpy(), sliced.to_numpy()),
        "`vecmat` with a sliced matrix fails",
    )

    var empty_vector = nm.empty[nm.f64](Shape(2, 0))
    var empty_matrix = nm.empty[nm.f64](Shape(1, 0, 3))
    check_is_close(
        nm.vecmat(empty_vector, empty_matrix),
        np.vecmat(
            np.empty(Python.tuple(2, 0)), np.empty(Python.tuple(1, 0, 3))
        ),
        "`vecmat` with an empty core dimension fails",
    )

    with assert_raises():
        _ = nm.vecmat(vector_f, nm.ones[nm.f64](Shape(4, 5)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
