# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Test indexing module `numojo.routines.indexing`.
"""
from python import Python
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close

from numojo.prelude import *


fn test_compress() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[i8](24).reshape(Shape(2, 3, 4))
    var anp = a.to_numpy()
    var b = nm.arange[i8](6)
    var bnp = b.to_numpy()

    check(
        numojo.indexing.compress(nm.array[boolean]("[0, 1, 1, 0]"), b),
        np.compress(np.array([0, 1, 1, 0]), bnp),
        "`compress` 1-d array is broken",
    )
    check(
        numojo.indexing.compress(
            nm.array[boolean]("[0,1,1,0,1,0,1,0,1,1,1]"), a
        ),
        np.compress(np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]), anp),
        "`compress` 3-d array without `axis` is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[0, 1]"), a, axis=0),
        np.compress(np.array([0, 1]), anp, axis=0),
        "`compress` 3-d array by axis 0 is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[1]"), a, axis=0),
        np.compress(np.array([1]), anp, axis=0),
        "`compress` 3-d array by axis 0 is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[1, 0, 1]"), a, axis=1),
        np.compress(np.array([1, 0, 1]), anp, axis=1),
        "`compress` 3-d array by axis 1 is broken (#1)",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[0, 1]"), a, axis=1),
        np.compress(np.array([0, 1]), anp, axis=1),
        "`compress` 3-d array by axis 1 is broken (#2)",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[1, 0, 1, 1]"), a, axis=2),
        np.compress(np.array([1, 0, 1, 1]), anp, axis=2),
        "`compress` 3-d array by axis 2 is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[0, 1]"), a, axis=2),
        np.compress(np.array([0, 1]), anp, axis=2),
        "`compress` 3-d array by axis 2 is broken",
    )


fn test_take_along_axis() raises:
    var np = Python.import_module("numpy")

    # Test 1-D array
    var a1d = nm.arange[i8](10)
    var a1d_np = a1d.to_numpy()
    var indices1d = nm.array[intp]("[2, 3, 1, 8]")
    var indices1d_np = indices1d.to_numpy()

    check(
        nm.indexing.take_along_axis(a1d, indices1d, axis=0),
        np.take_along_axis(a1d_np, indices1d_np, axis=0),
        "`take_along_axis` with 1-D array is broken",
    )

    # Test 2-D array with axis=0
    var a2d = nm.arange[i8](12).reshape(Shape(3, 4))
    var a2d_np = a2d.to_numpy()
    var indices2d_0 = nm.array[intp]("[[0, 1, 2, 0], [1, 2, 0, 1]]")
    var indices2d_0_np = indices2d_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a2d, indices2d_0, axis=0),
        np.take_along_axis(a2d_np, indices2d_0_np, axis=0),
        "`take_along_axis` with 2-D array on axis=0 is broken",
    )

    # Test 2-D array with axis=1
    var indices2d_1 = nm.array[intp](
        "[[3, 0, 2, 1], [1, 3, 0, 0], [2, 1, 0, 3]]"
    )
    var indices2d_1_np = indices2d_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a2d, indices2d_1, axis=1),
        np.take_along_axis(a2d_np, indices2d_1_np, axis=1),
        "`take_along_axis` with 2-D array on axis=1 is broken",
    )

    # Test 3-D array
    var a3d = nm.arange[i8](24).reshape(Shape(2, 3, 4))
    var a3d_np = a3d.to_numpy()

    # Test with axis=0
    var indices3d_0 = nm.zeros[intp](Shape(1, 3, 4))
    var indices3d_0_np = indices3d_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d, indices3d_0, axis=0),
        np.take_along_axis(a3d_np, indices3d_0_np, axis=0),
        "`take_along_axis` with 3-D array on axis=0 is broken",
    )

    # Test with axis=1
    var indices3d_1 = nm.array[intp](
        "[[[0, 1, 0, 2], [2, 1, 0, 1], [1, 2, 2, 0]], [[1, 0, 1, 2], [0, 2, 1,"
        " 0], [2, 0, 0, 1]]]"
    )
    var indices3d_1_np = indices3d_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d, indices3d_1, axis=1),
        np.take_along_axis(a3d_np, indices3d_1_np, axis=1),
        "`take_along_axis` with 3-D array on axis=1 is broken",
    )

    # Test with axis=2
    var indices3d_2 = nm.array[intp](
        "[[[2, 0, 3, 1], [1, 3, 0, 2], [3, 1, 2, 0]], [[0, 2, 1, 3], [2, 0, 3,"
        " 1], [1, 3, 0, 2]]]"
    )
    var indices3d_2_np = indices3d_2.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d, indices3d_2, axis=2),
        np.take_along_axis(a3d_np, indices3d_2_np, axis=2),
        "`take_along_axis` with 3-D array on axis=2 is broken",
    )

    # Test with negative axis
    check(
        nm.indexing.take_along_axis(a3d, indices3d_2, axis=-1),
        np.take_along_axis(a3d_np, indices3d_2_np, axis=-1),
        "`take_along_axis` with negative axis is broken",
    )


fn test_take_along_axis_fortran_order() raises:
    var np = Python.import_module("numpy")

    # Create 3-D F-order array
    var a3d_f = nm.arange[i8](24).reshape(Shape(2, 3, 4), order="F")
    var a3d_f_np = a3d_f.to_numpy()

    # Test with axis=0
    var indices3d_0 = nm.zeros[intp](Shape(1, 3, 4))
    var indices3d_0_np = indices3d_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, indices3d_0, axis=0),
        np.take_along_axis(a3d_f_np, indices3d_0_np, axis=0),
        "`take_along_axis` with 3-D F-order array on axis=0 is broken",
    )

    # Test with axis=1
    var indices3d_1 = nm.array[intp](
        "[[[0, 1, 0, 2], [2, 1, 0, 1], [1, 2, 2, 0]], [[1, 0, 1, 2], [0, 2, 1,"
        " 0], [2, 0, 0, 1]]]"
    )
    var indices3d_1_np = indices3d_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, indices3d_1, axis=1),
        np.take_along_axis(a3d_f_np, indices3d_1_np, axis=1),
        "`take_along_axis` with 3-D F-order array on axis=1 is broken",
    )

    # Test with axis=2
    var indices3d_2 = nm.array[intp](
        "[[[2, 0, 3, 1], [1, 3, 0, 2], [3, 1, 2, 0]], [[0, 2, 1, 3], [2, 0, 3,"
        " 1], [1, 3, 0, 2]]]"
    )
    var indices3d_2_np = indices3d_2.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, indices3d_2, axis=2),
        np.take_along_axis(a3d_f_np, indices3d_2_np, axis=2),
        "`take_along_axis` with 3-D F-order array on axis=2 is broken",
    )

    # Test with argsort use case on each axis
    var sorted_indices_0 = nm.argsort(a3d_f, axis=0)
    var sorted_indices_0_np = sorted_indices_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, sorted_indices_0, axis=0),
        np.take_along_axis(a3d_f_np, sorted_indices_0_np, axis=0),
        (
            "`take_along_axis` with argsorted indices on axis=0 for F-order"
            " array is broken"
        ),
    )

    var sorted_indices_1 = nm.argsort(a3d_f, axis=1)
    var sorted_indices_1_np = sorted_indices_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, sorted_indices_1, axis=1),
        np.take_along_axis(a3d_f_np, sorted_indices_1_np, axis=1),
        (
            "`take_along_axis` with argsorted indices on axis=1 for F-order"
            " array is broken"
        ),
    )

    var sorted_indices_2 = nm.argsort(a3d_f, axis=2)
    var sorted_indices_2_np = sorted_indices_2.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, sorted_indices_2, axis=2),
        np.take_along_axis(a3d_f_np, sorted_indices_2_np, axis=2),
        (
            "`take_along_axis` with argsorted indices on axis=2 for F-order"
            " array is broken"
        ),
    )
