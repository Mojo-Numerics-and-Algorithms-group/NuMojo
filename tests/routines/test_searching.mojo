from numojo.prelude import *
from python import Python, PythonObject
from utils_for_test import check, check_is_close, check_values_close
from testing import TestSuite


fn test_argmax() raises:
    var np = Python.import_module("numpy")

    # Test 1D array
    var a1d = nm.array[nm.f32]("[3.4, 1.2, 5.7, 0.9, 2.3]")
    var a1d_np = a1d.to_numpy()

    check_values_close(
        nm.argmax(a1d),
        np.argmax(a1d_np),
        "`argmax` with 1D array is broken",
    )

    # Test 2D array without specifying axis (flattened)
    var a2d = nm.array[nm.f32](
        "[[3.4, 1.2, 5.7], [0.9, 2.3, 4.1], [7.6, 0.5, 2.8]]"
    )
    var a2d_np = a2d.to_numpy()

    check_values_close(
        nm.argmax(a2d),
        np.argmax(a2d_np),
        "`argmax` with 2D array (flattened) is broken",
    )

    # Test 2D array with axis=0
    check(
        nm.argmax(a2d, axis=0),
        np.argmax(a2d_np, axis=0),
        "`argmax` with 2D array on axis=0 is broken",
    )

    # Test 2D array with axis=1
    check(
        nm.argmax(a2d, axis=1),
        np.argmax(a2d_np, axis=1),
        "`argmax` with 2D array on axis=1 is broken",
    )

    # Test 2D array with negative axis
    check(
        nm.argmax(a2d, axis=-1),
        np.argmax(a2d_np, axis=-1),
        "`argmax` with 2D array on negative axis is broken",
    )

    # Test 3D array
    var a3d = nm.random.randint(2, 3, 4, low=0, high=10)
    var a3d_np = a3d.to_numpy()

    check_values_close(
        nm.argmax(a3d),
        np.argmax(a3d_np),
        "`argmax` with 3D array (flattened) is broken",
    )

    check(
        nm.argmax(a3d, axis=0),
        np.argmax(a3d_np, axis=0),
        "`argmax` with 3D array on axis=0 is broken",
    )

    check(
        nm.argmax(a3d, axis=1),
        np.argmax(a3d_np, axis=1),
        "`argmax` with 3D array on axis=1 is broken",
    )

    check(
        nm.argmax(a3d, axis=2),
        np.argmax(a3d_np, axis=2),
        "`argmax` with 3D array on axis=2 is broken",
    )

    # Test with F-order array
    var a3d_f = nm.random.randint(2, 3, 4, low=0, high=10).reshape(
        Shape(2, 3, 4), order="F"
    )
    var a3d_f_np = a3d_f.to_numpy()

    for i in range(3):
        check(
            nm.argmax(a3d_f, axis=i),
            np.argmax(a3d_f_np, axis=i),
            String(
                "`argmax` with F-order 3D array on axis={} is broken"
            ).format(i),
        )


fn test_argmin() raises:
    var np = Python.import_module("numpy")

    # Test 1D array
    var a1d = nm.array[nm.f32]("[3.4, 1.2, 5.7, 0.9, 2.3]")
    var a1d_np = a1d.to_numpy()

    check_values_close(
        nm.argmin(a1d),
        np.argmin(a1d_np),
        "`argmin` with 1D array is broken",
    )

    # Test 2D array without specifying axis (flattened)
    var a2d = nm.array[nm.f32](
        "[[3.4, 1.2, 5.7], [0.9, 2.3, 4.1], [7.6, 0.5, 2.8]]"
    )
    var a2d_np = a2d.to_numpy()

    check_values_close(
        nm.argmin(a2d),
        np.argmin(a2d_np),
        "`argmin` with 2D array (flattened) is broken",
    )

    # Test 2D array with axis=0
    check(
        nm.argmin(a2d, axis=0),
        np.argmin(a2d_np, axis=0),
        "`argmin` with 2D array on axis=0 is broken",
    )

    # Test 2D array with axis=1
    check(
        nm.argmin(a2d, axis=1),
        np.argmin(a2d_np, axis=1),
        "`argmin` with 2D array on axis=1 is broken",
    )

    # Test 2D array with negative axis
    check(
        nm.argmin(a2d, axis=-1),
        np.argmin(a2d_np, axis=-1),
        "`argmin` with 2D array on negative axis is broken",
    )

    # Test 3D array
    var a3d = nm.random.randint(2, 3, 4, low=0, high=10)
    var a3d_np = a3d.to_numpy()

    check_values_close(
        nm.argmin(a3d),
        np.argmin(a3d_np),
        "`argmin` with 3D array (flattened) is broken",
    )

    check(
        nm.argmin(a3d, axis=0),
        np.argmin(a3d_np, axis=0),
        "`argmin` with 3D array on axis=0 is broken",
    )

    check(
        nm.argmin(a3d, axis=1),
        np.argmin(a3d_np, axis=1),
        "`argmin` with 3D array on axis=1 is broken",
    )

    check(
        nm.argmin(a3d, axis=2),
        np.argmin(a3d_np, axis=2),
        "`argmin` with 3D array on axis=2 is broken",
    )

    # Test with F-order array
    var a3d_f = nm.random.randint(2, 3, 4, low=0, high=10).reshape(
        Shape(2, 3, 4), order="F"
    )
    var a3d_f_np = a3d_f.to_numpy()

    for i in range(3):
        check(
            nm.argmin(a3d_f, axis=i),
            np.argmin(a3d_f_np, axis=i),
            String(
                "`argmin` with F-order 3D array on axis={} is broken"
            ).format(i),
        )


fn test_take_along_axis_with_argmax_argmin() raises:
    var np = Python.import_module("numpy")

    # Test with argmax to get maximum values
    var a2d = nm.random.randint(5, 4, low=0, high=10)
    var a2d_np = a2d.to_numpy()

    # Get indices of maximum values along axis=1
    var max_indices = nm.argmax(a2d, axis=1)
    var max_indices_np = np.argmax(a2d_np, axis=1)

    # Reshape indices for take_along_axis
    var reshaped_indices = max_indices.reshape(Shape(max_indices.shape[0], 1))
    var reshaped_indices_np = max_indices_np.reshape(max_indices_np.shape[0], 1)

    # Get maximum values using take_along_axis
    check(
        nm.indexing.take_along_axis(a2d, reshaped_indices, axis=1),
        np.take_along_axis(a2d_np, reshaped_indices_np, axis=1),
        "`take_along_axis` with argmax is broken",
    )

    # Test with argmin to get minimum values
    var min_indices = nm.argmin(a2d, axis=1)
    var min_indices_np = np.argmin(a2d_np, axis=1)

    # Reshape indices for take_along_axis
    var reshaped_min_indices = min_indices.reshape(
        Shape(min_indices.shape[0], 1)
    )
    var reshaped_min_indices_np = min_indices_np.reshape(
        min_indices_np.shape[0], 1
    )

    # Get minimum values using take_along_axis
    check(
        nm.indexing.take_along_axis(a2d, reshaped_min_indices, axis=1),
        np.take_along_axis(a2d_np, reshaped_min_indices_np, axis=1),
        "`take_along_axis` with argmin is broken",
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
