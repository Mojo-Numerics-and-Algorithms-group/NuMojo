from std.testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from std.python import Python, PythonObject
from std.testing import TestSuite

import numojo as nm
from numojo import *


def test_unique() raises:
    var np = Python.import_module("numpy")

    var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3, 2]")
    var anp = a.to_numpy()
    check_is_close(nm.unique(a), np.unique(anp), "`unique` 1-D fails.")

    var B = nm.reshape(
        nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3, 2, 1]"), Shape(2, 4)
    )
    var Bnp = B.to_numpy()
    check_is_close(
        nm.unique(B), np.unique(Bnp), "`unique` flattens multi-d array."
    )


def test_unique_counts() raises:
    var np = Python.import_module("numpy")

    var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3, 2]")
    var anp = a.to_numpy()
    var result = nm.unique_counts(a)
    var npresult = np.unique(anp, return_counts=True)
    check_is_close(result[0], npresult[0], "`unique_counts` values fails.")
    check_is_close(result[1], npresult[1], "`unique_counts` counts fails.")


def test_unique_axis() raises:
    var np = Python.import_module("numpy")

    var A = nm.fromstring[nm.i32]("[[1,2][3,4][1,2][3,4][5,6]]")
    var Anp = A.to_numpy()
    check_is_close(
        nm.unique(A, axis=0),
        np.unique(Anp, axis=0),
        "`unique` axis=0 fails.",
    )

    var result = nm.unique_counts(A, axis=0)
    var npresult = np.unique(Anp, axis=0, return_counts=True)
    check_is_close(
        result[0], npresult[0], "`unique_counts` axis=0 values fails."
    )
    check_is_close(
        result[1], npresult[1], "`unique_counts` axis=0 counts fails."
    )


def test_unique_index() raises:
    var np = Python.import_module("numpy")

    var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3, 2]")
    var anp = a.to_numpy()
    var result = nm.unique_index(a)
    var npresult = np.unique(anp, return_index=True)
    check_is_close(result[0], npresult[0], "`unique_index` values fails.")
    check_is_close(result[1], npresult[1], "`unique_index` index fails.")


def test_unique_inverse() raises:
    var np = Python.import_module("numpy")

    var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3, 2]")
    var anp = a.to_numpy()
    var result = nm.unique_inverse(a)
    var npresult = np.unique(anp, return_inverse=True)
    check_is_close(result[0], npresult[0], "`unique_inverse` values fails.")
    check_is_close(
        result[1],
        np.reshape(npresult[1], anp.shape),
        "`unique_inverse` inverse fails.",
    )


def test_unique_all() raises:
    var np = Python.import_module("numpy")

    var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3, 2]")
    var anp = a.to_numpy()
    var result = nm.unique_all(a)
    var npresult = np.unique(
        anp, return_index=True, return_inverse=True, return_counts=True
    )
    check_is_close(result[0], npresult[0], "`unique_all` values fails.")
    check_is_close(result[1], npresult[1], "`unique_all` index fails.")
    check_is_close(
        result[2],
        np.reshape(npresult[2], anp.shape),
        "`unique_all` inverse fails.",
    )
    check_is_close(result[3], npresult[3], "`unique_all` counts fails.")


def test_unique_index_axis() raises:
    var np = Python.import_module("numpy")

    var A = nm.fromstring[nm.i32]("[[1,2][3,4][1,2][3,4][5,6]]")
    var Anp = A.to_numpy()
    var result = nm.unique_index(A, axis=0)
    var npresult = np.unique(Anp, axis=0, return_index=True)
    check_is_close(
        result[0], npresult[0], "`unique_index` axis=0 values fails."
    )
    check_is_close(result[1], npresult[1], "`unique_index` axis=0 index fails.")


def test_unique_inverse_axis() raises:
    var np = Python.import_module("numpy")

    var A = nm.fromstring[nm.i32]("[[1,2][3,4][1,2][3,4][5,6]]")
    var Anp = A.to_numpy()
    var result = nm.unique_inverse(A, axis=0)
    var npresult = np.unique(Anp, axis=0, return_inverse=True)
    check_is_close(
        result[0], npresult[0], "`unique_inverse` axis=0 values fails."
    )
    check_is_close(
        result[1],
        np.reshape(npresult[1], Anp.shape[0]),
        "`unique_inverse` axis=0 inverse fails.",
    )


def test_unique_all_axis() raises:
    var np = Python.import_module("numpy")

    var A = nm.fromstring[nm.i32]("[[1,2][3,4][1,2][3,4][5,6]]")
    var Anp = A.to_numpy()
    var result = nm.unique_all(A, axis=0)
    var npresult = np.unique(
        Anp,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    check_is_close(result[0], npresult[0], "`unique_all` axis=0 values fails.")
    check_is_close(result[1], npresult[1], "`unique_all` axis=0 index fails.")
    check_is_close(
        result[2],
        np.reshape(npresult[2], Anp.shape[0]),
        "`unique_all` axis=0 inverse fails.",
    )
    check_is_close(result[3], npresult[3], "`unique_all` axis=0 counts fails.")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
