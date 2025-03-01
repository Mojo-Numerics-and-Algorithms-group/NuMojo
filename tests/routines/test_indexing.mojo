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
