# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Test functional programming module `numojo.routines.functional`.
"""
from python import Python
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from testing import TestSuite

from numojo.prelude import *


fn test_apply_along_axis() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.randn(Shape(4, 8, 16))
    var anp = a.to_numpy()
    var b = nm.reshape(a, a.shape, order="F")
    var bnp = b.to_numpy()

    for i in range(a.ndim):
        check(
            nm.apply_along_axis[nm.sorting.quick_sort_1d](a, axis=i),
            np.apply_along_axis(np.sort, axis=i, arr=anp),
            String(
                "`apply_along_axis` C-order array along axis {} is broken"
            ).format(i),
        )
        check(
            nm.apply_along_axis[nm.sorting.quick_sort_1d](b, axis=i),
            np.apply_along_axis(np.sort, axis=i, arr=bnp),
            String(
                "`apply_along_axis` F-order array along axis {} is broken"
            ).format(i),
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
