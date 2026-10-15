from std.python import Python, PythonObject
from std.testing.testing import assert_true
from utils_for_test import check, check_is_close
from std.testing import TestSuite

import numojo as nm
from numojo import Shape, Item


def test_sort() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(10)
    var B = nm.random.randn(2, 3)
    var C = nm.random.randn(2, 3, 4)
    var S = nm.random.randn(3, 3, 3, 3, 3, 3)
    check(
        nm.sort(A, axis=0), np.sort(A.to_numpy(), axis=0), "`sort` 1d is broken"
    )
    check(
        nm.sort(B, axis=0),
        np.sort(B.to_numpy(), axis=0),
        "`sort` 2d by axis 0 is broken",
    )
    check(
        nm.sort(B, axis=1),
        np.sort(B.to_numpy(), axis=1),
        "`sort` 2d by axis 1 is broken",
    )
    for i in range(3):
        check(
            nm.sort(C, axis=i),
            np.sort(C.to_numpy(), axis=i),
            String("`sort` 3d by axis {} is broken").format(i),
        )
    for i in range(6):
        check(
            nm.sort(S, axis=i),
            np.sort(S.to_numpy(), axis=i),
            String("`sort` 6d by axis {} is broken").format(i),
        )


def test_argsort() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(10)
    var B = nm.random.randn(2, 3)
    var C = nm.random.randn(2, 3, 4)
    var S = nm.random.randn(3, 3, 3, 3, 3, 3)
    var Sf = S.reshape(S.shape, order="F")

    check(
        nm.argsort(A, axis=0),
        np.argsort(A.to_numpy(), axis=0),
        "`argsort` 1d is broken",
    )
    check(
        nm.argsort(B, axis=0),
        np.argsort(B.to_numpy(), axis=0),
        "`argsort` 2d by axis 0 is broken",
    )
    check(
        nm.argsort(B, axis=1),
        np.argsort(B.to_numpy(), axis=1),
        "`argsort` 2d by axis 1 is broken",
    )
    for i in range(3):
        check_is_close(
            nm.argsort(C, axis=i),
            np.argsort(C.to_numpy(), axis=i),
            String("`argsort` 3d by axis {} is broken").format(i),
        )
    check(
        nm.argsort(S),
        np.argsort(S.to_numpy(), axis=PythonObject(None)),
        String("`argsort` 6d by axis None is broken"),
    )
    for i in range(6):
        check(
            nm.argsort(S, axis=i),
            np.argsort(S.to_numpy(), axis=i),
            String("`argsort` 6d by axis {} is broken").format(i),
        )

    check(
        nm.argsort(Sf),
        np.argsort(Sf.to_numpy(), axis=PythonObject(None)),
        String("`argsort` 6d F-order array by axis None is broken"),
    )
    for i in range(6):
        check(
            nm.argsort(Sf, axis=i),
            np.argsort(Sf.to_numpy(), axis=i),
            String("`argsort` 6d F-order by axis {} is broken").format(i),
        )


def test_inplace_sort() raises:
    var np = Python.import_module("numpy")
    var C = nm.random.randn(2, 3, 4)
    for i in range(3):
        C.sort(axis=i)
        var Cnp = C.to_numpy()
        Cnp.sort(axis=i)
        check(
            C,
            Cnp,
            String("`NDArray.sort()` 3d in-place by axis {} is broken").format(
                i
            ),
        )


def test_sort_stable() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(10)
    var B = nm.random.randn(2, 3)
    var C = nm.random.randn(2, 3, 4)
    var S = nm.random.randn(3, 3, 3, 3, 3, 3)

    check(
        nm.sort(A, axis=0, stable=True),
        np.sort(A.to_numpy(), axis=0, stable=PythonObject(True)),
        "`sort` 1d stably is broken",
    )
    check(
        nm.sort(B, axis=0, stable=True),
        np.sort(B.to_numpy(), axis=0, stable=PythonObject(True)),
        "`sort` 2d stably by axis 0 is broken",
    )
    check(
        nm.sort(B, axis=1, stable=True),
        np.sort(B.to_numpy(), axis=1, stable=PythonObject(True)),
        "`sort` 2d stably by axis 1 is broken",
    )
    for i in range(3):
        check(
            nm.sort(C, axis=i, stable=True),
            np.sort(C.to_numpy(), axis=i, stable=PythonObject(True)),
            String("`sort` 3d stably by axis {} is broken").format(i),
        )
    for i in range(6):
        check(
            nm.sort(S, axis=i, stable=True),
            np.sort(S.to_numpy(), axis=i, stable=PythonObject(True)),
            String("`sort` 6d stably by axis {} is broken").format(i),
        )


def test_partition() raises:
    var a = nm.fromstring[nm.i32]("[3, 1, 4, 1, 5, 9, 2, 6]")
    var sorted_a = nm.sort(a)

    for kth in range(a.size):
        var part = nm.partition(a, kth)
        assert_true(
            part.item(kth) == sorted_a.item(kth),
            "`partition` kth value mismatch",
        )
        for i in range(kth):
            assert_true(
                part.item(i) <= part.item(kth),
                "`partition` element before kth exceeds kth value",
            )
        for i in range(kth + 1, a.size):
            assert_true(
                part.item(i) >= part.item(kth),
                "`partition` element after kth is below kth value",
            )

    # Negative kth.
    var part_neg = nm.partition(a, -2)
    assert_true(
        part_neg.item(a.size - 2) == sorted_a.item(a.size - 2),
        "`partition` with negative kth is broken",
    )


def test_argpartition() raises:
    var a = nm.fromstring[nm.i32]("[3, 1, 4, 1, 5, 9, 2, 6]")
    var sorted_a = nm.sort(a)
    var kth = 3

    var indices = nm.argpartition(a, kth)
    var reordered = nm.NDArray[nm.i32](Shape(a.size))
    for i in range(a.size):
        reordered[Item(i)] = a.item(Int(indices.item(i)))

    assert_true(
        reordered.item(kth) == sorted_a.item(kth),
        "`argpartition` kth value mismatch",
    )
    for i in range(kth):
        assert_true(
            reordered.item(i) <= reordered.item(kth),
            "`argpartition` element before kth exceeds kth value",
        )
    for i in range(kth + 1, a.size):
        assert_true(
            reordered.item(i) >= reordered.item(kth),
            "`argpartition` element after kth is below kth value",
        )


def test_lexsort() raises:
    var np = Python.import_module("numpy")

    var surnames = nm.fromstring[nm.i32]("[1, 2, 1, 2]")
    var ages = nm.fromstring[nm.i32]("[30, 25, 20, 25]")

    var keys = List[nm.NDArray[nm.i32]]()
    keys.append(ages.copy())
    keys.append(surnames.copy())

    var order = nm.lexsort(keys)
    var order_np = np.lexsort(
        Python.tuple(ages.to_numpy(), surnames.to_numpy())
    )
    check(order, order_np, "`lexsort` is broken")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
