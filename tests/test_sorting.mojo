import numojo as nm
from python import Python, PythonObject
from utils_for_test import check, check_is_close


fn test_sorting() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(10)
    var B = nm.random.randn(2, 3)
    var C = nm.random.randn(2, 3, 4)
    var S = nm.random.randn(3, 3, 3, 3, 3, 3)  # 6d array

    # Sort
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

    # Argsort
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
    for i in range(6):
        check(
            nm.argsort(S, axis=i),
            np.argsort(S.to_numpy(), axis=i),
            String("`argsort` 6d by axis {} is broken").format(i),
        )
