import numojo as nm
from python import Python, PythonObject
from utils_for_test import check, check_is_close


fn test_sorting() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.rand[nm.i8](10, min=0, max=100)
    var B = nm.random.rand[nm.i8](2, 3, min=0, max=100)
    var C = nm.random.rand[nm.i8](2, 3, 4, min=0, max=100)

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
            String("`sort` 1d by axis {} is broken").format(i),
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
            String("`argsort` 1d by axis {} is broken").format(i),
        )
