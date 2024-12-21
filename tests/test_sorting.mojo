import numojo as nm
from python import Python, PythonObject
from utils_for_test import check, check_is_close


fn test_sorting() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.rand[nm.i8](10, min=0, max=100)
    var B = nm.random.rand[nm.i8](2, 3, min=0, max=100)
    var C = nm.random.rand[nm.i8](2, 3, 4, min=0, max=100)
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


# ND sorting currently works differently than numpy which has an on axis

# def test_sort_2d():
#     arr = nm.NDArray(5,5,random=True)
#     var np = Python.import_module("numpy")
#     arr_sorted = arr.sort()
#     print(arr_sorted)
#     np_arr_sorted  = np.sort(arr.to_numpy())
#     print(np_arr_sorted)
#     return check(arr_sorted,np_arr_sorted, "quick sort is broken")

# def main():
#     test_sort_1d()
#     # test_sort_2d()
