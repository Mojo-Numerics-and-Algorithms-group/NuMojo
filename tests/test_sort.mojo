import numojo as nm
from python import Python, PythonObject
from utils_for_test import check, check_is_close


def test_sort_1d():
    var arr = nm.core.random.rand[nm.i16](25, min=0, max=100)
    var np = Python.import_module("numpy")
    arr.sort()
    np_arr_sorted = np.sort(arr.to_numpy())
    return check[nm.i16](arr, np_arr_sorted, "quick sort is broken")


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
