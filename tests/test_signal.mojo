import numojo as nm
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from utils_for_test import check, check_is_close
from testing.testing import assert_raises


def test_convolve2d():
    var sp = Python.import_module("scipy")
    in1 = nm.random.rand(6, 6)
    in2 = nm.fromstring("[[1, 0], [0, -1]]")

    npin1 = in1.to_numpy()
    npin2 = in2.to_numpy()

    res1 = nm.signal.convolve2d(in1, in2)
    res2 = sp.signal.convolve2d(npin1, npin2, mode="valid")
    check(
        res1,
        res2,
        "test_convolve2d failed #2\n" + str(res1) + "\n" + str(res2),
    )
