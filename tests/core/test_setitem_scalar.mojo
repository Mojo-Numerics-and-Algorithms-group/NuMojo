from std.python import Python, PythonObject
from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *
from utils_for_test import check


# Mojo's __setitem__ overload resolution with variadic+keyword args requires:
#   - List[Slice] scalar: pass the list positionally, `val=` keyword works.
#   - *Slice scalar: use explicit `.__setitem__(s1, s2, ..., scalar=v)`.
#   - *Variant[Slice,Int] scalar: use explicit `.__setitem__(i, s, scalar=v)`.
# The `scalar` keyword name was chosen (instead of `val`) to avoid shadowing
# the existing `val: Self` overloads at the Mojo compiler level.


# ===== Step 3: __setitem__(List[Slice], val: Scalar) =====


def test_setitem_list_slice_scalar_1d() raises:
    """List[Slice] scalar backend: fills selected 1D range."""
    var a = nm.arange[nm.i32](0, 6, step=1)
    var sl = List[Slice]()
    sl.append(Slice(1, 4))
    a._setitem_slice_scalar(sl, Scalar[nm.i32](99))

    assert_equal(Int(a.item(0)), 0)
    assert_equal(Int(a.item(1)), 99)
    assert_equal(Int(a.item(2)), 99)
    assert_equal(Int(a.item(3)), 99)
    assert_equal(Int(a.item(4)), 4)
    assert_equal(Int(a.item(5)), 5)


def test_setitem_list_slice_scalar_2d_submatrix() raises:
    """List[Slice] scalar: fills a 2x2 sub-matrix, surrounding values unchanged.
    """
    var a = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))
    var sl = List[Slice]()
    sl.append(Slice(1, 3))
    sl.append(Slice(1, 3))
    a._setitem_slice_scalar(sl, Scalar[nm.i32](0))

    assert_equal(Int(a.item(1, 1)), 0)
    assert_equal(Int(a.item(1, 2)), 0)
    assert_equal(Int(a.item(2, 1)), 0)
    assert_equal(Int(a.item(2, 2)), 0)
    # Row 0 and col 0 untouched
    assert_equal(Int(a.item(0, 1)), 1)
    assert_equal(Int(a.item(1, 0)), 4)
    assert_equal(Int(a.item(3, 3)), 15)


def test_setitem_list_slice_scalar_3d() raises:
    """List[Slice] scalar: fills a 3D sub-region."""
    var a = nm.arange[nm.i32](0, 24, step=1).reshape(Shape(2, 3, 4))
    var sl = List[Slice]()
    sl.append(Slice(0, 2))
    sl.append(Slice(1, 3))
    sl.append(Slice(2, 4))
    a._setitem_slice_scalar(sl, Scalar[nm.i32](5))

    for i in range(2):
        for j in range(1, 3):
            for k in range(2, 4):
                assert_equal(
                    Int(a.item(i, j, k)),
                    5,
                    String("a[{},{},{}] should be 5").format(i, j, k),
                )
    # Outside region
    assert_equal(Int(a.item(0, 0, 0)), 0)
    assert_equal(Int(a.item(0, 0, 1)), 1)


def test_setitem_list_slice_scalar_step() raises:
    """List[Slice] scalar: step-stride fills every other row."""
    var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4))
    var sl = List[Slice]()
    sl.append(Slice(0, 3, 2))  # rows 0, 2
    sl.append(Slice(0, 4))
    a._setitem_slice_scalar(sl, Scalar[nm.i32](-1))

    for c in range(4):
        assert_equal(Int(a.item(0, c)), -1)
        assert_equal(Int(a.item(2, c)), -1)
    # Row 1 unchanged: 4 5 6 7
    assert_equal(Int(a.item(1, 0)), 4)
    assert_equal(Int(a.item(1, 3)), 7)


def test_setitem_list_slice_scalar_implicit_trailing_dim() raises:
    """List[Slice] scalar: fewer slices than ndim pads trailing dims as full."""
    var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4))
    var sl = List[Slice]()
    sl.append(Slice(1, 3))  # only row dim given
    a._setitem_slice_scalar(sl, Scalar[nm.i32](42))

    for c in range(4):
        assert_equal(Int(a.item(1, c)), 42)
        assert_equal(Int(a.item(2, c)), 42)
    # Row 0 unchanged
    for c in range(4):
        assert_equal(Int(a.item(0, c)), c)


def test_setitem_list_slice_scalar_single_element() raises:
    """List[Slice] scalar: unit-length slices write exactly one element."""
    var a = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))
    var sl = List[Slice]()
    sl.append(Slice(2, 3))
    sl.append(Slice(2, 3))
    a._setitem_slice_scalar(sl, Scalar[nm.i32](123))

    assert_equal(Int(a.item(2, 2)), 123)
    assert_equal(Int(a.item(2, 1)), 9)
    assert_equal(Int(a.item(2, 3)), 11)
    assert_equal(Int(a.item(1, 2)), 6)
    assert_equal(Int(a.item(3, 2)), 14)


def test_setitem_list_slice_scalar_whole_array() raises:
    """List[Slice] scalar: full-range slices zero the whole array."""
    var a = nm.arange[nm.i32](1, 10, step=1).reshape(Shape(3, 3))
    var sl = List[Slice]()
    sl.append(Slice(0, 3))
    sl.append(Slice(0, 3))
    a._setitem_slice_scalar(sl, Scalar[nm.i32](0))

    for i in range(3):
        for j in range(3):
            assert_equal(Int(a.item(i, j)), 0)


# ===== Step 3: __setitem__(*slices: Slice, scalar: Scalar) =====


def test_setitem_variadic_slice_scalar_2d() raises:
    """*Slice scalar wrapper produces same result as List[Slice] scalar."""
    var a = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))
    var b = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))

    var sl = List[Slice]()
    sl.append(Slice(1, 3))
    sl.append(Slice(1, 3))
    a._setitem_slice_scalar(sl, Scalar[nm.i32](77))

    b.__setitem__(Slice(1, 3), Slice(1, 3), scalar=Scalar[nm.i32](77))

    for i in range(4):
        for j in range(4):
            assert_equal(
                Int(a.item(i, j)),
                Int(b.item(i, j)),
                String(
                    "*Slice and List[Slice] scalar paths diverge at ({},{})"
                ).format(i, j),
            )


def test_setitem_variadic_slice_scalar_1d() raises:
    """*Slice scalar: 1D slice fill."""
    var a = nm.arange[nm.i32](0, 8, step=1)
    a.__setitem__(Slice(2, 6), scalar=Scalar[nm.i32](5))

    assert_equal(Int(a.item(0)), 0)
    assert_equal(Int(a.item(1)), 1)
    assert_equal(Int(a.item(2)), 5)
    assert_equal(Int(a.item(5)), 5)
    assert_equal(Int(a.item(6)), 6)
    assert_equal(Int(a.item(7)), 7)


def test_setitem_variadic_slice_scalar_implicit_trailing() raises:
    """*Slice scalar: single slice on 2D array, trailing dim implicit full."""
    var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4))
    a.__setitem__(Slice(0, 2), scalar=Scalar[nm.i32](9))

    for c in range(4):
        assert_equal(Int(a.item(0, c)), 9)
        assert_equal(Int(a.item(1, c)), 9)
    # Row 2 unchanged: 8 9 10 11
    assert_equal(Int(a.item(2, 0)), 8)
    assert_equal(Int(a.item(2, 3)), 11)


# ===== Step 4: __setitem__(*Variant[Slice, Int], scalar: Scalar) =====


def test_setitem_mixed_single_element_2d() raises:
    """All-Int fast path: single element write on 2D array."""
    var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4))
    a.__setitem__(1, 2, scalar=Scalar[nm.i32](99))

    assert_equal(Int(a.item(1, 2)), 99)
    assert_equal(Int(a.item(1, 1)), 5)
    assert_equal(Int(a.item(1, 3)), 7)
    assert_equal(Int(a.item(0, 2)), 2)
    assert_equal(Int(a.item(2, 2)), 10)


def test_setitem_mixed_single_element_3d() raises:
    """All-Int fast path: single element write on 3D array."""
    var a = nm.arange[nm.i32](0, 24, step=1).reshape(Shape(2, 3, 4))
    a.__setitem__(0, 1, 2, scalar=Scalar[nm.i32](77))

    assert_equal(Int(a.item(0, 1, 2)), 77)
    assert_equal(Int(a.item(0, 0, 0)), 0)
    assert_equal(Int(a.item(1, 2, 3)), 23)


def test_setitem_mixed_int_slice_row() raises:
    """Int + Slice: select row by int, column range by slice."""
    var a = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))
    a.__setitem__(1, Slice(1, 3), scalar=Scalar[nm.i32](55))

    assert_equal(Int(a.item(1, 1)), 55)
    assert_equal(Int(a.item(1, 2)), 55)
    assert_equal(Int(a.item(1, 0)), 4)
    assert_equal(Int(a.item(1, 3)), 7)
    assert_equal(Int(a.item(0, 1)), 1)
    assert_equal(Int(a.item(2, 1)), 9)


def test_setitem_mixed_slice_int_col() raises:
    """Slice + Int: select row range by slice, column by int."""
    var a = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))
    a.__setitem__(Slice(1, 3), 2, scalar=Scalar[nm.i32](33))

    assert_equal(Int(a.item(1, 2)), 33)
    assert_equal(Int(a.item(2, 2)), 33)
    assert_equal(Int(a.item(1, 0)), 4)
    assert_equal(Int(a.item(2, 3)), 11)
    assert_equal(Int(a.item(0, 2)), 2)
    assert_equal(Int(a.item(3, 2)), 14)


def test_setitem_mixed_negative_int() raises:
    """Negative integer index is normalised correctly."""
    var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4))
    a.__setitem__(-1, 2, scalar=Scalar[nm.i32](88))

    assert_equal(Int(a.item(2, 2)), 88)
    assert_equal(Int(a.item(0, 2)), 2)
    assert_equal(Int(a.item(1, 2)), 6)
    assert_equal(Int(a.item(2, 0)), 8)


def test_setitem_mixed_int_only_1d() raises:
    """All-Int fast path on 1D array."""
    var a = nm.arange[nm.i32](0, 6, step=1)
    a.__setitem__(3, scalar=Scalar[nm.i32](99))

    assert_equal(Int(a.item(3)), 99)
    assert_equal(Int(a.item(2)), 2)
    assert_equal(Int(a.item(4)), 4)


def test_setitem_mixed_partial_indices_3d() raises:
    """Two ints on 3D: trailing dim filled implicitly as full range."""
    var a = nm.arange[nm.i32](0, 24, step=1).reshape(Shape(2, 3, 4))
    a.__setitem__(0, 1, scalar=Scalar[nm.i32](11))

    for k in range(4):
        assert_equal(
            Int(a.item(0, 1, k)),
            11,
            String("a[0,1,{}] should be 11").format(k),
        )
    assert_equal(Int(a.item(0, 0, 0)), 0)
    assert_equal(Int(a.item(1, 1, 0)), 16)


def test_setitem_mixed_does_not_corrupt_neighbours() raises:
    """Single-element write must not corrupt adjacent elements."""
    var a = nm.zeros[nm.i32](Shape(4, 4))
    a.__setitem__(1, 2, scalar=Scalar[nm.i32](7))

    for r in range(4):
        for c in range(4):
            var expected = 7 if (r == 1 and c == 2) else 0
            assert_equal(
                Int(a.item(r, c)),
                expected,
                String("Element ({},{}) wrong after a[1,2]=7").format(r, c),
            )


def test_setitem_mixed_agrees_with_list_slice_scalar() raises:
    """Mixed int/slice scalar path matches List[Slice] scalar path."""
    var a = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))
    var b = nm.arange[nm.i32](0, 16, step=1).reshape(Shape(4, 4))

    a.__setitem__(1, Slice(1, 3), scalar=Scalar[nm.i32](55))

    var sl = List[Slice]()
    sl.append(Slice(1, 2))
    sl.append(Slice(1, 3))
    b._setitem_slice_scalar(sl, Scalar[nm.i32](55))

    for i in range(4):
        for j in range(4):
            assert_equal(
                Int(a.item(i, j)),
                Int(b.item(i, j)),
                String(
                    "Mixed and List[Slice] scalar paths diverge at ({},{})"
                ).format(i, j),
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
