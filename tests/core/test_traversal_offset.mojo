from std.testing import TestSuite
from std.testing.testing import assert_equal

import numojo as nm
from numojo.prelude import *

# ===== Step 7: F-order setitem noffset =====


def test_setitem_forder_top_left_region() raises:
    """F-order setitem: zero out [0:2, 1:3] — touches cols 1 and 2."""
    # Before: row0=[0,3,6,9], row1=[1,4,7,10], row2=[2,5,8,11]
    # After [0:2,1:3]=0: row0=[0,0,0,9], row1=[1,0,0,10], row2=[2,5,8,11]
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var val = nm.zeros[nm.i32](Shape(2, 2))
    a[Slice(0, 2), Slice(1, 3)] = val

    assert_equal(Int(a.item(0, 0)), 0)
    assert_equal(Int(a.item(0, 1)), 0)
    assert_equal(Int(a.item(0, 2)), 0)
    assert_equal(Int(a.item(0, 3)), 9)
    assert_equal(Int(a.item(1, 0)), 1)
    assert_equal(Int(a.item(1, 1)), 0)
    assert_equal(Int(a.item(1, 2)), 0)
    assert_equal(Int(a.item(1, 3)), 10)
    assert_equal(Int(a.item(2, 0)), 2)
    assert_equal(Int(a.item(2, 1)), 5)
    assert_equal(Int(a.item(2, 2)), 8)
    assert_equal(Int(a.item(2, 3)), 11)


def test_setitem_forder_bottom_right_region() raises:
    """F-order setitem: write 55 into [1:3, 2:4] — highest-offset corner."""
    # Before: row0=[0,3,6,9], row1=[1,4,7,10], row2=[2,5,8,11]
    # After [1:3,2:4]=55: row0=[0,3,6,9], row1=[1,4,55,55], row2=[2,5,55,55]
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var val = nm.full[nm.i32](Shape(2, 2), fill_value=Scalar[nm.i32](55))
    a[Slice(1, 3), Slice(2, 4)] = val

    assert_equal(Int(a.item(0, 0)), 0)
    assert_equal(Int(a.item(0, 2)), 6)
    assert_equal(Int(a.item(0, 3)), 9)
    assert_equal(Int(a.item(1, 0)), 1)
    assert_equal(Int(a.item(1, 1)), 4)
    assert_equal(Int(a.item(1, 2)), 55)
    assert_equal(Int(a.item(1, 3)), 55)
    assert_equal(Int(a.item(2, 0)), 2)
    assert_equal(Int(a.item(2, 1)), 5)
    assert_equal(Int(a.item(2, 2)), 55)
    assert_equal(Int(a.item(2, 3)), 55)


def test_setitem_forder_full_row() raises:
    """F-order setitem: overwrite rows 1 and 2 — [1:3, 0:4] = 7."""
    # Before: row1=[1,4,7,10], row2=[2,5,8,11]
    # After [1:3,0:4]=7: row1=[7,7,7,7], row2=[7,7,7,7]
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var val = nm.full[nm.i32](Shape(2, 4), fill_value=Scalar[nm.i32](7))
    a[Slice(1, 3), Slice(0, 4)] = val

    assert_equal(Int(a.item(0, 0)), 0)
    assert_equal(Int(a.item(0, 3)), 9)
    assert_equal(Int(a.item(1, 0)), 7)
    assert_equal(Int(a.item(1, 1)), 7)
    assert_equal(Int(a.item(1, 2)), 7)
    assert_equal(Int(a.item(1, 3)), 7)
    assert_equal(Int(a.item(2, 0)), 7)
    assert_equal(Int(a.item(2, 3)), 7)


def test_setitem_forder_full_col() raises:
    """F-order setitem: overwrite column 2 entirely — [0:3, 2:3] = 99."""
    # Before: col2=[6,7,8]
    # After [0:3,2:3]=99: col2=[99,99,99], others unchanged
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var val = nm.full[nm.i32](Shape(3, 1), fill_value=Scalar[nm.i32](99))
    a[Slice(0, 3), Slice(2, 3)] = val

    assert_equal(Int(a.item(0, 1)), 3)
    assert_equal(Int(a.item(0, 2)), 99)
    assert_equal(Int(a.item(0, 3)), 9)
    assert_equal(Int(a.item(1, 2)), 99)
    assert_equal(Int(a.item(2, 2)), 99)
    assert_equal(Int(a.item(2, 1)), 5)
    assert_equal(Int(a.item(2, 3)), 11)


def test_setitem_forder_non_trivial_source() raises:
    """F-order setitem: source is non-uniform — verifies values not just written once.
    """
    # Write [[10,11],[12,13]] into [0:2, 0:2] of F-order 3x4
    # Before: (0,0)=0,(0,1)=3,(1,0)=1,(1,1)=4
    # After:  (0,0)=10,(0,1)=11,(1,0)=12,(1,1)=13
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var val = nm.arange[nm.i32](10, 14).reshape(Shape(2, 2))
    a[Slice(0, 2), Slice(0, 2)] = val

    assert_equal(Int(a.item(0, 0)), 10)
    assert_equal(Int(a.item(0, 1)), 11)
    assert_equal(Int(a.item(1, 0)), 12)
    assert_equal(Int(a.item(1, 1)), 13)
    # Untouched
    assert_equal(Int(a.item(0, 2)), 6)
    assert_equal(Int(a.item(2, 0)), 2)
    assert_equal(Int(a.item(2, 3)), 11)


def test_setitem_forder_does_not_corrupt_untouched() raises:
    """F-order setitem: only the target region changes, rest stays."""
    # Write zeros into [0:2, 1:3] — a 2x2 region
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var original = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var val = nm.zeros[nm.i32](Shape(2, 2))
    a[Slice(0, 2), Slice(1, 3)] = val

    # Changed: rows 0-1, cols 1-2 → 0
    assert_equal(Int(a.item(0, 1)), 0)
    assert_equal(Int(a.item(0, 2)), 0)
    assert_equal(Int(a.item(1, 1)), 0)
    assert_equal(Int(a.item(1, 2)), 0)
    for r in range(3):
        for c in range(4):
            if not ((r == 0 or r == 1) and (c == 1 or c == 2)):
                assert_equal(
                    Int(a.item(r, c)),
                    Int(original.item(r, c)),
                    String("F-order setitem corrupted ({},{})").format(r, c),
                )


# ===== Step 8: traverse_iterative (getter) =====


def test_getitem_slice_corder_values() raises:
    """traverse_iterative getter: C-order slice returns correct element values.
    """
    # arange(24).reshape(4,6): row i, col j → value = i*6+j
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(4, 6))
    var b = a[Slice(1, 3), Slice(2, 5)]
    # b[i,j] = a[1+i, 2+j] = (1+i)*6 + (2+j)
    assert_equal(b.shape[0], 2)
    assert_equal(b.shape[1], 3)
    assert_equal(Int(b.item(0, 0)), 8)  # a[1,2]
    assert_equal(Int(b.item(0, 1)), 9)  # a[1,3]
    assert_equal(Int(b.item(0, 2)), 10)  # a[1,4]
    assert_equal(Int(b.item(1, 0)), 14)  # a[2,2]
    assert_equal(Int(b.item(1, 1)), 15)  # a[2,3]
    assert_equal(Int(b.item(1, 2)), 16)  # a[2,4]


def test_getitem_slice_forder_values() raises:
    """traverse_iterative getter: F-order slice returns correct element values.
    """
    # arange(12).reshape(3,4,order="F"):
    #   row0=[0,3,6,9], row1=[1,4,7,10], row2=[2,5,8,11]
    # b = a[0:2, 1:3] → row0=[3,6], row1=[4,7]
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var b = a[Slice(0, 2), Slice(1, 3)]

    assert_equal(b.shape[0], 2)
    assert_equal(b.shape[1], 2)
    assert_equal(Int(b.item(0, 0)), 3)
    assert_equal(Int(b.item(0, 1)), 6)
    assert_equal(Int(b.item(1, 0)), 4)
    assert_equal(Int(b.item(1, 1)), 7)


def test_getitem_slice_3d_corder() raises:
    """traverse_iterative getter: 3D C-order slice."""
    # arange(24).reshape(2,3,4): element [i,j,k] = i*12 + j*4 + k
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var b = a[Slice(0, 2), Slice(1, 3), Slice(1, 3)]
    # b[i,j,k] = a[i, 1+j, 1+k] = i*12 + (1+j)*4 + (1+k)
    assert_equal(b.shape[0], 2)
    assert_equal(b.shape[1], 2)
    assert_equal(b.shape[2], 2)
    assert_equal(Int(b.item(0, 0, 0)), 5)  # a[0,1,1]
    assert_equal(Int(b.item(0, 0, 1)), 6)  # a[0,1,2]
    assert_equal(Int(b.item(0, 1, 0)), 9)  # a[0,2,1]
    assert_equal(Int(b.item(1, 0, 0)), 17)  # a[1,1,1]
    assert_equal(Int(b.item(1, 1, 1)), 22)  # a[1,2,2]


# ===== Step 8: traverse_iterative_setter =====


def test_setitem_corder_non_trivial_source() raises:
    """traverse_iterative_setter: non-uniform source values written correctly.
    """
    # arange(16).reshape(4,4): write arange(10,14).reshape(2,2) into [1:3,1:3]
    # repl = [[10,11],[12,13]]
    # Expected: (1,1)=10, (1,2)=11, (2,1)=12, (2,2)=13
    var a = nm.arange[nm.i32](0, 16).reshape(Shape(4, 4))
    var repl = nm.arange[nm.i32](10, 14).reshape(Shape(2, 2))
    a[Slice(1, 3), Slice(1, 3)] = repl

    assert_equal(Int(a.item(1, 1)), 10)
    assert_equal(Int(a.item(1, 2)), 11)
    assert_equal(Int(a.item(2, 1)), 12)
    assert_equal(Int(a.item(2, 2)), 13)

    assert_equal(Int(a.item(0, 0)), 0)
    assert_equal(Int(a.item(1, 0)), 4)
    assert_equal(Int(a.item(3, 3)), 15)


def test_setitem_step_stride_corder() raises:
    """traverse_iterative_setter: step=2 skips correctly on C-order dest."""
    # arange(16).reshape(4,4): [::2, :] = -1 → rows 0,2 set to -1
    var a = nm.arange[nm.i32](0, 16).reshape(Shape(4, 4))
    var repl = nm.full[nm.i32](Shape(2, 4), fill_value=Scalar[nm.i32](-1))
    a[Slice(0, 4, 2), Slice(0, 4)] = repl

    for c in range(4):
        assert_equal(Int(a.item(0, c)), -1)
        assert_equal(Int(a.item(2, c)), -1)

    assert_equal(Int(a.item(1, 0)), 4)
    assert_equal(Int(a.item(1, 3)), 7)
    assert_equal(Int(a.item(3, 0)), 12)
    assert_equal(Int(a.item(3, 3)), 15)


def test_setitem_forder_non_trivial_source_step8() raises:
    """traverse_iterative_setter: F-order dest + non-uniform C-order source."""
    # arange(12).reshape(3,4,order="F"):
    #   row0=[0,3,6,9], row1=[1,4,7,10], row2=[2,5,8,11]
    # Write arange(10,14).reshape(2,2) = [[10,11],[12,13]] into [0:2,0:2]
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4), order="F")
    var repl = nm.arange[nm.i32](10, 14).reshape(Shape(2, 2))
    a[Slice(0, 2), Slice(0, 2)] = repl

    assert_equal(Int(a.item(0, 0)), 10)
    assert_equal(Int(a.item(0, 1)), 11)
    assert_equal(Int(a.item(1, 0)), 12)
    assert_equal(Int(a.item(1, 1)), 13)

    assert_equal(Int(a.item(0, 2)), 6)
    assert_equal(Int(a.item(2, 0)), 2)


def test_setitem_corder_source_count_correct() raises:
    """traverse_iterative_setter: all source elements consumed exactly once.

    Before Step 8, the loop ran narr.size (destination) times instead of
    orig.size (source) times. With a 1x4 source into a 2x4 destination
    (via step=2), the old code would have written 8 times from a 4-element
    source, producing wrong results for the second row.
    """
    # Destination: 4x4 zeros. Source: [10,20,30,40] (1x4).
    # Write into [0:4:2, 0:4] — rows 0 and 2.
    # Each of the 2 destination rows gets the same source row.
    var a = nm.zeros[nm.i32](Shape(4, 4))
    var repl = nm.array[nm.i32]("[[10, 20, 30, 40], [50, 60, 70, 80]]")
    a[Slice(0, 4, 2), Slice(0, 4)] = repl

    # Row 0 ← repl[0]
    assert_equal(Int(a.item(0, 0)), 10)
    assert_equal(Int(a.item(0, 1)), 20)
    assert_equal(Int(a.item(0, 2)), 30)
    assert_equal(Int(a.item(0, 3)), 40)
    # Row 2 ← repl[1]
    assert_equal(Int(a.item(2, 0)), 50)
    assert_equal(Int(a.item(2, 1)), 60)
    assert_equal(Int(a.item(2, 2)), 70)
    assert_equal(Int(a.item(2, 3)), 80)
    # Rows 1 and 3 untouched (zeros)
    assert_equal(Int(a.item(1, 0)), 0)
    assert_equal(Int(a.item(3, 3)), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
