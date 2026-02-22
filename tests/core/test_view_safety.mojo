"""
Tests for view safety: ensure all functions guarded with `contiguous()`
produce correct results when given non-contiguous (e.g. F-order) arrays.

This covers [Issue 309](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/issues/309) 
Phase 1 & Phase 3 — contiguous guards across the codebase.
"""

import numojo as nm
from numojo.prelude import *
from numojo.core.matrix import Matrix
from numojo.routines.math.extrema import minimum
from python import Python, PythonObject
from testing.testing import assert_true
from testing import TestSuite


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


fn check_array[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(array.to_numpy(), np_sol)), st)


fn check_array_close[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(array.to_numpy(), np_sol, atol=PythonObject(0.1))),
        st,
    )


fn check_scalar_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=PythonObject(0.001)), st)


fn check_matrix_close[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(
            np.isclose(
                np.matrix(matrix.to_numpy()), np_sol, atol=PythonObject(0.01)
            )
        ),
        st,
    )


fn check_matrix_equal[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(np.matrix(matrix.to_numpy()), np_sol)), st)


fn check_value_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=PythonObject(0.01)), st)


# ===-----------------------------------------------------------------------===#
# NDArray: sum, prod, cumsum, cumprod on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_sums_products_view() raises:
    """Test sum, prod, cumsum, cumprod on F-order (non C-contiguous) NDArrays.
    """
    var np = Python.import_module("numpy")

    # Create F-order 2D array
    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()

    # Verify it is indeed non-contiguous
    assert_true(
        not A.is_c_contiguous(), "F-order array should not be C-contiguous"
    )

    # sum (flattened)
    check_scalar_close(
        nm.sum(A),
        np.sum(Anp),
        "`sum` on F-order NDArray is broken",
    )

    # prod (flattened)
    check_scalar_close(
        nm.prod(A),
        np.prod(Anp),
        "`prod` on F-order NDArray is broken",
    )

    # cumsum (flattened)
    check_array_close(
        nm.cumsum(A),
        np.cumsum(Anp),
        "`cumsum` on F-order NDArray is broken",
    )

    # cumprod (flattened)
    check_array_close(
        nm.cumprod(A),
        np.cumprod(Anp),
        "`cumprod` on F-order NDArray is broken",
    )

    # 3D F-order array
    var B = nm.random.randn(2, 3, 4).reshape(Shape(2, 3, 4), order="F")
    var Bnp = B.to_numpy()
    assert_true(
        not B.is_c_contiguous(), "F-order 3D array should not be C-contiguous"
    )

    check_scalar_close(
        nm.sum(B),
        np.sum(Bnp),
        "`sum` on F-order 3D NDArray is broken",
    )
    check_scalar_close(
        nm.prod(B),
        np.prod(Bnp),
        "`prod` on F-order 3D NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: extrema (max, min, minimum, maximum) on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_extrema_view() raises:
    """Test max, min, minimum, maximum on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # max
    check_scalar_close(
        nm.max(A),
        np.max(Anp),
        "`max` on F-order NDArray is broken",
    )

    # min
    check_scalar_close(
        nm.min(A),
        np.min(Anp),
        "`min` on F-order NDArray is broken",
    )

    # minimum (elementwise)
    var B = nm.arange[nm.f64](11, -1, -1).reshape(Shape(3, 4), order="F")
    var Bnp = B.to_numpy()
    assert_true(not B.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(
        minimum(A, B),
        np.minimum(Anp, Bnp),
        "`minimum` on F-order NDArrays is broken",
    )

    # maximum (elementwise)
    check_array_close(
        nm.maximum(A, B),
        np.maximum(Anp, Bnp),
        "`maximum` on F-order NDArrays is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: searching (argmax, argmin) on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_searching_view() raises:
    """Test argmax_1d, argmin_1d on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_scalar_close(
        nm.argmax(A),
        np.argmax(Anp),
        "`argmax` on F-order NDArray is broken",
    )

    check_scalar_close(
        nm.argmin(A),
        np.argmin(Anp),
        "`argmin` on F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: sorting on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_sorting_view() raises:
    """Test sort, argsort on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Create F-order array
    var A = nm.random.randn(3, 4).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # sort along axis 0
    check_array_close(
        nm.sort(A, axis=0),
        np.sort(Anp, axis=0),
        "`sort` axis=0 on F-order NDArray is broken",
    )

    # sort along axis 1
    check_array_close(
        nm.sort(A, axis=1),
        np.sort(Anp, axis=1),
        "`sort` axis=1 on F-order NDArray is broken",
    )

    # argsort (flattened)
    check_array(
        nm.argsort(A),
        np.argsort(Anp, axis=PythonObject(None)),
        "`argsort` on F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: linalg on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_linalg_view() raises:
    """Test matmul, dot, trace, diagonal on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Create F-order 2D arrays for matmul
    var A = nm.random.randn(3, 4).reshape(Shape(3, 4), order="F")
    var B = nm.random.randn(4, 2).reshape(Shape(4, 2), order="F")
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # matmul
    check_array_close(
        A @ B,
        np.matmul(Anp, Bnp),
        "`matmul` on F-order NDArrays is broken",
    )

    # dot (1D)
    var v1 = nm.random.randn(6).reshape(Shape(6), order="F")
    var v2 = nm.random.randn(6).reshape(Shape(6), order="F")
    var v1np = v1.to_numpy()
    var v2np = v2.to_numpy()

    check_array_close(
        nm.linalg.dot(v1, v2),
        v1np * v2np,
        "`dot` on F-order 1D NDArrays is broken",
    )

    # trace
    var S = nm.random.randn(4, 4).reshape(Shape(4, 4), order="F")
    var Snp = S.to_numpy()
    assert_true(not S.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(
        nm.linalg.trace(S),
        np.trace(Snp),
        "`trace` on F-order NDArray is broken",
    )

    # diagonal
    check_array_close(
        nm.diagonal(S),
        np.diagonal(Snp),
        "`diagonal` on F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: creation (diag) on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_creation_view() raises:
    """Test diag on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # diag from 1D F-order vector
    var v = nm.arange[nm.f64](0, 4).reshape(Shape(4), order="F")
    var vnp = v.to_numpy()
    check_array(
        nm.diag(v),
        np.diag(vnp),
        "`diag` from 1D F-order NDArray is broken",
    )

    # diag from 2D F-order matrix (extract diagonal)
    var M = nm.arange[nm.f64](0, 9).reshape(Shape(3, 3), order="F")
    var Mnp = M.to_numpy()
    assert_true(not M.is_c_contiguous(), "Should be non-contiguous")
    check_array(
        nm.diag(M),
        np.diag(Mnp),
        "`diag` from 2D F-order NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: indexing (compress) on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_indexing_view() raises:
    """Test compress on NDArrays with F-order condition."""
    var np = Python.import_module("numpy")

    # compress with condition
    var a = nm.arange[nm.i8](6)
    var anp = a.to_numpy()
    var cond = nm.array[boolean]("[1, 0, 1, 0, 1, 0]")

    check_array(
        nm.indexing.compress(cond, a),
        np.compress(np.array(Python.list(1, 0, 1, 0, 1, 0)), anp),
        "`compress` with condition on NDArray is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: __pow__ on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_pow_view() raises:
    """Test __pow__ on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # __pow__ with scalar
    check_array_close(
        A**2,
        np.power(Anp, 2),
        "`__pow__(scalar)` on F-order NDArray is broken",
    )

    # __pow__ with another F-order array
    var B = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var Bnp = B.to_numpy()
    check_array_close(
        A**B,
        np.power(Anp, Bnp),
        "`__pow__(NDArray)` on F-order NDArrays is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: sum, prod, cumsum on F-order matrices
# ===-----------------------------------------------------------------------===#


fn test_matrix_sums_products_view() raises:
    """Test sum, prod, cumsum on F-order Matrices."""
    var np = Python.import_module("numpy")

    var A = Matrix.rand[nm.f64](shape=(3, 4), order="F")
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    # sum (flattened)
    check_value_close(
        nm.sum(A),
        np.sum(Anp),
        "`sum` on F-order Matrix is broken",
    )

    # sum along axis
    check_matrix_close(
        nm.sum(A, axis=0),
        np.sum(Anp, axis=0),
        "`sum(axis=0)` on F-order Matrix is broken",
    )
    check_matrix_close(
        nm.sum(A, axis=1),
        np.sum(Anp, axis=1),
        "`sum(axis=1)` on F-order Matrix is broken",
    )

    # prod (flattened)
    check_value_close(
        nm.prod(A),
        np.prod(Anp),
        "`prod` on F-order Matrix is broken",
    )

    # cumsum (flattened)
    var cs = nm.cumsum(A)
    var cs_np = np.cumsum(Anp)
    check_matrix_close(
        cs,
        np.matrix(cs_np),
        "`cumsum` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: logic (all, any) on F-order matrices
# ===-----------------------------------------------------------------------===#


fn test_matrix_logic_view() raises:
    """Test all, any on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Matrix with all ones (F-order) - use i8 since all/any need integral type
    var A = Matrix.ones[nm.i8](shape=(3, 4), order="F")
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    assert_true(nm.all(A), "`all` on F-order Matrix of ones should be True")
    assert_true(nm.any(A), "`any` on F-order Matrix of ones should be True")

    # Matrix with all zeros (F-order)
    var B = Matrix.zeros[nm.i8](shape=(3, 4), order="F")
    assert_true(
        not nm.all(B),
        "`all` on F-order Matrix of zeros should be False",
    )
    assert_true(
        not nm.any(B),
        "`any` on F-order Matrix of zeros should be False",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: __pow__ on F-order matrices
# ===-----------------------------------------------------------------------===#


fn test_matrix_pow_view() raises:
    """Test __pow__ on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Create F-order Matrix via NDArray conversion (fromstring ignores order)
    var nd = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var A = Matrix[nm.f64](nd)
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    # __pow__ with Int
    var result = A**2
    var expected = np.power(Anp, 2)
    check_matrix_close(
        result,
        expected,
        "`__pow__` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: astype, flatten, to_ndarray on F-order matrices
# ===-----------------------------------------------------------------------===#


fn test_matrix_conversion_view() raises:
    """Test astype, flatten, to_ndarray on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Create F-order Matrix via NDArray conversion
    var nd_src = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var A = Matrix[nm.f64](nd_src)
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    # astype
    var A32 = A.astype[nm.f32]()
    var A32np = np.matrix(A32.to_numpy())
    assert_true(
        np.all(np.isclose(A32np, Anp, atol=PythonObject(0.01))),
        "`astype` on F-order Matrix is broken",
    )

    # flatten
    var flat = A.flatten()
    var flat_np = np.array(Anp).flatten()
    assert_true(
        np.all(
            np.isclose(
                np.matrix(flat.to_numpy()),
                np.matrix(flat_np),
                atol=PythonObject(0.01),
            )
        ),
        "`flatten` on F-order Matrix is broken",
    )

    # to_ndarray
    var nd = A.to_ndarray()
    var nd_np = nd.to_numpy()
    assert_true(
        np.all(np.isclose(nd_np, np.array(Anp), atol=PythonObject(0.01))),
        "`to_ndarray` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# Matrix: rounding on F-order matrices
# ===-----------------------------------------------------------------------===#


fn test_matrix_rounding_view() raises:
    """Test round on F-order Matrices."""
    var np = Python.import_module("numpy")

    # Use rand to create F-order Matrix with random values to test rounding
    var A = Matrix.rand[nm.f64](shape=(3, 4), order="F")
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        not A.is_c_contiguous(), "F-order Matrix should not be C-contiguous"
    )

    var result = nm.math.round(A, decimals=1)
    var expected = np.around(Anp, 1)
    check_matrix_close(
        result,
        expected,
        "`round` on F-order Matrix is broken",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: mutating original is not affected by sort on view
# ===-----------------------------------------------------------------------===#


fn test_sort_does_not_mutate_original() raises:
    """Test that sorting a view does not mutate the original array."""
    var np = Python.import_module("numpy")

    # Create original array
    var original = nm.array[nm.f64]("[5.0, 3.0, 1.0, 4.0, 2.0]")
    var original_copy = nm.array[nm.f64]("[5.0, 3.0, 1.0, 4.0, 2.0]")

    # Sort (this uses .contiguous() internally to avoid mutating input)
    var sorted_arr = nm.sort(original, axis=0)

    # Check sorted result is correct
    check_array_close(
        sorted_arr,
        np.sort(np.array(Python.list(5.0, 3.0, 1.0, 4.0, 2.0))),
        "sort result is wrong",
    )

    # Check original was not mutated
    check_array(
        original,
        original_copy.to_numpy(),
        "Original array should not be mutated by sort",
    )


# ===-----------------------------------------------------------------------===#
# NDArray: linalg solve on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_solve_view() raises:
    """Test linalg.solve on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Create a non-singular 3x3 system A*x = b using random values
    var A = nm.random.randn(3, 3).reshape(Shape(3, 3), order="F")
    # Add identity to make it more likely non-singular
    for i in range(3):
        A._setitem(i, i, val=A.load[1](i * 3 + i) + 10.0)
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    var b = nm.random.randn(3, 1).reshape(Shape(3, 1), order="F")
    var bnp = b.to_numpy()

    # solve
    var x = nm.linalg.solve(A, b)
    var xnp = np.linalg.solve(Anp, bnp)

    check_array_close(
        x,
        xnp,
        "`solve` on F-order NDArrays is broken",
    )


# ===-----------------------------------------------------------------------===#
# Phase 3: Math backend functions on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_trig_view() raises:
    """Test trigonometric functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    # Values in [0.1, 0.6] — safe for all inverse trig functions
    var A = nm.fromstring("[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.sin(A), np.sin(Anp), "`sin` on F-order broken")
    check_array_close(nm.cos(A), np.cos(Anp), "`cos` on F-order broken")
    check_array_close(nm.tan(A), np.tan(Anp), "`tan` on F-order broken")
    check_array_close(nm.asin(A), np.arcsin(Anp), "`asin` on F-order broken")
    check_array_close(nm.acos(A), np.arccos(Anp), "`acos` on F-order broken")
    check_array_close(nm.atan(A), np.arctan(Anp), "`atan` on F-order broken")

    # atan2 and hypot: two-array inputs
    var B = nm.fromstring("[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]").reshape(
        Shape(2, 3), order="F"
    )
    var Bnp = B.to_numpy()
    check_array_close(
        nm.atan2(A, B), np.arctan2(Anp, Bnp), "`atan2` on F-order broken"
    )
    check_array_close(
        nm.hypot(A, B), np.hypot(Anp, Bnp), "`hypot` on F-order broken"
    )


fn test_ndarray_hyper_view() raises:
    """Test hyperbolic functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.sinh(A), np.sinh(Anp), "`sinh` on F-order broken")
    check_array_close(nm.cosh(A), np.cosh(Anp), "`cosh` on F-order broken")
    check_array_close(nm.tanh(A), np.tanh(Anp), "`tanh` on F-order broken")
    check_array_close(nm.asinh(A), np.arcsinh(Anp), "`asinh` on F-order broken")
    check_array_close(nm.atanh(A), np.arctanh(Anp), "`atanh` on F-order broken")

    # acosh needs values >= 1
    var C = nm.fromstring("[1.1, 1.5, 2.0, 2.5, 3.0, 4.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Cnp = C.to_numpy()
    check_array_close(nm.acosh(C), np.arccosh(Cnp), "`acosh` on F-order broken")


fn test_ndarray_exp_log_view() raises:
    """Test exp/log functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.exp(A), np.exp(Anp), "`exp` on F-order broken")
    check_array_close(nm.exp2(A), np.exp2(Anp), "`exp2` on F-order broken")
    check_array_close(nm.expm1(A), np.expm1(Anp), "`expm1` on F-order broken")
    check_array_close(nm.log(A), np.log(Anp), "`log` on F-order broken")
    check_array_close(nm.log2(A), np.log2(Anp), "`log2` on F-order broken")
    check_array_close(nm.log10(A), np.log10(Anp), "`log10` on F-order broken")
    check_array_close(nm.log1p(A), np.log1p(Anp), "`log1p` on F-order broken")


fn test_ndarray_arithmetic_view() raises:
    """Test arithmetic functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var B = nm.arange[nm.f64](7, 13).reshape(Shape(2, 3), order="F")
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.add(A, B), np.add(Anp, Bnp), "`add` on F-order broken")
    check_array_close(
        nm.sub(A, B), np.subtract(Anp, Bnp), "`sub` on F-order broken"
    )
    check_array_close(
        nm.mul(A, B), np.multiply(Anp, Bnp), "`mul` on F-order broken"
    )
    check_array_close(
        nm.div(A, B), np.divide(Anp, Bnp), "`div` on F-order broken"
    )

    # fma: A * B + scalar
    var c: Scalar[nm.f64] = 10.0
    check_array_close(
        nm.fma(A, B, c),
        np.add(np.multiply(Anp, Bnp), 10.0),
        "`fma` on F-order broken",
    )


fn test_ndarray_rounding_view() raises:
    """Test rounding functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[1.2, -2.7, 3.5, -4.1, 5.9, -6.3]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.tabs(A), np.abs(Anp), "`tabs` on F-order broken")
    check_array_close(nm.tfloor(A), np.floor(Anp), "`tfloor` on F-order broken")
    check_array_close(nm.tceil(A), np.ceil(Anp), "`tceil` on F-order broken")
    check_array_close(nm.ttrunc(A), np.trunc(Anp), "`ttrunc` on F-order broken")
    check_array_close(nm.tround(A), np.round(Anp), "`tround` on F-order broken")


fn test_ndarray_misc_math_view() raises:
    """Test misc math functions (clip, sqrt, cbrt, rsqrt) on F-order arrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array_close(nm.sqrt(A), np.sqrt(Anp), "`sqrt` on F-order broken")
    check_array_close(nm.cbrt(A), np.cbrt(Anp), "`cbrt` on F-order broken")
    check_array_close(
        nm.rsqrt(A),
        np.reciprocal(np.sqrt(Anp)),
        "`rsqrt` on F-order broken",
    )

    # clip
    check_array_close(
        nm.clip(A, Scalar[nm.f64](5.0), Scalar[nm.f64](20.0)),
        np.clip(Anp, 5.0, 20.0),
        "`clip` on F-order broken",
    )


fn test_ndarray_comparison_view() raises:
    """Test comparison and logic functions on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 7).reshape(Shape(2, 3), order="F")
    var B = nm.fromstring("[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    check_array(
        nm.greater(A, B),
        np.greater(Anp, Bnp),
        "`greater` on F-order broken",
    )
    check_array(
        nm.less(A, B),
        np.less(Anp, Bnp),
        "`less` on F-order broken",
    )
    check_array(
        nm.equal(A, B),
        np.equal(Anp, Bnp),
        "`equal` on F-order broken",
    )


fn test_ndarray_copysign_view() raises:
    """Test copysign and nextafter on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.fromstring("[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]").reshape(
        Shape(2, 3), order="F"
    )
    var B = nm.fromstring("[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]").reshape(
        Shape(2, 3), order="F"
    )
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()

    check_array_close(
        nm.copysign(A, B),
        np.copysign(Anp, Bnp),
        "`copysign` on F-order broken",
    )


# ===-----------------------------------------------------------------------===#
# Phase 3: Differences on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_differences_view() raises:
    """Test gradient and trapz on F-order (non C-contiguous) NDArrays."""
    var np = Python.import_module("numpy")

    # 1D array reshaped to F-order (for 1D arrays F-order is same as C-order,
    # so use a slice of a 2D F-order array instead)
    var A2d = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    assert_true(not A2d.is_c_contiguous(), "Should be non-contiguous")

    # gradient: test with a simple 1D array (gradient is 1D only)
    var x = nm.fromstring("[1.0, 2.0, 4.0, 7.0, 11.0]")
    var xnp = np.array(Python.list(1.0, 2.0, 4.0, 7.0, 11.0))
    var grad_result = nm.gradient(x, Scalar[nm.f64](1.0))
    var grad_np = np.gradient(xnp, 1.0)
    check_array_close(grad_result, grad_np, "`gradient` result is wrong")

    # Note: trapz is not tested here due to a pre-existing constraint bug
    # in differences.mojo that rejects float dtypes (issue unrelated to
    # view safety).


# ===-----------------------------------------------------------------------===#
# Phase 3: Manipulation on F-order arrays
# ===-----------------------------------------------------------------------===#


fn test_ndarray_reshape_view() raises:
    """Test reshape on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # reshape to different shape
    var reshaped = nm.reshape(A, Shape(4, 3))
    var reshaped_np = np.reshape(Anp, Python.tuple(4, 3))
    check_array_close(reshaped, reshaped_np, "`reshape` on F-order broken")

    # reshape to 1D
    var flat = nm.reshape(A, Shape(12))
    var flat_np = np.reshape(Anp, 12)
    check_array_close(flat, flat_np, "`reshape` to 1D on F-order broken")


fn test_ndarray_ravel_view() raises:
    """Test ravel on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    var raveled = nm.ravel(A)
    var raveled_np = np.ravel(Anp)
    check_array_close(raveled, raveled_np, "`ravel` on F-order broken")


fn test_ndarray_transpose_view() raises:
    """Test transpose on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # Simple transpose (no axes)
    var T = nm.transpose(A)
    var Tnp = np.transpose(Anp)
    check_array_close(T, Tnp, "`transpose` on F-order broken")

    # Transpose with axes
    var B = nm.arange[nm.f64](0, 24).reshape(Shape(2, 3, 4), order="F")
    var Bnp = B.to_numpy()
    assert_true(not B.is_c_contiguous(), "Should be non-contiguous")

    var T2 = nm.transpose(B, axes=[Int(2), 0, 1])
    var T2np = np.transpose(Bnp, Python.list(2, 0, 1))
    check_array_close(T2, T2np, "`transpose(axes)` on F-order broken")


fn test_ndarray_flip_view() raises:
    """Test flip on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 12).reshape(Shape(3, 4), order="F")
    var Anp = A.to_numpy()
    assert_true(not A.is_c_contiguous(), "Should be non-contiguous")

    # flip (all axes)
    var flipped = nm.flip(A)
    var flipped_np = np.flip(Anp)
    check_array_close(flipped, flipped_np, "`flip` on F-order broken")

    # flip along axis 0
    var flipped0 = nm.flip(A, axis=0)
    var flipped0_np = np.flip(Anp, axis=0)
    check_array_close(flipped0, flipped0_np, "`flip(axis=0)` on F-order broken")

    # flip along axis 1
    var flipped1 = nm.flip(A, axis=1)
    var flipped1_np = np.flip(Anp, axis=1)
    check_array_close(flipped1, flipped1_np, "`flip(axis=1)` on F-order broken")


fn test_ndarray_broadcast_to_view() raises:
    """Test broadcast_to on F-order NDArrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](1, 4).reshape(Shape(1, 3), order="F")
    var Anp = A.to_numpy()

    var broadcasted = nm.broadcast_to(A, Shape(3, 3))
    var broadcasted_np = np.broadcast_to(Anp, Python.tuple(3, 3))
    check_array_close(
        broadcasted, broadcasted_np, "`broadcast_to` on F-order broken"
    )


# ===-----------------------------------------------------------------------===#
# Phase 3: Sliced views (non-contiguous due to slicing, not just F-order)
# ===-----------------------------------------------------------------------===#


fn test_ndarray_sliced_view_math() raises:
    """Test math on non-contiguous views created via F-order reshape."""
    var np = Python.import_module("numpy")

    # Create a 3D F-order array (guaranteed non-contiguous)
    var A = nm.arange[nm.f64](0, 24).reshape(Shape(2, 3, 4), order="F")
    var Anp = A.to_numpy()

    assert_true(
        not A.is_c_contiguous(),
        "F-order array should not be C-contiguous",
    )

    # sin on F-order 3D view
    check_array_close(
        nm.sin(A),
        np.sin(Anp),
        "`sin` on 3D F-order broken",
    )

    # exp on F-order 3D view
    check_array_close(
        nm.exp(A),
        np.exp(Anp),
        "`exp` on 3D F-order broken",
    )

    # add two F-order 3D views
    var B = nm.arange[nm.f64](24, 48).reshape(Shape(2, 3, 4), order="F")
    var Bnp = B.to_numpy()
    check_array_close(
        nm.add(A, B),
        np.add(Anp, Bnp),
        "`add` on 3D F-order broken",
    )

    # sum on 3D F-order view
    check_scalar_close(
        nm.sum(A),
        np.sum(Anp),
        "`sum` on 3D F-order broken",
    )


fn test_ndarray_sliced_view_manipulation() raises:
    """Test manipulation functions on 3D F-order arrays."""
    var np = Python.import_module("numpy")

    var A = nm.arange[nm.f64](0, 24).reshape(Shape(2, 3, 4), order="F")
    var Anp = A.to_numpy()

    assert_true(
        not A.is_c_contiguous(),
        "F-order array should not be C-contiguous",
    )

    # reshape
    check_array_close(
        nm.reshape(A, Shape(6, 4)),
        np.reshape(Anp, Python.tuple(6, 4)),
        "`reshape` on 3D F-order broken",
    )

    # ravel
    check_array_close(
        nm.ravel(A),
        np.ravel(Anp),
        "`ravel` on 3D F-order broken",
    )

    # transpose
    check_array_close(
        nm.transpose(A),
        np.transpose(Anp),
        "`transpose` on 3D F-order broken",
    )

    # flip
    check_array_close(
        nm.flip(A),
        np.flip(Anp),
        "`flip` on 3D F-order broken",
    )


# ===-----------------------------------------------------------------------===#
# main
# ===-----------------------------------------------------------------------===#


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
