# import numojo as nj
import time
from benchmark.compiler import keep
from python import Python, PythonObject

from random import seed
from random.random import randint, random_float64

import numojo as nm
from numojo import *
from time import now


fn test_constructors1() raises:
    var arr1 = NDArray[f32](3, 4, 5)
    print(arr1)
    print("ndim: ", arr1.ndim)
    print("shape: ", arr1.shape)
    print("strides: ", arr1.stride)
    print("size: ", arr1.shape.ndsize)
    print("offset: ", arr1.stride.offset)
    print("dtype: ", arr1.dtype)

    var arr2 = NDArray[f32](VariadicList[Int](3, 4, 5))
    print(arr2)
    print("ndim: ", arr2.ndim)
    print("shape: ", arr2.shape)
    print("strides: ", arr2.stride)
    print("size: ", arr2.shape.ndsize)
    print("offset: ", arr2.stride.offset)
    print("dtype: ", arr2.dtype)

    var arr3 = NDArray[f32](VariadicList[Int](3, 4, 5), fill=Scalar[f32](10.0))
    print(arr3)
    print("ndim: ", arr3.ndim)
    print("shape: ", arr3.shape)
    print("strides: ", arr3.stride)
    print("size: ", arr3.shape.ndsize)
    print("offset: ", arr3.stride.offset)
    print("dtype: ", arr3.dtype)

    var arr4 = NDArray[f32](List[Int](3, 4, 5))
    print(arr4)
    print("ndim: ", arr4.ndim)
    print("shape: ", arr4.shape)
    print("strides: ", arr4.stride)
    print("size: ", arr4.shape.ndsize)
    print("offset: ", arr4.stride.offset)
    print("dtype: ", arr4.dtype)

    var arr5 = NDArray[f32](NDArrayShape(3, 4, 5))
    print(arr5)
    print("ndim: ", arr5.ndim)
    print("shape: ", arr5.shape)
    print("strides: ", arr5.stride)
    print("size: ", arr5.shape.ndsize)
    print("offset: ", arr5.stride.offset)
    print("dtype: ", arr5.dtype)

    var arr6 = NDArray[f32](
        data=List[SIMD[f32, 1]](1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        shape=List[Int](2, 5),
    )
    print(arr6)
    print("ndim: ", arr6.ndim)
    print("shape: ", arr6.shape)
    print("strides: ", arr6.stride)
    print("size: ", arr6.shape.ndsize)
    print("offset: ", arr6.stride.offset)
    print("dtype: ", arr6.dtype)

    var arr7 = nm.NDArray[nm.f32](String("[[1.0,0,1], [0,2,1], [1,1,1]]"))
    print(arr7)


fn test_constructors2() raises:
    # var fill_value: SIMD[i32, 1] = 10
    # var A = NDArray[i32](3, 2, fill=fill_value)
    # print(A)
    var np = Python.import_module("numpy")
    var A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    var B = NDArray[f32](data=A)
    print(B)


fn test_random() raises:
    var arr_variadic = nm.core.random.rand(
        shape=List[Int](10, 10, 10), min=1.0, max=2.0
    )
    print(arr_variadic)
    var random_array_var = nm.core.random.randn[i16](3, 2, mean=0, variance=5)
    print(random_array_var)
    var random_array_list = nm.core.random.randn[i16](
        List[Int](3, 2), mean=0, variance=1
    )
    print(random_array_list)

    var random_array_var1 = nm.core.random.rand[f16](3, 2, min=0, max=100)
    print(random_array_var1)
    var random_array_list1 = nm.core.random.rand[i32](
        List[Int](3, 2), min=0, max=100
    )
    print(random_array_list1)


fn test_arr_manipulation() raises:
    var np = Python.import_module("numpy")
    var A = np.arange(12)
    print(A)
    var temp = NDArray[f64](data=A)
    print(temp)
    temp.T()
    print(temp)
    A.reshape(2, 3, order="F")
    # nm.ravel(A)
    print(A)
    var B = arange[i16](0, 12, 1)
    B.reshape(3, 2, 2, order="F")
    ravel(B, order="C")
    print(B)

    var array = arange[i16](0, 12, 1)
    array.reshape(3, 2, 2)
    print(array)
    print("x[0]", array.item(0))
    print("x[1]", array.item(1))
    print("x[2]", array.item(2))
    # swapaxis(array, 0, -1)
    # print(array.shape)
    # swapaxis(array, 0, 2)
    # print(array.shape)
    # moveaxis(array, 0, 2)
    # array.reshape(2, 2, 3, order="C")
    # swapaxis(array, 0, 1)
    # print(array)
    # print("x[0]", array.item(0))
    # print("x[1]", array.item(1))
    # print("x[2]", array.item(2))
    # moveaxis(array, 0, 1)


fn test_bool_masks1() raises:
    var A = nm.core.random.rand[i16](3, 2, 2)
    print(A.shape)
    seed(10)
    var B = nm.core.random.rand[i16](3, 2, 2)
    print(B)
    var mask: NDArray[DType.bool] = A > Scalar[i16](10)
    A[mask] = B
    print(A)
    var gt = A > Scalar[i16](10)
    print(gt)
    var ge = A >= Scalar[i16](10)
    print(ge)
    var lt = A < Scalar[i16](10)
    print(lt)
    var le = A <= Scalar[i16](10)
    print(le)
    var eq = A == Scalar[i16](10)
    print(eq)
    var ne = A != Scalar[i16](10)
    print(ne)
    var mask1 = A[A > Scalar[i16](10)]
    print(mask1)
    seed(12)


fn test_bool_masks2() raises:
    var np = Python.import_module("numpy")
    var np_A = np.arange(0, 24, dtype=np.int16).reshape((3, 2, 4))
    var A = nm.arange[nm.i16](0, 24)
    A.reshape(3, 2, 4)

    # Test greater than
    var np_gt = np_A > 10
    var gt = A > Scalar[nm.i16](10)
    print(gt)
    print(np_gt)
    gt.to_numpy()
    print(gt)

    var AA = nm.core.random.rand[i16](3, 3)
    var BB = nm.core.random.rand[i16](3, 3)
    print(BB)
    var gta = AA > BB
    print(gta)
    var gte = AA >= BB
    print(gte)
    var lt = AA < BB
    print(lt)
    var le = AA <= BB
    print(le)
    var eq = AA == BB
    print(eq)
    var ne = AA != BB
    print(ne)
    var mask = AA[AA > BB]
    print(mask)
    var temp = AA * SIMD[i16, 1](2)
    print(temp)
    var temp1 = temp == SIMD[i16, 1](0)
    var temp2 = AA[temp1]
    print(temp2)
    var temp3 = AA[AA % SIMD[i16, 1](2) == SIMD[i16, 1](0)]
    print(temp3)
    print(temp3.get(0))
    print(temp3.shape, temp3.stride, temp3.shape.ndsize)


fn test_creation_routines() raises:
    var x = linspace[numojo.f32](0.0, 60.0, 60)
    var y = ones[numojo.f32](shape(3, 2))
    var z = logspace[numojo.f32](-3, 0, 60)
    var w = arange[f32](0.0, 24.0, step=1)
    print(x)
    print(y)
    print(z)
    print(w)


fn test_creation_routines1() raises:
    var rand_matrix = nm.random.rand[f32](3, 3, min=0, max=100)
    var diag_matrix = nm.diag[f32](rand_matrix, k=0)
    print("rand_matrix: ", rand_matrix)
    print("diag_matrix: ", diag_matrix)
    var diag_matrix_k1 = nm.diag[f32](rand_matrix, k=1)
    print("diag_matrix_k1: ", diag_matrix_k1)
    var diag_matrix_k2 = nm.diag[f32](rand_matrix, k=2)
    print("diag_matrix_k2: ", diag_matrix_k2)
    var diag_of_diag = nm.diag[f32](diag_matrix_k1, k=0)
    print("diag_of_diag: ", diag_of_diag)

    var diagflat_matrix = diagflat[f32](rand_matrix, k=0)
    print("diagflat_matrix: ", diagflat_matrix)
    var diagflat_matrix_km1 = diagflat[f32](rand_matrix, k=-1)
    print("diagflat_matrix_km1: ", diagflat_matrix_km1)

    var np = Python.import_module("numpy")
    var arange_matrix = nm.arange[nm.i64](0, 4, 1)
    arange_matrix.reshape(2, 2)
    var diagflat_k2 = nm.diagflat[nm.i64](arange_matrix, k=2)
    print()
    var diagflat_km1 = nm.diagflat[nm.i64](arange_matrix, k=-1)
    print("diagflat_k2: ", diagflat_k2)
    print("diagflat_km1: ", diagflat_km1)

    var tri_matrix = nm.tri[f32](4, 4, k=-2)
    print("tri_matrix: ", tri_matrix)
    var reshaped_arange = nm.arange[nm.i64](1, 13, 1)
    reshaped_arange.reshape(4, 3)
    print("reshaped_arange: ", reshaped_arange)
    var triu_matrix = nm.triu[nm.i64](reshaped_arange, k=-1)
    print("triu_matrix: ", triu_matrix)
    var array_input = nm.array[nm.i64](String("[1, 2, 3, 5]"))
    var vander_matrix = nm.vander[nm.i64](array_input, N=3)
    var vander_matrix_inc = nm.vander[nm.i64](array_input, N=3, increasing=True)
    print(vander_matrix)
    print(vander_matrix_inc)
    var full_vander = nm.vander[nm.i64](array_input)
    var full_vander_inc = nm.vander[nm.i64](array_input, increasing=True)
    print(full_vander)
    print(full_vander_inc)


fn test_slicing() raises:
    var raw = List[Int32]()
    for _ in range(16):
        raw.append(random.random_float64().cast[i32]() * 10)
    var arr1 = numojo.NDArray[numojo.i32](
        data=raw, shape=List[Int](4, 4), order="C"
    )
    print(arr1)
    print(arr1[0, 1])
    print(arr1[0:1, :])
    print(arr1[1:2, 3:4])

    # var w = arange[f32](0.0, 24.0, step=1)
    # w.reshape(2, 3, 4, order="C")
    # print(w[0, 0, 0], w[0, 0, 1], w[1, 1, 1], w[1, 2, 3])
    # print(w)
    # var slicedw = w[0:1, :, 1:2]
    # print(slicedw)
    # print()

    var y = arange[numojo.f32](0.0, 24.0, step=1)
    y.reshape(3, 2, 4, order="C")
    # print(y[0, 0, 0], y[0, 0, 1], y[1, 1, 1], y[1, 2, 3])
    print("y: ", y)
    print(y.order)

    # var start = time.now()
    # for _ in range(10):
    var slicedy = y[:, :, 1:2]
    print(slicedy)
    var slicedy1 = y[0:1, :, :]
    print(slicedy1)
    var slicedy2 = y[:, 0:1, :]
    print(slicedy2)
    var slicedy3 = y[0:1, 0:1, 0:1]
    print(slicedy3)
    # print("Time taken: ", (time.now() - start)/1e9/10)

    # var np = Python.import_module("numpy")
    # y = nm.arange[nm.f32](0.0, 24.0, step=1)
    # y.reshape(2, 3, 4, order="C")
    # np_y = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4, order="C")
    # print(y)
    # print(np_y)
    # print()
    # # Test slicing
    # slicedy = y[:, :, 1:2]
    # print("slicedy: ", slicedy)
    # np_slicedy = np.take(
    #     np.take(
    #         np.take(np_y, np.arange(0, 2), axis=0), np.arange(0, 3), axis=1
    #     ),
    #     np.arange(1, 2),
    #     axis=2,
    # )
    # print("np_slicedy: ", np_slicedy)
    # np_slicedy = np.squeeze(
    #     np_slicedy, axis=2
    # )  # Remove the dimension with size 1
    # var np_arr = slicedy.to_numpy()
    # print()
    # print(np_arr)
    # print(np_slicedy)
    # print(np.all(np.equal(np_arr, np_slicedy)))


fn test_rand_funcs[
    dtype: DType = DType.float64
](shape: List[Int], min: Scalar[dtype], max: Scalar[dtype]) raises -> NDArray[
    dtype
]:
    var result: NDArray[dtype] = NDArray[dtype](shape)
    if dtype.is_integral():
        randint[dtype](
            ptr=result._buf,
            size=result.shape.ndsize,
            low=int(min),
            high=int(max),
        )
    elif dtype.is_floating_point():
        for i in range(result.shape.ndsize):
            var temp: Scalar[dtype] = random.random_float64(
                min.cast[f64](), max.cast[f64]()
            ).cast[dtype]()
            result.set(i, temp)
    else:
        raise Error(
            "Invalid type provided. dtype must be either an integral or"
            " floating-point type."
        )
    return result


fn test_linalg() raises:
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 100)
    arr.reshape(10, 10)
    var np_arr = np.arange(0, 100).reshape(10, 10)
    print(arr)
    print(nm.math.linalg.matmul_naive[f64](arr, arr))
    print(np.matmul(np_arr, np_arr))
    # The only matmul that currently works is par (__matmul__)
    # check_is_close(nm.matmul_tiled_unrolled_parallelized(arr,arr),np.matmul(np_arr,np_arr),"TUP matmul is broken")


def test_inv():
    var np = Python.import_module("numpy")
    var arr = nm.core.random.rand(5, 5)
    var np_arr = arr.to_numpy()
    print("arr: ", arr)
    print("np_arr: ", np_arr)
    print(nm.math.linalg.inv(arr))
    print(np.linalg.inv(np_arr))


def test_solve():
    var np = Python.import_module("numpy")
    var A = nm.core.random.randn(10, 10)
    var B = nm.core.random.randn(10, 5)
    var A_np = A.to_numpy()
    var B_np = B.to_numpy()
    print(
        nm.math.linalg.solver.solve(A, B),
        np.linalg.solve(A_np, B_np),
    )


fn test_setter() raises:
    print("Testing setter")
    var A = nm.full[i16](shape(3, 3, 3), fill_value=1)
    var B = nm.full[i16](shape(3, 3), fill_value=2)
    A[0] = B
    print(A)

    var A1 = nm.full[i16](shape(3, 4, 5), fill_value=1)
    print("A1: ", A1)
    var D1 = nm.random.rand[i16](3, 5, min=0, max=100)
    A1[:, 0:1, :] = D1  # sets the elements of A[:, 0:1, :] with the array `D`
    print("A3: ", A1)
    var D = nm.random.rand[i16](4, 5, min=0, max=100)
    A1[1] = D  # sets the elements of A[1:2, :, :] with the array `D`
    print("A2: ", A1)


fn main() raises:
    # test_constructors1()
    # test_constructors2()
    # test_random()
    # test_arr_manipulation()
    # test_bool_masks1()
    # test_bool_masks2()
    # test_creation_routines()
    # test_creation_routines1()
    # test_slicing()
    # test_inv1()
    # test_inv()
    # test_solve()
    # test_linalg()
    # test_setter()
    var a = nm.linspace[nm.f64](0, 1, 3)
    var b = nm.linspace[nm.f64](0, 1, 2)
    # var c = nm.arange[nm.i32](20, 30, 1)
    var d = nm.meshgrid[nm.f64](List[NDArray[nm.f64]](a, b))
    print(d[0], d[1])
