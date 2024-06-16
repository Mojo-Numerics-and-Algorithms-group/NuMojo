from tensor import Tensor, TensorShape
import numojo as nj

fn main() raises:
    var x = Tensor[DType.float32](TensorShape(4), List[Float32](5.0, 7.0, 10.0, 12.0))
    var y = Tensor[DType.float32](TensorShape(4), List[Float32](17.0, 2.0, 3.0, 4.0))
    var trpzfloat = nj.trapz[DType.float32, DType.float32](y, x) 
    print(trpzfloat) # prints 33.5 that matches numpy

    var x1 = Tensor[DType.int32](TensorShape(4), List[Int32](10, 13, 15, 18))
    var y1 = Tensor[DType.int32](TensorShape(4), List[Int32](15, 18, 20, 24))
    var trpzint = nj.trapz[DType.int32, DType.float32](y1, x1) 
    print(trpzint) # prints 33.5 that matches numpy

    var arr_dff = nj.diff[DType.int32, DType.int32](x1, n=1)
    print(arr_dff) # prints [3,2,3]

    var xi = Tensor[DType.int32](TensorShape(3), List[Int32](1, 2, 3))
    var yi = Tensor[DType.int32](TensorShape(3), List[Int32](5, 7, 9))
    var arr_cross = nj.cross[DType.int32, DType.int32](xi, yi)
    print(arr_cross) # prints [-3,6,-3]

