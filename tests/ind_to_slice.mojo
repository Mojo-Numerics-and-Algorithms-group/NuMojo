# from numojo.core.ndarray import NDArray
from numojo import NDArray
from numojo.core.array_creation_routines import arange
from numojo.core.datatypes import *
# from numojo.math.linalg import matmul_naive, matmul_tiled_unrolled_parallelized, matmul_parallelized
def main():
    var x = numojo.arange[numojo.f32](0,1000)
    var y =  numojo.identity[numojo.i32](10)
    print(slice(0,1)[0])
    x.reshape(10,10,10)
    print(x[Slice(0,2),1])
    print()
    print(y[Slice(0,5),1])
