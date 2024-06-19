from numojo.core.ndarray import NDArray
import numojo
from numojo.math.linalg import matmul_naive, matmul_tiled_unrolled_parallelized, matmul_parallelized
def main():
    var x = numojo.identity[numojo.f32](10)
    
    # x.reshape(10,10,10)
    print(x[:2,:2])
    # print(matmul_parallelized(x,x))
    # print(x.info.shape[1])
    # var s = Slice(-1,0)
    # for i in range((s.end-s.start)//s.step):
    #     print(i)