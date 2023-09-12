from array2dpar import Array
from sys.info import simdwidthof
from benchmark import Benchmark
alias f32array = Array[DType.float32, simdwidthof[DType.float32]()]
# When intiallized with out any inputs we can access the methods
# until the cannot implicitly convert type to same type issue is resolved
# this will be how it works
let f32funcs = f32array()
def benchmark_add(M: Int, N: Int):
    
    var A: f32array = f32funcs.zeros(M,N) - 64
    var B: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        try:
            _ = A + B
        except:
            pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    _ = B
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")
def benchmark_sqrt(M: Int, N: Int):
    
    let A: f32array = f32funcs.zeros(M,N) + 64
    var B: f32array = f32funcs.zeros(M,N)
    @parameter
    fn test_fn():
        # try:
        _=f32funcs.sqrt(A,B)
        # except:
            # pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    _ = B
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")

def benchmark_abs(M: Int, N: Int):
    
    let A: f32array = f32funcs.zeros(M,N) + 64
    var B: f32array = f32funcs.zeros(M,N)
    @parameter
    fn test_fn():
        # try:
        _=f32funcs.abs(A,B)
        # except:
            # pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    _ = B
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")
def main():
    # var A: f32array = f32funcs.zeros(100,100)
    # let B: f32array = f32funcs.zeros(100,100) +10
    # A+=B
    # A.arr_print()
    # print(B.data.simd_load[10](10000))
    benchmark_add(1000,1000)
    print("Sqrt benchmark")
    benchmark_sqrt(1000,1000)
    print("Abs benchmark")
    benchmark_abs(1000,1000)
