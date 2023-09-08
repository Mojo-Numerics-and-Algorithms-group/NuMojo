from sys.info import simdwidthof
from ndarray.array2d import Array
from benchmark import Benchmark
# Here we get the f32array type
alias f32array = Array[DType.float32, simdwidthof[DType.float32]()]
# When intiallized with out any inputs we can access the methods
# until the cannot implicitly convert type to same type issue is resolved
# this will be how it works
let f32funcs = f32array()

def demo():
    # First lets create an Identity matrix
    var Arr: f32array = f32funcs.eye(10)
    print("Identity Matrix")
    # We can print it with .arr_print()
    Arr.arr_print()
    # Arimetic methods work for both inplace and regular
    print("Times 4")
    Arr*=4
    Arr.arr_print()
    print("Square root")
    Arr = f32funcs.sqrt(Arr)
    Arr.arr_print()
    print("Order of operations for now is all array operations to the left of all scalar opeartions")
    print("as __rmethods__ are not working how I would have expected")
    let C = f32funcs.sqrt(Arr+25)+6
    C.arr_print()
    print("The mojo CLI won't even run it in a try block if scalars are to the left")

def benchmark_add(M: Int, N: Int):
    
    var A: f32array = f32funcs.zeros(M,N) - 64
    var B: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        try:
            A+=B
        except:
            pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")

def benchmark_sqrt(M: Int, N: Int):
    
    let A: f32array = f32funcs.zeros(M,N) + 64
    @parameter
    fn test_fn():
        try:
            _=f32funcs.sqrt(A)
        except:
            pass
        

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = A
    
    let gflops = ((2*M*N/secs) / 1e9)
    _ = M
    _ = N
    print(gflops, "GFLOP/s, and ", secs," seconds")
from time import now
def main():
    demo()
    print("Add two arrays benchmark")
    benchmark_add(100,100)
    print("Sqrt benchmark")
    benchmark_sqrt(100,100)
    




