from Benchmark import Benchmark
from DType import DType
from Intrinsics import strided_load
from List import VariadicList
from Math import div_ceil, min, sqrt
from Memory import memset_zero
from Object import object, Attr
from Pointer import DTypePointer
from Random import rand, random_float64
from TargetInfo import simdwidthof
from Error import Error
from SIMD import SIMD
from Range import range
#st:StringLiteral)raises->(Bool,Error):
fn check_dims(rows:Int,cols:Int,x:Int,y:Int)raises->(Bool,Error):

        let safe: Bool = x>=rows and y>=cols
        let err:Error = Error("Index Outside of assigned array")
        return (safe,err)
    
struct Array[dtype:DType,opt_nelts:Int]:
    var data: DTypePointer[dtype]
    var rows: Int
    var cols: Int
    var size: Int
    
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        self.rows = rows
        self.cols = cols
        self.size = self.rows*self.cols
        
        self.zero()
        
    fn __copyinit__(inout self, other: Self):
        self.rows = other.rows
        self.cols = other.cols
        self.data = DTypePointer[dtype].alloc(self.rows * self.cols)
        self.size = other.rows * other.cols
        for i in range(0, self.size, opt_nelts):
            let other_data = other.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,other_data)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let other_data = other.data.load(i)
            self.data.store(i,other_data)
        
        
    fn __del__(owned self):
        self.data.free()

    fn fill(inout self, val: SIMD[dtype,1]):
        memset_zero(self.data, self.rows * self.cols)
        self+=val
        
    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)
    
    fn __imul__(inout self, rhs: SIMD[dtype,1]):
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data * rhs)   
            
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data = self.data.load(i)
            self.data.store(i, simd_data * rhs)
            
    fn __imul__(inout self, rhs: Array[dtype,opt_nelts]):
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data_self * simd_data_rhs)
            
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data_self = self.data.load(i)
            let simd_data_rhs = rhs.data.load(i)
            self.data.store(i, simd_data_self * simd_data_rhs)
    
    fn __mul__(self, rhs: SIMD[dtype,1])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data * rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data = self.data.load(i)
            result_array.data.store(i, simd_data * rhs)
        return result_array
    
    fn __mul__(self, rhs: Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self * simd_data_rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data_self = self.data.load(i)
            let simd_data_rhs = rhs.data.load(i)
            result_array.data.store(i, simd_data_self * simd_data_rhs)
        return result_array
        
    fn __iadd__(inout self, rhs: SIMD[dtype,1]):
        
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data + rhs)   
        if self.size//opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data + rhs)
        
    fn __iadd__(inout self, rhs: Array[dtype,opt_nelts]):
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,simd_data_self + simd_data_rhs)
        if self.size//opt_nelts != 0 :
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i,simd_data_self + simd_data_rhs)
        
    fn __add__(self, rhs: SIMD[dtype,1])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data + rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data = self.data.load(i)
            result_array.data.store(i, simd_data + rhs)
        return result_array
    
    fn __add__(self, rhs: Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self + simd_data_rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data_self = self.data.load(i)
            let simd_data_rhs = rhs.data.load(i)
            result_array.data.store(i, simd_data_self + simd_data_rhs)
        return result_array
    
    fn __ipow__(inout self, rhs: Int):
        
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data ** rhs)   
        if self.size//opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data ** rhs)
        
    fn __pow__(self, rhs: Int)->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data ** rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data = self.data.load(i)
            result_array.data.store(i, simd_data ** rhs)
        return result_array
    
    fn __itruediv__(inout self, rhs: SIMD[dtype,1]):
        
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data / rhs)   
        if self.size//opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data / rhs)
        
    fn __itruediv__(inout self, rhs: Array[dtype,opt_nelts]):
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,simd_data_self / simd_data_rhs)
        if self.size//opt_nelts != 0 :
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i,simd_data_self / simd_data_rhs)
        
    fn __truediv__(self, rhs: SIMD[dtype,1])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data / rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data = self.data.load(i)
            result_array.data.store(i, simd_data / rhs)
        return result_array
    
    fn __truediv__(self, rhs: Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self / simd_data_rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data_self = self.data.load(i)
            let simd_data_rhs = rhs.data.load(i)
            result_array.data.store(i, simd_data_self / simd_data_rhs)
        return result_array
    
    fn __ifloordiv__(inout self, rhs: SIMD[dtype,1]):
        
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data // rhs)   
        if self.size//opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data // rhs)
        
    fn __ifloordiv__(inout self, rhs: Array[dtype,opt_nelts]):
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,simd_data_self // simd_data_rhs)
        if self.size//opt_nelts != 0 :
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i,simd_data_self // simd_data_rhs)
        
    fn __floordiv__(self, rhs: SIMD[dtype,1])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data // rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data = self.data.load(i)
            result_array.data.store(i, simd_data // rhs)
        return result_array
    
    fn __floordiv__(self, rhs: Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, self.size, opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self // simd_data_rhs)
        for i in range(opt_nelts*(self.size//opt_nelts), self.size):
            let simd_data_self = self.data.load(i)
            let simd_data_rhs = rhs.data.load(i)
            result_array.data.store(i, simd_data_self // simd_data_rhs)
        return result_array
    
    @always_inline
    fn __getitem__(self, y: Int, x: Int) raises -> SIMD[dtype,1]:
        # let safe: Bool
        # let err: Error
        let safe: Bool = x>(self.rows-1) or y>(self.cols-1)
        if safe:
            raise Error("Index Outside of assigned array get item")

        return self.data.simd_load[1](y * self.cols + x)
    
    @always_inline
    fn __getitem__(self, x:Int) raises -> SIMD[dtype,1]:
        
        if self.cols>1:
            raise Error("Sub arrays not implemented for 2d Arrays")
        let safe: Bool = x>(self.rows-1)
        if safe:
            raise Error("Index Outside of assigned array get item")
        return self.data.simd_load[1](x)

    @ always_inline
    fn __getitem__(self, span:slice) raises -> Array[dtype,opt_nelts]:
        let new_size:Int = span.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_size,1)
        for i in range(new_size):
            new_Arr[i]=self[span[i]]
        return new_Arr

    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) raises -> SIMD[dtype, nelts]:

        return self.data.simd_load[nelts](y * self.cols + x)
    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: SIMD[dtype,1]) raises:
        let safe: Bool = x>(self.rows-1) or y>(self.cols-1)
        if safe:
            raise Error("Index Outside of assigned array set item")
        return self.data.simd_store[1](y * self.cols + x, val)
    @always_inline
    fn __setitem__(self,  x: Int, val: SIMD[dtype,1]) raises:
        if self.cols>1:
            raise Error("Sub arrays not implemented for 2d Arrays")
        if x>(self.rows-1):
            raise Error("Index Outside of assigned array set item")
        return self.data.simd_store[1]( x, val)

    @always_inline
    fn __setitem__(inout self,  span: slice, val: Array[dtype,opt_nelts]) raises:
        let new_size:Int = span.__len__()
        if val.size < new_size:
            raise Error("Set item slice array: val is not large enough to fill the array")
        let new_Arr: Array[dtype,opt_nelts] = self
        for i in range(new_size): 
            new_Arr[span[i]] = val[i]
        self=new_Arr
        
    @always_inline    
    fn __setitem__(inout self,  span: slice, val: SIMD[dtype,1]) raises:
        let new_size:Int = span.__len__()
        for i in range(new_size): 
            self.data.simd_store[1](i,va

fn varrange[dtype:DType,opt_nelts:Int](start:SIMD[dtype,1],end:SIMD[dtype,1],step:SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
    """Creates an endpoint inclusive range between start and end with in steps of step
        Stores to a Vectorized Array
        Partially replicates the np.arrange function
    """
    if start>=end:
        raise Error("End must be greater than start")
    let diff: SIMD[dtype,1] = end-start
    let number_of_steps: SIMD[dtype,1] = diff/step
    let int_number_of_steps: Int = number_of_steps.cast[DType.int32]().to_int() + 1
    let arr: Array[dtype,opt_nelts]=Array[dtype,opt_nelts](int_number_of_steps,1)
    # arr.fill(start)
    for i in range(int_number_of_steps):
        arr[i]=start+step*i
    return arr
alias Float64 =SIMD[DType.float64, 1]
def benchmark_varrange[dtype:DType,opt_nelts:Int]()-> (Float64,Int):
    var err_count:Int=0
    @parameter
    fn test_fn():
        try:
            _=varrange[dtype,opt_nelts](0,5000,0.25)
        except:
            pass
            err_count+=1
    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    
    return secs, err_count
def vsqrt[dtype:DType,opt_nelts:Int](arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
    let res_arr:Array[dtype,opt_nelts]=Array[dtype,opt_nelts](arr.rows,arr.cols)
    for i in range(0, arr.size, opt_nelts):
        let simd_data_sqrt =sqrt[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data_sqrt )
    for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
        let simd_data_sqrt = sqrt[dtype,1]( arr.data.simd_load[1](i))
        res_arr.data.simd_store[1](i, simd_data_sqrt)
    return res_arr

def vsqrt2[dtype:DType,opt_nelts:Int](arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
    for i in range(0, arr.size, opt_nelts):
        let simd_data_sqrt =sqrt[dtype,opt_nelts]( arr.data.simd_load[opt_nelts](i))
        arr.data.simd_store[opt_nelts](i,simd_data_sqrt )
    for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
        let simd_data_sqrt = sqrt[dtype,1]( arr.data.simd_load[1](i))
        arr.data.simd_store[1](i, simd_data_sqrt)
    return arr

def benchmark_vsqrt[dtype:DType,opt_nelts:Int]()-> (Float64,Int):
    var err_count:Int=0
    var dat = Array[dtype,opt_nelts](200,100)
    dat+=100000
    @parameter
    fn test_fn():
        try:
            _=vsqrt[dtype,opt_nelts](dat)
        except:
            pass
            err_count+=1
    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    return secs, err_count
