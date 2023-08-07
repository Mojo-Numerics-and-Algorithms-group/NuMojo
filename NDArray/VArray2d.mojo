from Benchmark import Benchmark
from DType import DType
from Intrinsics import strided_load
from List import VariadicList
from Math import mul,sub,add,div,clamp,abs,floor,ceil,ceildiv,trunc,sqrt,rsqrt,exp2,ldexp,exp,frexp,log,log2,copysign,erf,tanh,isclose,all_true,any_true,none_true,reduce_bit_count,iota,is_power_of_2,is_odd,is_even,fma,reciprocal,identity,greater,greater_equal,less,less_equal,equal,not_equal,select,max,min,pow,div_ceil,align_down,align_up,acos,asin,atan,atan2,cos,sin,tan,acosh,asinh,atanh,cosh,sinh,expm1,log10,log1p,logb,cbrt,hypot,erfc,lgamma,tgamma,nearbyint,rint,round,remainder,nextafter,j0,j1,y0,y1,scalb,gcd,lcm,factorial,nan,isnan
from Memory import memset_zero
from Object import object, Attr
from Pointer import DTypePointer
from Random import rand, random_float64
from TargetInfo import simdwidthof
from Error import Error
from SIMD import SIMD
from Range import range
from IO import print, put_new_line, print_no_newline
    
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
    
    # fn is_safe(self,x:Int, y:Int, st: StringLiteral) raises -> None:
    #     if x>(self.rows-1) or y>(cols-1):
    #         raise Error(st)
        
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

    fn fill(inout self, val: SIMD[dtype,1])raises:
        memset_zero(self.data, self.rows * self.cols)
        self+=val
        
    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)
        
    ## Math for floats and ints
    
    @always_inline
    fn __imul__(inout self, rhs: SIMD[dtype,1])raises:
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data * rhs)   
            
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data * rhs)
    
    @always_inline        
    fn __imul__(inout self, rhs: Array[dtype,opt_nelts])raises:
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data_self * simd_data_rhs)
            
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i, simd_data_self * simd_data_rhs)
    
    @always_inline
    fn __mul__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data * rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                result_array.data.store(i, simd_data * rhs)
        return result_array
    
    @always_inline
    fn __mul__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self * simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, simd_data_self * simd_data_rhs)
        return result_array
    
    @always_inline    
    fn __iadd__(inout self, rhs: SIMD[dtype,1])raises:
        
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data + rhs)   
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data + rhs)
    
    @always_inline    
    fn __iadd__(inout self, rhs: Array[dtype,opt_nelts])raises:
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,simd_data_self + simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i,simd_data_self + simd_data_rhs)
    
    @always_inline    
    fn __add__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data + rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                result_array.data.store(i, simd_data + rhs)
        return result_array
    
    @always_inline
    fn __add__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self + simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, simd_data_self + simd_data_rhs)
        return result_array
    
    @always_inline
    fn __isub__(inout self, rhs: SIMD[dtype,1])raises:
        
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data - rhs)   
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data - rhs)
    
    @always_inline    
    fn __isub__(inout self, rhs: Array[dtype,opt_nelts])raises:
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,simd_data_self - simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i,simd_data_self - simd_data_rhs)
    
    @always_inline    
    fn __sub__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data - rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                result_array.data.store(i, simd_data - rhs)
        return result_array
    
    @always_inline
    fn __sub__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self - simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, simd_data_self - simd_data_rhs)
        return result_array
    
    @always_inline
    fn __ipow__(inout self, rhs: Int)raises:
        
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data ** rhs)   
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data ** rhs)
    
    @always_inline    
    fn __pow__(self, rhs: Int)raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data ** rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                result_array.data.store(i, simd_data ** rhs)
        return result_array
    
    @always_inline
    fn __itruediv__(inout self, rhs: SIMD[dtype,1])raises:
        
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data / rhs)   
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data / rhs)
    
    @always_inline    
    fn __itruediv__(inout self, rhs: Array[dtype,opt_nelts])raises:
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,simd_data_self / simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i,simd_data_self / simd_data_rhs)
    
    @always_inline    
    fn __truediv__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data / rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                result_array.data.store(i, simd_data / rhs)
        return result_array
    
    @always_inline
    fn __truediv__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self / simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, simd_data_self / simd_data_rhs)
        return result_array
    
    @always_inline
    fn __ifloordiv__(inout self, rhs: SIMD[dtype,1])raises:
        
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, simd_data // rhs)   
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                self.data.store(i, simd_data // rhs)
    
    @always_inline
    fn __ifloordiv__(inout self, rhs: Array[dtype,opt_nelts])raises:
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,simd_data_self // simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i,simd_data_self // simd_data_rhs)
    
    @always_inline    
    fn __floordiv__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data // rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                result_array.data.store(i, simd_data // rhs)
        return result_array
    
    @always_inline
    fn __floordiv__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, simd_data_self // simd_data_rhs)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, simd_data_self // simd_data_rhs)
        return result_array
    
    #Get Items
    
    @always_inline
    fn __getitem__(self, y: Int, x: Int) raises -> SIMD[dtype,1]:
        # let safe: Bool
        # let err: Error
        let safe: Bool = x>(self.rows-1) or y>(self.cols-1)
        if safe:
            raise Error("Index Outside of assigned array get item")
        # return (safe,err)#,"get item")
        # if not safe:
        #     raise err
        return self.data.simd_load[1](y * self.cols + x)
    
    @always_inline
    fn __getitem__(self, xspan:slice, y:Int) raises -> Array[dtype,opt_nelts]:
        let new_cols:Int = xspan.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_cols,1)
        for i in range(new_cols):
            new_Arr[i]=self[xspan[i],y]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, x:Int, yspan:slice) raises -> Array[dtype,opt_nelts]:
        let new_rows:Int = yspan.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_rows,1)
        for i in range(new_rows):
            new_Arr[i]=self[x,yspan[i]]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, xspan:slice, yspan:slice) raises -> Array[dtype,opt_nelts]:
        let new_cols:Int = xspan.__len__()
        let new_rows:Int = yspan.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_rows,new_cols)
        for i in range(new_cols):
            for j in range(new_rows):
                new_Arr[i,j]=self[xspan[i],yspan[j]]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, x:Int) raises -> SIMD[dtype,1]:
        # let safe: Bool
        # let err: Error
        if self.cols>1:
            raise Error("Sub arrays not implemented for 2d Arrays")
        let safe: Bool = x>(self.rows-1)
        if safe:
            raise Error("Index Outside of assigned array get item")
        # return (safe,err)#,"get item")
        # if not safe:
        #     raise err
        return self.data.simd_load[1](x)
    
    @always_inline
    fn __getitem__(self, span:slice) raises -> Array[dtype,opt_nelts]:
        let new_size:Int = span.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_size,1)
        for i in range(new_size):
            new_Arr[i]=self[span[i]]
        return new_Arr
    
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
            raise Error("Index Outside of assigned array set item 1d single")
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
    fn __setitem__(inout self, y: Int, xspan: slice, val: SIMD[dtype,1]) raises:
        let new_size:Int = xspan.__len__()
        for i in range(new_size): 
            self[y,xspan[i]] = val
    
    @always_inline
    fn __setitem__(inout self, y: Int,  xspan: slice, val: Array[dtype,opt_nelts]) raises:
        let new_size:Int = xspan.__len__()
        if val.size < new_size:
            raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_size): 
            self[y, xspan[i]] = val[i]
    
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  x: Int, val: SIMD[dtype,1]) raises:
        let new_size:Int = yspan.__len__()
        for i in range(new_size): 
            self[yspan[i], x] = val
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  x: Int, val: Array[dtype,opt_nelts]) raises:
        let new_size:Int = yspan.__len__()
        if val.size < new_size:
            raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_size): 
            self[yspan[i], x] = val[i]
    
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  xspan: slice, val: SIMD[dtype,1]) raises:
        let new_cols:Int = yspan.__len__()
        let new_rows:Int = xspan.__len__()
        for i in range(new_cols): 
            for j in range(new_rows):
                self[yspan[i], xspan[j]] = val
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  xspan: slice, val: Array[dtype,opt_nelts]) raises:
        let new_cols:Int = yspan.__len__()
        let new_rows:Int = xspan.__len__()
        # if val.size < new_size:
        #     raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_cols): 
            for j in range(new_rows):
                self[yspan[i], xspan[j]] = val[i,j]
    
    @always_inline    
    fn __setitem__(inout self,  span: slice, val: SIMD[dtype,1]) raises:
        let new_size:Int = span.__len__()
        for i in range(new_size): 
            self.data.simd_store[1](i,val)
    
    #Additional Methods
    
    @always_inline
    fn transpose(self) raises ->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.cols,self.rows)
        for i in range(self.cols):
            for j in range(self.rows):
                result_array[j,i]=self[i,j]
        return result_array
    
    @always_inline
    fn shape(self) raises:
        print("cols: ",self.cols," rows: ",self.rows)
    
    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) raises -> SIMD[dtype, nelts]:

        return self.data.simd_load[nelts](y * self.cols + x)
    
    @always_inline
    fn store[nelts:Int](self, y: Int, x: Int, val: SIMD[dtype, nelts]) raises:
        # let safe: Bool = x>(self.rows-1) or y>(self.cols-1)
        # if safe:
        #     raise Error("Index Outside of assigned array load")
        # let safe2: Bool =(y * self.cols + x+nelts)>self.size
        # if safe2:
        #     raise Error("Span of attempted load excedes size of Array")
        self.data.simd_store[nelts](y * self.cols + x, val)
    
    @always_inline
    fn arr_print(self)raises:
        for i in range(self.rows):
            print_no_newline("[ ")
            for j in range(self.cols):
            
                print_no_newline(self[j,i])
                if j != (self.rows - 1):
                    print_no_newline(", ")
            print_no_newline("]")
            put_new_line()
            
    def to_numpy(self) -> PythonObject:
        let np = Python.import_module("numpy")
        let numpy_array = np.zeros((self.rows, self.cols), np.float32)
        for col in range(self.cols):
            for row in range(self.rows):
                numpy_array.itemset((row, col), self[col, row])
        return numpy_array 
    
alias dtype = DType.float32
alias opt_nelts = simdwidthof[dtype]()
alias f32array = Array[dtype, opt_nelts]
            
fn arrange[dtype:DType,opt_nelts:Int](start:SIMD[dtype,1],end:SIMD[dtype,1],step:SIMD[dtype,1])raises->Array[dtype,opt_nelts]:

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

fn abs(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = abs[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = abs[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn floor(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = floor[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = floor[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn ceil(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = ceil[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = ceil[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr


fn trunc(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = trunc[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = trunc[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn sqrt(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = sqrt[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = sqrt[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr
fn rsqrt(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = rsqrt[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = rsqrt[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn exp2(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = exp2[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = exp2[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn exp(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = exp[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = exp[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr


fn log(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = log[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = log[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn log2(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = log2[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = log2[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn erf(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = erf[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = erf[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn tanh(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = tanh[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = tanh[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn reciprocal(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = reciprocal[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = reciprocal[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn acos(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = acos[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = acos[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn asin(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = asin[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = asin[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn atan(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = atan[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = atan[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn cos(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = cos[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = cos[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn sin(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = sin[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = sin[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn tan(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = tan[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = tan[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn acosh(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = acosh[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = acosh[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn asinh(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = asinh[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = asinh[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr
fn atanh(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = atanh[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = atanh[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn cosh(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = cosh[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = cosh[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn sinh(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = sinh[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = sinh[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn expm1(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = expm1[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = expm1[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr
fn log10(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = log10[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = log10[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn log1p(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = log1p[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = log1p[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn logb(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = logb[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = logb[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn cbrt(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = cbrt[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = cbrt[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn erfc(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = erfc[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = erfc[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn lgamma(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = lgamma[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = lgamma[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn tgamma(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = tgamma[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = tgamma[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn nearbyint(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = nearbyint[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = nearbyint[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn rint(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = rint[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = rint[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr

fn round(arr:f32array)->f32array:
    let res_arr:f32array=f32array(arr.rows,arr.cols)
    for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        let simd_data = round[dtype,opt_nelts](arr.data.simd_load[opt_nelts](i))
        res_arr.data.simd_store[opt_nelts](i,simd_data )

    if arr.size%opt_nelts != 0 :
        for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
            let simd_data = round[dtype,1]( arr.data.simd_load[1](i))
            res_arr.data.simd_store[1](i, simd_data)
    return res_arr
