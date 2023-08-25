from math import mul,sub,add,div,clamp,abs,floor,ceil,ceildiv,trunc,sqrt,rsqrt,exp2,ldexp,exp,frexp,log,log2,copysign,erf,tanh,isclose,all_true,any_true,none_true,reduce_bit_count,iota,is_power_of_2,is_odd,is_even,fma,reciprocal,identity,greater,greater_equal,less,less_equal,equal,not_equal,select,max,min,pow,div_ceil,align_down,align_up,acos,asin,atan,atan2,cos,sin,tan,acosh,asinh,atanh,cosh,sinh,expm1,log10,log1p,logb,cbrt,hypot,erfc,lgamma,tgamma,nearbyint,rint,round,remainder,nextafter,j0,j1,y0,y1,scalb,gcd,lcm,factorial,nan,isnan
from memory.unsafe import DTypePointer
from sys.info import simdwidthof
# from Error import Error
# from simd import SIMD
# from range import range
# from io import print, put_new_line, print_no_newline
from python import Python
from python.object import PythonObject
# from Bool import Bool

alias bool_nelts  = simdwidthof[DType.bool]()
alias dt_bool = DType.bool

struct BoolArray:
    var data: DTypePointer[dt_bool]
    var rows: Int
    var cols: Int
    var size: Int
    
    @always_inline
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[dt_bool].alloc(rows * cols)
        self.rows = rows
        self.cols = cols
        self.size = self.rows*self.cols
    
    @always_inline    
    fn __copyinit__(inout self, other: Self):
        self.rows = other.rows
        self.cols = other.cols
        self.data = DTypePointer[dt_bool].alloc(self.rows * self.cols)
        self.size = other.rows * other.cols
         for i in range(0, bool_nelts*(self.size//bool_nelts), bool_nelts):
            let other_data = other.data.simd_load[bool_nelts](i)
            self.data.simd_store[bool_nelts](i,other_data)
        if self.size%bool_nelts != 0 :    
            for i in range(bool_nelts*(self.size//bool_nelts), self.size):
                let other_data = other.data.load(i)
                self.data.store(i,other_data)
    @always_inline
    fn count_true(self)raises->Int:
        var count:Int = 0
        for i in range(self.size):
            if self[i]==True:
                count+=1
        return count
        
    @always_inline
    fn all_false(inout self):
        let zero: SIMD[dt_bool,1] = 0 
        for i in range(0, bool_nelts*(self.size//bool_nelts), bool_nelts):
            self.data.simd_store[bool_nelts](i, False) 
        if self.size%bool_nelts != 0 :    
            for i in range(bool_nelts*(self.size//bool_nelts), self.size):
                self.data.simd_store[bool_nelts](i,False)
    
    @always_inline
    fn all_true(inout self):
        let zero: SIMD[dt_bool,1] = 0 
        for i in range(0, bool_nelts*(self.size//bool_nelts), bool_nelts):
            self.data.simd_store[bool_nelts](i, True) 
        if self.size%bool_nelts != 0 :    
            for i in range(bool_nelts*(self.size//bool_nelts), self.size):
                self.data.simd_store[bool_nelts](i,True)
    
    @always_inline
    fn __and__(self, rhs: BoolArray)->BoolArray:
        let result_array: BoolArray = BoolArray(self.rows,self.cols)
        for i in range(0, bool_nelts*(self.size//bool_nelts), bool_nelts):
            let simd_data_self = self.data.simd_load[bool_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[bool_nelts](i)
            result_array.data.simd_store[bool_nelts](i, simd_data_self & simd_data_rhs)
        if self.size%bool_nelts != 0 :    
            for i in range(bool_nelts*(self.size//bool_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, simd_data_self & simd_data_rhs)
        return result_array
    
    @always_inline
    fn __and__(self, rhs: Bool)->BoolArray:
        let result_array: BoolArray = BoolArray(self.rows,self.cols)
        for i in range(0, bool_nelts*(self.size//bool_nelts), bool_nelts):
            let simd_data_self = self.data.simd_load[bool_nelts](i)
            # let simd_data_rhs = rhs.data.simd_load[bool_nelts](i)
            result_array.data.simd_store[bool_nelts](i, simd_data_self & rhs)
        if self.size%bool_nelts != 0 :    
            for i in range(bool_nelts*(self.size//bool_nelts), self.size):
                let simd_data_self = self.data.load(i)
                # let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, simd_data_self & rhs)
        return result_array
    
    @always_inline    
    fn __iand__(inout self, rhs: BoolArray)raises:
        self = self & rhs
        # let result_array: BoolArray = BoolArray(self.rows,self.cols)
        # for i in range(0, bool_nelts*(self.size//bool_nelts), bool_nelts):
        #     let simd_data_self = self.data.simd_load[bool_nelts](i)
        #     let simd_data_rhs = rhs.data.simd_load[bool_nelts](i)
        #     result_array.data.simd_store[bool_nelts](i, simd_data_self & simd_data_rhs)
        # if self.size%bool_nelts != 0 :    
        #     for i in range(bool_nelts*(self.size//bool_nelts), self.size):
        #         let simd_data_self = self.data.load(i)
        #         let simd_data_rhs = rhs.data.load(i)
        #         result_array.data.store(i, simd_data_self & simd_data_rhs)
        # return result_array
    
    @always_inline
    fn __iand__(inout self, rhs: Bool)raises:
        self = self & rhs
        # let result_array: BoolArray = BoolArray(self.rows,self.cols)
        # for i in range(0, bool_nelts*(self.size//bool_nelts), bool_nelts):
        #     let simd_data_self = self.data.simd_load[bool_nelts](i)
        #     # let simd_data_rhs = rhs.data.simd_load[bool_nelts](i)
        #     result_array.data.simd_store[bool_nelts](i, simd_data_self & rhs)
        # if self.size%bool_nelts != 0 :    
        #     for i in range(bool_nelts*(self.size//bool_nelts), self.size):
        #         let simd_data_self = self.data.load(i)
        #         # let simd_data_rhs = rhs.data.load(i)
        #         result_array.data.store(i, simd_data_self & rhs)
        # return result_array
    
    @always_inline
    fn __getitem__(self, y: Int, x: Int) raises -> SIMD[dt_bool,1]:
        # let safe: Bool
        # let err: Error
        let safe: Bool = x>(self.rows-1) or y>(self.cols-1)
        # if safe:
            # raise Error("Index Outside of assigned array get item")
        # return (safe,err)#,"get item")
        # if not safe:
        #     raise err
        return self.data.simd_load[1](y * self.cols + x)
    
    @always_inline
    fn __getitem__(self, xspan:slice, y:Int) raises -> BoolArray:
        let new_cols:Int = xspan.__len__()
        let new_Arr: BoolArray = BoolArray(new_cols,1)
        for i in range(new_cols):
            new_Arr[i]=self[xspan[i],y]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, x:Int, yspan:slice) raises -> BoolArray:
        let new_rows:Int = yspan.__len__()
        let new_Arr: BoolArray = BoolArray(new_rows,1)
        for i in range(new_rows):
            new_Arr[i]=self[x,yspan[i]]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, xspan:slice, yspan:slice) raises -> BoolArray:
        let new_cols:Int = xspan.__len__()
        let new_rows:Int = yspan.__len__()
        let new_Arr: BoolArray = BoolArray(new_rows,new_cols)
        for i in range(new_cols):
            for j in range(new_rows):
                new_Arr[i,j]=self[xspan[i],yspan[j]]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, x:Int) raises -> SIMD[dt_bool,1]:
        # let safe: Bool
        # let err: Error
        # if self.cols>1:
        #     raise Error("Sub arrays not implemented for 2d BoolArrays")
        # let safe: Bool = x>(self.rows-1)
        # if safe:
        #     raise Error("Index Outside of assigned array get item")
        # return (safe,err)#,"get item")
        # if not safe:
        #     raise err
        return self.data.simd_load[1](x)
    
    @always_inline
    fn __getitem__(self, span:slice) raises -> BoolArray:
        let new_size:Int = span.__len__()
        let new_Arr: BoolArray = BoolArray(new_size,1)
        for i in range(new_size):
            new_Arr[i]=self[span[i]]
        return new_Arr
    
    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: SIMD[dt_bool,1]) raises:
        # let safe: Bool = x>(self.rows-1) or y>(self.cols-1)
        # if safe:
        #     raise Error("Index Outside of assigned array set item")
        return self.data.simd_store[1](y * self.cols + x, val)
    
    @always_inline
    fn __setitem__(self,  x: Int, val: SIMD[dt_bool,1]) raises:
        # if self.cols>1:
        #     raise Error("Sub arrays not implemented for 2d BoolArrays")
        # if x>(self.rows-1):
        #     raise Error("Index Outside of assigned array set item 1d single")
        return self.data.simd_store[1]( x, val)
    
    @always_inline
    fn __setitem__(inout self,  span: slice, val: BoolArray) raises:
        let new_size:Int = span.__len__()
        # if val.size < new_size:
        #     raise Error("Set item slice array: val is not large enough to fill the array")
        let new_Arr: BoolArray = self
        for i in range(new_size): 
            new_Arr[span[i]] = val[i]
        self=new_Arr
        
    
    @always_inline
    fn __setitem__(inout self, y: Int, xspan: slice, val: SIMD[dt_bool,1]) raises:
        let new_size:Int = xspan.__len__()
        for i in range(new_size): 
            self[y,xspan[i]] = val
    
    @always_inline
    fn __setitem__(inout self, y: Int,  xspan: slice, val: BoolArray) raises:
        let new_size:Int = xspan.__len__()
        # if val.size < new_size:
        #     raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_size): 
            self[y, xspan[i]] = val[i]
    
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  x: Int, val: SIMD[dt_bool,1]) raises:
        let new_size:Int = yspan.__len__()
        for i in range(new_size): 
            self[yspan[i], x] = val
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  x: Int, val: BoolArray) raises:
        let new_size:Int = yspan.__len__()
        # if val.size < new_size:
        #     raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_size): 
            self[yspan[i], x] = val[i]
    
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  xspan: slice, val: SIMD[dt_bool,1]) raises:
        let new_cols:Int = yspan.__len__()
        let new_rows:Int = xspan.__len__()
        for i in range(new_cols): 
            for j in range(new_rows):
                self[yspan[i], xspan[j]] = val
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  xspan: slice, val: BoolArray) raises:
        let new_cols:Int = yspan.__len__()
        let new_rows:Int = xspan.__len__()
        # if val.size < new_size:
        #     raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_cols): 
            for j in range(new_rows):
                self[yspan[i], xspan[j]] = val[i,j]
    
    @always_inline    
    fn __setitem__(inout self,  span: slice, val: SIMD[dt_bool,1]) raises:
        let new_size:Int = span.__len__()
        for i in range(new_size): 
            self.data.simd_store[1](i,val)
    
    #Additional Methods
    
    @always_inline
    fn transpose(self) raises ->BoolArray:
        let result_array: BoolArray = BoolArray(self.cols,self.rows)
        for i in range(self.cols):
            for j in range(self.rows):
                result_array[j,i]=self[i,j]
        return result_array
    
    @always_inline
    fn shape(self) raises:
        print("cols: ",self.cols," rows: ",self.rows)
    
    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) raises -> SIMD[dt_bool, nelts]:

        return self.data.simd_load[nelts](y * self.cols + x)
    
    @always_inline
    fn store[nelts:Int](self, y: Int, x: Int, val: SIMD[dt_bool, nelts]) raises:
        # let safe: Bool = x>(self.rows-1) or y>(self.cols-1)
        # if safe:
        #     raise Error("Index Outside of assigned array load")
        # let safe2: Bool =(y * self.cols + x+nelts)>self.size
        # if safe2:
        #     raise Error("Span of attempted load excedes size of BoolArray")
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
