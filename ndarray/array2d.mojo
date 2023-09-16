from math import mul,sub,add,div,clamp,abs,floor,ceil,ceildiv,trunc,sqrt,rsqrt,exp2,ldexp,exp,frexp,log,log2,copysign,erf,tanh,isclose,all_true,any_true,none_true,reduce_bit_count,iota,is_power_of_2,is_odd,is_even,fma,reciprocal,identity,greater,greater_equal,less,less_equal,equal,not_equal,select,max,min,pow,div_ceil,align_down,align_up,acos,asin,atan,atan2,cos,sin,tan,acosh,asinh,atanh,cosh,sinh,expm1,log10,log1p,logb,cbrt,hypot,erfc,lgamma,tgamma,nearbyint,rint,round,remainder,nextafter,j0,j1,y0,y1,scalb,gcd,lcm,factorial,nan,isnan
from memory.unsafe import DTypePointer
from sys.info import simdwidthof
# from Error import Error
# from simd import SIMD
# from range import range
# from io import print, put_new_line, print_no_newline
from python import Python
from python.object import PythonObject
from math.limit import inf, neginf
# from Bool import Bool
# from runtime.llcl import num_cores, Runtime
# from algorithm.functional import parallelize

#Primary Array struct, valid for numeric types other than Boolean
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
        
    
    fn __init__(inout self):
        let rows:Int = 1
        let cols: Int =1
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
        
         for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let other_data = other.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i,other_data)
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let other_data = other.data.load(i)
                self.data.store(i,other_data)
        
        
    fn __del__(owned self):
        self.data.free()

    fn fill(inout self, val: SIMD[dtype,1])raises:
        self.zero()
        self+=val
        
    fn zero(inout self):
        let zero: SIMD[dtype,1] = 0 
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            self.data.simd_store[opt_nelts](i, zero) 
        
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                self.data.simd_store(i,zero)
    #Get Items
    
    @always_inline
    fn __getitem__(self, y: Int, x: Int) raises -> SIMD[dtype,1]:
        
        if x>(self.rows-1) or y>(self.cols-1):
            raise Error("Index Outside of assigned array get item")
        return self.data.simd_load[1](y * self.cols + x)
    
    @always_inline
    fn __getitem__(self, y:Int, xspan:slice) raises ->Array[dtype,opt_nelts]:
        if y > self.cols - 1:
            raise Error("y excedes allocated columns")
        if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > self.rows:
            raise Error("xspan slice points outside of assigned memory")
        let new_cols:Int = xspan.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_cols,1)
        for i in range(new_cols):
            new_Arr[i]=self[y,xspan[i]]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, yspan:slice, x:Int) raises -> Array[dtype,opt_nelts]:
        if x > self.rows - 1:
            raise Error("x excedes allocated columns")
        if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > self.cols:
            raise Error("yspan slice points outside of assigned memory")
        let new_rows:Int = yspan.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_rows,1)
        for i in range(new_rows):
            new_Arr[i]=self[yspan[i], x]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, yspan:slice, xspan:slice) raises -> Array[dtype,opt_nelts]:
        if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > self.cols:
            raise Error("yspan slice points outside of assigned memory")
        if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > self.rows:
            raise Error("xspan slice points outside of assigned memory")
        let new_cols:Int = xspan.__len__()
        let new_rows:Int = yspan.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_rows,new_cols)
        for i in range(new_cols):
            for j in range(new_rows):
                new_Arr[i,j]=self[yspan[j], xspan[i]]
        return new_Arr
    
    @always_inline
    fn __getitem__(self, x:Int) raises -> SIMD[dtype,1]:
        if x>(self.size-1):
            raise Error("Index Outside of assigned array get item")
        return self.data.simd_load[1](x)
    
    @always_inline
    fn __getitem__(self, span:slice) raises -> Array[dtype,opt_nelts]:
        if span[0]+span.__len__()*(span[1]-span[0]) > self.size:
            raise Error("span slice points outside of assigned memory")
        let new_size:Int = span.__len__()
        let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_size,1)
        for i in range(new_size):
            new_Arr[i]=self[span[i]]
        return new_Arr
    
#     @always_inline
#     fn __getitem__(self, mask:BoolArray) raises -> Array[dtype,opt_nelts]:
#         # let safe: Bool = self.size ==  mask.size
#         # if  safe:
#         #     raise Error("Index Outside of assigned array set item")
#         let new_size:Int = mask.count_true()
#         let new_Arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](new_size,1)
#         var new_item_index:Int = 0
#         for i in range(self.size):
#             if mask[i] == True:
#                 new_Arr[i] = self[new_item_index]
#                 new_item_index += 1
#         return new_Arr
    
#     @always_inline
#     fn mask(inout self, mask:BoolArray, val: SIMD[dtype,1]) raises:
#         # let safe: Bool = self.size ==  mask.size
#         if  mask.size !=self.size :
#             raise Error("Masks must be the same size as the array")
#         for i in range(self.size):
#             if mask[i] == True:
#                 self[i] = val
    
    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: SIMD[dtype,1]) raises:
        if x>(self.rows-1) or y>(self.cols-1):
            raise Error("Index Outside of assigned array set item")
        return self.data.simd_store[1](y * self.cols + x, val)
    
    @always_inline
    fn __setitem__(self,  x: Int, val: SIMD[dtype,1]) raises:
        if x>(self.size-1):
            raise Error("Index Outside of assigned array set item 1d single")
        return self.data.simd_store[1]( x, val)
    
    @always_inline
    fn __setitem__(inout self,  span: slice, val: Array[dtype,opt_nelts]) raises:
        if span[0]+span.__len__()*(span[1]-span[0]) > self.size:
            raise Error("span slice points outside of assigned memory")
        if span[0]+span.__len__()*(span[1]-span[0]) > val.size:
            raise Error("span slice points outside of assigned memory for val")
        let new_size:Int = span.__len__()
        if val.size != new_size:
            raise Error("Set item slice array: val is not large enough to fill the array")
        let new_Arr: Array[dtype,opt_nelts] = self
        for i in range(new_size): 
            new_Arr[span[i]] = val[i]
        self=new_Arr
        
    
    @always_inline
    fn __setitem__(inout self, y: Int, xspan: slice, val: SIMD[dtype,1]) raises:
        if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > self.rows:
            raise Error("xspan slice points outside of assigned memory")
        let new_size:Int = xspan.__len__()
        for i in range(new_size): 
            self[y,xspan[i]] = val
    
    @always_inline
    fn __setitem__(inout self, y: Int,  xspan: slice, val: Array[dtype,opt_nelts]) raises:
        if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > self.rows:
            raise Error("xspan slice points outside of assigned memory")
        # if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > val.rows:
        #     raise Error("xspan slice points outside of assigned memory of val")
        let new_size:Int = xspan.__len__()
        # if val.size < new_size:
        #     raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_size): 
            self[y, xspan[i]] = val[i]
    
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  x: Int, val: SIMD[dtype,1]) raises:
        if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > self.rows:
            raise Error("yspan slice points outside of assigned memory")
        let new_size:Int = yspan.__len__()
        for i in range(new_size): 
            self[yspan[i], x] = val
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  x: Int, val: Array[dtype,opt_nelts]) raises:
        # if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > val.rows:
        #     raise Error("yspan slice points outside of assigned memory of val")
        if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > self.rows:
            raise Error("yspan slice points outside of assigned memory")
        let new_size:Int = yspan.__len__()
        if val.size != new_size:
            raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_size): 
            self[yspan[i], x] = val[i]
    
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  xspan: slice, val: SIMD[dtype,1]) raises:
        if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > self.cols:
            raise Error("yspan slice points outside of assigned memory")
        if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > self.rows:
            raise Error("xspan slice points outside of assigned memory")
        let new_cols:Int = yspan.__len__()
        let new_rows:Int = xspan.__len__()
        for i in range(new_cols): 
            for j in range(new_rows):
                self[yspan[i], xspan[j]] = val
    
    @always_inline
    fn __setitem__(inout self, yspan: slice,  xspan: slice, val: Array[dtype,opt_nelts]) raises:
        if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > self.cols:
            raise Error("yspan slice points outside of assigned memory")
        if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > self.rows:
            raise Error("xspan slice points outside of assigned memory")
        # if yspan[0]+yspan.__len__()*(yspan[1]-yspan[0]) > val.cols:
        #     raise Error("yspan slice points outside of assigned memory of val")
        # if xspan[0]+xspan.__len__()*(xspan[1]-xspan[0]) > val.rows:
        #     raise Error("xspan slice points outside of assigned memory of val")
        let new_cols:Int = yspan.__len__()
        let new_rows:Int = xspan.__len__()
        # if val.size < new_size:
        #     raise Error("Set item slice array: val is not large enough to fill the array")
        for i in range(new_cols): 
            for j in range(new_rows):
                self[yspan[i], xspan[j]] = val[i,j]
    
    @always_inline    
    fn __setitem__(inout self,  span: slice, val: SIMD[dtype,1]) raises:
        if span[0]+span.__len__()*(span[1]-span[0]) > self.size:
            raise Error("span slice points outside of assigned memory")
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
    fn get_shape(self) raises:
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
    fn iarithmatic_Array_Array[func: fn[type:DType, simd_w:Int](SIMD[type, simd_w],SIMD[type, simd_w])->SIMD[type, simd_w]](inout self, rhs: Array[dtype,opt_nelts])raises:
        # @parameter
        # fn simd_arith(i:Int):
        #     let ind:Int = i * opt_nelts
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, func[dtype,opt_nelts](simd_data_self, simd_data_rhs))
        
        # let rt =Runtime()
        # parallelize[simd_arith](rt, (self.size//opt_nelts), num_cores())
        # rt._destroy() 
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                self.data.store(i, func[dtype,1](simd_data_self, simd_data_rhs))
    
    @always_inline        
    fn iarithmatic_Array_SIMD[func: fn[type:DType, simd_w:Int](SIMD[type, simd_w],SIMD[type, simd_w])->SIMD[type, simd_w]](inout self, rhs:SIMD[dtype,1])raises:
        # @parameter
        # fn simd_arith(i:Int):
        #     let ind:Int = i * opt_nelts
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, func[dtype,opt_nelts](simd_data_self, rhs))

        # let rt =Runtime()
        # parallelize[simd_arith](rt, (self.size//opt_nelts), num_cores())
        # rt._destroy()     
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                
                self.data.store(i, func[dtype,1](simd_data_self, rhs))
    
    @always_inline
    fn arithmatic_Array_SIMD[func: fn[type:DType, simd_w:Int](SIMD[type, simd_w],SIMD[type, simd_w])->SIMD[type, simd_w]](self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        # @parameter
        # fn simd_arith(i:Int):
        #     let ind:Int = i * opt_nelts
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data = self.data.simd_load[opt_nelts](i)
            result_array.data.simd_store[opt_nelts](i, func[dtype,opt_nelts](simd_data, rhs))
        
        # let rt =Runtime()
        # parallelize[simd_arith](rt, (self.size//opt_nelts), num_cores())
        # rt._destroy() 
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data = self.data.load(i)
                result_array.data.store(i, func[dtype,1](simd_data, rhs))
        return result_array

    @always_inline
    fn arithmatic_Array_Array[func: fn[type:DType, simd_w:Int](SIMD[type, simd_w],SIMD[type, simd_w])->SIMD[type, simd_w]](self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        let result_array: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](self.rows,self.cols)
        # @parameter
        # fn simd_arith(i:Int):
        #     let ind:Int = i * opt_nelts
        for i in range(0, opt_nelts*(self.size//opt_nelts), opt_nelts):
            let simd_data_self = self.data.simd_load[opt_nelts](i)
            let simd_data_rhs = rhs.data.simd_load[opt_nelts](i)
            self.data.simd_store[opt_nelts](i, func[dtype,opt_nelts](simd_data_self, simd_data_rhs))
            
        
        # let rt =Runtime()
        # parallelize[simd_arith](rt, (self.size//opt_nelts), num_cores())
        # rt._destroy() 
        if self.size%opt_nelts != 0 :    
            for i in range(opt_nelts*(self.size//opt_nelts), self.size):
                let simd_data_self = self.data.load(i)
                let simd_data_rhs = rhs.data.load(i)
                result_array.data.store(i, func[dtype,1](simd_data_self, simd_data_rhs))
        return result_array

    @always_inline
    fn math_func[func: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        let res_arr:Array[dtype,opt_nelts]=arr
        # let arr_c = arr
        # @parameter
        for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
        # fn simd_arith(i:Int):
            # let ind:Int = i * opt_nelts
        
            let simd_data = res_arr.data.simd_load[opt_nelts](i)
            res_arr.data.simd_store[opt_nelts](i,func[dtype,opt_nelts](simd_data))
        
        # parallelize[simd_arith](Runtime(), (self.size//opt_nelts), num_cores())
        if arr.size%opt_nelts != 0:
            for i in range(opt_nelts*(arr.size//opt_nelts), arr.size):
                let simd_data = func[dtype,1]( arr.data.simd_load[1](i))
                res_arr.data.simd_store[1](i, simd_data)
        return res_arr

    
    @always_inline
    fn __imul__(inout self, rhs: SIMD[dtype,1])raises:
        self.iarithmatic_Array_SIMD[SIMD.__mul__](rhs)
    
    @always_inline        
    fn __imul__(inout self, rhs: Array[dtype,opt_nelts])raises:
        self.iarithmatic_Array_Array[SIMD.__mul__](rhs)
    
    @always_inline
    fn __mul__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_SIMD[SIMD.__mul__](rhs)

    @always_inline
    fn __mul__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_Array[SIMD.__mul__](rhs)
    
    
    @always_inline
    fn __iadd__(inout self, rhs: SIMD[dtype,1])raises:
        self.iarithmatic_Array_SIMD[SIMD.__add__](rhs)
    
    @always_inline        
    fn __iadd__(inout self, rhs: Array[dtype,opt_nelts])raises:
        self.iarithmatic_Array_Array[SIMD.__add__](rhs)
    
    @always_inline
    fn __add__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_SIMD[SIMD.__add__](rhs)

    @always_inline
    fn __add__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_Array[SIMD.__add__](rhs)
    
    
    @always_inline
    fn __isub__(inout self, rhs: SIMD[dtype,1])raises:
        self.iarithmatic_Array_SIMD[SIMD.__sub__](rhs)
    
    @always_inline        
    fn __isub__(inout self, rhs: Array[dtype,opt_nelts])raises:
        self.iarithmatic_Array_Array[SIMD.__sub__](rhs)
    
    @always_inline
    fn __sub__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_SIMD[SIMD.__sub__](rhs)

    @always_inline
    fn __sub__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_Array[SIMD.__sub__](rhs)
    
    
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
        self.iarithmatic_Array_SIMD[SIMD.__truediv__](rhs)
    
    @always_inline        
    fn __itruediv__(inout self, rhs: Array[dtype,opt_nelts])raises:
        self.iarithmatic_Array_Array[SIMD.__truediv__](rhs)
    
    @always_inline
    fn __truediv__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_SIMD[SIMD.__truediv__](rhs)

    @always_inline
    fn __truediv__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_Array[SIMD.__truediv__](rhs)
    
    
    @always_inline
    fn __ifloordiv__(inout self, rhs: SIMD[dtype,1])raises:
        self.iarithmatic_Array_SIMD[SIMD.__floordiv__](rhs)
    
    @always_inline        
    fn __ifloordiv__(inout self, rhs: Array[dtype,opt_nelts])raises:
        self.iarithmatic_Array_Array[SIMD.__floordiv__](rhs)
    
    @always_inline
    fn __floordiv__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_SIMD[SIMD.__floordiv__](rhs)

    @always_inline
    fn __floordiv__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_Array[SIMD.__floordiv__](rhs)
    
    
    @always_inline
    fn __imod__(inout self, rhs: SIMD[dtype,1])raises:
        self.iarithmatic_Array_SIMD[SIMD.__mod__](rhs)
    
    @always_inline        
    fn __imod__(inout self, rhs: Array[dtype,opt_nelts])raises:
        self.iarithmatic_Array_Array[SIMD.__mod__](rhs)
    
    @always_inline
    fn __mod__(self, rhs: SIMD[dtype,1])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_SIMD[SIMD.__mod__](rhs)

    @always_inline
    fn __mod__(self, rhs: Array[dtype,opt_nelts])raises->Array[dtype,opt_nelts]:
        return self.arithmatic_Array_Array[SIMD.__mod__](rhs)
    
    
    @always_inline
    fn abs(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[abs](arr)

    @always_inline
    fn floor(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[floor](arr)

    @always_inline
    fn ceil(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[ceil](arr)

    @always_inline
    fn trunc(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[trunc](arr)

    @always_inline
    fn sqrt(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[sqrt](arr)

    @always_inline
    fn rsqrt(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[rsqrt](arr)

    @always_inline
    fn exp2(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[exp2](arr)

    @always_inline
    fn exp(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[exp](arr)

    @always_inline
    fn log(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[log](arr)

    @always_inline
    fn log2(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[log2](arr)

    @always_inline
    fn erf(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[erf](arr)

    @always_inline
    fn tanh(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[tanh](arr)

    @always_inline
    fn reciprocal(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[reciprocal](arr)

    @always_inline
    fn acos(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[acos](arr)

    @always_inline
    fn asin(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[asin](arr)

    @always_inline
    fn atan(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[atan](arr)

    @always_inline
    fn cos(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[cos](arr)

    @always_inline
    fn sin(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[sin](arr)

    @always_inline
    fn tan(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[tan](arr)

    @always_inline
    fn acosh(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[acosh](arr)

    @always_inline
    fn asinh(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[asinh](arr)

    @always_inline
    fn atanh(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[atanh](arr)

    @always_inline
    fn cosh(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[cosh](arr)

    @always_inline
    fn sinh(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[sinh](arr)

    @always_inline
    fn expm1(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[expm1](arr)

    @always_inline
    fn log10(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[log10](arr)

    @always_inline
    fn log1p(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[log1p](arr)

    @always_inline
    fn logb(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[logb](arr)

    @always_inline
    fn cbrt(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[cbrt](arr)

    @always_inline
    fn erfc(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[erfc](arr)

    @always_inline
    fn lgamma(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[lgamma](arr)

    @always_inline
    fn tgamma(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[tgamma](arr)

    @always_inline
    fn nearbyint(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[nearbyint](arr)

    @always_inline
    fn rint(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[rint](arr)

    @always_inline
    fn round(self,arr:Array[dtype,opt_nelts])->Array[dtype,opt_nelts]:
        return self.math_func[round](arr)
    
    # Reduce functions
    @always_inline
    fn sum(self, arr:Array[dtype,opt_nelts])raises->SIMD[dtype,1]:
        var total:SIMD[dtype,1] = 0
        for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
            let simd_data = arr.data.simd_load[opt_nelts](i)
            total += simd_data.reduce_add()
        for i in range(opt_nelts*(arr.size//opt_nelts),arr.size):
            total += arr[i]
        return total
    
    @always_inline
    fn prod(self, arr:Array[dtype,opt_nelts])raises->SIMD[dtype,1]:
        var total:SIMD[dtype,1] = 0
        for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
            let simd_data = arr.data.simd_load[opt_nelts](i)
            total *= simd_data.reduce_mul()
        for i in range(opt_nelts*(arr.size//opt_nelts),arr.size):
            total *= arr[i]
        return total

    @always_inline
    fn avg(self, arr:Array[dtype,opt_nelts])raises->SIMD[dtype,1]:
        return self.sum(arr)/arr.size
    
    @always_inline
    fn max(self, arr:Array[dtype,opt_nelts])raises->SIMD[dtype,1]:
        var max: SIMD[dtype,1] = 0
        for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
            let simd_data = arr.data.simd_load[opt_nelts](i)
            let p_max = simd_data.reduce_max()
            if i==0 or p_max>max:
                max=p_max
        for i in range(opt_nelts*(arr.size//opt_nelts),arr.size):
            if arr[i]>max:
                max = arr[i]
        return max
    
    @always_inline
    fn min(self, arr:Array[dtype,opt_nelts])raises->SIMD[dtype,1]:
        var min: SIMD[dtype,1] = 0
        for i in range(0, opt_nelts*(arr.size//opt_nelts), opt_nelts):
            let simd_data = arr.data.simd_load[opt_nelts](i)
            let p_min = simd_data.reduce_max()
            if i==0 or p_min<min:
                min=p_min
        for i in range(opt_nelts*(arr.size//opt_nelts),arr.size):
            if arr[i]<min:
                min = arr[i]
        return min
    
    @always_inline
    fn `var`(self, arr:Array[dtype,opt_nelts])raises->SIMD[dtype,1]:
        let n:SIMD[dtype,1]=arr.size
        return self.sum((arr-self.avg(arr))**2)/n
    
    @always_inline
    fn std(self, arr:Array[dtype,opt_nelts])raises->SIMD[dtype,1]:
        let n:SIMD[dtype,1]=arr.size
        return sqrt[dtype,1](self.sum((arr-self.avg(arr))**2)/n)


    # Array Creation
    fn arrange(self,start:SIMD[dtype,1],end:SIMD[dtype,1],step:SIMD[dtype,1])raises->Array[dtype,opt_nelts]:

        # if start>=end:
        #     raise Error("End must be greater than start")
            let diff: SIMD[dtype,1] = end-start
            let number_of_steps: SIMD[dtype,1] = diff/step
            let int_number_of_steps: Int = number_of_steps.cast[DType.int32]().to_int() + 1
            let arr: Array[dtype,opt_nelts]=Array[dtype,opt_nelts](int_number_of_steps,1)
        # arr.fill(start)
        
            for i in range(int_number_of_steps):
                arr[i]=start+step*i
            
            return arr
    
    fn eye(self,n:Int)raises->Array[dtype,opt_nelts]:
        
        let ident: Array[dtype, opt_nelts] = Array[dtype, opt_nelts](n,n)
        for i in range(n):
            ident[i,i] = 1
        return ident
    
    fn zeros(self,rows:Int, cols:Int)raises->Array[dtype,opt_nelts]:
        return Array[dtype,opt_nelts](rows, cols)
    
    fn ones(self,rows:Int, cols:Int)raises->Array[dtype,opt_nelts]:
        var arr: Array[dtype,opt_nelts] = Array[dtype,opt_nelts](rows,cols)
        arr.fill(1)
        return arr
    
    # Convieneince
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

    