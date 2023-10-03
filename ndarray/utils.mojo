from math import mul,sub,add,div,clamp,abs,floor,ceil,ceildiv,trunc,sqrt,rsqrt,exp2,ldexp,exp,frexp,log,log2,copysign,erf,tanh,isclose,all_true,any_true,none_true,reduce_bit_count,iota,is_power_of_2,is_odd,is_even,fma,reciprocal,identity,greater,greater_equal,less,less_equal,equal,not_equal,select,max,min,pow,div_ceil,align_down,align_up,acos,asin,atan,atan2,cos,sin,tan,acosh,asinh,atanh,cosh,sinh,expm1,log10,log1p,logb,cbrt,hypot,erfc,lgamma,tgamma,nearbyint,rint,round,remainder,nextafter,j0,j1,y0,y1,scalb,gcd,lcm,factorial,nan,isnan
from memory.unsafe import DTypePointer
from sys.info import simdwidthof
from python import Python
from python.object import PythonObject
from math.limit import inf, neginf
from .array2d import Array
from time import now

alias f32array = Array[DType.float32, simdwidthof[DType.float32]()]
let f32funcs = f32array()
alias f64array = Array[DType.float64, simdwidthof[DType.float64]()]
let f64funcs = f64array()


@always_inline
fn time_correction()raises->Float64:
    let time_Array: f64array = f64funcs.zeros(1,200)   
    for i in range(200):
        let t1:Float64 = now()
        let t2:Float64 =now()
        time_Array[i] = t2-t1 
    let secs = (f64funcs.avg(time_Array))/10**9
    return secs

def benchmark[func:fn()capturing->None](cycles:Int=200)->(Float64, Float64):
    let time_correction = time_correction()
    let time_Array: f64array = f64funcs.zeros(1,cycles)   
    for i in range(cycles):
        let t1:Float64 = now()
        func()
        let t2:Float64 =now()
        time_Array[i] = t2-t1 
    let secs = ((f64funcs.avg(time_Array))-time_correction)/10**9
    let std = f64funcs.std(time_Array-time_correction)/10**9
    return secs, std
    