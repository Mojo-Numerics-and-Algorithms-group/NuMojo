from math import mul,sub,add,div,clamp,abs,floor,ceil,ceildiv,trunc,sqrt,rsqrt,exp2,ldexp,exp,frexp,log,log2,copysign,erf,tanh,isclose,all_true,any_true,none_true,reduce_bit_count,iota,is_power_of_2,is_odd,is_even,fma,reciprocal,identity,greater,greater_equal,less,less_equal,equal,not_equal,select,max,min,pow,div_ceil,align_down,align_up,acos,asin,atan,atan2,cos,sin,tan,acosh,asinh,atanh,cosh,sinh,expm1,log10,log1p,logb,cbrt,hypot,erfc,lgamma,tgamma,nearbyint,rint,round,remainder,nextafter,j0,j1,y0,y1,scalb,gcd,lcm,factorial,nan,isnan
from memory.unsafe import DTypePointer
from sys.info import simdwidthof
from python import Python
from python.object import PythonObject
from math.limit import inf, neginf
from .array2d import Array





@always_inline
fn secant(func:fn(t:Float64)->Float64, xg0:Float64,xg1:Float64,tol:Float64=10**-8,max_iter:Int=2000)->(Bool,Float64):
    let x2: Float64
    var x0: Float64 = xg0
    var x1: Float64 = xg1
    for i in range(max_iter):
        
        x2 = x1 - ((func(x1) * (x1-x0))/(func(x1)-func(x0)))
        if abs(x1-x0) < tol:
            return (True,x2)
        if func(x1) - func(x0)==0:
            return (True, x0)
        x0, x1 = x1, x2
    return (False, Float64(0.0))

@always_inline
fn secant(func:fn(t:Float32)->Float32, xg0:Float32,xg1:Float32,tol:Float32=10**-8,max_iter:Int=2000)->(Bool,Float32):
    let x2: Float32
    var x0: Float32 = xg0
    var x1: Float32 = xg1
    for i in range(max_iter):
        
        x2 = x1 - ((func(x1) * (x1-x0))/(func(x1)-func(x0)))
        if abs(x1-x0) < tol:
            return (True,x2)
        if func(x1) - func(x0)==0:
            return (True, x0)
        x0, x1 = x1, x2
    return (False, Float32(0.0))

@always_inline
fn newton_raphson(f:fn(t:Float64)->Float64,df:fn(t:Float64)->Float64, x:Float64, tol:Float64=10**-8,max_iter:Int=2000)->(Bool,Float64):
    let x1: Float64
    let y: Float64
    let yp: Float64
    var x0: Float64 = x
    for i in range(max_iter):
        y = f(x0)
        yp = df(x0)
        if abs(yp)<10**-9:
            break
        x1 = x0 -(y/yp)
        if abs(x1-x0) < tol:
            return (True,x1)
        if f(x1) - f(x0)==0:
            return (True, x1)
        x0 = x1    
    return (False, Float64(0.0))

@always_inline
fn newton_raphson(f:fn(t:Float32)->Float32,df:fn(t:Float32)->Float32, x:Float32, tol:Float32=10**-8,max_iter:Int=2000)->(Bool,Float32):
    let x1: Float32
    let y: Float32
    let yp: Float32
    var x0: Float32 = x
    for i in range(max_iter):
        y = f(x0)
        yp = df(x0)
        if abs(yp)<10**-9:
            break
        x1 = x0 -(y/yp)
        if abs(x1-x0) < tol:
            return (True,x1)
        if f(x1) - f(x0)==0:
            return (True, x1)
        x0 = x1    
    return (False, Float32(0.0))

@always_inline
fn halley(f:fn(t:Float64)->Float64, df:fn(t:Float64)->Float64,ddf:fn(t:Float64)->Float64, x:Float64, tol:Float64=10**-8,max_iter:Int=2000)->(Bool,Float64):
    let x1: Float64
    let y: Float64
    let yp: Float64
    let ypp: Float64
    var x0: Float64 = x
    for i in range(max_iter):
        y = f(x0)
        yp = df(x0)
        ypp = ddf(x0)
        x1 = x0 -((2*y*yp)/(2*(yp**2) - y * ypp))
        if abs(x1-x0) < tol:
            return (True,x1)
        if f(x1) - f(x0)==0:
            return (True, x1)
        x0 = x1    
    return (False, Float64(0.0))

@always_inline
fn halley(f:fn(t:Float32)->Float32, df:fn(t:Float32)->Float32,ddf:fn(t:Float32)->Float32, x:Float32, tol:Float32=10**-8,max_iter:Int=2000)->(Bool,Float32):
    let x1: Float32
    let y: Float32
    let yp: Float32
    let ypp: Float32
    var x0: Float32 = x
    for i in range(max_iter):
        y = f(x0)
        yp = df(x0)
        ypp = ddf(x0)
        x1 = x0 -((2*y*yp)/(2*(yp**2) - y * ypp))
        if abs(x1-x0) < tol:
            return (True,x1)
        if f(x1) - f(x0)==0:
            return (True, x1)
        x0 = x1    
    return (False, Float32(0.0))


# def g(f:fn(t:Float32)->Float32, x: Float32, fx: Float32) -> Func:
#     """First-order divided difference function.

#     Arguments:
#         f: Function input to g
#         x: Point at which to evaluate g
#         fx: Function f evaluated at x 
#     """
#     return lambda x: f(x + fx) / fx - 1

def steff(f:fn(t:Float64)->Float64, x: Float64, max_iter:Int=2000, tol:Float64=10**-8) -> Float64:
    """Steffenson algorithm for finding roots.

    This recursive generator yields the x_{n+1} value first then, when the generator iterates,
    it yields x_{n+2} from the next level of recursion.

    Arguments:
        f: Function whose root we are searching for
        x: Starting value upon first call, each level n that the function recurses x is x_n
    """
    var gx:Float64
    fn g(f:fn(t:Float64)->Float64,x:Float64, fx:Float64)->Float64:
        return f(x+fx)/fx -1
    for _ in range(max_iter):    
        fx = f(x)
        gx = g(f, x, fx)
        if gx == 0:
            return x
        x = x - fx / gx
        if abs(f(x))<tol:
            return x
    return x
def steff(f:fn(t:Float32)->Float32, x: Float32, max_iter:Int=2000, tol:Float32=10**-8) -> Float32:
    """Steffenson algorithm for finding roots.

    This recursive generator yields the x_{n+1} value first then, when the generator iterates,
    it yields x_{n+2} from the next level of recursion.

    Arguments:
        f: Function whose root we are searching for
        x: Starting value upon first call, each level n that the function recurses x is x_n
    """
    var gx:Float32
    fn g(f:fn(t:Float32)->Float32,x:Float32, fx:Float32)->Float32:
        return f(x+fx)/fx -1
    while True:    
        fx = f(x)
        gx = g(f, x, fx)
        if gx == 0:
            return x
        x = x - fx / gx
        if abs(f(x))<tol:
            return x
    return x
        
        