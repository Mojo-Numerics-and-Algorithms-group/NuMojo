import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep

fn _math_func[dtype:DType, func: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](tensor: Tensor[dtype])->Tensor[dtype]:
    var result_tensor: Tensor[dtype]=Tensor[dtype](tensor.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(0, opt_nelts*(tensor.num_elements()//opt_nelts), opt_nelts):
        var simd_data = tensor.load[width=opt_nelts](i)
        result_tensor.store[width=opt_nelts](i,func[dtype,opt_nelts](simd_data))
        
       
    if tensor.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor.num_elements()//opt_nelts), tensor.num_elements()):
                var simd_data = func[dtype,1](tensor.load[width=1](i))
                result_tensor.store[width=1](i, simd_data)
    return result_tensor

fn abs[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.abs](tensor)

fn floor[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.floor](tensor)

fn ceil[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.ceil](tensor)

fn trunc[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.trunc](tensor)

fn round[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.round](tensor)

fn roundeven[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.roundeven](tensor)

fn round_half_down[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.round_half_down](tensor)

fn round_half_up[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.round_half_up](tensor)

fn rsqrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.rsqrt](tensor)

fn exp2[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.exp2](tensor)

fn exp[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.exp](tensor)

fn log[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.log](tensor)

fn log2[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.log2](tensor)

fn erf[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.erf](tensor)

fn tanh[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.tanh](tensor)

fn reciprocal[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.reciprocal](tensor)

fn identity[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.identity](tensor)

fn acos[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.acos](tensor)

fn asin[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.asin](tensor)

fn atan[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.atan](tensor)

fn cos[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.cos](tensor)

fn sin[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.sin](tensor)

fn tan[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.tan](tensor)

fn acosh[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.acosh](tensor)

fn asinh[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.asinh](tensor)

fn atanh[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.atanh](tensor)

fn cosh[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.cosh](tensor)

fn sinh[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.sinh](tensor)

fn expm1[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.expm1](tensor)

fn log10[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.log10](tensor)

fn log1p[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.log1p](tensor)

fn cbrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.cbrt](tensor)

fn erfc[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.erfc](tensor)

fn lgamma[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.lgamma](tensor)

fn tgamma[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.tgamma](tensor)

fn nearbyint[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.nearbyint](tensor)

fn rint[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.rint](tensor)

fn j0[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.j0](tensor)

fn j1[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.j1](tensor)

fn y0[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.y0](tensor)

fn y1[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.y1](tensor)

fn ulp[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.ulp](tensor)

# To test uncomment main build and and run
# fn main():
#     var tens:Tensor[DType.float32] = Tensor[DType.float32](100,100)
#     for i in range(10_000):
#         tens[i]= SIMD[DType.float32,1](3.141592/4)
#     var res:Tensor[DType.float32]
#     fn test_abs()capturing:
#         res = abs[DType.float32](tens)
#         keep(res.data())
#     var report_abs = benchmark.run[test_abs]()
#     print('abs f32 100x100')
#     report_abs.print()
#     fn test_floor()capturing:
#         res = floor[DType.float32](tens)
#         keep(res.data())
#     var report_floor = benchmark.run[test_floor]()
#     print('floor f32 100x100')
#     report_floor.print()
#     fn test_ceil()capturing:
#         res = ceil[DType.float32](tens)
#         keep(res.data())
#     var report_ceil = benchmark.run[test_ceil]()
#     print('ceil f32 100x100')
#     report_ceil.print()
#     fn test_trunc()capturing:
#         res = trunc[DType.float32](tens)
#         keep(res.data())
#     var report_trunc = benchmark.run[test_trunc]()
#     print('trunc f32 100x100')
#     report_trunc.print()
#     fn test_round()capturing:
#         res = round[DType.float32](tens)
#         keep(res.data())
#     var report_round = benchmark.run[test_round]()
#     print('round f32 100x100')
#     report_round.print()
#     fn test_roundeven()capturing:
#         res = roundeven[DType.float32](tens)
#         keep(res.data())
#     var report_roundeven = benchmark.run[test_roundeven]()
#     print('roundeven f32 100x100')
#     report_roundeven.print()
#     fn test_round_half_down()capturing:
#         res = round_half_down[DType.float32](tens)
#         keep(res.data())
#     var report_round_half_down = benchmark.run[test_round_half_down]()
#     print('round_half_down f32 100x100')
#     report_round_half_down.print()
#     fn test_round_half_up()capturing:
#         res = round_half_up[DType.float32](tens)
#         keep(res.data())
#     var report_round_half_up = benchmark.run[test_round_half_up]()
#     print('round_half_up f32 100x100')
#     report_round_half_up.print()
#     fn test_rsqrt()capturing:
#         res = rsqrt[DType.float32](tens)
#         keep(res.data())
#     var report_rsqrt = benchmark.run[test_rsqrt]()
#     print('rsqrt f32 100x100')
#     report_rsqrt.print()
#     fn test_exp2()capturing:
#         res = exp2[DType.float32](tens)
#         keep(res.data())
#     var report_exp2 = benchmark.run[test_exp2]()
#     print('exp2 f32 100x100')
#     report_exp2.print()
#     fn test_exp()capturing:
#         res = exp[DType.float32](tens)
#         keep(res.data())
#     var report_exp = benchmark.run[test_exp]()
#     print('exp f32 100x100')
#     report_exp.print()
#     fn test_log()capturing:
#         res = log[DType.float32](tens)
#         keep(res.data())
#     var report_log = benchmark.run[test_log]()
#     print('log f32 100x100')
#     report_log.print()
#     fn test_log2()capturing:
#         res = log2[DType.float32](tens)
#         keep(res.data())
#     var report_log2 = benchmark.run[test_log2]()
#     print('log2 f32 100x100')
#     report_log2.print()
#     fn test_erf()capturing:
#         res = erf[DType.float32](tens)
#         keep(res.data())
#     var report_erf = benchmark.run[test_erf]()
#     print('erf f32 100x100')
#     report_erf.print()
#     fn test_tanh()capturing:
#         res = tanh[DType.float32](tens)
#         keep(res.data())
#     var report_tanh = benchmark.run[test_tanh]()
#     print('tanh f32 100x100')
#     report_tanh.print()
#     fn test_reciprocal()capturing:
#         res = reciprocal[DType.float32](tens)
#         keep(res.data())
#     var report_reciprocal = benchmark.run[test_reciprocal]()
#     print('reciprocal f32 100x100')
#     report_reciprocal.print()
#     fn test_identity()capturing:
#         res = identity[DType.float32](tens)
#         keep(res.data())
#     var report_identity = benchmark.run[test_identity]()
#     print('identity f32 100x100')
#     report_identity.print()
#     fn test_acos()capturing:
#         res = acos[DType.float32](tens)
#         keep(res.data())
#     var report_acos = benchmark.run[test_acos]()
#     print('acos f32 100x100')
#     report_acos.print()
#     fn test_asin()capturing:
#         res = asin[DType.float32](tens)
#         keep(res.data())
#     var report_asin = benchmark.run[test_asin]()
#     print('asin f32 100x100')
#     report_asin.print()
#     fn test_atan()capturing:
#         res = atan[DType.float32](tens)
#         keep(res.data())
#     var report_atan = benchmark.run[test_atan]()
#     print('atan f32 100x100')
#     report_atan.print()
#     fn test_cos()capturing:
#         res = cos[DType.float32](tens)
#         keep(res.data())
#     var report_cos = benchmark.run[test_cos]()
#     print('cos f32 100x100')
#     report_cos.print()
#     fn test_sin()capturing:
#         res = sin[DType.float32](tens)
#         keep(res.data())
#     var report_sin = benchmark.run[test_sin]()
#     print('sin f32 100x100')
#     report_sin.print()
#     fn test_tan()capturing:
#         res = tan[DType.float32](tens)
#         keep(res.data())
#     var report_tan = benchmark.run[test_tan]()
#     print('tan f32 100x100')
#     report_tan.print()
#     fn test_acosh()capturing:
#         res = acosh[DType.float32](tens)
#         keep(res.data())
#     var report_acosh = benchmark.run[test_acosh]()
#     print('acosh f32 100x100')
#     report_acosh.print()
#     fn test_asinh()capturing:
#         res = asinh[DType.float32](tens)
#         keep(res.data())
#     var report_asinh = benchmark.run[test_asinh]()
#     print('asinh f32 100x100')
#     report_asinh.print()
#     fn test_atanh()capturing:
#         res = atanh[DType.float32](tens)
#         keep(res.data())
#     var report_atanh = benchmark.run[test_atanh]()
#     print('atanh f32 100x100')
#     report_atanh.print()
#     fn test_cosh()capturing:
#         res = cosh[DType.float32](tens)
#         keep(res.data())
#     var report_cosh = benchmark.run[test_cosh]()
#     print('cosh f32 100x100')
#     report_cosh.print()
#     fn test_sinh()capturing:
#         res = sinh[DType.float32](tens)
#         keep(res.data())
#     var report_sinh = benchmark.run[test_sinh]()
#     print('sinh f32 100x100')
#     report_sinh.print()
#     fn test_expm1()capturing:
#         res = expm1[DType.float32](tens)
#         keep(res.data())
#     var report_expm1 = benchmark.run[test_expm1]()
#     print('expm1 f32 100x100')
#     report_expm1.print()
#     fn test_log10()capturing:
#         res = log10[DType.float32](tens)
#         keep(res.data())
#     var report_log10 = benchmark.run[test_log10]()
#     print('log10 f32 100x100')
#     report_log10.print()
#     fn test_log1p()capturing:
#         res = log1p[DType.float32](tens)
#         keep(res.data())
#     var report_log1p = benchmark.run[test_log1p]()
#     print('log1p f32 100x100')
#     report_log1p.print()
#     fn test_cbrt()capturing:
#         res = cbrt[DType.float32](tens)
#         keep(res.data())
#     var report_cbrt = benchmark.run[test_cbrt]()
#     print('cbrt f32 100x100')
#     report_cbrt.print()
#     fn test_erfc()capturing:
#         res = erfc[DType.float32](tens)
#         keep(res.data())
#     var report_erfc = benchmark.run[test_erfc]()
#     print('erfc f32 100x100')
#     report_erfc.print()
#     fn test_lgamma()capturing:
#         res = lgamma[DType.float32](tens)
#         keep(res.data())
#     var report_lgamma = benchmark.run[test_lgamma]()
#     print('lgamma f32 100x100')
#     report_lgamma.print()
#     fn test_tgamma()capturing:
#         res = tgamma[DType.float32](tens)
#         keep(res.data())
#     var report_tgamma = benchmark.run[test_tgamma]()
#     print('tgamma f32 100x100')
#     report_tgamma.print()
#     fn test_nearbyint()capturing:
#         res = nearbyint[DType.float32](tens)
#         keep(res.data())
#     var report_nearbyint = benchmark.run[test_nearbyint]()
#     print('nearbyint f32 100x100')
#     report_nearbyint.print()
#     fn test_rint()capturing:
#         res = rint[DType.float32](tens)
#         keep(res.data())
#     var report_rint = benchmark.run[test_rint]()
#     print('rint f32 100x100')
#     report_rint.print()
#     fn test_j0()capturing:
#         res = j0[DType.float32](tens)
#         keep(res.data())
#     var report_j0 = benchmark.run[test_j0]()
#     print('j0 f32 100x100')
#     report_j0.print()
#     fn test_j1()capturing:
#         res = j1[DType.float32](tens)
#         keep(res.data())
#     var report_j1 = benchmark.run[test_j1]()
#     print('j1 f32 100x100')
#     report_j1.print()
#     fn test_y0()capturing:
#         res = y0[DType.float32](tens)
#         keep(res.data())
#     var report_y0 = benchmark.run[test_y0]()
#     print('y0 f32 100x100')
#     report_y0.print()
#     fn test_y1()capturing:
#         res = y1[DType.float32](tens)
#         keep(res.data())
#     var report_y1 = benchmark.run[test_y1]()
#     print('y1 f32 100x100')
#     report_y1.print()
#     fn test_ulp()capturing:
#         res = ulp[DType.float32](tens)
#         keep(res.data())
#     var report_ulp = benchmark.run[test_ulp]()
#     print('ulp f32 100x100')
#     report_ulp.print()
