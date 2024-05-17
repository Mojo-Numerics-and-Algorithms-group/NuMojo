from numojo import *
import benchmark
from benchmark.compiler import keep
fn main():
    var tens1:Tensor[DType.float32] = Tensor[DType.float32](100,100)
    var tens2:Tensor[DType.float32] = Tensor[DType.float32](100,100)
    for i in range(10_000):
        tens1[i]= SIMD[DType.float32,1](3.141592/4)
        tens2[i]= SIMD[DType.float32,1](3.141592)
    var res:Tensor[DType.float32]
    fn test_mod()capturing:
        try:
            res = mod[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('mod: Failed shape error')
    var report_mod = benchmark.run[test_mod]()
    print('mod f32 100x100')
    report_mod.print()
    fn test_mul()capturing:
        try:
            res = mul[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('mul: Failed shape error')
    var report_mul = benchmark.run[test_mul]()
    print('mul f32 100x100')
    report_mul.print()
    fn test_sub()capturing:
        try:
            res = sub[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('sub: Failed shape error')
    var report_sub = benchmark.run[test_sub]()
    print('sub f32 100x100')
    report_sub.print()
    fn test_add()capturing:
        try:
            res = add[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('add: Failed shape error')
    var report_add = benchmark.run[test_add]()
    print('add f32 100x100')
    report_add.print()
    fn test_div()capturing:
        try:
            res = div[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('div: Failed shape error')
    var report_div = benchmark.run[test_div]()
    print('div f32 100x100')
    report_div.print()
    fn test_copysign()capturing:
        try:
            res = copysign[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('copysign: Failed shape error')
    var report_copysign = benchmark.run[test_copysign]()
    print('copysign f32 100x100')
    report_copysign.print()
    fn test_atan2()capturing:
        try:
            res = atan2[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('atan2: Failed shape error')
    var report_atan2 = benchmark.run[test_atan2]()
    print('atan2 f32 100x100')
    report_atan2.print()
    fn test_hypot()capturing:
        try:
            res = hypot[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('hypot: Failed shape error')
    var report_hypot = benchmark.run[test_hypot]()
    print('hypot f32 100x100')
    report_hypot.print()
    fn test_nextafter()capturing:
        try:
            res = nextafter[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('nextafter: Failed shape error')
    var report_nextafter = benchmark.run[test_nextafter]()
    print('nextafter f32 100x100')
    report_nextafter.print()
    fn test_scalb()capturing:
        try:
            res = scalb[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('scalb: Failed shape error')
    var report_scalb = benchmark.run[test_scalb]()
    print('scalb f32 100x100')
    report_scalb.print()
    fn test_remainder()capturing:
        try:
            res = remainder[DType.float32](tens1, tens2)
            keep(res.data())
        except:
            print('remainder: Failed shape error')
    var report_remainder = benchmark.run[test_remainder]()
    print('remainder f32 100x100')
    report_remainder.print()
    var tens:Tensor[DType.float32] = Tensor[DType.float32](100,100)
    for i in range(10_000):
        tens[i]= SIMD[DType.float32,1](3.141592/4)
    fn test_abs()capturing:
        res = abs[DType.float32](tens)
        keep(res.data())
    var report_abs = benchmark.run[test_abs]()
    print('abs f32 100x100')
    report_abs.print()
    fn test_floor()capturing:
        res = floor[DType.float32](tens)
        keep(res.data())
    var report_floor = benchmark.run[test_floor]()
    print('floor f32 100x100')
    report_floor.print()
    fn test_ceil()capturing:
        res = ceil[DType.float32](tens)
        keep(res.data())
    var report_ceil = benchmark.run[test_ceil]()
    print('ceil f32 100x100')
    report_ceil.print()
    fn test_trunc()capturing:
        res = trunc[DType.float32](tens)
        keep(res.data())
    var report_trunc = benchmark.run[test_trunc]()
    print('trunc f32 100x100')
    report_trunc.print()
    fn test_round()capturing:
        res = round[DType.float32](tens)
        keep(res.data())
    var report_round = benchmark.run[test_round]()
    print('round f32 100x100')
    report_round.print()
    fn test_roundeven()capturing:
        res = roundeven[DType.float32](tens)
        keep(res.data())
    var report_roundeven = benchmark.run[test_roundeven]()
    print('roundeven f32 100x100')
    report_roundeven.print()
    fn test_round_half_down()capturing:
        res = round_half_down[DType.float32](tens)
        keep(res.data())
    var report_round_half_down = benchmark.run[test_round_half_down]()
    print('round_half_down f32 100x100')
    report_round_half_down.print()
    fn test_round_half_up()capturing:
        res = round_half_up[DType.float32](tens)
        keep(res.data())
    var report_round_half_up = benchmark.run[test_round_half_up]()
    print('round_half_up f32 100x100')
    report_round_half_up.print()
    fn test_rsqrt()capturing:
        res = rsqrt[DType.float32](tens)
        keep(res.data())
    var report_rsqrt = benchmark.run[test_rsqrt]()
    print('rsqrt f32 100x100')
    report_rsqrt.print()
    fn test_exp2()capturing:
        res = exp2[DType.float32](tens)
        keep(res.data())
    var report_exp2 = benchmark.run[test_exp2]()
    print('exp2 f32 100x100')
    report_exp2.print()
    fn test_exp()capturing:
        res = exp[DType.float32](tens)
        keep(res.data())
    var report_exp = benchmark.run[test_exp]()
    print('exp f32 100x100')
    report_exp.print()
    fn test_log()capturing:
        res = log[DType.float32](tens)
        keep(res.data())
    var report_log = benchmark.run[test_log]()
    print('log f32 100x100')
    report_log.print()
    fn test_log2()capturing:
        res = log2[DType.float32](tens)
        keep(res.data())
    var report_log2 = benchmark.run[test_log2]()
    print('log2 f32 100x100')
    report_log2.print()
    # fn test_erf()capturing:
    #     res = erf[DType.float32](tens)
    #     keep(res.data())
    # var report_erf = benchmark.run[test_erf]()
    # print('erf f32 100x100')
    # report_erf.print()
    fn test_tanh()capturing:
        res = tanh[DType.float32](tens)
        keep(res.data())
    var report_tanh = benchmark.run[test_tanh]()
    print('tanh f32 100x100')
    report_tanh.print()
    fn test_reciprocal()capturing:
        res = reciprocal[DType.float32](tens)
        keep(res.data())
    var report_reciprocal = benchmark.run[test_reciprocal]()
    print('reciprocal f32 100x100')
    report_reciprocal.print()
    # fn test_identity()capturing:
    #     res = identity[DType.float32](tens)
    #     keep(res.data())
    # var report_identity = benchmark.run[test_identity]()
    # print('identity f32 100x100')
    # report_identity.print()
    fn test_acos()capturing:
        res = acos[DType.float32](tens)
        keep(res.data())
    var report_acos = benchmark.run[test_acos]()
    print('acos f32 100x100')
    report_acos.print()
    fn test_asin()capturing:
        res = asin[DType.float32](tens)
        keep(res.data())
    var report_asin = benchmark.run[test_asin]()
    print('asin f32 100x100')
    report_asin.print()
    fn test_atan()capturing:
        res = atan[DType.float32](tens)
        keep(res.data())
    var report_atan = benchmark.run[test_atan]()
    print('atan f32 100x100')
    report_atan.print()
    fn test_cos()capturing:
        res = cos[DType.float32](tens)
        keep(res.data())
    var report_cos = benchmark.run[test_cos]()
    print('cos f32 100x100')
    report_cos.print()
    fn test_sin()capturing:
        res = sin[DType.float32](tens)
        keep(res.data())
    var report_sin = benchmark.run[test_sin]()
    print('sin f32 100x100')
    report_sin.print()
    fn test_tan()capturing:
        res = tan[DType.float32](tens)
        keep(res.data())
    var report_tan = benchmark.run[test_tan]()
    print('tan f32 100x100')
    report_tan.print()
    fn test_acosh()capturing:
        res = acosh[DType.float32](tens)
        keep(res.data())
    var report_acosh = benchmark.run[test_acosh]()
    print('acosh f32 100x100')
    report_acosh.print()
    fn test_asinh()capturing:
        res = asinh[DType.float32](tens)
        keep(res.data())
    var report_asinh = benchmark.run[test_asinh]()
    print('asinh f32 100x100')
    report_asinh.print()
    fn test_atanh()capturing:
        res = atanh[DType.float32](tens)
        keep(res.data())
    var report_atanh = benchmark.run[test_atanh]()
    print('atanh f32 100x100')
    report_atanh.print()
    fn test_cosh()capturing:
        res = cosh[DType.float32](tens)
        keep(res.data())
    var report_cosh = benchmark.run[test_cosh]()
    print('cosh f32 100x100')
    report_cosh.print()
    fn test_sinh()capturing:
        res = sinh[DType.float32](tens)
        keep(res.data())
    var report_sinh = benchmark.run[test_sinh]()
    print('sinh f32 100x100')
    report_sinh.print()
    fn test_expm1()capturing:
        res = expm1[DType.float32](tens)
        keep(res.data())
    var report_expm1 = benchmark.run[test_expm1]()
    print('expm1 f32 100x100')
    report_expm1.print()
    fn test_log10()capturing:
        res = log10[DType.float32](tens)
        keep(res.data())
    var report_log10 = benchmark.run[test_log10]()
    print('log10 f32 100x100')
    report_log10.print()
    fn test_log1p()capturing:
        res = log1p[DType.float32](tens)
        keep(res.data())
    var report_log1p = benchmark.run[test_log1p]()
    print('log1p f32 100x100')
    report_log1p.print()
    fn test_cbrt()capturing:
        res = cbrt[DType.float32](tens)
        keep(res.data())
    var report_cbrt = benchmark.run[test_cbrt]()
    print('cbrt f32 100x100')
    report_cbrt.print()
    # fn test_erfc()capturing:
    #     res = erfc[DType.float32](tens)
    #     keep(res.data())
    # var report_erfc = benchmark.run[test_erfc]()
    # print('erfc f32 100x100')
    # report_erfc.print()
    # fn test_lgamma()capturing:
    #     res = lgamma[DType.float32](tens)
    #     keep(res.data())
    # var report_lgamma = benchmark.run[test_lgamma]()
    # print('lgamma f32 100x100')
    # report_lgamma.print()
    # fn test_tgamma()capturing:
    #     res = tgamma[DType.float32](tens)
    #     keep(res.data())
    # var report_tgamma = benchmark.run[test_tgamma]()
    # print('tgamma f32 100x100')
    # report_tgamma.print()
    # fn test_nearbyint()capturing:
    #     res = nearbyint[DType.float32](tens)
    #     keep(res.data())
    # var report_nearbyint = benchmark.run[test_nearbyint]()
    # print('nearbyint f32 100x100')
    # report_nearbyint.print()
    # fn test_rint()capturing:
    #     res = rint[DType.float32](tens)
    #     keep(res.data())
    # var report_rint = benchmark.run[test_rint]()
    # print('rint f32 100x100')
    # report_rint.print()
    # fn test_j0()capturing:
    #     res = j0[DType.float32](tens)
    #     keep(res.data())
    # var report_j0 = benchmark.run[test_j0]()
    # print('j0 f32 100x100')
    # report_j0.print()
    # fn test_j1()capturing:
    #     res = j1[DType.float32](tens)
    #     keep(res.data())
    # var report_j1 = benchmark.run[test_j1]()
    # print('j1 f32 100x100')
    # report_j1.print()
    # fn test_y0()capturing:
    #     res = y0[DType.float32](tens)
    #     keep(res.data())
    # var report_y0 = benchmark.run[test_y0]()
    # print('y0 f32 100x100')
    # report_y0.print()
    # fn test_y1()capturing:
    #     res = y1[DType.float32](tens)
    #     keep(res.data())
    # var report_y1 = benchmark.run[test_y1]()
    # print('y1 f32 100x100')
    # report_y1.print()
    # fn test_ulp()capturing:
    #     res = ulp[DType.float32](tens)
    #     keep(res.data())
    # var report_ulp = benchmark.run[test_ulp]()
    # print('ulp f32 100x100')
    # report_ulp.print()

