with open("m1.txt") as funcs_1_inout:
    funcs_1_inout_list =funcs_1_inout.readlines()
with open("tensor_funcs_1_input_1_output.mojo","a") as out:
    for func in funcs_1_inout_list:
        func = func.replace("\n","")
        out.write(
            f"""fn {func}[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:\n    return _math_func[dtype,math.{func}](tensor)\n\n"""
            )
    out.write("""fn main():\n    var tens:Tensor[DType.float32] = Tensor[DType.float32](100,100)\n    for i in range(10_000):\n        tens[i]= SIMD[DType.float32,1](3.141592/4)\n""")
    out.write("    var res:Tensor[DType.float32]\n")
    for func in funcs_1_inout_list:
        func = func.replace("\n","")
        out.write(f"    fn test_{func}()capturing:\n")
        out.write(f"        res = {func}[DType.float32](tens)\n")
        out.write("        keep(res.data())\n")
        out.write(f"    var report_{func} = benchmark.run[test_{func}]()\n")
        out.write(f"    print('{func} f32 100x100')\n")
        out.write(f"    report_{func}.print()\n")
   