import math as mt

#TODO: fix the problem where precision > number of digits in the mantissa results in a not so exact value.
fn format_float_scientific[dtype: DType = DType.float64](x: Scalar[dtype], precision: Int = 10, sign: Bool = False) raises -> String:
    """
    Format a float in scientific notation.

    Parameters:
        dtype: Datatype of the float.
    
    Args:
        x: The float to format.
        precision: The number of decimal places to include in the mantissa.
        sign: Whether to include the sign of the float in the result. Defaults to False.

    Returns:
        A string representation of the float in scientific notation.
    """
    if dtype.is_integral():
        raise Error("Invalid type provided. dtype must be a floating-point type.")

    var power: Int = int(mt.log10(abs(x)))
    var mantissa: Scalar[dtype] = x / (10 ** power)
    var m_string: String = String("{0}").format(mantissa)
    var result: String = m_string[0] + '.'
    print(len(m_string))
    for i in range(2, precision + 2):
        print(m_string[i])
        result += m_string[i]
        if i >= len(m_string):
            break
    if x < 0:
        return String("-{0}e{0}").format(result, power)
    
    if sign:
        return String("+{0}e{1}").format(result, power)
    else:
        return String("{0}e{1}").format(result, power)
    




