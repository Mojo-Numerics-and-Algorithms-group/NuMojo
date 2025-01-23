import math as mt

alias DEFAULT_PRECISION = 4
alias DEFAULT_LINEWIDTH = 10
alias DEFAULT_SUPPRESS_SMALL = False
alias DEFAULT_SEPARATOR = "\t"
alias DEFAULT_PADDING = "\t"
alias DEFAULT_ITEMS_TO_DISPLAY = 6

var GLOBAL_PRINT_OPTIONS: PrintOptions = PrintOptions(
    precision=DEFAULT_PRECISION,
    linewidth=DEFAULT_LINEWIDTH,
    suppress_small=DEFAULT_SUPPRESS_SMALL,
    separator=DEFAULT_SEPARATOR,
    padding=DEFAULT_PADDING,
    items_to_display=DEFAULT_ITEMS_TO_DISPLAY,
)

alias printoptions = PrintOptions


@value
struct PrintOptions:
    var precision: Int
    var linewidth: Int
    var suppress_small: Bool
    var separator: String
    var padding: String
    var items_to_display: Int

    fn __init__(
        mut self,
        precision: Int = DEFAULT_PRECISION,
        linewidth: Int = DEFAULT_LINEWIDTH,
        suppress_small: Bool = DEFAULT_SUPPRESS_SMALL,
        separator: String = DEFAULT_SEPARATOR,
        padding: String = DEFAULT_PADDING,
        items_to_display: Int = DEFAULT_ITEMS_TO_DISPLAY,
    ):
        self.precision = precision
        self.linewidth = linewidth
        self.suppress_small = suppress_small
        self.separator = separator
        self.padding = padding
        self.items_to_display = items_to_display

    fn set_options(
        mut self,
        precision: Int = DEFAULT_PRECISION,
        linewidth: Int = DEFAULT_LINEWIDTH,
        suppress_small: Bool = DEFAULT_SUPPRESS_SMALL,
        separator: String = DEFAULT_SEPARATOR,
        padding: String = DEFAULT_PADDING,
        items_to_display: Int = DEFAULT_ITEMS_TO_DISPLAY,
    ):
        self.precision = precision
        self.linewidth = linewidth
        self.suppress_small = suppress_small
        self.separator = separator
        self.padding = padding
        self.items_to_display = items_to_display

    fn __enter__(mut self) -> Self:
        GLOBAL_PRINT_OPTIONS.set_options(
            precision=self.precision,
            linewidth=self.linewidth,
            suppress_small=self.suppress_small,
            separator=self.separator,
            padding=self.padding,
            items_to_display=self.items_to_display,
        )
        return GLOBAL_PRINT_OPTIONS

    fn __exit__(mut self):
        GLOBAL_PRINT_OPTIONS.set_options(
            precision=DEFAULT_PRECISION,
            linewidth=DEFAULT_LINEWIDTH,
            suppress_small=DEFAULT_SUPPRESS_SMALL,
            separator=DEFAULT_SEPARATOR,
            padding=DEFAULT_PADDING,
            items_to_display=DEFAULT_ITEMS_TO_DISPLAY,
        )


fn set_printoptions(
    precision: Int = DEFAULT_PRECISION,
    linewidth: Int = DEFAULT_LINEWIDTH,
    suppress_small: Bool = DEFAULT_SUPPRESS_SMALL,
    separator: String = DEFAULT_SEPARATOR,
    padding: String = DEFAULT_PADDING,
    items_to_display: Int = DEFAULT_ITEMS_TO_DISPLAY,
):
    GLOBAL_PRINT_OPTIONS.set_options(
        precision,
        linewidth,
        suppress_small,
        separator,
        padding,
        items_to_display,
    )


# TODO: fix the problem where precision > number of digits in the mantissa results in a not so exact value.
fn format_float_scientific[
    dtype: DType = DType.float64
](x: Scalar[dtype], precision: Int = 10, sign: Bool = False) raises -> String:
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
        raise Error(
            "Invalid type provided. dtype must be a floating-point type."
        )

    var power: Int = int(mt.log10(abs(x)))
    var mantissa: Scalar[dtype] = x / (10**power)
    var m_string: String = String("{0}").format(mantissa)
    var result: String = m_string[0] + "."
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


fn format_floating_values[dtype: DType](value: Scalar[dtype]) raises -> String:
    """
    Format a floating-point value to the specified precision.
    """
    var precision = GLOBAL_PRINT_OPTIONS.precision
    var suppress_small = GLOBAL_PRINT_OPTIONS.suppress_small

    if suppress_small and abs(value) < 1e-10:
        return "0.0"

    scaling_factor = 10**precision
    rounded_value = round(value * scaling_factor) / scaling_factor

    integer_part = int(rounded_value)
    fractional_part = abs(rounded_value - integer_part)

    result = str(integer_part)

    if precision > 0:
        result += "."
        for _ in range(precision):
            fractional_part *= 10
            digit = int(fractional_part)
            result += str(digit)
            fractional_part -= digit

    return result
