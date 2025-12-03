from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from sympy import sympify, Symbol, Eq, integrate, diff, solveset, S, latex

# Initialize MCP server
mcp = FastMCP("calculator")

def _safe_expr(expr: str):
    """
    Parse a math expression safely into a SymPy object.
    Examples:
      "sin(x) + x^2"  -> use ** for power: x**2
      "exp(-x)*cos(x)"
    """
    expr = expr.replace("^", "**")
    return sympify(expr, convert_xor=True)

def _fmt(obj) -> str:
    """Human-friendly string with LaTeX (optional) for clarity."""
    try:
        return f"{str(obj)}\nLaTeX: {latex(obj)}"
    except Exception:
        return str(obj)

@mcp.tool()
async def evaluate(expression: str) -> str:
    """
    Evaluate a numeric/symbolic expression.
    Args:
        expression: e.g., "2+3*4", "sin(pi/3)", "limit(sin(x)/x, x, 0)"
    """
    try:
        val = _safe_expr(expression)
        # Try to evaluate to a number if possible
        evaluated = val.evalf() if val.free_symbols == set() else val
        return f"Result:\n{_fmt(evaluated)}"
    except Exception as e:
        return f"Error: could not evaluate expression. Details: {e}"

@mcp.tool()
async def differentiate(expression: str, variable: str = "x", order: int = 1) -> str:
    """
    Differentiate an expression wrt a variable.
    Args:
        expression: e.g., "sin(x)*exp(x)"
        variable:   e.g., "x"
        order:      e.g., 2 for second derivative
    """
    try:
        # TODO: (optional) validate order >= 1
        if order >= 1:
            pass
        else:
            return "Incorrect order input since it must be a positive integer."

        # TODO: create a SymPy symbol from `variable`
        # x = ...
        x = Symbol(variable)

        # TODO: parse `expression` safely to a SymPy object using _safe_expr
        # f = ...
        f = _safe_expr(expression)

        # TODO: compute the derivative of given order
        # diff = ... using diff(f, x, order)
        diff_exp = diff(f, x, order)

        # format the result using _fmt and return
        return f"diff^{order}/diff{variable}^{order} of {expression}:\n{_fmt(diff_exp)}"
    except Exception as e:
        return f"Error: differentiation failed. Details: {e}"

@mcp.tool()
async def integrate_expr(expression: str, variable: str = "x",
                         lower: Optional[str] = None,
                         upper: Optional[str] = None) -> str:
    """
    Integrate an expression (indefinite or definite).
    Args:
        expression: e.g., "exp(-x^2)"
        variable:   e.g., "x"
        lower:      e.g., "0" (omit for indefinite)
        upper:      e.g., "1" (omit for indefinite)
    """
    try:
        # TODO (optional): if exactly one of lower/upper is provided, return a helpful error.
        if (lower is None) != (upper is None):
            return "Invalid input: both lower and upper bounds must be provided for definite integrals."
            
        # TODO: create a SymPy symbol from `variable`
        # x = ...
        x = Symbol(variable)

        # TODO: parse `expression` safely to a SymPy object using _safe_expr
        # f = ...
        f = _safe_expr(expression)

        # TODO: compute the integral
        if lower is not None and upper is not None:
            # lower_bound = ... using _safe_expr
            # upper_bound = ... using _safe_expr
            # definite_int = ... using integrate(f, (x, lower_bound, upper_bound))
            lower_bound = _safe_expr(lower)
            upper_bound = _safe_expr(upper)
            definite_int = integrate(f, (x, lower_bound, upper_bound))
            return f"∫_{lower}^{upper} {expression} d{variable} =\n{_fmt(definite_int)}"
        else:
            # indefinite_int = ... using integrate(f, x)
            indefinite_int = integrate(f, x)
            return f"∫ {expression} d{variable} =\n{_fmt(indefinite_int)} + C"
    except Exception as e:
        return f"Error: integration failed. Details: {e}"

@mcp.tool()
async def solve_equation(equation: str, variable: str = "x", domain: str = "C") -> str:
    """
    Solve equation(s) for a variable.
    Args:
        equation: e.g., "x^2 - 2 = 0" or "sin(x)=1/2"
        variable: variable to solve for
        domain:   'R' (reals) or 'C' (complex)
    """
    try:
        # TODO: create a SymPy symbol for the variable
        # x = ...
        x = Symbol(variable)

        # TODO: handle input string
        # If equation contains "=", split into lhs and rhs
        if "=" in equation:
            # lhs, rhs = using .split()
            # expr = ... using Eq(...)
            lhs_str, rhs_str = equation.split("=", 1)
            lhs = _safe_expr(lhs_str.strip())
            rhs = _safe_expr(rhs_str.strip())
            expr = Eq(lhs, rhs)
        # Otherwise, treat as "expression = 0" using Eq(...)
        else:
            # expr = ... using Eq(...) with S.Zero
            expr = Eq(_safe_expr(equation), S.Zero)


#        TODO: select domain (S.Reals if domain == "R", else S.Complexes)
        # dom = ...
        if domain.upper() == "R":
            dom = S.Reals
        else:  
            dom = S.Complexes

        # TODO: solve using solveset
        # sol = ... 
        sol = solveset(expr, x, domain=dom)


        return f"Solutions for {equation} over {domain}:\n{_fmt(sol)}"
    except Exception as e:
        return f"Error: solving failed. Details: {e}"
    
from sympy.stats import Normal, Uniform, Bernoulli, E, variance

@mcp.tool()
async def stats_simple(dist: str, p1: float, p2: Optional[float] = None) -> str:
    """
    Compute expectation and variance for a distribution (normal, uniform, or bernoulli).

    dist = "normal", "uniform", or "bernoulli"
    p1 and p2 = parameters depend on distribution
    """
    try:
        match dist.lower():
            case "normal":
                X = Normal("X", p1, p2)

            case "uniform":
                X = Uniform("X", p1, p2)

            case "bernoulli":
                X = Bernoulli("X", p1)

            case _:
                return "Error: Not suppported distribution. Use 'normal', 'uniform', or 'bernoulli' only."

        mean = E(X)
        var = variance(X)

        return f"Mean: {mean}\nVariance: {var}"

    except Exception as e:
        return f"Tool computing error: {e}"


def main():
    # Use stdio transport for Claude Desktop integration
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
