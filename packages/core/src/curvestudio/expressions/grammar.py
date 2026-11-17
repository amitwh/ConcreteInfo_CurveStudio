"""The whitelisted expression grammar and its validator.

An expression is the right-hand side of ``y = f(x; parameters)`` written in a small
arithmetic language: numbers, the variable and parameter names the caller allows, the
constants in :data:`CONSTANTS`, the operators ``+ - * / **`` (and unary ``+ -``), and calls
of the functions in :data:`UNARY_FUNCTIONS` / :data:`BINARY_FUNCTIONS`.

Validation parses with :mod:`ast` (which never executes anything) and walks the tree
against that whitelist. Every failure is a ``ValueError`` whose message names the problem
and, where there is an obvious fix, says what to write instead. Size is bounded before
any recursion: text length, node count and nesting depth (:data:`MAX_EXPRESSION_LENGTH`,
:data:`MAX_NODES`, :data:`MAX_DEPTH`).
"""

import ast
import difflib
import re
import unicodedata
import warnings
from collections.abc import Callable
from typing import Any

import sympy

MAX_EXPRESSION_LENGTH = 1000
"""Longest accepted expression, in characters (bounds parse time and recursion depth)."""

MAX_NODES = 300
"""Most expression nodes (numbers, names, operators, calls) in one expression."""

MAX_DEPTH = 40
"""Deepest nesting of expression nodes; a left-to-right sum counts one level per term."""

_MAX_CONSTANT_EXPONENT = 1000.0
"""Largest exponent allowed when it is a pure number (``x**1e9`` would stall SymPy)."""

UNARY_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.log,
    "log10": lambda arg: sympy.log(arg) / sympy.log(10),
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "Abs": sympy.Abs,
}
"""Functions called with exactly one argument, mapped to their SymPy builders."""

BINARY_FUNCTIONS: dict[str, Callable[..., Any]] = {}
"""Functions called with exactly two arguments, mapped to their SymPy builders."""

CONSTANTS: dict[str, Any] = {"pi": sympy.pi, "E": sympy.E}
"""Bare names that stand for a number."""

RESERVED_NAMES = frozenset(UNARY_FUNCTIONS) | frozenset(BINARY_FUNCTIONS) | frozenset(CONSTANTS)
RESERVED_NAMES |= {"Integer", "Float"}
"""Names a variable or parameter may not take (functions, constants, SymPy number types)."""

_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY_OPS = (ast.UAdd, ast.USub)

_IMPLICIT_PRODUCT = re.compile(r"\d(?:\.\d*)?\s*[A-Za-z_(]|[A-Za-z0-9_)]\s+[A-Za-z0-9_(]")
"""Two operands side by side (``2x``, ``a x``, ``2(x+1)``), a syntax error to Python."""

_EQUATION_PREFIX = re.compile(r"\s*(?:y|f\(\s*x\s*\))\s*=(?!=)")


def _function_names() -> list[str]:
    return sorted(UNARY_FUNCTIONS | BINARY_FUNCTIONS)


def _suggest(name: str, candidates: list[str]) -> list[str]:
    """Close matches for ``name``, case-insensitive matches first."""
    folded = [c for c in candidates if c.lower() == name.lower() and c != name]
    close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
    return list(dict.fromkeys([*folded, *close]))


def _quoted(items: list[str]) -> str:
    return " or ".join(repr(item) for item in items)


def validate_expression(text: str, names: frozenset[str]) -> None:
    """Reject anything outside the whitelisted arithmetic grammar with ``ValueError``.

    ``names`` are the variable and parameter names that may appear as bare names.
    Nothing is executed; a valid expression returns ``None``.
    """
    parse_expression(text, names)


def parse_expression(text: str, names: frozenset[str]) -> ast.expr:
    """Validate ``text`` (see :func:`validate_expression`) and return its parsed syntax tree."""
    if not text.strip():
        raise ValueError("expression is empty")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression is too long (limit {MAX_EXPRESSION_LENGTH} characters)")
    if unicodedata.normalize("NFKC", text) != text:
        odd = sorted({ch for ch in text if unicodedata.normalize("NFKC", ch) != ch})
        raise ValueError(
            f"expression contains characters {''.join(odd)!r} that Unicode would silently "
            "rewrite (look-alike letters, superscripts, full-width forms); type plain ASCII"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # e.g. "2(x)": we report it ourselves
            tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(_syntax_message(text, exc)) from exc
    except (ValueError, RecursionError, MemoryError, OverflowError) as exc:
        raise ValueError(f"expression {text!r} is not valid: {exc}") from exc
    _check_size(tree.body)
    try:
        _check_node(tree.body, names)
    except (OverflowError, ZeroDivisionError, TypeError, RecursionError) as exc:
        raise ValueError(f"expression {text!r} has an unusable constant: {exc}") from exc
    return tree.body


def _syntax_message(text: str, exc: SyntaxError) -> str:
    """Turn Python's syntax error into a hint about what the author most likely meant."""
    stripped = text.strip()
    if stripped.endswith("="):
        return f"expression {text!r} ends with '='; write only the right-hand side"
    if "=" in stripped and not any(op in stripped for op in ("==", "<=", ">=", "!=", ":=")):
        return (
            f"expression {text!r} contains '='; write only the right-hand side "
            "(remove the 'y =' prefix)"
        )
    non_ascii = sorted({ch for ch in stripped if ord(ch) > 127})
    if non_ascii:
        return (
            f"expression {text!r} contains the non-ASCII character(s) {''.join(non_ascii)!r}; "
            "use plain ASCII operators (* / ** + -)"
        )
    if _IMPLICIT_PRODUCT.search(stripped):
        return (
            f"expression {text!r} is not valid: implicit multiplication is not supported, "
            "write '*' between factors (for example 2*x, a*x, sin(x) not 'sin x')"
        )
    return f"expression {text!r} is not valid: {exc.msg}"


def _check_size(root: ast.expr) -> None:
    """Bound node count and depth with an explicit stack, before any recursion."""
    nodes = 0
    stack: list[tuple[ast.AST, int]] = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, ast.expr):
            nodes += 1
            if nodes > MAX_NODES:
                raise ValueError(
                    f"expression has too many terms (limit {MAX_NODES} operations, numbers "
                    "and names); simplify it"
                )
            if depth > MAX_DEPTH:
                raise ValueError(
                    f"expression is nested too deeply (limit {MAX_DEPTH} levels); simplify it"
                )
        child_depth = depth + 1 if isinstance(node, ast.expr) else depth
        stack.extend((child, child_depth) for child in ast.iter_child_nodes(node))


def _plain_number(value: object) -> float | None:
    """``value`` as a float if it is a plain int or float (not bool, complex, str...)."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _numeric_value(node: ast.expr) -> float | None:
    """Evaluate a name-free arithmetic subtree in floats, or return ``None`` if it has names.

    Floats overflow loudly (``OverflowError``) instead of building astronomically large
    integers, which is what makes ``9**9**9`` safe to reject. A non-finite result is an error.
    """
    if isinstance(node, ast.Constant):
        return _plain_number(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _numeric_value(node.operand)
        if operand is None:
            return None
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left, right = _numeric_value(node.left), _numeric_value(node.right)
        if left is None or right is None:
            return None
        match node.op:
            case ast.Add():
                result = left + right
            case ast.Sub():
                result = left - right
            case ast.Mult():
                result = left * right
            case ast.Div():
                result = left / right
            case _:
                result = float(left**right)
        if not -float("inf") < result < float("inf"):
            raise OverflowError("constant sub-expression is not finite")
        return result
    return None


def _check_node(node: ast.expr, names: frozenset[str]) -> None:
    match node:
        case ast.BinOp():
            if isinstance(node.op, ast.BitXor):
                raise ValueError("'^' is not a power operator here; use ** for powers (x**2)")
            if not isinstance(node.op, _BINARY_OPS):
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            if isinstance(node.op, ast.Pow):
                exponent = _numeric_value(node.right)
                if exponent is not None and abs(exponent) > _MAX_CONSTANT_EXPONENT:
                    raise ValueError(f"constant exponent {exponent:g} is too large")
            _numeric_value(node)  # a name-free subtree must fold to a finite number
            _check_node(node.left, names)
            _check_node(node.right, names)
        case ast.UnaryOp():
            if not isinstance(node.op, _UNARY_OPS):
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            _check_node(node.operand, names)
        case ast.Constant():
            number = _plain_number(node.value)
            if number is None:
                raise ValueError(f"constant {node.value!r} is not a plain number")
            if not -float("inf") < number < float("inf"):
                raise ValueError(f"constant {node.value!r} is too large to use")
        case ast.Name():
            _check_name(node.id, names)
        case ast.Call():
            _check_call(node, names)
        case _:
            raise ValueError(f"{type(node).__name__} is not allowed in a model expression")


def _check_name(name: str, names: frozenset[str]) -> None:
    if name in UNARY_FUNCTIONS:
        raise ValueError(f"function {name!r} must be called, e.g. {name}(x)")
    if name in BINARY_FUNCTIONS:
        raise ValueError(f"function {name!r} must be called, e.g. {name}(x, y)")
    if name in names or name in CONSTANTS:
        return
    known = sorted(names | CONSTANTS.keys())
    close = _suggest(name, known)
    hint = f"did you mean {_quoted(close)}?" if close else f"known names: {', '.join(known)}"
    raise ValueError(f"unknown name {name!r}; {hint}")


def _check_call(node: ast.Call, names: frozenset[str]) -> None:
    func = node.func
    if not isinstance(func, ast.Name):
        raise ValueError(
            "only whitelisted functions may be called; for multiplication write '*' "
            "(for example 2*(x+1))"
        )
    if func.id in names or func.id in CONSTANTS:
        raise ValueError(
            f"{func.id!r} is a parameter, variable or constant, not a function; "
            f"write '*' for multiplication (for example {func.id}*(x+1))"
        )
    if func.id in UNARY_FUNCTIONS:
        arity = 1
    elif func.id in BINARY_FUNCTIONS:
        arity = 2
    else:
        close = _suggest(func.id, _function_names())
        hint = (
            f"did you mean {_quoted(close)}?"
            if close
            else f"allowed functions: {', '.join(_function_names())}"
        )
        raise ValueError(f"unknown function {func.id!r}; {hint}")
    if (
        node.keywords
        or len(node.args) != arity
        or any(isinstance(a, ast.Starred) for a in node.args)
    ):
        plural = "argument" if arity == 1 else "arguments"
        raise ValueError(f"{func.id}() takes exactly {arity} positional {plural}")
    for argument in node.args:
        _check_node(argument, names)


def split_equation(text: str) -> str:
    """Strip an optional leading ``y =`` or ``f(x) =`` and return the right-hand side.

    Any other ``=`` (a second one, a trailing one, ``==``, ``<=``, ``:=``) is a
    ``ValueError``; it is never reinterpreted.
    """
    body = _EQUATION_PREFIX.sub("", text, count=1)
    if "=" in body:
        raise ValueError(
            f"expression {text!r} has a stray '='; write one equation as 'y = <expression>' "
            "or just the right-hand side"
        )
    if not body.strip():
        raise ValueError("expression is empty")
    return body.strip()
