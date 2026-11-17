"""Build SymPy expressions from a validated expression, without ``eval``.

:func:`to_sympy` validates the text against the grammar, then walks the parsed
:mod:`ast` and constructs SymPy nodes directly from the whitelist tables, so user text
never reaches ``eval``, ``exec`` or ``sympy.parse_expr``.
"""

import ast
import math
from collections.abc import Mapping
from typing import Any

import sympy

from curvestudio.expressions.grammar import (
    CONSTANTS,
    FUNCTIONS,
    MAX_CONSTANT_EXPONENT,
    parse_expression,
)

_MAX_RESULT_BITS = 10_000.0


def to_sympy(text: str, symbols: Mapping[str, sympy.Symbol]) -> sympy.Expr:
    """SymPy expression for ``text``; ``symbols`` maps each allowed name to its symbol.

    Raises ``ValueError`` for anything outside the grammar (see ``validate_expression``) and
    for an expression that is undefined, infinite or complex whatever the variables are.
    """
    body = parse_expression(text, frozenset(symbols))
    expr = _build(body, symbols)
    if expr.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo, sympy.I):
        raise ValueError(
            f"expression {text!r} is not defined for real numbers (it contains an undefined, "
            "infinite or complex constant sub-expression such as 1/0, log(0) or sqrt(-1))"
        )
    _check_powers(expr, text)
    return expr


def _check_powers(expr: sympy.Expr, text: str) -> None:
    """Second line of defence: refuse powers whose constant exponent SymPy built up.

    SymPy merges ``(x**1000)**1000`` into ``x**1000000`` and ``x**600*x**600`` into
    ``x**1200``, so the per-operator cap in the grammar can be outrun by composition. A power
    with a numeric exponent is limited to the same cap, and a numeric base to a result of
    at most :data:`_MAX_RESULT_BITS` bits.
    """
    for node in sympy.preorder_traversal(expr):
        if not isinstance(node, sympy.Pow) or not node.exp.is_Number:
            continue
        exponent = float(node.exp)
        if abs(exponent) > MAX_CONSTANT_EXPONENT:
            raise ValueError(
                f"expression {text!r} builds a power with constant exponent {exponent:g} "
                f"(limit {MAX_CONSTANT_EXPONENT:g}); simplify the exponents"
            )
        if node.base.is_Number and node.base != 0:
            bits = abs(exponent) * math.log2(abs(float(node.base)))
            if abs(bits) > _MAX_RESULT_BITS:
                raise ValueError(
                    f"expression {text!r} contains the constant power {node.base}**{node.exp}, "
                    "which is too large to evaluate"
                )


def _build(node: ast.expr, symbols: Mapping[str, sympy.Symbol]) -> Any:
    match node:
        case ast.Constant(value=int() as value):
            return sympy.Integer(value)
        case ast.Constant(value=float() as value):
            return sympy.Float(value)
        case ast.Name(id=name):
            return symbols[name] if name in symbols else CONSTANTS[name]
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_build(operand, symbols)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _build(operand, symbols)
        case ast.BinOp(left=left, op=op, right=right):
            lhs, rhs = _build(left, symbols), _build(right, symbols)
            match op:
                case ast.Add():
                    return lhs + rhs
                case ast.Sub():
                    return lhs - rhs
                case ast.Mult():
                    return lhs * rhs
                case ast.Div():
                    return lhs / rhs
                case ast.Pow():
                    return lhs**rhs
        case ast.Call(func=ast.Name(id=name), args=args):
            built = [_build(arg, symbols) for arg in args]
            return FUNCTIONS[name](*built)
    raise ValueError(f"{type(node).__name__} is not allowed in a model expression")
