"""AST to SymPy conversion: same trees as parse_expr, no dynamic execution."""

import builtins
from typing import Any

import pytest
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from curvestudio.expressions import to_sympy
from curvestudio.expressions.grammar import CONSTANTS, UNARY_FUNCTIONS
from curvestudio.models.starter import STARTER_MODELS


def _symbols(*names: str) -> dict[str, sympy.Symbol]:
    return {name: sympy.Symbol(name, real=True) for name in names}


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda s: s.id)
def test_matches_parse_expr_on_the_starter_models(spec: Any) -> None:
    symbols = _symbols(*spec.variables, *(p.name for p in spec.parameters))
    old = parse_expr(
        spec.expression,
        local_dict={**UNARY_FUNCTIONS, **CONSTANTS, **symbols},
        global_dict={"Integer": sympy.Integer, "Float": sympy.Float},
        transformations=standard_transformations,
    )
    new = to_sympy(spec.expression, symbols)
    assert sympy.simplify(old - new) == 0
    assert old == new  # identical trees, not just equivalent ones


def test_to_sympy_never_calls_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("dynamic execution attempted")

    symbols = _symbols("x", "a", "b")
    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    expr = to_sympy("a*exp(-b*x) + sqrt(x)/2 - 1.5e-3*x**2 + pi", symbols)
    assert expr.free_symbols == set(symbols.values())


def test_numbers_become_exact_sympy_numbers() -> None:
    x = sympy.Symbol("x", real=True)
    expr = to_sympy("2*x + 0.5", {"x": x})
    assert expr == 2 * x + sympy.Float(0.5)
    assert isinstance(to_sympy("3", {}), sympy.Integer)
    assert isinstance(to_sympy("3.0", {}), sympy.Float)


def test_operators_and_unary_signs() -> None:
    x, a = sympy.Symbol("x", real=True), sympy.Symbol("a", real=True)
    symbols = {"x": x, "a": a}
    assert to_sympy("-x", symbols) == -x
    assert to_sympy("+x", symbols) == x
    assert to_sympy("a - x", symbols) == a - x
    assert to_sympy("a / x", symbols) == a / x
    assert to_sympy("x ** 2", symbols) == x**2


def test_constants_and_function_aliases() -> None:
    x = sympy.Symbol("x", real=True)
    assert to_sympy("ln(x)", {"x": x}) == sympy.log(x) == to_sympy("log(x)", {"x": x})
    assert to_sympy("E", {}) == sympy.E
    assert to_sympy("pi", {}) == sympy.pi


def test_invalid_input_is_validated_before_conversion() -> None:
    with pytest.raises(ValueError):
        to_sympy("__import__('os')", {})
    with pytest.raises(ValueError, match="unknown name"):
        to_sympy("a*x", _symbols("x"))
