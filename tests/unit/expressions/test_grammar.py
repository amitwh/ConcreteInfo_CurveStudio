"""Grammar validation: friendly errors for typos, caps for hostile input."""

import pytest

from curvestudio.expressions import (
    MAX_DEPTH,
    MAX_EXPRESSION_LENGTH,
    MAX_NODES,
    split_equation,
    validate_expression,
)

NAMES = frozenset({"x", "a", "b", "k"})


def _error(text: str) -> str:
    with pytest.raises(ValueError) as info:
        validate_expression(text, NAMES)
    return str(info.value)


def test_limits_have_the_documented_values() -> None:
    assert (MAX_EXPRESSION_LENGTH, MAX_NODES, MAX_DEPTH) == (1000, 300, 40)


def test_valid_expressions_pass() -> None:
    for text in ("a*exp(b*x)", "a + b*ln(x)", "k/(1+exp(-a*(x-b)))", "a*x**2.5 + 1e-3", " a*x "):
        validate_expression(text, NAMES)


def test_caret_error_suggests_double_star() -> None:
    assert "**" in _error("a*x^2")


@pytest.mark.parametrize("text", ["2x", "a x", "2 x", "a*2x", "sin x", "2(x+1)", "(a)(x)", "x a"])
def test_implicit_multiplication_is_rejected_with_hint(text: str) -> None:
    message = _error(text)
    assert "*" in message, message


def test_function_name_alone_is_rejected() -> None:
    message = _error("a*exp")
    assert "exp" in message and "exp(" in message


def test_parameter_used_as_a_function_is_explained() -> None:
    message = _error("a(x)")
    assert "'a'" in message and "*" in message


def test_unknown_function_suggests_close_matches() -> None:
    assert "exp" in _error("Exp(x)")
    assert "sqrt" in _error("sqr(x)")


def test_unknown_name_suggests_close_matches() -> None:
    message = _error("a*xx")
    assert "'xx'" in message and "'x'" in message


def test_unknown_name_without_close_match_lists_known_names() -> None:
    message = _error("a*zzzzz")
    assert "zzzzz" in message and "known names" in message


def test_equation_prefix_hint() -> None:
    message = _error("y = a*x")
    assert "right-hand side" in message


def test_trailing_equals_hint() -> None:
    assert "=" in _error("a*x =")


def _balanced_sum(leaves: int) -> str:
    """A sum of ``leaves`` x's nested as a balanced tree: ``2*leaves - 1`` nodes, shallow."""
    if leaves == 1:
        return "x"
    half = leaves // 2
    return f"({_balanced_sum(half)}+{_balanced_sum(leaves - half)})"


def test_node_and_depth_caps() -> None:
    validate_expression(_balanced_sum(150), NAMES)  # 299 nodes
    validate_expression("-" + _balanced_sum(150), NAMES)  # exactly 300 nodes
    with pytest.raises(ValueError, match="too many"):
        validate_expression(_balanced_sum(151), NAMES)  # 301 nodes
    with pytest.raises(ValueError, match=r"too many|nested"):
        validate_expression("+".join(["x"] * 400), NAMES)  # flat sums nest one level per term
    validate_expression("-" * (MAX_DEPTH - 1) + "x", NAMES)
    with pytest.raises(ValueError, match="nested"):
        validate_expression("-" * MAX_DEPTH + "x", NAMES)  # 41 deep


def test_length_cap_applies_before_parsing() -> None:
    with pytest.raises(ValueError, match="too long"):
        validate_expression("(" * 1001, NAMES)


def test_non_finite_and_overflowing_constants_are_rejected() -> None:
    for text in ("1e999", "a*1e999", "1e308*10", "10**1000"):
        with pytest.raises(ValueError):
            validate_expression(text, NAMES)


def test_lookalike_characters_are_not_silently_normalised() -> None:
    # a full-width x would be folded to ASCII x by the Python parser
    assert "ASCII" in _error("a*\uff58")
    assert "ASCII" in _error("a\u00d7x")
    assert "unknown name" in _error("a*\ua7b5")


def test_equation_prefix_is_stripped() -> None:
    assert split_equation("y = a*x") == "a*x"
    assert split_equation("  y=a*x  ") == "a*x"
    assert split_equation("f(x) = a*x") == "a*x"
    assert split_equation("f( x )=a*x") == "a*x"
    assert split_equation("a*x") == "a*x"


@pytest.mark.parametrize(
    "text", ["y = a = x", "a*x =", "= a*x", "a == x", "a <= x", "x:=1", "z = a*x", "y = "]
)
def test_split_equation_rejects_other_equals_signs(text: str) -> None:
    with pytest.raises(ValueError):
        split_equation(text)
