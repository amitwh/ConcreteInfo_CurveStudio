"""``analyze_expression``: parameter inference, LaTeX and the normalised form."""

import pytest

from curvestudio.expressions import (
    ExpressionReport,
    analyze_expression,
    validate_expression,
)


def test_parameters_inferred_in_order() -> None:
    report = analyze_expression("a*exp(-b*x)+c")
    assert isinstance(report, ExpressionReport)
    assert report.parameters == ("a", "b", "c")


def test_names_that_are_constants_or_functions_are_not_parameters() -> None:
    report = analyze_expression("a*sin(2*pi*x/E)")
    assert report.parameters == ("a",)


def test_latex_and_normalised_expression() -> None:
    report = analyze_expression("y = a*x + b*x**2 + 3")
    assert report.expression == "a*x + b*x**2 + 3"
    assert report.latex == r"a x + b x^{2} + 3"
    assert report.node_count > 0
    # the normalised form is SymPy's canonical spelling, not the author's layout
    assert analyze_expression("a*x*x").expression == "a*x**2"


def test_misspelled_function_name_suggests_the_whitelisted_one() -> None:
    with pytest.raises(ValueError, match=r"unknown function 'Exp'.*did you mean 'exp'\?"):
        analyze_expression("Exp(x)")


def _balanced_product(factors: int) -> str:
    """A balanced binary tree of ``sin(p_i*x)`` factors: 299 nodes, ~9 levels deep."""

    def build(lo: int, hi: int) -> str:
        if hi - lo == 1:
            return f"sin(p{lo}*x)"
        mid = (lo + hi) // 2
        return f"({build(lo, mid)}*{build(mid, hi)})"

    return build(0, factors)


def test_normalised_expression_always_revalidates() -> None:
    # SymPy flattens the balanced 50-factor product into one flat Mul and sstr prints it as
    # a single 50-factor chain, which the grammar's depth cap refuses. The report must
    # never hand out text the user did not write and cannot reuse: fall back to the
    # author's own (validated) right-hand side.
    text = _balanced_product(50)
    report = analyze_expression(text)
    assert report.parameters == tuple(f"p{i}" for i in range(50))
    validate_expression(report.expression, frozenset({"x", *report.parameters}))
    assert report.expression == text  # the author's spelling, since SymPy's would not parse
