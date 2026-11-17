"""Static analysis of a validated expression: free parameters, LaTeX, normalised text.

:func:`analyze_expression` discovers the free names itself, so a caller that does not
yet know which letters the author used as parameters (the custom-equation builder) can
validate the text and learn them in one call. It is still a pure grammar-layer module:
no ``eval``, and nothing outside the whitelist is ever executed.
"""

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass

import sympy

from curvestudio.expressions.convert import to_sympy
from curvestudio.expressions.grammar import (
    RESERVED_NAMES,
    parse_expression,
    split_equation,
    validate_expression,
)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""Every identifier-shaped token in the text; a superset of the names the AST can use."""

_CALLED_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
"""Identifier-shaped tokens in call position: names, never parameters or variables.

Keeping them out of the candidate set lets the grammar report a misspelled function
(``Exp(x)``) as an unknown function with the right suggestion, instead of treating it
as a parameter that must not be called.
"""


@dataclass(frozen=True)
class ExpressionReport:
    """What :func:`analyze_expression` found out about an expression.

    ``expression`` is the normalised right-hand side (SymPy's canonical spelling, any
    ``y =`` prefix removed; the author's own validated text when SymPy's spelling would
    not re-validate), ``parameters`` lists the free names — those that are neither
    variables nor constants — in order of first appearance, ``latex`` renders the
    right-hand side, and ``node_count`` is the size of the SymPy tree.
    """

    expression: str
    parameters: tuple[str, ...]
    latex: str
    node_count: int


def analyze_expression(text: str, variables: Sequence[str] = ("x",)) -> ExpressionReport:
    """Validate ``text`` and report its parameters, LaTeX form, normalised text and size.

    The free names are discovered from the text, so the caller does not have to know them
    first; names that are variables (``variables``) or constants (``pi``, ``E``) are not
    parameters. Raises ``ValueError`` exactly as
    :func:`~curvestudio.expressions.validate_expression` does.
    """
    body = split_equation(text)
    variable_set = set(variables)
    # Identifier-shaped tokens are a superset of the names a parseable expression can use
    # (they include pieces of attribute paths or quoted text, which the grammar then
    # rejects on its own). Validating against the superset keeps every syntax and
    # whitelist check while letting any genuinely free name through.
    candidates = set(_IDENTIFIER.findall(body)) - set(_CALLED_NAME.findall(body))
    candidates -= RESERVED_NAMES
    tree = parse_expression(body, frozenset(candidates | variable_set))
    free = [
        name
        for name in _names_in_order(tree)
        if name not in RESERVED_NAMES and name not in variable_set
    ]
    symbols = {name: sympy.Symbol(name, real=True) for name in sorted(candidates | variable_set)}
    expr = to_sympy(body, symbols)
    return ExpressionReport(
        expression=_normalised_text(expr, body, candidates | variable_set),
        parameters=tuple(free),
        latex=sympy.latex(expr),
        node_count=sum(1 for _ in sympy.preorder_traversal(expr)),
    )


def _normalised_text(expr: sympy.Expr, body: str, names: set[str]) -> str:
    """SymPy's spelling of ``expr``, unless that spelling would not re-validate.

    SymPy flattens a balanced product of many factors into one flat ``Mul`` and prints it
    as a single ``a*b*c*...`` chain — text whose left-leaning parse is deeper than the
    author's balanced original and can trip the grammar's depth cap. The report's
    ``expression`` is whatever the user wrote (the validated right-hand side) whenever the
    normalised spelling does not itself validate, so it always round-trips.
    """
    normalised: str = sympy.sstr(expr)
    try:
        validate_expression(normalised, frozenset(names))
    except ValueError:
        return body
    return normalised


def _names_in_order(tree: ast.expr) -> list[str]:
    """Bare names in the tree, in order of first appearance (left to right in the text)."""
    names: list[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name) and child.id not in names:
                names.append(child.id)
            visit(child)

    visit(tree)
    return names
