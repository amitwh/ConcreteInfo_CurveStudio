"""Safe model-expression grammar: validate text, convert to SymPy, never ``eval``.

This is a pure layer: it imports nothing else from ``curvestudio``. The model layer
(:mod:`curvestudio.models`) builds :class:`~curvestudio.models.spec.ModelSpec` objects on top of it.
"""

from curvestudio.expressions.analyze import ExpressionReport, analyze_expression
from curvestudio.expressions.convert import to_sympy
from curvestudio.expressions.grammar import (
    MAX_DEPTH,
    MAX_EXPRESSION_LENGTH,
    MAX_NODES,
    RESERVED_NAMES,
    split_equation,
    validate_expression,
)

__all__ = [
    "MAX_DEPTH",
    "MAX_EXPRESSION_LENGTH",
    "MAX_NODES",
    "RESERVED_NAMES",
    "ExpressionReport",
    "analyze_expression",
    "split_equation",
    "to_sympy",
    "validate_expression",
]
