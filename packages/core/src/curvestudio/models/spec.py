"""Declarative model specification: one SymPy expression gives value, Jacobian and text.

Model authors write the right-hand side of ``y = f(x; parameters)`` as a string. The
string is untrusted input: :func:`curvestudio.expressions.to_sympy` checks it against a
strict grammar and builds the SymPy tree by walking the parsed syntax tree, so nothing
is ever passed to ``eval``.
"""

import keyword
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray

from curvestudio.expressions import RESERVED_NAMES, to_sympy

FloatArray = NDArray[np.float64]

_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TOKEN = re.compile(rf"{_NUMBER}|[A-Za-z_][A-Za-z0-9_]*")
"""Numeric literals come first so the ``e`` of ``1e-3`` is never read as a name."""


@dataclass(frozen=True)
class Parameter:
    """A fitted parameter: its name in the expression, documentation and bounds."""

    name: str
    description: str = ""
    unit: str = ""
    lower: float = -math.inf
    upper: float = math.inf


@dataclass(frozen=True)
class _Compiled:
    """SymPy expression plus its NumPy-callable value and per-parameter derivatives."""

    expr: Any
    value: Callable[..., Any]
    derivatives: tuple[Callable[..., Any], ...]


def _compile(spec: ModelSpec) -> _Compiled:
    names = frozenset(spec.variables) | frozenset(p.name for p in spec.parameters)
    symbols = {name: sympy.Symbol(name, real=True) for name in sorted(names)}
    expr = to_sympy(spec.expression, symbols)
    args = [symbols[name] for name in (*spec.variables, *(p.name for p in spec.parameters))]
    # lambdify compiles source that SymPy prints from `expr`; that tree was built from the
    # whitelist and its symbols have grammar-checked identifiers, so no user text is executed.
    return _Compiled(
        expr=expr,
        value=sympy.lambdify(args, expr, modules="numpy"),
        derivatives=tuple(
            sympy.lambdify(args, sympy.diff(expr, symbols[p.name]), modules="numpy")
            for p in spec.parameters
        ),
    )


def _format_number(value: float, digits: int) -> str:
    text = f"{value:.{digits}g}"
    return f"({text})" if value < 0 else text


@dataclass(frozen=True)
class ModelSpec:
    """A named, documented model ``y = expression(variables; parameters)``.

    The expression is parsed once at construction (invalid or unsafe expressions raise
    ``ValueError`` there). ``initial_guess`` is stored but never called here.
    For a single variable ``x`` is a 1-D array; with several variables it is a 2-D
    array whose columns follow ``variables``.
    """

    id: str
    name: str
    category: str
    expression: str
    parameters: tuple[Parameter, ...]
    initial_guess: Callable[[FloatArray, FloatArray], dict[str, float]]
    variables: tuple[str, ...] = ("x",)
    domain_check: Callable[[FloatArray, FloatArray], list[str]] | None = None
    references: tuple[str, ...] = ()
    assumptions: str = ""
    _compiled: _Compiled = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("model id must not be empty")
        if not self.parameters:
            raise ValueError(f"model {self.id!r} needs at least one parameter")
        if not self.variables:
            raise ValueError(f"model {self.id!r} needs at least one variable")
        names = [*self.variables, *(p.name for p in self.parameters)]
        for name in names:
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
                raise ValueError(f"{name!r} is not a usable variable/parameter name")
            if name in RESERVED_NAMES:
                raise ValueError(f"{name!r} is reserved for a function or constant")
        if len(set(names)) != len(names):
            raise ValueError(f"variable and parameter names must be unique: {names}")
        for p in self.parameters:
            if not p.lower <= p.upper:
                raise ValueError(f"parameter {p.name!r} has lower bound above upper bound")
        object.__setattr__(self, "_compiled", _compile(self))

    # -- evaluation -------------------------------------------------------------------

    def _arguments(self, x: FloatArray, params: Mapping[str, float]) -> tuple[list[Any], int]:
        expected = {p.name for p in self.parameters}
        missing = sorted(expected - params.keys())
        unknown = sorted(params.keys() - expected)
        if missing or unknown:
            raise ValueError(
                f"parameters mismatch for {self.id!r}: missing {missing}, unexpected {unknown}"
            )
        data = np.asarray(x, dtype=np.float64)
        if len(self.variables) == 1:
            columns = [data]
        else:
            if data.ndim != 2 or data.shape[1] != len(self.variables):
                raise ValueError(
                    f"model {self.id!r} needs a 2-D array with one column per variable "
                    f"{self.variables}; got shape {data.shape}"
                )
            columns = [data[:, i] for i in range(len(self.variables))]
        values = [float(params[p.name]) for p in self.parameters]
        return [*columns, *values], data.shape[0]

    def evaluate(self, x: FloatArray, **params: float) -> FloatArray:
        """Model values at ``x``; non-finite entries mark points outside the domain."""
        args, n = self._arguments(x, params)
        with np.errstate(all="ignore"):
            out = self._compiled.value(*args)
        return np.array(np.broadcast_to(np.asarray(out, dtype=np.float64), (n,)))

    def jacobian(self, x: FloatArray, **params: float) -> FloatArray:
        """Analytic ``d value / d parameter``, shape ``(n, k)``, columns in ``parameters`` order."""
        args, n = self._arguments(x, params)
        jac = np.empty((n, len(self.parameters)), dtype=np.float64)
        with np.errstate(all="ignore"):
            for j, derivative in enumerate(self._compiled.derivatives):
                jac[:, j] = np.broadcast_to(np.asarray(derivative(*args), dtype=np.float64), (n,))
        return jac

    # -- presentation -----------------------------------------------------------------

    def latex(self) -> str:
        """LaTeX for the model equation, e.g. ``y = a e^{b x}``."""
        return f"y = {sympy.latex(self._compiled.expr)}"

    def equation_text(self, params: Mapping[str, float], digits: int = 6) -> str:
        """The original expression with fitted values substituted, e.g. ``y = 2*exp(0.5*x)``.

        Substitution is textual (whole-word, single pass) so the author's layout survives
        and a substituted number such as ``1e-07`` is never re-scanned for names. Numbers
        carry up to ``digits`` significant digits; negative values are parenthesised.
        """
        if digits < 1:
            raise ValueError("digits must be at least 1")
        missing = [p.name for p in self.parameters if p.name not in params]
        if missing:
            raise ValueError(f"equation_text is missing values for parameters {missing}")

        known = {p.name for p in self.parameters}

        def substitute(match: re.Match[str]) -> str:
            name = match.group(0)
            if name in known:
                return _format_number(float(params[name]), digits)
            return name  # numeric literal, variable, function or constant: keep verbatim

        text = _TOKEN.sub(substitute, self.expression.strip())
        return f"y = {text}"

    def applicability(self, x: FloatArray, y: FloatArray) -> list[str]:
        """Reasons this model does not suit the data; an empty list means it applies."""
        return [] if self.domain_check is None else list(self.domain_check(x, y))
