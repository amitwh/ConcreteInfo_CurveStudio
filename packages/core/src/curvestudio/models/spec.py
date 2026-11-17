"""Declarative model specification: one SymPy expression gives value, Jacobian and text.

Model authors write the right-hand side of ``y = f(x; parameters)`` as a string. The
string is untrusted input: :func:`curvestudio.expressions.validate_expression` checks it
against a strict grammar when the spec is built, and :func:`curvestudio.expressions.to_sympy`
builds the SymPy tree by walking the parsed syntax tree, so nothing is ever passed to ``eval``.
The (slow) SymPy compilation is deferred to first use; see :meth:`ModelSpec.compile`.
"""

import keyword
import math
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray

from curvestudio.expressions import RESERVED_NAMES, to_sympy, validate_expression
from curvestudio.expressions.special import NUMPY_CALLABLES

FloatArray = NDArray[np.float64]

_LAMBDIFY_MODULES: list[Any] = [NUMPY_CALLABLES, "scipy", "numpy"]
"""Namespaces for compiled expressions: our special functions, then SciPy, then NumPy."""

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


@dataclass(frozen=True)
class ModelExample:
    """A worked example: true parameter values and the x range to sample them on.

    The example drives noiseless parameter-recovery tests and gives documentation and the
    UI a realistic curve to show. ``n`` evenly spaced x values are used.

    The example is immutable and hashable (``params`` becomes a read-only mapping), so a
    ``ModelSpec`` holding one stays hashable.
    """

    params: Mapping[str, float]
    x_range: tuple[float, float]
    n: int = 60

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.params.items())), self.x_range, self.n))

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(sorted(self.params.items()))))
        low, high = self.x_range
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f"example x_range must be finite with low < high, got {self.x_range}")
        if self.n < 2:
            raise ValueError(f"example needs at least 2 points, got n={self.n}")
        if not all(math.isfinite(v) for v in self.params.values()):
            raise ValueError("example parameter values must be finite")


def _compile(spec: ModelSpec) -> _Compiled:
    names = frozenset(spec.variables) | frozenset(p.name for p in spec.parameters)
    symbols = {name: sympy.Symbol(name, real=True) for name in sorted(names)}
    expr = to_sympy(spec.expression, symbols)
    # lambdify compiles source that SymPy prints from the tree; that tree was built from the
    # whitelist, so no user text is executed. Its namespace also contains one entry per
    # argument name, which would shadow printed helper names (reduce, select, numpy, ...) if a
    # user called a parameter that. The compiled copy therefore uses private argument names.
    ordered = [*spec.variables, *(p.name for p in spec.parameters)]
    # Numbered in sorted-name order so SymPy orders terms exactly as it would for the user's
    # names (floating-point summation order, and so fitted digits, stay unchanged).
    private = {
        name: sympy.Symbol(f"_arg{i:04d}", real=True) for i, name in enumerate(sorted(ordered))
    }
    compiled = expr.xreplace({symbols[name]: private[name] for name in ordered})
    args = [private[name] for name in ordered]
    return _Compiled(
        expr=expr,
        value=sympy.lambdify(args, compiled, modules=_LAMBDIFY_MODULES),
        derivatives=tuple(
            sympy.lambdify(args, sympy.diff(compiled, private[p.name]), modules=_LAMBDIFY_MODULES)
            for p in spec.parameters
        ),
    )


def _format_number(value: float, digits: int) -> str:
    text = f"{value:.{digits}g}"
    return f"({text})" if value < 0 else text


@dataclass(frozen=True)
class ModelSpec:
    """A named, documented model ``y = expression(variables; parameters)``.

    The expression is checked against the grammar at construction (invalid or unsafe
    expressions raise ``ValueError`` there). Building the SymPy tree, the Jacobian and the
    NumPy callables is the expensive part, so it happens on first use (``evaluate``,
    ``jacobian``, ``latex``) or when :meth:`compile` is called; a catalogue of hundreds of
    specs therefore costs nothing at import. ``initial_guess`` is stored but never called
    here. For a single variable ``x`` is a 1-D array; with several variables it is a 2-D
    array whose columns follow ``variables``.

    ``aliases`` are other names of the same equation (for search), ``description`` is a
    one-sentence summary and ``example`` holds true parameters for a noiseless demo curve.
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
    aliases: tuple[str, ...] = ()
    description: str = ""
    example: ModelExample | None = None
    _compiled: _Compiled | None = field(init=False, default=None, repr=False, compare=False)
    _compile_lock: Any = field(
        init=False, default_factory=threading.Lock, repr=False, compare=False
    )

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
        validate_expression(self.expression, frozenset(names))
        self._validate_aliases()
        if self.example is not None:
            self._validate_example(self.example)

    def _validate_aliases(self) -> None:
        for alias in self.aliases:
            if not alias.strip():
                raise ValueError(f"model {self.id!r} has an empty alias")
        if len(set(self.aliases)) != len(self.aliases) or self.id in self.aliases:
            raise ValueError(f"model {self.id!r} has duplicate aliases: {list(self.aliases)}")

    def _validate_example(self, example: ModelExample) -> None:
        expected = {p.name for p in self.parameters}
        if set(example.params) != expected:
            raise ValueError(
                f"example of {self.id!r} must give exactly the parameters {sorted(expected)}; "
                f"got {sorted(example.params)}"
            )
        if len(self.variables) != 1:
            raise ValueError(f"example of {self.id!r}: only one-variable models can have one")

    # -- compilation ------------------------------------------------------------------

    def compile(self) -> None:
        """Build the SymPy expression, Jacobian and NumPy callables now (idempotent).

        Normally this happens on first use. Call it to surface an expression that passes the
        grammar but cannot be turned into a usable real-valued SymPy expression early.

        Raises:
            ValueError: the expression cannot be compiled (for example it is undefined for
                every real input, or SymPy cannot differentiate it).
        """
        self._ready()

    def _ready(self) -> _Compiled:
        compiled = self._compiled
        if compiled is not None:
            return compiled
        with self._compile_lock:  # concurrent first uses compile exactly once
            if self._compiled is not None:
                return self._compiled
            try:
                compiled = _compile(self)
            except ValueError:
                raise
            except Exception as exc:  # SymPy raises many types; report them uniformly
                raise ValueError(
                    f"model {self.id!r}: expression {self.expression!r} cannot be compiled: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            object.__setattr__(self, "_compiled", compiled)
            return compiled

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
            out = self._ready().value(*args)
        return np.array(np.broadcast_to(np.asarray(out, dtype=np.float64), (n,)))

    def jacobian(self, x: FloatArray, **params: float) -> FloatArray:
        """Analytic ``d value / d parameter``, shape ``(n, k)``, columns in ``parameters`` order."""
        args, n = self._arguments(x, params)
        jac = np.empty((n, len(self.parameters)), dtype=np.float64)
        with np.errstate(all="ignore"):
            for j, derivative in enumerate(self._ready().derivatives):
                jac[:, j] = np.broadcast_to(np.asarray(derivative(*args), dtype=np.float64), (n,))
        return jac

    def example_data(self) -> tuple[FloatArray, FloatArray]:
        """Noiseless ``(x, y)`` of the model at its example parameters.

        Raises:
            ValueError: the model has no ``example``.
        """
        if self.example is None:
            raise ValueError(f"model {self.id!r} has no example")
        x = np.linspace(self.example.x_range[0], self.example.x_range[1], self.example.n)
        return x, self.evaluate(x, **self.example.params)

    # -- presentation -----------------------------------------------------------------

    def latex(self) -> str:
        """LaTeX for the model equation, e.g. ``y = a e^{b x}``."""
        return f"y = {sympy.latex(self._ready().expr)}"

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
