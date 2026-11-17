"""Build a fittable :class:`~curvestudio.models.spec.ModelSpec` from a user equation.

The definition (:class:`CustomModelDefinition`) is plain, JSON-round-trippable data;
:func:`build_custom_model` turns it into a compiled model. Everything the user wrote is
checked by the expression grammar — there is no ``eval`` — and the builder refuses
names, bounds and derivative work that cannot produce a usable model, with messages
that name the problem and the fix.
"""

import difflib
import math
import re
import unicodedata
from collections.abc import Sequence

import sympy
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from curvestudio.expressions import RESERVED_NAMES, analyze_expression, to_sympy
from curvestudio.expressions.grammar import CONSTANTS, FUNCTIONS
from curvestudio.models.spec import FloatArray, ModelSpec, Parameter

MAX_DERIVATIVE_NODES = 20_000
"""Most SymPy tree nodes the first-order parameter derivatives may together use.

Calibrated against a 60-term product of sines with 8 parameters (~21 600 derivative
nodes): such an equation is already refused by the grammar's 300-node input cap, and
the largest sin-product the grammar admits measures about 7 800, so genuine equations
stay far below the limit while pathological ones fail at build time instead of
stalling compilation.
"""

_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
"""Model ids are lowercase snake_case: the Auto-fit layer addresses custom models by them."""

_SYMPY_SPECIAL_NAMES = frozenset({"oo", "zoo", "nan", "inf", "I", "S"})
"""SymPy's spellings of infinity, the undefined and the imaginary unit.

A custom-equation parameter called one of these could never evaluate to a fitted real
number, so the builder refuses them. They are reserved here rather than in the grammar's
:data:`~curvestudio.expressions.grammar.RESERVED_NAMES` because that set also gates
direct :class:`~curvestudio.models.spec.ModelSpec` construction, which the model
catalogues use with single-letter engineering symbols of their own (sorptivity ``S``).
"""


def _has_control_characters(text: str) -> bool:
    return any(unicodedata.category(char).startswith("C") for char in text)


class ParameterDefinition(BaseModel):
    """A fitted parameter as the user declares it: start value, bounds, unit, help text.

    Bounds are floats, unbounded by default; ``None`` is accepted on input and produced
    on serialisation for an infinite bound, so a definition is always valid strict JSON.
    Frozen, so a :class:`CustomModelDefinition` that holds parameters stays hashable (the
    Auto-fit planner caches the compiled model by definition).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    initial: float | None = None
    lower: float = -math.inf
    upper: float = math.inf
    unit: str = ""
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_is_short(cls, value: str) -> str:
        if len(value) > 32:
            raise ValueError("name must be at most 32 characters")
        return value

    @field_validator("unit")
    @classmethod
    def _unit_is_short_and_printable(cls, value: str) -> str:
        if len(value) > 32:
            raise ValueError("unit must be at most 32 characters")
        if _has_control_characters(value):
            raise ValueError("unit must not contain control characters")
        return value

    @field_validator("description")
    @classmethod
    def _description_is_short(cls, value: str) -> str:
        if len(value) > 500:
            raise ValueError("description must be at most 500 characters")
        return value

    @field_validator("initial")
    @classmethod
    def _initial_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("initial must be a finite number or null")
        return value

    @field_validator("lower", mode="before")
    @classmethod
    def _lower_from_none(cls, value: object) -> object:
        return -math.inf if value is None else value

    @field_validator("upper", mode="before")
    @classmethod
    def _upper_from_none(cls, value: object) -> object:
        return math.inf if value is None else value

    @field_validator("lower", "upper")
    @classmethod
    def _bound_is_a_real_number(cls, value: float) -> float:
        if math.isnan(value):
            raise ValueError("bounds must be numbers, not NaN")
        return value

    @field_serializer("lower")
    def _lower_to_json(self, value: float) -> float | None:
        return None if value == -math.inf else value

    @field_serializer("upper")
    def _upper_to_json(self, value: float) -> float | None:
        return None if value == math.inf else value


class CustomModelDefinition(BaseModel):
    """A user equation as storable data; ``parameters=None`` infers them from the text."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    expression: str
    variables: tuple[str, ...] = ("x",)
    parameters: tuple[ParameterDefinition, ...] | None = None

    @field_validator("name")
    @classmethod
    def _name_is_short(cls, value: str) -> str:
        if len(value) > 120:
            raise ValueError("name must be at most 120 characters")
        return value

    @field_validator("id")
    @classmethod
    def _id_is_lowercase_snake_case(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError(
                "id must be lowercase snake_case matching ^[a-z][a-z0-9_]{0,63}$: a lowercase "
                "letter, then letters, digits or underscores, at most 64 characters"
            )
        return value


def _close_matches(name: str, candidates: Sequence[str]) -> list[str]:
    return difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)


def _check_names(
    declared: Sequence[ParameterDefinition],
    variables: Sequence[str],
    expression_parameters: Sequence[str],
) -> None:
    """Reject reserved, duplicated, colliding and unknown parameter names, with hints."""
    variable_set = set(variables)
    seen: set[str] = set()
    for definition in declared:
        name = definition.name
        if name in CONSTANTS:
            raise ValueError(f"{name!r} is a constant, not a parameter")
        if name in FUNCTIONS:
            raise ValueError(f"{name!r} is a function, not a parameter")
        if name in RESERVED_NAMES or name in _SYMPY_SPECIAL_NAMES:
            # lambdify-printed helpers, SymPy number types and the oo/nan/I/... spellings
            raise ValueError(f"parameter {name!r} is reserved; pick another name")
        if name in variable_set:
            raise ValueError(f"{name!r} is a variable of this model, not a parameter")
        if name in seen:
            raise ValueError(f"parameter {name!r} is declared twice")
        seen.add(name)
    known = [*expression_parameters, *variables]
    for definition in declared:
        if definition.name not in expression_parameters:
            close = _close_matches(definition.name, known)
            hint = (
                f"did you mean {close[0]!r}?"
                if close
                else f"parameters appearing in the expression: {list(expression_parameters)}"
            )
            raise ValueError(f"unknown parameter {definition.name!r}; {hint}")
    missing = [name for name in expression_parameters if name not in seen]
    if missing:
        raise ValueError(
            f"the expression uses parameter(s) {missing} that the definition does not "
            "declare; declare them (or fix their spelling)"
        )


def _start_value(definition: ParameterDefinition) -> float:
    """The parameter's ``initial``, else ``1.0`` clipped into its bounds."""
    if not definition.lower < definition.upper:
        raise ValueError(
            f"parameter {definition.name!r} needs lower < upper bounds, "
            f"got ({definition.lower}, {definition.upper})"
        )
    if definition.initial is not None:
        if not definition.lower <= definition.initial <= definition.upper:
            raise ValueError(
                f"parameter {definition.name!r} has initial {definition.initial} outside "
                f"its bounds ({definition.lower}, {definition.upper})"
            )
        return definition.initial
    return min(max(1.0, definition.lower), definition.upper)


def _compile_with_derivative_cap(
    spec: ModelSpec, expression: str, parameter_names: Sequence[str]
) -> None:
    """Differentiate once, refuse enormous trees, and hand the work to the spec.

    The builder must measure the derivatives (its complexity cap) and the spec must own
    them (its Jacobian), and differentiating the largest admissible expressions is by far
    the slowest step of a build. The trees are therefore built here exactly once and
    adopted as the spec's compilation (:meth:`ModelSpec._adopt_compilation`), so the lazy
    path reuses them instead of recomputing.
    """
    names = sorted([*spec.variables, *parameter_names])
    symbols = {name: sympy.Symbol(name, real=True) for name in names}
    expr = to_sympy(expression, symbols)
    derivatives = []
    total = 0
    for name in parameter_names:
        derivative = sympy.diff(expr, symbols[name])
        total += sum(1 for _ in sympy.preorder_traversal(derivative))
        if total > MAX_DERIVATIVE_NODES:
            raise ValueError(
                f"model too complex: the parameter derivatives of {expression!r} need more "
                f"than {MAX_DERIVATIVE_NODES} SymPy operations; simplify the equation"
            )
        derivatives.append(derivative)
    spec._adopt_compilation(symbols, expr, tuple(derivatives))


def build_custom_model(definition: CustomModelDefinition) -> ModelSpec:
    """Compile a :class:`CustomModelDefinition` into a fittable model (category ``custom``).

    Parameters left as ``None`` are inferred from the expression in order of first
    appearance. The start vector is fixed: each parameter's ``initial`` if given, else
    ``1.0`` clipped into its bounds — a deliberately rough universal start. When that
    start converges poorly the remedy is multistart (``FitConfig(n_starts=...)``), not a
    smarter guess here. The returned spec is already compiled, so an equation that
    cannot be differentiated or lambdified fails here rather than mid-fit.

    Raises:
        ValueError: the expression, names, bounds or derivative size are unusable (see
            :func:`~curvestudio.expressions.validate_expression` and the checks above).
    """
    report = analyze_expression(definition.expression, definition.variables)
    variables = tuple(definition.variables)
    if definition.parameters is None:
        declared = tuple(ParameterDefinition(name=name) for name in report.parameters)
    else:
        declared = definition.parameters
    _check_names(declared, variables, report.parameters)
    if not declared:
        raise ValueError(
            f"no free parameter in the expression {definition.expression!r}: every name is a "
            "variable, constant or function, so there is nothing to fit — add a free name "
            "(for example the 'a' in 'a*x')"
        )
    parameters = tuple(
        Parameter(
            name=d.name,
            description=d.description,
            unit=d.unit,
            lower=d.lower,
            upper=d.upper,
        )
        for d in declared
    )
    starts = {d.name: _start_value(d) for d in declared}

    def initial_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        return dict(starts)

    spec = ModelSpec(
        id=definition.id,
        name=definition.name,
        category="custom",
        expression=report.expression,
        parameters=parameters,
        initial_guess=initial_guess,
        variables=variables,
    )
    _compile_with_derivative_cap(spec, report.expression, [d.name for d in declared])
    spec.compile()  # already compiled by adoption; idempotent, and a guard against regressions
    return spec
