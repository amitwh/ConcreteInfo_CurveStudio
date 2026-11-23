"""What the methods text says was fitted, and its estimates rounded by one rule.

:func:`_fitted_model` names the model from its result. :func:`_show` turns the parameters
into the numbers the text prints: a value and its standard error share decimal places,
chosen so that the error shows two significant digits (:func:`_places`). Nothing here is
computed from the data: every number is one the fit result carries.
"""

import math
from dataclasses import dataclass

from curvestudio.fitting import FitResult, ParameterEstimate
from curvestudio.models import ModelSpec, get_model
from curvestudio.tools._thresholds import POLYNOMIAL_ID

ERROR_DIGITS = 2
"""Significant digits a standard error is shown with. Its value shares the decimal places."""
VALUE_DIGITS = 6
"""Significant digits shown for a value when no parameter has a standard error to set the
decimal places: the fitting layer's own precision for a fitted equation."""


@dataclass(frozen=True)
class _Model:
    """What was fitted, as the result and the registry tell it."""

    description: str
    """``a polynomial of degree 2``, a registered model's name, or ``a custom equation (...)``."""
    nonlinear: bool
    """Fitted by an iterative solver, whose name the paragraph then gives."""
    registered: ModelSpec | None
    references: tuple[str, ...]
    assumptions: str
    powers: tuple[int, ...]
    """A polynomial's powers, one per coefficient; empty for any other model."""


@dataclass(frozen=True)
class _Shown:
    """One parameter's numbers as the text shows them; ``None`` where none could be determined."""

    name: str
    value: str
    std_err: str | None
    lower: str | None
    upper: str | None


def _fitted_model(result: FitResult) -> _Model:
    """Name the model from its result: the polynomial id, the registry, or a custom equation."""
    polynomial = POLYNOMIAL_ID.fullmatch(result.model_id)
    if polynomial is not None:
        degree = int(polynomial.group(1))
        through = " through the origin" if polynomial.group(2) else ""
        # A polynomial's coefficients are named b<power> by the fitting layer.
        powers = tuple(int(parameter.name[1:]) for parameter in result.parameters)
        return _Model(f"a polynomial of degree {degree}{through}", False, None, (), "", powers)
    try:
        model = get_model(result.model_id)
    except KeyError:
        equation = result.equation
        return _Model(f"a custom equation ({equation})", True, None, (), "", ())
    return _Model(model.name, True, model, model.references, model.assumptions, ())


def _unsigned_zero(text: str) -> str:
    """A number that rounds to zero is not shown with a sign (``-0.00`` -> ``0.00``)."""
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def _fixed(value: float, places: int) -> str:
    """``value`` at ``places`` decimal places; below zero places it is a whole number."""
    if places < 0:
        return _unsigned_zero(f"{round(value, places):.0f}")
    return _unsigned_zero(f"{value:.{places}f}")


def _significant(text: str) -> int:
    """Significant digits ``text`` shows (``0.00123`` -> 3, ``1.2e+03`` -> 2)."""
    return len(text.partition("e")[0].lstrip("-+").replace(".", "").lstrip("0"))


def _places(value: float, std_err: float) -> int:
    """Decimal places for ``value`` and its ``std_err``, so the error shows two significant
    digits (``0.0123`` -> 3 places).

    A zero standard error (an exact fit) has no significant digits to show, so the value's
    ``VALUE_DIGITS`` significant digits decide instead.
    """
    if std_err <= 0.0:
        leading = math.floor(math.log10(abs(value))) if value != 0.0 else 0
        return max(VALUE_DIGITS - 1 - leading, 0)
    places = 1 - math.floor(math.log10(std_err))
    while places > 0 and _significant(_fixed(std_err, places)) > ERROR_DIGITS:
        places -= 1  # 0.0999 at three places is 0.100: one place fewer shows 0.10
    return places


def _show(parameters: tuple[ParameterEstimate, ...]) -> tuple[_Shown, ...]:
    """Every parameter's numbers, rounded by :func:`_places`.

    A parameter without a standard error is shown at the widest precision the others have, so
    one column keeps one rounding; when none has one, by significant digits.
    """
    widest: int | None = None
    for parameter in parameters:
        if parameter.std_err is not None:
            places = _places(parameter.value, parameter.std_err)
            widest = places if widest is None else max(widest, places)
    rows: list[_Shown] = []
    for parameter in parameters:
        if parameter.std_err is None:
            value = (
                _unsigned_zero(f"{parameter.value:.{VALUE_DIGITS}g}")
                if widest is None
                else _fixed(parameter.value, widest)
            )
            rows.append(_Shown(parameter.name, value, None, None, None))
            continue
        places = _places(parameter.value, parameter.std_err)
        rows.append(
            _Shown(
                parameter.name,
                _fixed(parameter.value, places),
                _fixed(parameter.std_err, places),
                None if parameter.ci_low is None else _fixed(parameter.ci_low, places),
                None if parameter.ci_high is None else _fixed(parameter.ci_high, places),
            )
        )
    return tuple(rows)
