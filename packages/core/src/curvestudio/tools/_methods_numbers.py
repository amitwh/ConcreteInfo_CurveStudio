"""What the methods text says was fitted, and its estimates rounded by one rule.

:func:`_fitted_model` names the model from its result. :func:`_show` turns the parameters
into the numbers the text prints, each by :func:`_written`: a value and its standard error
share decimal places, chosen so that the error shows two significant digits
(:func:`_places`), unless that would take rounding for precision. Nothing here is computed
from the data: every number is one the fit result carries.
"""

import math
from dataclasses import dataclass

from curvestudio.fitting import FitResult, ParameterEstimate
from curvestudio.models import ModelSpec, get_model
from curvestudio.tools._thresholds import POLYNOMIAL_ID

ERROR_DIGITS = 2
"""Significant digits a standard error is shown with. Its value shares the decimal places."""
MAX_DECIMALS = 6
"""The most decimal places a value and its standard error share. An error that would need
more is at rounding level, and its number is written another way (:func:`_written`)."""
WIDE_DIGITS = 4
"""Significant digits a value takes when its standard error is written in scientific
notation."""
VALUE_DIGITS = 6
"""Significant digits shown for a value that has no standard error when no other parameter
has one either: the fitting layer's own precision for a fitted equation."""


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


def _general(value: float, digits: int) -> str:
    """``value`` with ``digits`` significant digits (``0.0000123`` -> ``1.23e-05``)."""
    return _unsigned_zero(f"{value:.{digits}g}")


def _significant(text: str) -> int:
    """Significant digits ``text`` shows (``0.00123`` -> 3, ``1.2e+03`` -> 2)."""
    return len(text.partition("e")[0].lstrip("-+").replace(".", "").lstrip("0"))


def _places(std_err: float) -> int:
    """Decimal places that show ``std_err``, which is above zero, with ``ERROR_DIGITS``
    significant digits (``0.0123`` -> 3)."""
    places = 1 - math.floor(math.log10(std_err))
    while places > 0 and _significant(_fixed(std_err, places)) > ERROR_DIGITS:
        places -= 1  # 0.0999 at three places is 0.100: one place fewer shows 0.10
    return places


def _written(
    name: str, value: float, std_err: float, lower: float | None, upper: float | None
) -> tuple[_Shown, int | None]:
    """One parameter's numbers, and the decimal places they share (``None`` when they share
    none).

    The rule. A value and its standard error share decimal places, chosen so that the error
    shows ``ERROR_DIGITS`` significant digits. When that would need more than
    ``MAX_DECIMALS`` places (an error at rounding level), or the error is zero (an exact
    fit), the value is written with ``WIDE_DIGITS`` significant digits and the error in
    scientific notation with ``ERROR_DIGITS`` of them (``1.022`` and ``3.2e-17``); a zero
    error reads ``0``. The interval's bounds are written as the value is.
    """
    if std_err > 0.0:
        places = _places(std_err)
        if places <= MAX_DECIMALS:
            return (
                _Shown(
                    name,
                    _fixed(value, places),
                    _fixed(std_err, places),
                    None if lower is None else _fixed(lower, places),
                    None if upper is None else _fixed(upper, places),
                ),
                places,
            )
    error = "0" if std_err == 0.0 else f"{std_err:.{ERROR_DIGITS - 1}e}"
    return (
        _Shown(
            name,
            _general(value, WIDE_DIGITS),
            error,
            None if lower is None else _general(lower, WIDE_DIGITS),
            None if upper is None else _general(upper, WIDE_DIGITS),
        ),
        None,
    )


def _show(parameters: tuple[ParameterEstimate, ...]) -> tuple[_Shown, ...]:
    """Every parameter's numbers by :func:`_written`.

    A parameter without a standard error takes the widest decimal places the others share,
    so one column keeps one rounding; when none of them shares any, it is written by
    significant digits (``VALUE_DIGITS``).
    """
    rows: list[_Shown | None] = [None for _ in parameters]
    places: list[int | None] = [None for _ in parameters]
    for index, parameter in enumerate(parameters):
        if parameter.std_err is not None:
            rows[index], places[index] = _written(
                parameter.name,
                parameter.value,
                parameter.std_err,
                parameter.ci_low,
                parameter.ci_high,
            )
    widest = max((each for each in places if each is not None), default=None)
    for index, parameter in enumerate(parameters):
        if rows[index] is None:
            value = (
                _general(parameter.value, VALUE_DIGITS)
                if widest is None
                else _fixed(parameter.value, widest)
            )
            rows[index] = _Shown(parameter.name, value, None, None, None)
    return tuple(row for row in rows if row is not None)
