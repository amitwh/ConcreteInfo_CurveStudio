"""``FitSpec``: a complete, JSON-serialisable description of how to fit one model."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from curvestudio.fitting.metrics import FiniteFloat

MAX_POLYNOMIAL_DEGREE = 10
MAX_CUSTOM_PARAMETERS = 20
MAX_SEED = 2**53 - 1
"""Largest integer a JSON number (an IEEE double) carries exactly."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PolynomialRef(_Frozen):
    """``y = b0 + b1*x + ... + b<degree>*x**degree``, fitted by linear least squares."""

    kind: Literal["polynomial"] = "polynomial"
    degree: int = Field(ge=1, le=MAX_POLYNOMIAL_DEGREE)


class RegisteredRef(_Frozen):
    """A model from the registry, by id (see :func:`curvestudio.workflow.list_models`)."""

    kind: Literal["registered"] = "registered"
    id: str = Field(min_length=1, max_length=100)


class CustomRef(_Frozen):
    """A user equation: ``equation`` is ``y = <expression>`` or just the right-hand side.

    ``parameters`` are the names to fit; ``variable`` is the independent variable. The text
    is checked by the safe expression grammar and is never evaluated as Python.
    """

    kind: Literal["custom"] = "custom"
    equation: str = Field(min_length=1, max_length=2000)
    parameters: list[str] = Field(max_length=MAX_CUSTOM_PARAMETERS)
    variable: str = Field(default="x", min_length=1, max_length=64)


ModelRef = Annotated[PolynomialRef | RegisteredRef | CustomRef, Field(discriminator="kind")]


class FitSpec(_Frozen):
    """What to fit and how.

    ``bounds`` maps a parameter to ``(lower, upper)``; ``None`` means unbounded on that side
    (JSON has no infinity). ``initial`` overrides the starting value per parameter; a custom
    equation starts every parameter at 1.0 unless told otherwise. ``use_weights`` weights
    the fit by the data's weights column (an error if the data has none); with ``False``
    any weights in the data are ignored and a warning says so. ``seed`` (default 0) is
    recorded in the result. ``time_limit`` is a wall-clock budget in seconds for
    registered and custom models; a fit that exceeds it is reported ``failed``.

    Polynomials are fitted in closed form, so ``method``, ``loss``, ``bounds`` and ``initial``
    must be left at their defaults for them.
    """

    model: ModelRef
    method: str = "auto"
    loss: str = "linear"
    bounds: dict[str, tuple[FiniteFloat | None, FiniteFloat | None]] = Field(default_factory=dict)
    initial: dict[str, FiniteFloat] = Field(default_factory=dict)
    use_weights: bool = False
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    time_limit: float | None = Field(default=None, gt=0, allow_inf_nan=False)
