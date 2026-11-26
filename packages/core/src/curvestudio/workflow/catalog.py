"""The model catalogue: registered models plus the polynomial family, as plain data."""

import math

from pydantic import BaseModel, ConfigDict

from curvestudio.models import ModelSpec
from curvestudio.models import get_model as _registered_model
from curvestudio.models import list_models as _registered_models
from curvestudio.workflow import catalogue_cache
from curvestudio.workflow.errors import WorkflowError
from curvestudio.workflow.spec import MAX_POLYNOMIAL_DEGREE

POLYNOMIAL_ID = "polynomial"


class ParameterInfo(BaseModel):
    """A model parameter; ``None`` bounds mean unbounded on that side."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    unit: str
    lower: float | None
    upper: float | None


class ModelInfo(BaseModel):
    """What a UI needs to offer a model: identity, equation and parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    category: str
    equation: str
    latex: str
    parameters: tuple[ParameterInfo, ...]
    variables: tuple[str, ...]
    references: tuple[str, ...]
    assumptions: str
    is_family: bool


_POLYNOMIAL = ModelInfo(
    id=POLYNOMIAL_ID,
    name="Polynomial",
    category="polynomial",
    equation="y = b0 + b1*x + ... + bn*x**n",
    latex=r"y = b_0 + b_1 x + \dots + b_n x^n",
    parameters=(
        ParameterInfo(
            name="b0", description="Intercept (constant term)", unit="", lower=None, upper=None
        ),
        ParameterInfo(name="b1", description="Coefficient of x", unit="", lower=None, upper=None),
        ParameterInfo(
            name="bn",
            description="Coefficient of x**n, where n is the chosen degree",
            unit="",
            lower=None,
            upper=None,
        ),
    ),
    variables=("x",),
    references=(),
    assumptions=(
        f"Degree n is chosen by the caller (1 to {MAX_POLYNOMIAL_DEGREE}); fitted by linear "
        "least squares. High degrees are ill-conditioned and extrapolate badly."
    ),
    is_family=True,
)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _info(spec: ModelSpec) -> ModelInfo:
    return ModelInfo(
        id=spec.id,
        name=spec.name,
        category=spec.category,
        equation=f"y = {spec.expression}",
        latex=spec.latex(),
        parameters=tuple(
            ParameterInfo(
                name=p.name,
                description=p.description,
                unit=p.unit,
                lower=_finite_or_none(p.lower),
                upper=_finite_or_none(p.upper),
            )
            for p in spec.parameters
        ),
        variables=spec.variables,
        references=spec.references,
        assumptions=spec.assumptions,
        is_family=False,
    )


def list_models() -> list[ModelInfo]:
    """Every registered model and the polynomial family, sorted by category then name.

    Built on a miss, served from the per-user disk cache when the fingerprint matches:
    the catalogue depends on nothing but the contributing packages and their entry
    points (see :mod:`curvestudio.workflow.catalogue_cache`), so a repeat start-up skips
    the plugin registration and the per-model LaTeX work entirely. Any mismatch,
    corruption or failure rebuilds here and rewrites the cache.
    """
    return catalogue_cache.catalogue(_build_catalogue)


def _build_catalogue() -> list[ModelInfo]:
    """The ordinary scan: load the plugins, describe every model, add the family."""
    infos = [_info(spec) for spec in _registered_models()]
    infos.append(_POLYNOMIAL)
    return sorted(infos, key=lambda m: (m.category, m.name, m.id))


def get_model(model_id: str) -> ModelInfo:
    """One catalogue entry by id (``"polynomial"`` for the family).

    Raises:
        WorkflowError: ``UNKNOWN_MODEL``, with near-miss suggestions when there are any.
    """
    if model_id == POLYNOMIAL_ID:
        return _POLYNOMIAL
    return _info(registered_spec(model_id))


def registered_spec(model_id: str) -> ModelSpec:
    """The registry's :class:`ModelSpec`, or ``WorkflowError("UNKNOWN_MODEL")``."""
    try:
        return _registered_model(model_id)
    except KeyError as exc:
        raise WorkflowError("UNKNOWN_MODEL", str(exc.args[0])) from exc
