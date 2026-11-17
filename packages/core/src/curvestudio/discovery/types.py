"""Public value types of model discovery: configuration, candidate references, results."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.fitting.result import ParameterEstimate
from curvestudio.models.custom import CustomModelDefinition


class AutoFitError(ValueError):
    """Auto-fit cannot run on the given data or settings.

    ``code`` is a stable machine-readable identifier (``TOO_FEW_POINTS``, ...), as for
    ``StatisticsError``, so callers and UIs can react without parsing the message.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)  # args=(code, message): survives pickling
        self.code = code

    def __str__(self) -> str:
        return str(self.args[1])


class RankBy(StrEnum):
    """Criterion the final list is ordered by."""

    BEST_OVERALL = "best_overall"
    R2 = "r2"
    ADJ_R2 = "adj_r2"
    RMSE = "rmse"
    AICC = "aicc"
    BIC = "bic"
    CV_RMSE = "cv_rmse"


class AutoFitConfig(BaseModel):
    """Settings of one Auto-fit run.

    Attributes:
        rank_by: ordering criterion of the result.
        families: keep only models whose category equals one of these or has it as its first
            dotted segment (``"engineering"`` matches ``"engineering.strength"``);
            ``None`` keeps every category.
        include_polynomials: also try polynomials (``polyN`` and ``polyN_origin``).
        include_generated: also try models of lazily generated families.
        exhaustive: include the long tail of generated families and raise the parameter cap.
        max_params: largest parameter count tried. ``None`` means ``n_active - 3`` when
            ``families`` is explicit (you asked for those models; only the data can rule one
            out), ``min(10, n_active - 3)`` when exhaustive, else
            ``min(6, max(2, n_active // 3))``. Whatever the value, a model never gets more
            than ``n_active - 3`` parameters. Every candidate the cap leaves out is listed in
            the result's ``skipped`` with its parameter count.
        n_starts: multistart count per candidate.
        seed: base seed; a candidate's seed is derived from it and the candidate id, so
            results do not depend on scheduling order.
        time_limit: seconds allowed per candidate.
        workers: worker processes; ``None`` means ``max(1, cpu_count - 1)`` and ``1`` runs
            in-process.
        cv_folds: folds of the cross-validation RMSE; ``None`` disables it.
        staged_starts: fit every candidate from two starts first and give the full ``n_starts``
            only to those that could still change the ranking (AICc within 10 of the best, or
            not converged but with an SSE within three times the best converged SSE); the
            rest keep their two-start outcome (``stage="screen"``). Turn off to give every
            candidate the full multistart.
        custom_models: user equations to fit alongside the catalogue. They are always
            candidates (neither ``families`` nor ``max_params`` filters them); the data
            alone decides whether one can be fitted. Ids must be unique and must not
            collide with a catalogue or polynomial id.
        curve_top_n: how many top-ranked candidates (under each ranking criterion) get
            their fitted curve and bands sampled into the result so a UI can plot without
            refitting; ``0`` samples none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank_by: RankBy = RankBy.BEST_OVERALL
    families: tuple[str, ...] | None = None
    include_polynomials: bool = True
    include_generated: bool = True
    exhaustive: bool = False
    max_params: Annotated[int, Field(ge=1)] | None = None
    n_starts: Annotated[int, Field(ge=1)] = 8
    seed: int = 0
    time_limit: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 20.0
    workers: Annotated[int, Field(ge=1)] | None = None
    cv_folds: Annotated[int, Field(ge=2)] | None = 5
    staged_starts: bool = True
    custom_models: tuple[CustomModelDefinition, ...] = ()
    curve_top_n: Annotated[int, Field(ge=0, le=50)] = 5

    @field_validator("families")
    @classmethod
    def _families_not_empty(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and not value:
            raise ValueError("families must list at least one category, or be None for all")
        return value


class PolynomialRef(BaseModel):
    """A polynomial candidate, ``polyN`` or (with ``through_origin``) ``polyN_origin``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["polynomial"] = "polynomial"
    degree: Annotated[int, Field(ge=1)]
    through_origin: bool = False

    @property
    def id(self) -> str:
        return f"poly{self.degree}_origin" if self.through_origin else f"poly{self.degree}"

    @property
    def parameter_count(self) -> int:
        return self.degree if self.through_origin else self.degree + 1


class RegisteredRef(BaseModel):
    """A model addressed by its registry id (hand-written, plugin or generated)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["registered"] = "registered"
    model_id: Annotated[str, Field(min_length=1)]

    @property
    def id(self) -> str:
        return self.model_id


ModelRef = RegisteredRef
"""Name used by the plan for :class:`RegisteredRef`."""


class CustomRef(BaseModel):
    """A user-defined equation; the definition travels with the job so a worker can build it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["custom"] = "custom"
    definition: CustomModelDefinition

    @property
    def id(self) -> str:
        return self.definition.id


CandidateRef = Annotated[PolynomialRef | RegisteredRef | CustomRef, Field(discriminator="kind")]
"""What one worker needs to rebuild a candidate; each variant has ``kind`` and ``id``."""


class CandidateMetrics(BaseModel):
    """Goodness-of-fit numbers of one fitted candidate; ``None`` means not defined."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int
    n: int
    r2: FiniteFloat | None = None
    adj_r2: FiniteFloat | None = None
    rmse: FiniteFloat | None = None
    aicc: FiniteFloat | None = None
    bic: FiniteFloat | None = None
    cv_rmse: FiniteFloat | None = None


class RankedFit(BaseModel):
    """One row of the ranked result.

    ``rank`` starts at 1; rows whose fitted curves coincide (``same_curve_as`` lists the
    others) share the rank of the best of them. ``status`` is the candidate's fit status as
    text (``"exact_fit"`` for a fit with SSE = 0). ``parameters`` are the fitted estimates;
    ``warnings`` are the fit's own; ``overfitting_note`` names a much simpler candidate that
    fits almost as well, when there is one.

    ``delta_aicc`` (AICc minus the lowest AICc of the run) and ``akaike_weight`` (relative
    likelihood normalised over every candidate with an AICc) are ``None`` exactly when the
    row has no AICc (exact fits, tiny samples); they use the true AICc, never the ranking
    adjustment for unreliable fits. ``degenerate`` marks a fit whose optimum collapsed onto a
    simpler model (a parameter at its bound with negligible effect, or a singular Jacobian);
    such rows rank below every non-degenerate row. ``generated`` says the model comes from a
    generated family, not a hand-written one. ``shape_parameters`` counts its free-frequency
    parameters, which the ranking charges as extra effective parameters.
    ``redundant_with`` names the canonical row when
    this row draws the same curve as a model with fewer parameters: it is then not tied with
    that row but listed as redundant with it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int
    model_id: str
    name: str
    category: str = ""
    equation_fitted: str | None = None
    parameters: tuple[ParameterEstimate, ...] = ()
    metrics: CandidateMetrics
    same_curve_as: tuple[str, ...] = ()
    status: str = "converged"
    warnings: tuple[str, ...] = ()
    overfitting_note: str | None = None
    generated: bool = False
    shape_parameters: int = 0
    degenerate: bool = False
    delta_aicc: FiniteFloat | None = None
    akaike_weight: FiniteFloat | None = None
    redundant_with: str | None = None
