"""Public value types of model discovery: configuration, candidate references, results."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from curvestudio.fitting.metrics import FiniteFloat


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
        max_params: largest parameter count tried. ``None`` means
            ``min(6, max(1, n_active // 4))``, or ``min(10, n_active - 3)`` when exhaustive.
            Whatever the value, a model never gets more than ``n_active - 3`` parameters.
        n_starts: multistart count per candidate.
        seed: base seed; a candidate's seed is derived from it and the candidate id, so
            results do not depend on scheduling order.
        time_limit: seconds allowed per candidate.
        workers: worker processes; ``None`` means ``max(1, cpu_count - 1)`` and ``1`` runs
            in-process.
        cv_folds: folds of the cross-validation RMSE; ``None`` disables it.
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

CandidateRef = Annotated[PolynomialRef | RegisteredRef, Field(discriminator="kind")]
"""What one worker needs to rebuild a candidate; each variant has ``kind`` and ``id``.

The ``CustomRef`` variant (a user-defined equation) is added by Task 4.14, which brings
``CustomModelDefinition``: one more class with ``kind: Literal["custom"]`` and one more
member of this union, so existing code keeps working unchanged.
"""


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
    others) share the rank of the best of them. ``status`` is the candidate's fit status
    as text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int
    model_id: str
    name: str
    equation_fitted: str | None = None
    metrics: CandidateMetrics
    same_curve_as: tuple[str, ...] = ()
    status: str = "converged"
