"""``describe_fit``: one call that bundles every statistic a fit deserves.

The bundle never turns a refused fit into a partial report: if the fit is not converged or
does not belong to the data, :func:`~curvestudio.statistics.build_view` raises
:class:`~curvestudio.statistics.StatisticsError` and so does :func:`describe_fit`. Nothing in
the report says a model is "valid": it collects numbers and the tests' own assumptions and
limitations, and leaves the judgement to the reader.
"""

import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from curvestudio.data import XY
from curvestudio.diagnostics.hypothesis_tests import DiagnosticsReport, run_diagnostics
from curvestudio.diagnostics.residuals import ResidualSet, residual_set
from curvestudio.fitting import FitResult
from curvestudio.models import ModelSpec
from curvestudio.statistics import (
    CrossValidation,
    Fitter,
    InformationCriteria,
    build_view,
    information_criteria,
    k_fold_cv,
)


class FitReport(BaseModel):
    """Everything :func:`describe_fit` computed for one converged fit.

    ``result_summary`` holds ``model_id`` and ``status`` (strings) and ``n``, ``k``, ``df``,
    ``rmse`` plus, where defined, ``r2`` and ``adj_r2`` (numbers). A statistic that is not
    defined is left out of the mapping, never stored as ``0.0`` or NaN. ``cv`` is ``None``
    when no ``fitter`` was supplied. ``warnings`` merges the fit's own warnings with what the
    bundle found: incomplete cross-validation, tests that did not apply and flagged points.

    Like :class:`~curvestudio.fitting.FitResult` the model is frozen but its lists are not
    deep-frozen; :meth:`to_json` re-checks finiteness, so a mutation that sneaks in NaN/inf
    fails loudly instead of writing a file that cannot be read back.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_summary: dict[str, float | str]
    criteria: InformationCriteria
    residuals: ResidualSet
    diagnostics: DiagnosticsReport
    cv: CrossValidation | None
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        """Serialise to standard JSON (floats keep full ``repr`` precision).

        Raises:
            ValueError: a non-finite number is present (only possible through mutation of a
                list field).
        """
        return json.dumps(self.model_dump(mode="python"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse :meth:`to_json` output; invalid JSON or non-finite values raise ``ValueError``."""

        def reject_constant(token: str) -> Any:
            raise ValueError(f"non-finite JSON constant {token!r} is not allowed")

        return cls.model_validate(json.loads(text, parse_constant=reject_constant))


def _summary(result: FitResult) -> dict[str, float | str]:
    stats = result.stats
    if stats is None:  # unreachable after build_view; keeps the type checker honest
        raise ValueError("a converged fit always carries statistics")
    summary: dict[str, float | str] = {
        "model_id": result.model_id,
        "status": result.status.value,
        "n": float(stats.n),
        "k": float(stats.k),
        "df": float(stats.df),
        "rmse": stats.rmse,
    }
    if stats.r2 is not None:
        summary["r2"] = stats.r2
    if stats.adj_r2 is not None:
        summary["adj_r2"] = stats.adj_r2
    return summary


def _findings(
    result: FitResult,
    residuals: ResidualSet,
    diagnostics: DiagnosticsReport,
    cv: CrossValidation | None,
) -> tuple[str, ...]:
    messages = list(result.warnings)
    for test in diagnostics.tests:
        if test.status == "not_applicable":
            messages.append(f"{test.name} does not apply: {test.reason}")
    if cv is not None and cv.status == "incomplete":
        messages.append(
            f"cross-validation is incomplete: {cv.n_failed} of {cv.folds} folds failed "
            f"({'; '.join(cv.reasons)})"
        )
    if residuals.flagged_high_leverage:
        messages.append(
            f"{len(residuals.flagged_high_leverage)} point(s) have high leverage "
            f"(h > {residuals.leverage_threshold:.4g})"
        )
    if residuals.flagged_high_influence:
        messages.append(
            f"{len(residuals.flagged_high_influence)} point(s) are influential "
            f"(Cook's distance > {residuals.cooks_threshold:.4g})"
        )
    return tuple(messages)


def describe_fit(
    data: XY,
    result: FitResult,
    model: ModelSpec | str | None = None,
    fitter: Fitter | None = None,
    cv_folds: int = 5,
    seed: int = 0,
) -> FitReport:
    """Bundle information criteria, residual diagnostics, tests and (optionally) CV.

    Cross-validation runs only when a ``fitter`` is given, with ``cv_folds`` folds and the
    recorded ``seed``. Every other statistic is deterministic.

    Raises:
        StatisticsError: the fit is refused (``FIT_NOT_CONVERGED``, ``DATA_MISMATCH``,
            ``MODEL_MISMATCH``) or a section cannot be computed (for example
            ``INVALID_FOLDS``); no partial report is returned.
        KeyError: no ``model`` was given and ``result.model_id`` is not registered.
    """
    view = build_view(data, result, model)
    criteria = information_criteria(result)
    residuals = residual_set(view)
    diagnostics = run_diagnostics(view)
    cv = None if fitter is None else k_fold_cv(data, fitter, folds=cv_folds, seed=seed)
    return FitReport(
        result_summary=_summary(result),
        criteria=criteria,
        residuals=residuals,
        diagnostics=diagnostics,
        cv=cv,
        warnings=_findings(result, residuals, diagnostics, cv),
    )
