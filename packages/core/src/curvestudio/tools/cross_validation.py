"""Cross-validation for the audit: the held-out error of the model a stored fit estimates.

A stored fit does not carry its cross-validated error, so :func:`~curvestudio.tools.audit_fit`
reads ``not_applicable`` until it is given one. :func:`cross_validate` produces it by
refitting through the core's own functions (:func:`~curvestudio.statistics.k_fold_cv` and
:func:`~curvestudio.statistics.loo_cv`) with the fitter the workflow uses for the
specification, so every fold is fitted with the fit's own bounds, loss and weights.
"""

from typing import Literal

from curvestudio.data import XY
from curvestudio.fitting import FitStatus
from curvestudio.statistics import (
    CrossValidation,
    StatisticsError,
    k_fold_cv,
    loo_cv,
)
from curvestudio.tools._errors import ToolError
from curvestudio.workflow import FitBundle, FitSpec, WorkflowError, fitter_for, snapshot_xy

LOO_MAX_N = 2000
"""Leave-one-out refits once per point: it is run on at most this many active points."""


def _active_points(data: XY) -> int:
    """The points a fit of ``data`` uses: every row, or those with a weight above zero."""
    if data.weights is None:
        return int(data.x.size)
    return int((data.weights > 0.0).sum())


def cross_validate(
    bundle: FitBundle,
    spec: FitSpec,
    *,
    data: XY | None = None,
    folds: int | Literal["loo"] = 5,
    seed: int | None = None,
) -> CrossValidation:
    """Cross-validate the model a stored fit estimates, on the data it was fitted to.

    Args:
        bundle: a fit that converged.
        spec: the specification the fit ran with. The fitter is the one the workflow uses
            for it (:func:`~curvestudio.workflow.fitter_for`), given the stored fit, so each
            fold is fitted with the fit's own bounds, loss, starting values and weights.
        data: the data to cross-validate. The default is what the bundle carries; a compact
            bundle (``bundle.data.subsampled``) holds only a subsample and needs the full
            data here.
        folds: the number of folds (5 by default), or ``"loo"`` for leave-one-out, which
            needs at most ``LOO_MAX_N`` active points.
        seed: the seed that decides the folds. The default is the fit's own
            (``bundle.result.provenance.seed``, which the core records for every fit).

    Returns:
        The core's :class:`~curvestudio.statistics.CrossValidation`: the held-out RMSE of
        each fold and the pooled one, with every failed fold counted and named.

    Raises:
        ToolError: ``FIT_NOT_CONVERGED_INPUT`` for a fit that did not converge;
            ``DATA_REQUIRED`` for a compact bundle without ``data``; ``INVALID_FOLDS`` for
            a count below 2 or above the number of active points, or for ``"loo"`` above
            ``LOO_MAX_N``; and, with its own code, anything the workflow refuses in the
            specification (``INVALID_SPEC``, ``UNKNOWN_MODEL``, ...) or the core in the
            settings (``INVALID_SEED``, ...).
    """
    result = bundle.result
    if result.status is not FitStatus.CONVERGED:
        raise ToolError(
            "FIT_NOT_CONVERGED_INPUT",
            f"the fit is {result.status.value} ({result.termination_reason}): a fit that did "
            "not converge has no model to cross-validate",
        )
    if data is None and bundle.data.subsampled:
        raise ToolError(
            "DATA_REQUIRED",
            "the bundle holds only a subsample of its data (compact mode): pass data= with "
            "the full data the fit was computed from",
        )
    xy = snapshot_xy(bundle) if data is None else data
    try:
        fitter, fitted = fitter_for(spec, xy, result=result)
    except WorkflowError as exc:
        raise ToolError(exc.code, exc.message) from exc
    if seed is None:
        seed = result.provenance.seed
    if folds == "loo":
        active = _active_points(fitted)
        if active > LOO_MAX_N:
            raise ToolError(
                "INVALID_FOLDS",
                f"leave-one-out cross-validation refits once per point and is run on at "
                f"most {LOO_MAX_N} active points; these data have {active}. Pass a number "
                "of folds instead",
            )
    try:
        if folds == "loo":
            return loo_cv(fitted, fitter)
        return k_fold_cv(fitted, fitter, folds=folds, seed=seed)
    except StatisticsError as exc:
        raise ToolError(exc.code, str(exc)) from exc
