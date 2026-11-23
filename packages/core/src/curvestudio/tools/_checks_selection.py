"""Checks that set the fit beside other fits.

Cross-validation holds the held-out error against what a correct model would show, and an
Auto-fit ranking is searched for a simpler model that fits as well.
"""

import math

from curvestudio.discovery import AutoFitResult, RankedFit
from curvestudio.tools._common import Check, Inputs, audit_check, skipped, title_for
from curvestudio.tools._thresholds import (
    CV_CAUTION_RATIO,
    CV_FAIL_RATIO,
    NO_STATISTICS,
    SIMPLE_DELTA_AICC,
)
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import CV_NEXT_STEP, SIMPLER_NEXT_STEP, num


def expected_cv_ratio(n: int, k: int, folds: int) -> float:
    """The cross-validation RMSE a correct linear model shows, as a multiple of its in-sample RMSE.

    With ``n`` points, ``k`` parameters and ``folds`` folds, a fold's fit uses ``m = n - n/folds``
    training points (the mean training size: ``k_fold_cv`` splits the points into folds that
    differ by at most one) and predicts a new point with expected squared error
    ``sigma^2 (1 + k/m)``. The in-sample mean square, ``SSE / n`` as the core defines it, is
    ``sigma^2 (1 - k/n)`` in expectation. The root of their ratio is what a model with nothing
    missing shows. Leave-one-out (``folds == n``) trains every fold on ``n - 1`` points.

    Raises:
        ValueError: fewer than two folds, more folds than points, or as many parameters as
            points.
    """
    if folds < 2 or folds > n:
        raise ValueError(f"folds must lie between 2 and the number of points ({n}), got {folds}")
    if k >= n:
        raise ValueError(f"a fit needs fewer parameters than points: {k} for {n} points")
    if folds == n:
        expected_cv = 1.0 + k / (n - 1)
    else:
        training = n - n / folds
        expected_cv = 1.0 + k * folds / (training * (folds - 1))
    return math.sqrt(expected_cv / (1.0 - k / n))


def check_cross_validation(inp: Inputs) -> AuditCheck:
    """Is the held-out error larger than a correct model with these parameters would show?"""
    stats = inp.result.stats
    if stats is None:
        return skipped(CROSS_VALIDATION, NO_STATISTICS)
    cv = inp.cv
    if cv is None:
        return skipped(
            CROSS_VALIDATION,
            "cross-validation has not been run",
            "Run cross-validation to check this.",
        )
    if cv.rmse is None:
        return skipped(
            CROSS_VALIDATION,
            f"cross-validation is incomplete: {cv.n_failed} of {cv.folds} folds failed, so "
            "there is no held-out error to compare",
            "Run cross-validation with fewer folds, or on more points, so that every fold "
            "can be fitted.",
        )
    n, k = inp.n, inp.k
    if cv.folds < 2 or cv.folds > n or k >= n:
        return skipped(
            CROSS_VALIDATION,
            f"cross-validation ran with {cv.folds} folds on {n} points for {k} parameters, "
            "so there is no correct model to expect of it",
        )
    expected = expected_cv_ratio(n, k, cv.folds)
    if stats.rmse > 0.0:
        ratio = cv.rmse / stats.rmse
    elif cv.rmse > 0.0:
        ratio = math.inf  # a perfect in-sample fit that predicts badly
    else:
        ratio = 0.0
    relative = ratio / expected
    if relative <= CV_CAUTION_RATIO:
        verdict = Verdict.PASS
    elif relative <= CV_FAIL_RATIO:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.FAIL
    compared = (
        f"In-sample RMSE {num(stats.rmse)}; cross-validation RMSE {num(cv.rmse)}, against "
        f"{num(expected * stats.rmse)} expected from a correct model with {n} points and "
        f"{cv.folds} folds"
    )
    if math.isfinite(relative):
        evidence = f"{compared} ({num(relative)} times that)."
        statistic = f"CV RMSE / RMSE = {num(ratio)}, relative to a correct model = {num(relative)}"
    else:
        evidence = f"{compared} (no finite multiple: the in-sample RMSE is 0)."
        statistic = "CV RMSE / RMSE is unbounded: the in-sample RMSE is 0"
    return audit_check(
        CROSS_VALIDATION,
        verdict,
        title_for(CROSS_VALIDATION_TITLES, verdict),
        evidence,
        statistic=statistic,
        threshold=(
            f"ratio to a correct model <= {num(CV_CAUTION_RATIO)} pass; "
            f"<= {num(CV_FAIL_RATIO)} caution; above {num(CV_FAIL_RATIO)} fail"
        ),
        next_step=None if verdict is Verdict.PASS else CV_NEXT_STEP,
    )


CROSS_VALIDATION_TITLES = (
    "Held-out error is close to what a correct model shows",
    "Held-out error is somewhat above what a correct model shows",
    "Held-out error is far above what a correct model shows",
)

CROSS_VALIDATION = Check(
    "cross_validation",
    "Predictive error was not assessed",
    "Cross-validation refits the model on subsets of the data and scores the points left "
    "out. A held-out error well above what a correct model would show means the model "
    "follows noise.",
    check_cross_validation,
)


def charged(row: RankedFit) -> int:
    """The parameter count the ranking charges: fitted parameters plus shape parameters."""
    return row.metrics.k + row.shape_parameters


def simplest(ranking: AutoFitResult | None, model_id: str) -> tuple[RankedFit | None, str | None]:
    """The ranked model with the fewest parameters among those within ``SIMPLE_DELTA_AICC`` of
    the best that are neither degenerate nor redundant, or the reason there is none.

    Ties go to the lower cross-validation error (a missing one counts as worse), then to the
    lower rank.
    """
    if ranking is None or not any(row.model_id == model_id for row in ranking.ranked):
        return None, "no Auto-fit ranking for this dataset"
    usable = [
        row
        for row in ranking.ranked
        if row.delta_aicc is not None
        and row.delta_aicc <= SIMPLE_DELTA_AICC
        and not row.degenerate
        and row.redundant_with is None
    ]
    if not usable:
        return None, (
            f"every model within {num(SIMPLE_DELTA_AICC)} AICc of the best is degenerate "
            "or redundant"
        )
    fewest = min(
        usable,
        key=lambda row: (
            charged(row),
            row.metrics.cv_rmse is None,
            row.metrics.cv_rmse or 0.0,
            row.rank,
        ),
    )
    return fewest, None


def check_simpler_model(inp: Inputs) -> AuditCheck:
    fewest = inp.simplest
    if fewest is None or inp.ranking is None:
        return skipped(SIMPLER_MODEL, inp.simplest_reason or "no Auto-fit ranking for this dataset")
    audited = next(row for row in inp.ranking.ranked if row.model_id == inp.result.model_id)
    here, there = charged(audited), charged(fewest)
    delta = fewest.delta_aicc if fewest.delta_aicc is not None else 0.0
    within = f"within {num(SIMPLE_DELTA_AICC)} AICc of the best"
    statistic = f"{here} parameters here; the simplest {within} has {there} ({fewest.model_id})"
    threshold = f"delta AICc <= {num(SIMPLE_DELTA_AICC)}"
    if fewest.model_id == inp.result.model_id:
        verdict, title = Verdict.PASS, "This is the simplest model within 2 AICc of the best"
        evidence = (
            f"This is the simplest model {within}: {here} parameters, "
            f"{num(delta)} AICc from the best."
        )
    elif there > here:
        verdict, title = Verdict.PASS, "No simpler model fits as well"
        evidence = (
            "No model within 2 AICc of the best that is not degenerate or redundant has "
            f"fewer parameters than this one; {fewest.name} ({fewest.model_id}) has {there}."
        )
    elif there == here:
        verdict, title = Verdict.PASS, "An equally simple model fits as well"
        evidence = (
            f"{fewest.name} ({fewest.model_id}) has the same number of parameters ({there}) "
            f"and lies {num(delta)} AICc from the best."
        )
    else:
        verdict, title = Verdict.CAUTION, "A simpler model fits as well"
        evidence = (
            f"{fewest.name} ({fewest.model_id}) reaches the same fit with {there} parameters "
            f"instead of {here} and lies {num(delta)} AICc from the best."
        )
        if fewest.metrics.cv_rmse is not None:
            evidence += f" Its cross-validation RMSE is {num(fewest.metrics.cv_rmse)}."
    return audit_check(
        SIMPLER_MODEL,
        verdict,
        title,
        evidence,
        statistic=statistic,
        threshold=threshold,
        next_step=SIMPLER_NEXT_STEP if verdict is Verdict.CAUTION else None,
    )


SIMPLER_MODEL = Check(
    "simpler_model",
    "Simpler models were not compared",
    "Of the models that fit about equally well, the one with the fewest parameters is usually "
    "the better description. A difference of 2 AICc or less is usually taken as no "
    "difference.",
    check_simpler_model,
)
