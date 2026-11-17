"""Information criteria and model comparison for fitted curves.

The criteria use one Gaussian convention everywhere, chosen so that a fit's *error
variance* is charged as a parameter (``k' = k + 1``) and the constants common to every
model are dropped. Only differences between criteria are therefore meaningful; an
absolute AIC is not a probability or a quality score.

The formulas need nothing but a :class:`~curvestudio.fitting.FitResult`'s statistics, so
they work without the data. A fit that did not converge is refused rather than ranked, and
an exact fit (``SSE = 0``) is refused too: its Gaussian likelihood is unbounded, so no
finite information criterion exists.
"""

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from curvestudio.fitting import FitResult, FitStatus
from curvestudio.statistics.errors import StatisticsError

_CONVENTION = (
    "Gaussian, AIC = n·ln(SSE/n) + 2k', BIC = n·ln(SSE/n) + k'·ln(n), "
    "AICc = AIC + 2k'(k'+1)/(n-k'-1) when n-k'-1 > 0, with k' = k + 1 (the error variance "
    "counts), n = active observations and SSE weighted. Constants common to all models "
    "(n·ln 2π + n) are dropped, so only differences are meaningful."
)

_DATASETS_DIFFER = (
    "DATASETS_DIFFER: the results were not fitted to the same dataset or observation count; "
    "information criteria are only comparable on the same response data"
)


class InformationCriteria(BaseModel):
    """AIC, AICc and BIC of one converged fit under the shared Gaussian convention.

    ``k_total`` is ``k' = k + 1``: the fitted parameters plus the error variance.
    ``aicc`` is ``None`` when ``n - k' - 1 <= 0``, because the small-sample correction is
    undefined there (it would divide by zero or turn negative), not because the fit failed.
    ``convention`` records the exact formulas so a stored result stays self-describing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    k_total: int
    sse: float
    aic: float
    aicc: float | None
    bic: float
    convention: str


class ComparisonRow(BaseModel):
    """One comparable model in a :class:`ModelComparison`.

    ``delta_aicc`` and ``akaike_weight`` are ``None`` exactly when ``aicc`` is ``None``;
    ``delta_bic`` is always defined. Deltas are relative to the best score in the compared
    set, and the Akaike weights are normalised over the rows with a defined AICc only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    aic: float
    aicc: float | None
    bic: float
    delta_aicc: float | None
    akaike_weight: float | None
    delta_bic: float


class ModelComparison(BaseModel):
    """Ranking of several fits by information criteria, with refusals kept visible.

    ``rows`` is ordered by AICc, falling back to BIC for a row whose AICc is undefined.
    ``excluded`` maps a model id to the reason its result could not be ranked (a
    :class:`StatisticsError` code and message) instead of dropping it silently; when the
    same model id appears more than once, the last failing result's reason is kept.
    ``warnings`` records ``DATASETS_DIFFER`` when the batch mixes datasets or observation
    counts, but the comparison is still computed because the caller may know better.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[ComparisonRow, ...]
    excluded: dict[str, str]
    warnings: tuple[str, ...]


def information_criteria(result: FitResult) -> InformationCriteria:
    """Compute AIC, AICc and BIC from ``result.stats`` alone.

    ``n`` is the fit's active observation count and ``SSE`` its weighted residual sum of
    squares, so zero-weight points are already excluded. No data are needed and none are
    inspected. ``aicc`` is ``None`` when ``n - k' - 1 <= 0``.

    The criteria assume Gaussian errors with constant variance and compare models fitted to
    the same response data. They rank models; they are not a goodness-of-fit test and do not
    establish that any model is valid.

    Raises:
        StatisticsError: ``FIT_NOT_CONVERGED`` when the status is not ``CONVERGED``;
            ``NO_STATS`` when ``stats`` is ``None``; ``PERFECT_FIT`` when ``SSE == 0``
            (the Gaussian likelihood is unbounded, so no finite criterion exists).
    """
    if result.status is not FitStatus.CONVERGED:
        raise StatisticsError(
            "FIT_NOT_CONVERGED",
            f"information criteria need a converged fit; status is {result.status.value!r} "
            f"({result.termination_reason})",
        )
    stats = result.stats
    if stats is None:
        raise StatisticsError(
            "NO_STATS",
            "the result carries no fit statistics (stats is None), so no likelihood is defined",
        )
    if stats.sse == 0.0:
        raise StatisticsError(
            "PERFECT_FIT",
            "the fit is exact (SSE = 0), so the Gaussian likelihood is unbounded and "
            "information criteria are not defined",
        )

    n = stats.n
    k_total = stats.k + 1
    # ln(SSE) - ln(n) rather than ln(SSE/n): identical in exact arithmetic, but it cannot
    # underflow to ln(0) for subnormal residuals.
    base = n * (math.log(stats.sse) - math.log(n))
    aic = base + 2.0 * k_total
    bic = base + k_total * math.log(n)
    denominator = n - k_total - 1
    aicc = aic + 2.0 * k_total * (k_total + 1) / denominator if denominator > 0 else None
    return InformationCriteria(
        n=n,
        k_total=k_total,
        sse=stats.sse,
        aic=aic,
        aicc=aicc,
        bic=bic,
        convention=_CONVENTION,
    )


def compare_models(results: Sequence[FitResult]) -> ModelComparison:
    """Rank ``results`` by AICc (BIC where a row's AICc is undefined).

    Non-converged, statistics-free and perfect fits are listed in ``excluded`` with the
    :class:`StatisticsError` code and message; they never appear as rows. AICc deltas and
    Akaike weights are computed over the rows with a defined AICc and are ``None``
    elsewhere; BIC deltas are always defined. A ``DATASETS_DIFFER`` warning is recorded
    when the dataset fingerprints or observation counts differ, but the comparison still
    runs.

    Limitations: the ranking is conditional on the supplied models, does not check that
    they are nested, and says nothing about whether the best of them is adequate. The
    caller is responsible for comparing fits of the same response variable.
    """
    compared: list[tuple[str, InformationCriteria]] = []
    excluded: dict[str, str] = {}
    for result in results:
        try:
            compared.append((result.model_id, information_criteria(result)))
        except StatisticsError as exc:
            excluded[result.model_id] = f"{exc.code}: {exc}"

    warnings: list[str] = []
    fingerprints = {result.provenance.dataset_fingerprint for result in results}
    observation_counts = {result.stats.n for result in results if result.stats is not None}
    if len(fingerprints) > 1 or len(observation_counts) > 1:
        warnings.append(_DATASETS_DIFFER)

    if not compared:
        return ModelComparison(rows=(), excluded=excluded, warnings=tuple(warnings))

    aicc_by_index: dict[int, float] = {}
    for index, (_, criteria) in enumerate(compared):
        if criteria.aicc is not None:
            aicc_by_index[index] = criteria.aicc
    delta_aicc: dict[int, float] = {}
    akaike_weight: dict[int, float] = {}
    if aicc_by_index:
        best_aicc = min(aicc_by_index.values())
        delta_aicc = {index: value - best_aicc for index, value in aicc_by_index.items()}
        terms = {index: math.exp(-0.5 * delta) for index, delta in delta_aicc.items()}
        total = sum(terms.values())
        akaike_weight = {index: term / total for index, term in terms.items()}

    best_bic = min(criteria.bic for _, criteria in compared)
    rows = [
        ComparisonRow(
            model_id=model_id,
            aic=criteria.aic,
            aicc=criteria.aicc,
            bic=criteria.bic,
            delta_aicc=delta_aicc.get(index),
            akaike_weight=akaike_weight.get(index),
            delta_bic=criteria.bic - best_bic,
        )
        for index, (model_id, criteria) in enumerate(compared)
    ]
    rows.sort(key=lambda row: row.aicc if row.aicc is not None else row.bic)
    return ModelComparison(rows=tuple(rows), excluded=excluded, warnings=tuple(warnings))
