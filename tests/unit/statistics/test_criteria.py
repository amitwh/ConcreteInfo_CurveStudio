"""Tests for information criteria and model comparison."""

import math

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import (
    FitResult,
    FitStatus,
    ParameterEstimate,
    collect_provenance,
    fit_polynomial,
    fit_statistics,
)
from curvestudio.statistics import (
    ComparisonRow,
    InformationCriteria,
    ModelComparison,
    StatisticsError,
    compare_models,
    information_criteria,
)


def make_result(
    *,
    model_id: str,
    n: int = 10,
    k: int = 2,
    sse: float = 2.5,
    status: FitStatus = FitStatus.CONVERGED,
    fingerprint: str = "fp-criteria",
    with_stats: bool = True,
) -> FitResult:
    """A result carrying exactly these statistics; no data is involved.

    Information criteria read only ``result.stats``, so a synthetic result is the honest
    unit-test input: the residual vector is chosen to produce the requested ``sse``.
    """
    y = np.full(n, math.sqrt(sse / n))
    stats = fit_statistics(y, np.zeros(n), k) if with_stats else None
    return FitResult(
        model_id=model_id,
        equation="y = 0",
        equation_fitted=None,
        parameters=tuple(ParameterEstimate(name=f"b{j}", value=0.0) for j in range(k)),
        covariance=None,
        correlation=None,
        stats=stats,
        fitted=[0.0] * n,
        residuals=y.tolist(),
        status=status,
        termination_reason="synthetic result for statistics tests",
        method="ols",
        loss="linear",
        initial_params={},
        n_starts=1,
        warnings=(),
        provenance=collect_provenance(fingerprint, seed=0),
    )


def test_formulae_on_hand_computed_values() -> None:
    # n = 10, SSE = 2.5, k = 2 -> k' = k + 1 = 3 and SSE/n = 0.25.
    # ln(0.25) = -1.3862943611198906, so n*ln(SSE/n) = -13.862943611198906.
    # AIC  = -13.862943611198906 + 2*3                  = -7.862943611198906
    # BIC  = -13.862943611198906 + 3*ln(10)             = -6.955188332216768
    # AICc = AIC + 2*3*4/(10 - 3 - 1) = AIC + 24/6      = -3.862943611198906
    criteria = information_criteria(make_result(model_id="hand", n=10, k=2, sse=2.5))

    assert isinstance(criteria, InformationCriteria)
    assert criteria.n == 10
    assert criteria.k_total == 3
    assert criteria.sse == pytest.approx(2.5, rel=1e-15)
    assert criteria.aic == pytest.approx(-7.862943611198906, rel=1e-15)
    assert criteria.bic == pytest.approx(-6.955188332216768, rel=1e-15)
    assert criteria.aicc == pytest.approx(-3.862943611198906, rel=1e-15)
    assert "AIC = n·ln(SSE/n) + 2k'" in criteria.convention
    assert "k' = k + 1" in criteria.convention


@pytest.mark.parametrize("k", [8, 9])
def test_aicc_is_none_when_correction_undefined(k: int) -> None:
    # k' = k + 1, so n - k' - 1 = 10 - (k + 1) - 1 = 8 - k: zero for k=8, negative for k=9.
    criteria = information_criteria(make_result(model_id="wide", n=10, k=k))

    assert criteria.aicc is None
    assert math.isfinite(criteria.aic)
    assert math.isfinite(criteria.bic)


def test_perfect_fit_is_refused() -> None:
    with pytest.raises(StatisticsError) as info:
        information_criteria(make_result(model_id="exact", sse=0.0))

    assert info.value.code == "PERFECT_FIT"
    assert "unbounded" in str(info.value).lower()


@pytest.mark.parametrize(
    "status", [FitStatus.NOT_CONVERGED, FitStatus.FAILED, FitStatus.INVALID_INPUT]
)
def test_non_converged_fit_is_refused(status: FitStatus) -> None:
    with pytest.raises(StatisticsError) as info:
        information_criteria(make_result(model_id="bad", status=status))

    assert info.value.code == "FIT_NOT_CONVERGED"


def test_missing_stats_is_refused() -> None:
    with pytest.raises(StatisticsError) as info:
        information_criteria(make_result(model_id="no-stats", with_stats=False))

    assert info.value.code == "NO_STATS"


def test_compare_models_orders_by_aicc_and_weights_sum_to_one() -> None:
    results = [
        make_result(model_id="m1", sse=2.5),
        make_result(model_id="m2", sse=3.6),
        make_result(model_id="m3", sse=5.0),
    ]
    comparison = compare_models(results)

    assert isinstance(comparison, ModelComparison)
    assert comparison.excluded == {}
    assert comparison.warnings == ()
    assert [row.model_id for row in comparison.rows] == ["m1", "m2", "m3"]
    assert all(isinstance(row, ComparisonRow) for row in comparison.rows)

    best = comparison.rows[0].aicc
    assert best is not None
    assert comparison.rows[0].delta_aicc == pytest.approx(0.0, abs=1e-15)
    assert comparison.rows[0].delta_bic == pytest.approx(0.0, abs=1e-15)

    weights = [row.akaike_weight for row in comparison.rows]
    assert all(weight is not None for weight in weights)
    assert sum(weight for weight in weights if weight is not None) == pytest.approx(1.0, rel=1e-12)
    assert weights[0] == max(weight for weight in weights if weight is not None)

    for row in comparison.rows:
        assert row.aicc is not None and row.delta_aicc is not None
        assert row.delta_aicc == pytest.approx(row.aicc - best, rel=1e-15)
        expected_weight = math.exp(-0.5 * row.delta_aicc) / sum(
            math.exp(-0.5 * other.delta_aicc)
            for other in comparison.rows
            if other.delta_aicc is not None
        )
        assert row.akaike_weight == pytest.approx(expected_weight, rel=1e-15)


def test_comparison_warns_when_datasets_differ() -> None:
    same = compare_models(
        [
            make_result(model_id="a", fingerprint="fp-same"),
            make_result(model_id="b", fingerprint="fp-same"),
        ]
    )
    assert same.warnings == ()

    other_fingerprint = compare_models(
        [
            make_result(model_id="a", fingerprint="fp-a"),
            make_result(model_id="b", fingerprint="fp-b"),
        ]
    )
    assert any("DATASETS_DIFFER" in warning for warning in other_fingerprint.warnings)
    assert len(other_fingerprint.rows) == 2  # it still compares; the user may know better

    other_count = compare_models(
        [
            make_result(model_id="a", n=10, fingerprint="fp-same"),
            make_result(model_id="b", n=12, fingerprint="fp-same"),
        ]
    )
    assert any("DATASETS_DIFFER" in warning for warning in other_count.warnings)
    assert len(other_count.rows) == 2


def test_non_converged_results_are_excluded_with_reason() -> None:
    comparison = compare_models(
        [
            make_result(model_id="failed", status=FitStatus.FAILED),
            make_result(model_id="good"),
            make_result(model_id="stopped", status=FitStatus.NOT_CONVERGED),
        ]
    )

    assert [row.model_id for row in comparison.rows] == ["good"]
    assert set(comparison.excluded) == {"failed", "stopped"}
    assert comparison.excluded["failed"].startswith("FIT_NOT_CONVERGED")
    assert "'failed'" in comparison.excluded["failed"]
    assert comparison.excluded["stopped"].startswith("FIT_NOT_CONVERGED")
    assert "'not_converged'" in comparison.excluded["stopped"]


def test_a_row_without_aicc_never_outranks_a_row_with_one() -> None:
    # n = 10, k = 8 -> k' = 9 -> n - k' - 1 = 0, so AICc is undefined but BIC is not.
    undefined = make_result(model_id="wide", n=10, k=8, sse=0.1)
    defined = make_result(model_id="narrow", n=10, k=2, sse=5.0)
    comparison = compare_models([undefined, defined])

    assert undefined_has_lower_bic(comparison)  # the trap: BIC alone would rank "wide" first
    assert [row.model_id for row in comparison.rows] == ["narrow", "wide"]
    assert comparison.rows[1].aicc is None
    assert comparison.rows[1].delta_aicc is None
    assert comparison.rows[1].akaike_weight is None
    assert comparison.rows[0].delta_aicc == pytest.approx(0.0, abs=1e-15)
    assert comparison.rows[0].akaike_weight == pytest.approx(1.0)


def undefined_has_lower_bic(comparison: ModelComparison) -> bool:
    by_id = {row.model_id: row for row in comparison.rows}
    return by_id["wide"].bic < by_id["narrow"].bic


def test_rows_are_ranked_by_bic_when_no_aicc_is_defined() -> None:
    worse = make_result(model_id="worse", n=10, k=8, sse=0.5)
    better = make_result(model_id="better", n=10, k=8, sse=0.1)
    comparison = compare_models([worse, better])

    assert [row.model_id for row in comparison.rows] == ["better", "worse"]
    assert all(row.aicc is None and row.akaike_weight is None for row in comparison.rows)


def test_a_known_truth_dataset_prefers_the_true_degree() -> None:
    rng = np.random.default_rng(2026)
    x = np.linspace(-3.0, 3.0, 40)
    y = 1.0 - 2.0 * x + 0.8 * x**2 + rng.normal(0.0, 0.7, x.size)
    data = XY(
        x=x,
        y=y,
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="fp-known-truth",
    )
    results = [fit_polynomial(data, degree) for degree in (1, 2, 3, 4)]
    assert all(result.status is FitStatus.CONVERGED for result in results)

    comparison = compare_models(results)

    assert comparison.rows[0].model_id == "poly2"
    assert min(comparison.rows, key=lambda row: row.bic).model_id == "poly2"
