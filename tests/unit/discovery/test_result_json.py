"""Tests for the Auto-fit result container: JSON, re-ranking, curves, streaming ranking."""

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    RankBy,
    autofit,
    iter_autofit,
    partial_ranking,
)
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()


def data(n: int = 40) -> XY:
    x = np.linspace(0.0, 6.0, n)
    rng = np.random.default_rng(11)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, n)
    return XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint="result-test")


def run(**overrides: Any) -> AutoFitResult:
    base: dict[str, Any] = {
        "workers": 1,
        "n_starts": 3,
        "cv_folds": 4,
        "families": ("exponential", "saturation", "polynomial"),
        "max_params": 3,
    }
    base.update(overrides)
    return autofit(data(), AutoFitConfig(**base))


def test_json_round_trip_is_exact_apart_from_elapsed() -> None:
    result = run()
    restored = AutoFitResult.from_json(result.to_json())
    assert restored == result
    assert restored.ranked_by(RankBy.R2) == result.ranked_by(RankBy.R2)
    assert restored.config == result.config


def test_to_json_is_standard_json_without_non_finite_numbers() -> None:
    text = run().to_json()
    parsed = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant {c}"))
    assert set(parsed) >= {
        "config",
        "dataset_fingerprint",
        "n_active",
        "ranked",
        "outcomes",
        "skipped",
        "unconverged",
        "exact_fits",
        "warnings",
        "cancelled",
        "curves",
        "elapsed_seconds",
    }
    row = parsed["ranked"][0]
    assert set(row) >= {
        "rank",
        "model_id",
        "name",
        "category",
        "equation_fitted",
        "metrics",
        "same_curve_as",
        "status",
        "warnings",
        "overfitting_note",
        "parameters",
    }
    assert set(row["metrics"]) >= {"k", "n", "r2", "adj_r2", "rmse", "aicc", "bic", "cv_rmse"}


def test_from_json_rejects_non_finite_constants_and_bad_input() -> None:
    text = run().to_json().replace('"elapsed_seconds": ', '"elapsed_seconds": NaN, "x": ', 1)
    with pytest.raises(ValueError):
        AutoFitResult.from_json(text)
    with pytest.raises(ValueError):
        AutoFitResult.from_json("{not json")


def test_slim_json_drops_the_per_point_vectors_but_keeps_every_row() -> None:
    result = run()
    full = result.to_json()
    slim = result.to_json(include_fits=False)
    assert len(slim) < len(full)
    restored = AutoFitResult.from_json(slim)
    assert restored.ranked == result.ranked
    assert all(not o.fit.fitted for o in restored.outcomes if o.fit is not None)
    assert restored.ranked_by(RankBy.RMSE) == result.ranked_by(RankBy.RMSE)
    assert restored.curves == result.curves


def test_curves_exist_for_the_top_rows_only() -> None:
    result = run(curve_top_n=3)
    assert result.curves
    top = {r.model_id for rb in RankBy for r in result.ranked_by(rb)[:3]}
    assert set(result.curves) == top
    assert len(result.curves) < len(result.ranked)
    curve = result.curves[result.ranked[0].model_id]
    assert len(curve.x) == len(curve.y) == 200
    assert curve.confidence_lower is not None and len(curve.confidence_lower) == 200
    assert curve.prediction_upper is not None
    lows, highs = curve.prediction_lower or [], curve.prediction_upper or []
    assert len(lows) == len(highs) == 200
    assert all(lo <= y <= hi for lo, y, hi in zip(lows, curve.y, highs, strict=True))
    assert curve.x[0] < 0.0 + 1e-12 and curve.x[-1] >= 6.0  # data range plus a margin


def test_curve_top_n_zero_samples_nothing() -> None:
    assert run(curve_top_n=0).curves == {}


def test_ranked_by_never_refits_and_respects_the_ranking_rules() -> None:
    result = run()
    by_r2 = result.ranked_by(RankBy.R2)
    r2 = [r.metrics.r2 for r in by_r2]
    assert r2 == sorted(r2, key=lambda v: -(v if v is not None else -1e300))
    by_aicc = result.ranked_by(RankBy.AICC)
    aicc = [r.metrics.aicc for r in by_aicc if r.metrics.aicc is not None]
    assert aicc == sorted(aicc)


def test_outcome_buckets_partition_all_candidates() -> None:
    result = run()
    names = {o.candidate_id for o in result.outcomes}
    ranked = {r.model_id for r in result.ranked}
    assert ranked <= names
    for cid in (*result.unconverged, *result.skipped, *result.exact_fits):
        assert cid in names and cid not in ranked
    assert result.failed.keys() <= result.unconverged.keys()
    assert result.timed_out.keys() <= result.unconverged.keys()


def test_overfitting_note_on_rows_that_buy_little_with_many_parameters() -> None:
    x = np.linspace(0.0, 10.0, 40)
    rng = np.random.default_rng(4)
    line = XY(x, 2.0 * x + 1.0 + rng.normal(0.0, 0.5, 40), None, "x", "y", "line-test")
    cfg = AutoFitConfig(workers=1, n_starts=2, cv_folds=4, families=("polynomial",), max_params=6)
    result = autofit(line, cfg)
    rows = {r.model_id: r for r in result.ranked}
    assert rows["poly1"].overfitting_note is None
    note = rows["poly5"].overfitting_note
    assert note is not None
    assert "poly1" in note or "poly2" in note  # names a much simpler candidate
    assert "noise" in note


def test_partial_ranking_of_streamed_outcomes_matches_the_final_ranking() -> None:
    cfg = AutoFitConfig(
        workers=1, n_starts=3, cv_folds=4, families=("exponential", "saturation"), curve_top_n=0
    )
    seen = []
    snapshots = []
    for outcome in iter_autofit(data(), cfg):
        seen.append(outcome)
        snapshots.append(partial_ranking(seen, RankBy.BEST_OVERALL))
    final = autofit(data(), cfg)
    assert snapshots[-1] == final.ranked
    assert len(snapshots[0]) <= 1
    assert [len(s) for s in snapshots] == sorted(len(s) for s in snapshots)
