"""Auto-fit hardening: exact data, degenerate optima, the parameter cap, staging, large data."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitError,
    AutoFitResult,
    CandidateOutcome,
    RankBy,
    RegisteredRef,
    autofit,
    iter_autofit,
    partial_ranking,
)
from curvestudio.discovery.result import _overfitting_note
from curvestudio.discovery.types import CandidateMetrics, RankedFit
from curvestudio.discovery.worker import CandidateJob, run_candidate
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models


def make_xy(x: Any, y: Any, weights: Any = None, fingerprint: str = "hardening") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=fingerprint,
    )


def config(**overrides: Any) -> AutoFitConfig:
    base: dict[str, Any] = {"workers": 1, "n_starts": 4, "cv_folds": 4}
    base.update(overrides)
    return AutoFitConfig(**base)


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()


def ids(rows: Any) -> list[str]:
    return [r.model_id for r in rows]


# -- C1: noiseless data --------------------------------------------------------------------


def test_noiseless_line_ranks_the_straight_line_first() -> None:
    x = np.linspace(0.0, 10.0, 20)
    result = autofit(make_xy(x, 2.0 * x + 1.0), config(max_params=4))
    top = result.ranked[0]
    assert top.model_id == "poly1"
    assert top.status == "exact_fit"
    assert top.metrics.k == 2
    assert top.metrics.aicc is None and top.delta_aicc is None and top.akaike_weight is None
    assert "poly1" in result.exact_fits
    # every exact row sits above every non-exact row, and none has an information criterion
    statuses = [r.status for r in result.ranked]
    assert statuses == sorted(statuses, key=lambda s: s != "exact_fit")
    assert all(r.metrics.aicc is None for r in result.ranked if r.status == "exact_fit")
    assert any("noiseless" in w for w in result.warnings)


@pytest.mark.real_plugins
def test_noiseless_exponential_headline_is_the_three_parameter_model() -> None:
    x = np.linspace(0.0, 6.0, 40)
    result = autofit(
        make_xy(x, 3.0 * np.exp(-0.8 * x) + 1.0), config(families=("exponential",), n_starts=3)
    )
    top = result.ranked[0]
    assert top.metrics.k == 3
    assert top.model_id in {"exp_decay_offset", "exp_offset"}
    assert top.status == "exact_fit" and top.metrics.aicc is None
    # the four-parameter near-copy is demoted behind it, not tied with it
    by_id = {r.model_id: r for r in result.ranked}
    copy = by_id["biexponential_difference_offset"]
    assert copy.rank > top.rank
    assert copy.redundant_with in {"exp_decay_offset", "exp_offset"} or copy.degenerate


def test_exactness_is_a_property_of_the_dataset_not_of_one_candidate() -> None:
    # tiny but real noise: the data are not exact, so every row keeps its AICc
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 10.0, 40)
    result = autofit(make_xy(x, 2.0 * x + 1.0 + rng.normal(0.0, 0.01, 40)), config(max_params=3))
    assert result.exact_fits == ()
    assert all(r.metrics.aicc is not None for r in result.ranked)


def test_same_curve_tolerance_is_relaxed_in_the_exact_regime() -> None:
    x = np.linspace(0.0, 10.0, 20)
    result = autofit(make_xy(x, 2.0 * x + 1.0), config(max_params=4))
    by_id = {r.model_id: r for r in result.ranked}
    # the exact quadratic fit and the line differ by ~1e-12, far inside 1e-6 of the range
    assert "poly1" in by_id["poly2"].same_curve_as
    assert by_id["poly2"].redundant_with == "poly1"


# -- C1: degenerate optima -----------------------------------------------------------------


def _ricker_like() -> ModelSpec:
    return ModelSpec(
        id="ricker_like",
        name="Ricker-like",
        category="peak",
        expression="a + c*x*exp(-b*x)",
        parameters=(Parameter("a"), Parameter("b", lower=0.0), Parameter("c")),
        initial_guess=lambda x, y: {"a": float(y[0]), "b": 0.1, "c": 1.0},
    )


def test_a_model_that_collapses_onto_a_line_is_flagged_degenerate_and_ranks_below() -> None:
    register(_ricker_like())
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 10.0, 24)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, 24)
    job = CandidateJob(
        x, y, None, "x", "y", "f", RegisteredRef(model_id="ricker_like"), 0, 4, 20, 4
    )
    outcome = run_candidate(job)
    assert outcome.fit is not None
    assert any(w.startswith("DEGENERATE_FIT") for w in outcome.fit.warnings)
    result = autofit(make_xy(x, y), config(families=("peak", "polynomial"), max_params=3))
    by_id = {r.model_id: r for r in result.ranked}
    assert by_id["ricker_like"].degenerate
    clean_ranks = [r.rank for r in result.ranked if not r.degenerate]
    assert by_id["ricker_like"].rank > max(clean_ranks)
    assert any("collapsed" in w for w in result.warnings)


# -- C2: constant data ---------------------------------------------------------------------


def test_autofit_refuses_constant_response_and_constant_x() -> None:
    x = np.linspace(0.0, 5.0, 12)
    with pytest.raises(AutoFitError) as info:
        autofit(make_xy(x, np.full(12, 5.0)), config())
    assert info.value.code == "CONSTANT_RESPONSE"
    with pytest.raises(AutoFitError) as info:
        iter_autofit(make_xy(x, np.zeros(12)), config())
    assert info.value.code == "CONSTANT_RESPONSE"
    with pytest.raises(AutoFitError) as info:
        autofit(make_xy(np.full(12, 3.0), x), config())
    assert info.value.code == "NO_X_VARIATION"


# -- I1: the parameter cap hides nothing ---------------------------------------------------


def _params_model(model_id: str, k: int) -> ModelSpec:
    names = [f"p{i}" for i in range(k)]
    return ModelSpec(
        id=model_id,
        name=model_id,
        category="dose_response",
        expression=" + ".join(f"{n}*x**{i}" for i, n in enumerate(names)),
        parameters=tuple(Parameter(n) for n in names),
        initial_guess=lambda x, y: dict.fromkeys(names, 0.5),
    )


def test_default_cap_follows_n_over_three_and_lists_every_capped_model() -> None:
    for k in (4, 5, 7):
        register(_params_model(f"model_{k}p", k))
    x = np.linspace(0.0, 4.0, 12)
    y = 1.0 + 2.0 * x + 0.5 * np.sin(3.0 * x)
    result = autofit(make_xy(x, y), config(include_polynomials=False))
    assert "model_4p" in {o.candidate_id for o in result.outcomes if o.outcome != "skipped"}
    for capped in ("model_5p", "model_7p"):
        assert result.skipped[capped] == (f"needs {capped[6]} parameters; cap is 4 for 12 points",)
    # 15 points: cap 5
    x15 = np.linspace(0.0, 4.0, 15)
    result15 = autofit(make_xy(x15, 1.0 + 2.0 * x15 + 0.5 * np.sin(3.0 * x15)), config())
    assert "model_5p" not in result15.skipped
    assert result15.skipped["model_7p"] == ("needs 7 parameters; cap is 5 for 15 points",)


def test_explicit_families_or_max_params_lift_the_default_cap() -> None:
    register(_params_model("model_7p", 7))
    x = np.linspace(0.0, 6.0, 30)
    y = 1.0 + x + 0.3 * np.sin(2.0 * x)
    base = autofit(make_xy(x, y), config(include_polynomials=False))
    assert "model_7p" in base.skipped  # default cap at 30 points is 6
    by_family = autofit(make_xy(x, y), config(families=("dose_response",)))
    assert "model_7p" not in by_family.skipped
    by_cap = autofit(make_xy(x, y), config(include_polynomials=False, max_params=7))
    assert "model_7p" not in by_cap.skipped


def test_more_than_fifty_capped_models_are_summarised_in_one_entry() -> None:
    for i in range(60):
        register(_params_model(f"wide_{i:02d}", 8))
    x = np.linspace(0.0, 6.0, 30)
    result = autofit(make_xy(x, 1.0 + x + 0.3 * np.sin(2.0 * x)), config(include_polynomials=False))
    entries = [k for k in result.skipped if k.startswith("wide_")]
    assert entries == []
    (summary,) = result.skipped["(parameter cap)"]
    assert "60 models" in summary and "cap is 6 for 30 points" in summary


# -- I2: Akaike deltas and weights ---------------------------------------------------------


def test_delta_aicc_and_akaike_weights_describe_the_whole_run() -> None:
    rng = np.random.default_rng(8)
    x = np.linspace(0.0, 6.0, 40)
    result = autofit(
        make_xy(x, 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, 40)),
        config(families=("exponential", "saturation", "polynomial")),
    )
    rows = [r for r in result.ranked if r.metrics.aicc is not None]
    assert min(r.delta_aicc for r in rows if r.delta_aicc is not None) == 0.0
    assert sum(r.akaike_weight or 0.0 for r in rows) == pytest.approx(1.0)
    for r in rows:
        assert r.delta_aicc == pytest.approx(
            r.metrics.aicc - min(x.metrics.aicc or 0 for x in rows)
        )
    flagged = [r for r in result.ranked if r.metrics.aicc is None]
    assert all(r.delta_aicc is None and r.akaike_weight is None for r in flagged)


def test_flagged_free_frequency_fit_does_not_beat_the_line_on_linear_noise() -> None:
    from curvestudio.models import register as reg

    reg(
        ModelSpec(
            id="sine_drift",
            name="Sine with drift",
            category="harmonic",
            expression="a + b*x + c*sin(w*x)",
            parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("w", lower=0.0)),
            initial_guess=lambda x, y: {"a": 0.0, "b": 1.0, "c": 0.1, "w": 2.0},
        )
    )
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 10.0, 20)
    result = autofit(
        make_xy(x, 2.0 * x + 1.0 + rng.normal(0.0, 0.3, 20)),
        config(families=("harmonic", "polynomial"), max_params=4, n_starts=8),
    )
    order = ids(result.ranked)
    sine = next(r for r in result.ranked if r.model_id == "sine_drift")
    if any(w.startswith(("MULTISTART_DISAGREE", "PARAMETER_AT_BOUND")) for w in sine.warnings):
        assert order.index("poly1") < order.index("sine_drift")


def _row(model_id: str, k: int, r2: float, delta: float | None, name: str = "") -> RankedFit:
    return RankedFit(
        rank=0,
        model_id=model_id,
        name=name or model_id,
        metrics=CandidateMetrics(k=k, n=30, r2=r2, aicc=1.0),
        delta_aicc=delta,
    )


def test_overfitting_note_on_the_aicc_winner_does_not_tell_it_to_compare_aicc() -> None:
    winner = _row("big", 6, 0.995, 0.0)
    simple = _row("line", 2, 0.994, 4.0)
    pool = [(winner, "centered"), (simple, "centered")]
    note = _overfitting_note(winner, pool, "centered")
    assert note is not None
    assert "lowest AICc" in note and "compare AICc" not in note and "line" in note
    other = _row("big2", 6, 0.995, 3.0)
    note2 = _overfitting_note(other, [(other, "centered"), (simple, "centered")], "centered")
    assert note2 is not None and "compare AICc" in note2


# -- SPEED: staged starts ------------------------------------------------------------------


def _exp_data(n: int = 60, seed: int = 4) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, n)
    return make_xy(x, 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, n))


def test_staged_starts_give_only_promising_candidates_the_full_multistart() -> None:
    cfg = config(families=("exponential", "saturation", "polynomial"), n_starts=6)
    staged = autofit(_exp_data(), cfg)
    full = autofit(_exp_data(), cfg.model_copy(update={"staged_starts": False}))
    stages = {o.candidate_id: o.stage for o in staged.outcomes if o.fit is not None}
    assert "screen" in stages.values() and "final" in stages.values()
    for o in staged.outcomes:
        if o.fit is not None and o.stage == "screen" and not o.candidate_id.startswith("poly"):
            assert o.fit.n_starts == 2
    assert all(o.stage == "final" for o in full.outcomes if o.fit is not None)
    # the headline is the same, and the top rows agree
    assert ids(staged.ranked[:5]) == ids(full.ranked[:5])
    assert set(ids(staged.ranked[:10])) >= set(ids(full.ranked[:10])) - {ids(full.ranked)[9]}


def test_staging_does_not_depend_on_completion_order() -> None:
    cfg = config(families=("exponential", "saturation"), n_starts=5)
    first = autofit(_exp_data(), cfg)
    second = autofit(_exp_data(), cfg)
    assert [o.stage for o in first.outcomes] == [o.stage for o in second.outcomes]
    assert [r.model_dump() for r in first.ranked] == [r.model_dump() for r in second.ranked]


def test_partial_ranking_counts_a_replaced_candidate_once() -> None:
    cfg = config(families=("exponential", "saturation"), n_starts=5, curve_top_n=0)
    seen: list[CandidateOutcome] = []
    for outcome in iter_autofit(_exp_data(), cfg):
        seen.append(outcome)
    assert len(seen) > len({o.candidate_id for o in seen})
    final = autofit(_exp_data(), cfg)
    assert partial_ranking(seen, RankBy.BEST_OVERALL) == final.ranked


# -- I7: very large data -------------------------------------------------------------------


def test_large_data_is_screened_on_a_subsample_and_the_best_refitted_on_all_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import curvestudio.discovery.engine as engine

    monkeypatch.setattr(engine, "REFIT_TOP", 2)
    rng = np.random.default_rng(6)
    n = 6000
    x = np.linspace(0.0, 6.0, n)
    xy = make_xy(x, 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, n), fingerprint="big")
    result = autofit(xy, config(families=("exponential", "saturation"), n_starts=3))
    assert result.subsample_n == 2000
    assert result.n_active == n
    assert any("subsample" in w for w in result.warnings)
    full = [r for r in result.ranked if r.metrics.n == n]
    sub = [r for r in result.ranked if r.metrics.n == 2000]
    assert full and sub
    assert ids(result.ranked[: len(full)]) == ids(full)  # full-data rows rank first
    assert "exp_decay_offset" in ids(full) or "exp_offset" in ids(full)
    assert len(full) <= 2 * len(RankBy)
    # per-point vectors only for the rows that were drawn
    with_vectors = {o.candidate_id for o in result.outcomes if o.fit is not None and o.fit.fitted}
    assert with_vectors == set(result.curves)
    assert len(result.curves) <= 5 * len(RankBy)
    restored = AutoFitResult.from_json(result.to_json(include_fits=False))
    assert restored.ranked == result.ranked
    assert len(result.to_json(include_fits=False)) < 3_000_000


def test_subsample_is_deterministic_and_keeps_the_ends() -> None:
    from curvestudio.discovery.engine import subsample

    x = np.linspace(0.0, 1.0, 9000)
    a, b = subsample(make_xy(x, x)), subsample(make_xy(x, x))
    assert np.array_equal(a.x, b.x) and a.x[0] == 0.0 and a.x[-1] == 1.0
    assert a.x.size == 2000


# -- I6 and minor warnings -----------------------------------------------------------------


def test_extreme_scale_warns_and_does_not_crash() -> None:
    x = np.linspace(0.0, 6.0, 30)
    y = 3.0 * np.exp(-0.8 * x) + 1.0
    rng = np.random.default_rng(0)
    for scale_x, scale_y in ((1e9, 1.0), (1.0, 1e9), (1e-9, 1e-9), (1e9, 1e9)):
        noisy = y + rng.normal(0.0, 0.02, 30)
        result = autofit(
            make_xy(x * scale_x, noisy * scale_y), config(families=("exponential", "polynomial"))
        )
        assert any("scale is extreme" in w for w in result.warnings), (scale_x, scale_y)
        assert "NaN" not in result.to_json()


def test_a_gross_outlier_in_the_data_raises_a_warning() -> None:
    rng = np.random.default_rng(9)
    x = np.linspace(0.0, 10.0, 40)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, 40)
    y[17] += 8.0
    result = autofit(make_xy(x, y), config(max_params=3))
    assert any("possible outlier" in w and "studentised" in w for w in result.warnings)
    clean = autofit(make_xy(x, 2.0 * x + 1.0 + rng.normal(0.0, 0.1, 40)), config(max_params=3))
    assert not any("possible outlier" in w for w in clean.warnings)


# -- free-frequency parameters count as extra effective parameters ------------


def _spec(model_id: str, expression: str, names: tuple[str, ...]) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=model_id,
        category="harmonic",
        expression=expression,
        parameters=tuple(Parameter(n) for n in names),
        initial_guess=lambda x, y: dict.fromkeys(names, 1.0),
    )


def test_shape_parameters_are_free_frequencies_only() -> None:
    from curvestudio.discovery.worker import shape_parameter_count

    assert shape_parameter_count(_spec("a", "a + b*sin(c*x)", ("a", "b", "c"))) == 1
    assert (
        shape_parameter_count(_spec("b", "a*cos(c*x + d)", ("a", "c", "d"))) == 1
    )  # not the phase
    assert shape_parameter_count(_spec("c", "a*x**b", ("a", "b"))) == 0  # a power law is fine
    assert shape_parameter_count(_spec("d", "a*exp(-b*x) + c", ("a", "b", "c"))) == 0  # a rate
    assert (
        shape_parameter_count(
            _spec("e", "a + b*sin(c*x) + d*sin((c+g)*x)", ("a", "b", "c", "d", "g"))
        )
        == 2
    )
    assert shape_parameter_count(_spec("f", "a + b*x**2", ("a", "b"))) == 0  # fixed exponent
    assert shape_parameter_count(None) == 0


def test_a_free_frequency_costs_an_effective_parameter_in_the_ranking_only() -> None:
    from curvestudio.discovery.ranking import rank

    plain = _row("plain", 3, 0.99, 0.0)
    sine = _row("sine", 3, 0.99, 0.0).model_copy(update={"shape_parameters": 1})
    sine = sine.model_copy(
        update={"metrics": sine.metrics.model_copy(update={"aicc": plain.metrics.aicc - 1.5})}
    )
    ranked = rank([sine, plain], RankBy.BEST_OVERALL)
    assert ids(ranked) == ["plain", "sine"]  # 1.5 better AICc does not pay for the frequency
    assert ranked[1].metrics.aicc == plain.metrics.aicc - 1.5  # the reported value is untouched
