"""Tests for the Auto-fit engine, run in-process (``workers=1``) so they stay fast."""

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitError,
    AutoFitResult,
    CancelToken,
    CandidateOutcome,
    RankBy,
    autofit,
    iter_autofit,
)
from curvestudio.models import (
    CustomModelDefinition,
    ModelSpec,
    Parameter,
    register,
)
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models


def make_xy(x: Any, y: Any, weights: Any = None) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="engine-test",
    )


def exp_decay_data(n: int = 40, sd: float = 0.05, seed: int = 7) -> XY:
    """Seeded ``3*exp(-0.8*x) + 1`` plus Gaussian noise."""
    x = np.linspace(0.0, 6.0, n)
    rng = np.random.default_rng(seed)
    return make_xy(x, 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, sd, n))


def config(**overrides: Any) -> AutoFitConfig:
    base: dict[str, Any] = {"workers": 1, "n_starts": 3, "cv_folds": 4}
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


def test_known_truth_small_catalogue() -> None:
    result = autofit(exp_decay_data(), config(families=("exponential", "saturation")))
    assert result.ranked, "nothing was ranked"
    assert "exp_decay_offset" in ids(result.ranked[:3])
    assert result.n_active == 40
    assert result.dataset_fingerprint == "engine-test"
    assert not result.cancelled
    for row in result.ranked:
        assert row.metrics.aicc is not None
        assert row.equation_fitted
        assert row.category in {"exponential", "saturation"}
    # every candidate has one outcome, ordered by id
    assert ids_of_outcomes(result) == sorted(ids_of_outcomes(result))


def ids_of_outcomes(result: AutoFitResult) -> list[str]:
    return [o.candidate_id for o in result.outcomes]


def test_equivalent_models_tie_on_the_same_curve() -> None:
    result = autofit(exp_decay_data(), config(families=("exponential",)))
    by_id = {r.model_id: r for r in result.ranked}
    assert "exp_offset" in by_id["exp_decay_offset"].same_curve_as
    assert by_id["exp_offset"].rank == by_id["exp_decay_offset"].rank


def test_non_converged_never_ranked_and_listed_with_reason() -> None:
    # a and b only enter as a + b: the Jacobian is rank deficient, so no unique fit exists
    register(
        ModelSpec(
            id="degenerate_pair",
            name="Degenerate pair",
            category="kinetics",
            expression="(a + b)*x",
            parameters=(Parameter("a"), Parameter("b")),
            initial_guess=lambda x, y: {"a": 1.0, "b": 1.0},
        )
    )
    x = np.linspace(0.0, 5.0, 25)
    result = autofit(
        make_xy(x, 2.0 * x + 1.0 + 0.01 * np.cos(7 * x)), config(families=("kinetics",))
    )
    assert "degenerate_pair" in result.unconverged
    assert result.unconverged["degenerate_pair"]
    assert "degenerate_pair" not in ids(result.ranked)
    (outcome,) = [o for o in result.outcomes if o.candidate_id == "degenerate_pair"]
    assert outcome.outcome == "not_converged"
    assert outcome.metrics is None


def test_skipped_models_listed_with_reasons() -> None:
    x = np.linspace(-1.0, 5.0, 30)  # not strictly positive: ln and power models cannot apply
    result = autofit(make_xy(x, 0.5 * x + 1.0 + 0.1 * np.sin(5 * x)), config())
    assert "log_natural" in result.skipped
    assert result.skipped["log_natural"]
    assert "log_natural" not in ids(result.ranked)
    (outcome,) = [o for o in result.outcomes if o.candidate_id == "log_natural"]
    assert outcome.outcome == "skipped"
    assert outcome.fit is None
    assert outcome.reason


def test_models_with_too_many_parameters_are_skipped_not_fitted() -> None:
    result = autofit(
        make_xy(np.arange(7.0), [1.0, 2.2, 2.9, 4.1, 5.0, 5.9, 7.2]), config(max_params=5)
    )
    assert result.skipped  # k > n - 3 candidates are listed, not silently dropped, when listed
    for cid, reasons in result.skipped.items():
        assert reasons, cid
    for row in result.ranked:
        assert row.metrics.k <= 7 - 3


def test_too_few_points_refuses_the_run() -> None:
    with pytest.raises(AutoFitError) as info:
        autofit(make_xy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), config())
    assert info.value.code == "TOO_FEW_POINTS"
    zero_weights = make_xy(np.arange(10.0), np.arange(10.0), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(AutoFitError) as info:
        iter_autofit(zero_weights, config())  # raised when called, not when first iterated
    assert info.value.code == "TOO_FEW_POINTS"


def test_exact_fit_listed_separately_not_ranked_by_aicc() -> None:
    x = np.arange(20.0)
    result = autofit(
        make_xy(x, 1.0 + 2.0 * x + 3.0 * x**2), config(families=("polynomial",), max_params=5)
    )
    assert "poly2" in result.exact_fits
    assert not set(result.exact_fits) & set(ids(result.ranked))
    for cid in result.exact_fits:
        (outcome,) = [o for o in result.outcomes if o.candidate_id == cid]
        assert outcome.outcome == "exact_fit"
        assert outcome.metrics is not None
        assert outcome.metrics.aicc is None
        assert outcome.metrics.bic is None
        assert outcome.metrics.r2 == pytest.approx(1.0)
    assert any("exact" in w for w in result.warnings)
    # under an ordering that needs no information criterion the exact fits are ranked
    by_rmse = result.ranked_by(RankBy.RMSE)
    assert set(result.exact_fits) <= set(ids(by_rmse))
    assert all(r.status == "exact_fit" for r in by_rmse if r.model_id in result.exact_fits)
    assert set(ids(by_rmse[: len(result.exact_fits)])) == set(result.exact_fits)


def test_same_seed_same_numbers() -> None:
    cfg = config(families=("exponential", "saturation"))
    first = autofit(exp_decay_data(), cfg)
    second = autofit(exp_decay_data(), cfg)
    assert [r.model_dump() for r in first.ranked] == [r.model_dump() for r in second.ranked]
    assert first.exact_fits == second.exact_fits
    assert first.unconverged == second.unconverged


def test_different_seed_changes_cv_but_not_the_catalogue() -> None:
    a = autofit(exp_decay_data(), config(families=("exponential",), seed=1))
    b = autofit(exp_decay_data(), config(families=("exponential",), seed=2))
    assert ids_of_outcomes(a) == ids_of_outcomes(b)


def test_rank_by_resort_matches_a_fresh_run_with_that_rank_by() -> None:
    cfg = config(families=("exponential", "saturation", "polynomial"), max_params=3)
    base = autofit(exp_decay_data(), cfg)
    for rank_by in RankBy:
        fresh = autofit(exp_decay_data(), cfg.model_copy(update={"rank_by": rank_by}))
        assert base.ranked_by(rank_by) == fresh.ranked, rank_by
        assert fresh.ranked == fresh.ranked_by(rank_by)


def test_cancel_returns_partial_result_with_flag() -> None:
    token = CancelToken()
    seen: list[str] = []

    def on_outcome(outcome: CandidateOutcome) -> None:
        seen.append(outcome.candidate_id)
        token.cancel()

    cfg = config(families=("exponential", "saturation", "polynomial"), max_params=4)
    result = autofit(exp_decay_data(), cfg, cancel=token, on_outcome=on_outcome)
    assert token.cancelled
    assert result.cancelled
    assert len(seen) == 1
    assert len(result.ranked) + len(result.exact_fits) + len(result.unconverged) <= 1
    unfitted = [o for o in result.outcomes if o.outcome == "skipped" and "cancel" in o.reason]
    assert unfitted, "candidates that were never fitted must be listed with the reason"
    assert any("cancel" in w for w in result.warnings)


def test_cancel_before_start_fits_nothing() -> None:
    token = CancelToken()
    token.cancel()
    result = autofit(exp_decay_data(), config(families=("exponential",)), cancel=token)
    assert result.cancelled
    assert result.ranked == ()


def test_iter_autofit_streams_every_attempted_candidate_once() -> None:
    cfg = config(families=("exponential", "saturation"))
    streamed = list(iter_autofit(exp_decay_data(), cfg))
    assert all(isinstance(o, CandidateOutcome) for o in streamed)
    final = autofit(exp_decay_data(), cfg)
    attempted = sorted(o.candidate_id for o in final.outcomes if o.outcome != "skipped")
    assert sorted(o.candidate_id for o in streamed) == attempted
    assert len({o.candidate_id for o in streamed}) == len(streamed)


def test_on_outcome_sees_each_attempted_candidate() -> None:
    seen: list[str] = []
    result = autofit(
        exp_decay_data(),
        config(families=("exponential",)),
        on_outcome=lambda o: seen.append(o.candidate_id),
    )
    assert sorted(seen) == sorted(o.candidate_id for o in result.outcomes if o.outcome != "skipped")


def test_custom_model_candidate_runs_and_is_ranked() -> None:
    mine = CustomModelDefinition(id="my_decay", name="My decay", expression="a*exp(-b*x) + c")
    result = autofit(exp_decay_data(), config(families=("polynomial",), custom_models=(mine,)))
    assert "my_decay" in ids(result.ranked[:3])
    row = next(r for r in result.ranked if r.model_id == "my_decay")
    assert row.name == "My decay"
    assert row.category == "custom"
    assert row.metrics.k == 3


def test_custom_model_with_a_clashing_or_invalid_definition_is_refused() -> None:
    clash = CustomModelDefinition(id="exp_offset", name="Clash", expression="a*x + b")
    with pytest.raises(AutoFitError) as info:
        autofit(exp_decay_data(), config(custom_models=(clash,)))
    assert info.value.code == "DUPLICATE_MODEL_ID"
    broken = CustomModelDefinition(id="broken", name="Broken", expression="a*x**")
    with pytest.raises(AutoFitError) as info:
        autofit(exp_decay_data(), config(custom_models=(broken,)))
    assert info.value.code == "INVALID_CUSTOM_MODEL"


def test_custom_model_the_data_cannot_support_is_skipped_with_reason() -> None:
    wide = CustomModelDefinition(
        id="wide", name="Wide", expression="a*x + b*x**2 + c*x**3 + d*x**4 + e*x**5"
    )
    result = autofit(
        make_xy(np.arange(6.0), [1.0, 3.1, 5.2, 6.9, 9.1, 11.0]),
        config(families=("polynomial",), custom_models=(wide,)),
    )
    assert "wide" in result.skipped


def test_time_limited_candidate_is_timed_out_not_ranked() -> None:
    spec = ModelSpec(
        id="slow_exp",
        name="Slow exponential",
        category="kinetics",
        expression="a*exp(b*x)",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.0, "b": 0.1},
    )
    register(spec)
    real_evaluate = spec.evaluate

    def slow_evaluate(x: Any, **params: float) -> Any:
        time.sleep(0.03)
        return real_evaluate(x, **params)

    object.__setattr__(spec, "evaluate", slow_evaluate)
    x = np.linspace(0.0, 5.0, 25)
    result = autofit(
        make_xy(x, 5.0 * np.sin(3.0 * x)), config(families=("kinetics",), time_limit=0.1)
    )
    (outcome,) = [o for o in result.outcomes if o.candidate_id == "slow_exp"]
    assert outcome.outcome == "timed_out"
    assert "time limit" in outcome.reason
    assert "slow_exp" not in ids(result.ranked)
    assert "slow_exp" in result.unconverged
    assert "slow_exp" in result.timed_out
    assert any("time limit" in w for w in result.warnings)
