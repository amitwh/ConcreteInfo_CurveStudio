"""Auto-fit through the real spawn process pool (slow: workers are fresh interpreters)."""

import os
import threading
import time
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    CancelToken,
    CandidateJob,
    CandidateOutcome,
    CustomRef,
    PolynomialRef,
    RankBy,
    RegisteredRef,
    autofit,
    iter_autofit,
    run_candidate,
)
from curvestudio.discovery.engine import stream_pool
from curvestudio.models import CustomModelDefinition, get_model

pytestmark = [pytest.mark.slow, pytest.mark.real_plugins]


def noisy_exp(n: int = 40, seed: int = 21) -> XY:
    x = np.linspace(0.0, 6.0, n)
    rng = np.random.default_rng(seed)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, n)
    return XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint="pool-test")


def comparable(result: Any) -> list[dict[str, Any]]:
    """Everything numeric about the ranked rows (no timings, no timestamps)."""
    return [r.model_dump() for r in result.ranked]


def test_pool_matches_in_process() -> None:
    mine = CustomModelDefinition(id="my_decay", name="My decay", expression="a*exp(-b*x) + c")
    base: dict[str, Any] = {
        "n_starts": 3,
        "cv_folds": 4,
        "families": ("exponential", "polynomial"),
        "max_params": 3,
        "custom_models": (mine,),
        "curve_top_n": 2,
    }
    inline = autofit(noisy_exp(), AutoFitConfig(workers=1, **base))
    pooled = autofit(noisy_exp(), AutoFitConfig(workers=2, **base))
    assert inline.ranked, "nothing converged"
    assert comparable(pooled) == comparable(inline)
    assert pooled.unconverged == inline.unconverged
    assert pooled.exact_fits == inline.exact_fits
    assert [o.candidate_id for o in pooled.outcomes] == [o.candidate_id for o in inline.outcomes]
    assert pooled.curves == inline.curves


def test_plugins_load_in_workers() -> None:
    cfg = AutoFitConfig(workers=2, n_starts=2, cv_folds=3, families=("exponential",), max_params=3)
    result = autofit(noisy_exp(), cfg)
    plugin_ids = [r.model_id for r in result.ranked if r.model_id not in _core_ids()]
    assert plugin_ids, "no library model was fitted inside a spawned worker"


def _core_ids() -> set[str]:
    from curvestudio.models.starter import STARTER_MODELS

    return {m.id for m in STARTER_MODELS}


def _job(ref: Any) -> CandidateJob:
    xy = noisy_exp()
    return CandidateJob(xy.x, xy.y, None, "x", "y", xy.fingerprint, ref, 0, 2, 20.0, 3)


def crash_on_marker(job: CandidateJob) -> CandidateOutcome:
    """Module-level so a spawned worker can import it: kills the worker for one candidate."""
    if job.ref.id == "poly2":
        os._exit(13)
    return run_candidate(job)


def test_a_worker_that_dies_fails_its_candidate_and_the_run_continues() -> None:
    jobs = [_job(PolynomialRef(degree=d)) for d in (1, 2, 3)]
    jobs += [_job(RegisteredRef(model_id="exp_offset"))]
    outcomes = list(stream_pool(jobs, workers=1, cancel=None, run=crash_on_marker))
    by_id = {o.candidate_id: o for o in outcomes}
    assert set(by_id) == {"poly1", "poly2", "poly3", "exp_offset"}
    assert by_id["poly2"].outcome == "failed"
    assert "worker" in by_id["poly2"].reason
    assert by_id["exp_offset"].outcome == "converged"


def slow_job(job: CandidateJob) -> CandidateOutcome:
    time.sleep(30.0)
    return run_candidate(job)


def test_cancel_stops_a_pool_with_running_workers_quickly() -> None:
    token = CancelToken()
    jobs = [_job(PolynomialRef(degree=d)) for d in (1, 2, 3, 4)]
    timer = threading.Timer(1.5, token.cancel)
    started = time.perf_counter()
    timer.start()
    try:
        assert list(stream_pool(jobs, workers=2, cancel=token, run=slow_job)) == []
    finally:
        timer.cancel()
    assert token.cancelled
    assert time.perf_counter() - started < 20.0  # the jobs sleep 30 s: workers were terminated


def test_cancel_mid_run_returns_partial_result() -> None:
    token = CancelToken()
    cfg = AutoFitConfig(workers=2, n_starts=2, cv_folds=3, families=("exponential", "polynomial"))
    result = autofit(noisy_exp(), cfg, cancel=token, on_outcome=lambda o: token.cancel())
    assert result.cancelled
    assert any("cancel" in w for w in result.warnings)


def test_iter_autofit_closed_early_shuts_the_pool_down() -> None:
    cfg = AutoFitConfig(workers=2, n_starts=2, cv_folds=3, families=("exponential", "polynomial"))
    stream = iter_autofit(noisy_exp(), cfg)
    first = next(stream)
    assert isinstance(first, CandidateOutcome)
    stream.close()  # must not hang


def test_full_default_run_on_30_points_finishes_and_records_timing() -> None:
    xy = noisy_exp(30)
    started = time.perf_counter()
    result = autofit(xy, AutoFitConfig())
    wall = time.perf_counter() - started
    print(
        f"\nAuto-fit default run: {len(result.outcomes)} candidates, "
        f"{len(result.ranked)} ranked, {wall:.1f} s wall, {result.elapsed_seconds:.1f} s recorded"
    )
    assert not result.cancelled
    assert result.elapsed_seconds > 0.0
    assert len(result.outcomes) > 50
    assert all(o.elapsed_seconds >= 0.0 for o in result.outcomes)


def _truth_rank(truth: str, seed: int) -> int:
    """Rank of the generating model on seeded data made from its own library example."""
    spec = get_model(truth)
    assert spec.example is not None
    x = np.linspace(spec.example.x_range[0], spec.example.x_range[1], 60)
    clean = spec.evaluate(x, **spec.example.params)
    rng = np.random.default_rng(seed)
    y = clean + rng.normal(0.0, 0.02 * float(np.ptp(clean)), x.size)
    xy = XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint=f"truth-{truth}-{seed}")
    result = autofit(xy, AutoFitConfig(rank_by=RankBy.BEST_OVERALL))
    best = next(r for r in result.ranked if r.model_id == truth or truth in r.same_curve_as)
    print(f"\n{truth} seed {seed}: rank {best.rank} of {len(result.ranked)} candidates")
    return best.rank


@pytest.mark.parametrize("truth", ["exp_decay_offset", "logistic3", "michaelis_menten", "gaussian"])
def test_known_truth_ranks_in_the_top_three_with_the_real_library(truth: str) -> None:
    """60 points, 2 % noise, the full library, three noise draws per generating model.

    One draw in a dozen puts a flexible model well ahead of the truth by chance (an AICc
    gap of 5-8 units from noise alone), so the claim is "usually", not "always": at least
    two of three draws. Over 12 draws for each of five library models the generating model
    was in the top three in 56 of 60 runs (see ``ranking._tie_order``).
    """
    ranks = [_truth_rank(truth, seed) for seed in (100, 101, 102)]
    assert sum(1 for r in ranks if r <= 3) >= 2, ranks


def test_custom_candidate_runs_in_a_worker() -> None:
    mine = CustomModelDefinition(id="my_decay", name="My decay", expression="a*exp(-b*x) + c")
    cfg = AutoFitConfig(
        workers=2, n_starts=2, cv_folds=3, families=("polynomial",), custom_models=(mine,)
    )
    result = autofit(noisy_exp(), cfg)
    assert "my_decay" in [r.model_id for r in result.ranked[:3]]
    assert CustomRef(definition=mine).id == "my_decay"
