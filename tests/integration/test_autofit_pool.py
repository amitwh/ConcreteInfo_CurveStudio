"""Auto-fit through the real spawn process pool (slow: workers are fresh interpreters)."""

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
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
from curvestudio.discovery.pool import AutoFitPool, default_deadline
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


class _ScriptedExecutor:
    """A stand-in for a process pool: scripted submits, the rest run in this process.

    The script is shared by every executor a pool builds; each ``"broken"`` entry makes the
    next submit raise :class:`BrokenProcessPool`, which is what a real pool does once a worker
    has died (the failure is reported asynchronously, so it surfaces at the next submit).
    """

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.ran: list[CandidateJob] = []
        self.terminations = 0

    def submit(self, fn: Any, *args: Any) -> Future[CandidateOutcome]:
        if self.script and self.script.pop(0) == "broken":
            raise BrokenProcessPool("simulated: a worker died between submitting and running")
        future: Future[CandidateOutcome] = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # a raising job is reported as an outcome, never propagated
            future.set_exception(exc)
        self.ran.append(args[0])
        return future

    def terminate_workers(self) -> None:
        self.terminations += 1


class _ScriptedPool(AutoFitPool):
    """An :class:`AutoFitPool` whose executors are scripted: no worker processes are started."""

    def __init__(self, script: list[str], workers: int = 2) -> None:
        super().__init__(workers)
        self.script = script
        self.executors: list[_ScriptedExecutor] = []

    def executor(self) -> Any:
        if self._executor is None:
            self._executor = _ScriptedExecutor(self.script)
            self.executors.append(self._executor)
        return self._executor


def test_a_broken_submit_is_isolated_and_every_job_reports() -> None:
    """A submit raising BrokenProcessPool must not end the run.

    The pool dies asynchronously, so the failure surfaces at the next submit: the job that
    could not be submitted and every job already in flight are re-run as suspects in a fresh
    executor, and every job still reports exactly once.
    """
    jobs = [_job(PolynomialRef(degree=d)) for d in (1, 2, 3)]
    jobs += [_job(RegisteredRef(model_id="exp_offset"))]
    pool = _ScriptedPool(["run", "broken"])  # poly1 is in flight when the pool dies
    outcomes = list(stream_pool(jobs, workers=2, cancel=None, pool=pool))
    assert sorted(o.candidate_id for o in outcomes) == ["exp_offset", "poly1", "poly2", "poly3"]
    assert all(o.outcome == "converged" for o in outcomes), [
        (o.candidate_id, o.outcome, o.reason) for o in outcomes
    ]


def test_a_broken_submit_discards_the_dead_pool_and_a_later_run_still_works() -> None:
    jobs = [_job(PolynomialRef(degree=d)) for d in (1, 2)]
    pool = _ScriptedPool(["broken"])
    first = list(stream_pool(jobs, workers=2, cancel=None, pool=pool))
    assert {o.candidate_id for o in first} == {"poly1", "poly2"}
    # the dead executor was terminated and a fresh one was built for the rest of the run
    assert [len(pool.executors), pool.executors[0].terminations] == [2, 1]
    again = list(stream_pool(jobs, workers=2, cancel=None, pool=pool))
    assert {o.candidate_id for o in again} == {"poly1", "poly2"}
    assert all(o.outcome == "converged" for o in again)


def test_a_pool_that_never_recovers_ends_with_lost_outcomes_not_an_exception() -> None:
    jobs = [_job(PolynomialRef(degree=d)) for d in (1, 2, 3)]
    pool = _ScriptedPool(["broken"] * 100)
    outcomes = list(stream_pool(jobs, workers=2, cancel=None, pool=pool))
    assert sorted(o.candidate_id for o in outcomes) == ["poly1", "poly2", "poly3"]
    assert all(o.outcome == "failed" for o in outcomes)
    assert all("kept dying" in o.reason for o in outcomes)


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
    # x starts at 0: models that cannot take x = 0 are skipped with a reason, never failed
    assert not [cid for cid in result.failed if cid.startswith("engineering_")], result.failed
    assert any(c.startswith("engineering_") for c in result.skipped)
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


def hang_on_marker(job: CandidateJob) -> CandidateOutcome:
    """Module-level so a spawned worker can import it: one candidate never comes back."""
    if job.ref.id == "poly2":
        time.sleep(3600.0)
    return run_candidate(job)


def test_the_watchdog_stops_a_hung_candidate_and_the_run_continues() -> None:
    jobs = [_job(PolynomialRef(degree=d)) for d in (1, 2, 3)]
    jobs += [_job(RegisteredRef(model_id="exp_offset"))]
    started = time.perf_counter()
    outcomes = list(
        stream_pool(jobs, workers=2, cancel=None, run=hang_on_marker, deadline_of=lambda job: 4.0)
    )
    wall = time.perf_counter() - started
    by_id = {o.candidate_id: o for o in outcomes}
    assert set(by_id) == {"poly1", "poly2", "poly3", "exp_offset"}
    assert by_id["poly2"].outcome == "timed_out"
    assert "watchdog" in by_id["poly2"].reason
    for other in ("poly1", "poly3", "exp_offset"):
        assert by_id[other].outcome == "converged", by_id[other]
    assert wall < 60.0


def test_default_watchdog_deadline_is_twice_the_time_limit_plus_five_seconds() -> None:
    assert default_deadline(_job(PolynomialRef(degree=1))) == 45.0  # time_limit is 20 s


def test_a_warm_pool_is_reused_across_runs_and_shut_down_explicitly() -> None:
    cfg = AutoFitConfig(n_starts=2, cv_folds=3, families=("exponential",), max_params=3)
    with AutoFitPool(2) as pool:
        first = autofit(noisy_exp(), cfg, pool=pool)
        pids = pool.worker_pids()
        assert pids and pool.running
        second = autofit(noisy_exp(), cfg, pool=pool)
        assert pool.worker_pids() == pids  # the same interpreters served both runs
        assert comparable(first) == comparable(second)
    assert not pool.running
    for pid in pids:
        assert not _alive(pid)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def test_a_cancelled_run_leaves_no_busy_workers_in_the_warm_pool() -> None:
    token = CancelToken()
    cfg = AutoFitConfig(n_starts=2, cv_folds=3, families=("exponential", "polynomial"))
    with AutoFitPool(2) as pool:
        result = autofit(
            noisy_exp(), cfg, cancel=token, on_outcome=lambda o: token.cancel(), pool=pool
        )
        assert result.cancelled
        assert not pool.running  # discarded: busy workers are never handed to the next run
        again = autofit(noisy_exp(), cfg, pool=pool)
        assert not again.cancelled and again.ranked


_ORPHAN_SCRIPT = """
import sys, time
from curvestudio.discovery.pool import AutoFitPool, stream_pool
from curvestudio.discovery.worker import CandidateJob
from curvestudio.discovery import PolynomialRef
import numpy as np

def sleeper(job):
    time.sleep(600)

if __name__ == "__main__":
    x = np.linspace(0, 1, 10)
    jobs = [CandidateJob(x, x, None, "x", "y", "f", PolynomialRef(degree=1), 0, 1, 20.0, None)] * 2
    pool = AutoFitPool(2)
    stream = stream_pool(jobs, 2, None, run=sleeper, pool=pool)
    import threading
    threading.Thread(target=lambda: list(stream), daemon=True).start()
    while len(pool.worker_pids()) < 2:
        time.sleep(0.2)
    time.sleep(3)
    print(*pool.worker_pids(), flush=True)
    time.sleep(600)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and SIGKILL")
def test_workers_exit_when_the_parent_is_killed(tmp_path: Any) -> None:
    script = tmp_path / "orphan_parent.py"
    script.write_text(_ORPHAN_SCRIPT)
    parent = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True, cwd=str(tmp_path)
    )
    try:
        assert parent.stdout is not None
        pids = [int(p) for p in parent.stdout.readline().split()]
        assert len(pids) == 2 and all(_alive(p) for p in pids)
        parent.kill()  # SIGKILL: nothing in the parent gets to clean up
        parent.wait(timeout=10)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.2)
        assert not any(_alive(p) for p in pids), "workers outlived their parent by 10 s"
    finally:
        parent.kill()


def _reference(n: int) -> XY:
    x = np.linspace(0.0, 6.0, n)
    rng = np.random.default_rng(1)
    return XY(x, 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, n), None, "x", "y", f"ref{n}")


@pytest.mark.parametrize("n", [30, 100])
def test_staged_starts_keep_the_top_ten_of_the_full_multistart_run(n: int) -> None:
    started = time.perf_counter()
    staged = autofit(_reference(n), AutoFitConfig())
    staged_wall = time.perf_counter() - started
    full = autofit(_reference(n), AutoFitConfig(staged_starts=False))
    top_staged = {r.model_id for r in staged.ranked[:10]}
    top_full = {r.model_id for r in full.ranked[:10]}
    print(
        f"\nn={n}: staged {staged_wall:.1f} s, full {full.elapsed_seconds:.1f} s, "
        f"top-10 overlap {len(top_staged & top_full)}/10"
    )
    assert len(top_staged & top_full) >= 9
    assert [r.model_id for r in staged.ranked[:3]] == [r.model_id for r in full.ranked[:3]]


def test_noiseless_data_headline_is_the_generating_model_with_the_full_library() -> None:
    x = np.linspace(0.0, 6.0, 40)
    exp = autofit(
        XY(x, 3.0 * np.exp(-0.8 * x) + 1.0, None, "x", "y", "noiseless-exp"), AutoFitConfig()
    )
    top = exp.ranked[0]
    assert top.model_id in {"exp_decay_offset", "exp_offset"}, [r.model_id for r in exp.ranked[:5]]
    assert top.status == "exact_fit" and top.metrics.aicc is None and top.metrics.k == 3
    xl = np.linspace(0.0, 10.0, 20)
    line = autofit(XY(xl, 2.0 * xl + 1.0, None, "x", "y", "noiseless-line"), AutoFitConfig())
    assert line.ranked[0].model_id == "poly1", [r.model_id for r in line.ranked[:5]]
    assert line.ranked[0].status == "exact_fit"
    # nothing degenerate or free-frequency leads a clean headline
    assert not any(r.degenerate for r in line.ranked[:3])


def test_small_samples_still_try_the_four_and_five_parameter_models() -> None:
    for n, family_model in ((12, "dose_response_4pl"), (15, "logistic4")):
        x = np.linspace(0.5, 10.0, n)  # positive: the dose-response models need x > 0
        rng = np.random.default_rng(3)
        y = 1.0 + 4.0 / (1.0 + np.exp(-(x - 5.0))) + rng.normal(0.0, 0.05, n)
        result = autofit(XY(x, y, None, "x", "y", f"small-{n}"), AutoFitConfig())
        assert family_model not in result.skipped or "cap" not in result.skipped[family_model][0], (
            n,
            family_model,
            result.skipped.get(family_model),
        )
        assert family_model in {o.candidate_id for o in result.outcomes if o.outcome != "skipped"}
        for cid, reasons in result.skipped.items():  # nothing was dropped silently
            assert reasons, cid
