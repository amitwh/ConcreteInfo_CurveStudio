"""The cooperative wall-clock time limit (``FitConfig.time_limit``).

The limit is cooperative: it is checked inside the residual evaluation, so a run stops
within one model call of the deadline. A start that crosses the deadline ends
``NOT_CONVERGED`` with a termination reason naming the limit; with several starts, a
converged start found before the deadline stays the winner and the remaining starts are
skipped with a warning. ``time_limit=None`` (the default) changes nothing.

The slow evaluations are produced by patching ``ModelSpec.evaluate`` to sleep, so the
tests measure the engine's stopping behaviour, not any particular model's cost.
"""

import json
import math
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit
from curvestudio.models import ModelSpec, Parameter
from curvestudio.models.registry import _clear_registry

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

SLEEP = 0.02
"""Cost of one patched model evaluation in the slow tests (seconds)."""


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    _clear_registry()
    yield
    _clear_registry()


def make_xy(x: Any, y: Any) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="test-fingerprint",
    )


EXP_DECAY = ModelSpec(
    id="exp_decay",
    name="Exponential decay with offset",
    category="test",
    expression="a*exp(-b*x) + c",
    parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    initial_guess=lambda x, y: {"a": 1.0, "b": 0.1, "c": float(np.min(y))},
)


def prd_data() -> XY:
    """PRD section 20 example: 5*exp(-0.4x) + 2 with N(0, 0.1) noise, default_rng(42)."""
    x = np.linspace(0.0, 10.0, 100)
    y = 5.0 * np.exp(-0.4 * x) + 2.0 + np.random.default_rng(42).normal(0.0, 0.1, x.size)
    return make_xy(x, y)


def values(result: Any) -> dict[str, float]:
    return {p.name: p.value for p in result.parameters}


def _without_created_at(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    del payload["provenance"]["created_at"]
    return payload


def test_tiny_limit_stops_a_slow_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ModelSpec.evaluate

    def slow(self: ModelSpec, x: FloatArray, **params: float) -> FloatArray:
        time.sleep(SLEEP)
        return original(self, x, **params)

    monkeypatch.setattr(ModelSpec, "evaluate", slow)
    began = time.perf_counter()
    res = fit(EXP_DECAY, prd_data(), FitConfig(time_limit=0.05))
    elapsed = time.perf_counter() - began
    assert elapsed < 0.5, elapsed
    assert res.status is FitStatus.NOT_CONVERGED
    assert "time limit" in res.termination_reason
    assert res.parameters == ()  # an interrupted start reports no parameter values


def test_converged_start_before_deadline_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start 1 sits exactly on the true parameters, so it converges without ever paying
    # the sleep; every perturbed restart (b != 0.4 to float precision) pays SLEEP per
    # evaluation and cannot converge inside the deadline. The converged start must stay
    # the winner, the restart that crossed the deadline is skipped-with-warning, and the
    # 18 starts after it never run.
    original = ModelSpec.evaluate
    slow_calls = {"n": 0}

    def slow_away_from_truth(self: ModelSpec, x: FloatArray, **params: float) -> FloatArray:
        if abs(params["b"] - 0.4) > 1e-9:
            slow_calls["n"] += 1
            time.sleep(0.05)
        return original(self, x, **params)

    monkeypatch.setattr(ModelSpec, "evaluate", slow_away_from_truth)
    x = np.linspace(0.0, 10.0, 100)
    data = make_xy(x, 5.0 * np.exp(-0.4 * x) + 2.0)
    began = time.perf_counter()
    res = fit(
        EXP_DECAY,
        data,
        FitConfig(initial={"a": 5.0, "b": 0.4, "c": 2.0}, n_starts=20, seed=0, time_limit=0.25),
    )
    elapsed = time.perf_counter() - began
    assert res.status is FitStatus.CONVERGED, res.termination_reason
    assert "time limit reached after 2 of 20 starts" in res.warnings
    v = values(res)
    assert v["a"] == pytest.approx(5.0, rel=1e-6)
    assert v["b"] == pytest.approx(0.4, rel=1e-6)
    assert v["c"] == pytest.approx(2.0, rel=1e-6)
    # the 18 starts after the timed-out one never ran: without the skip each would pay
    # at least its starting-value evaluation (18 x 0.05 s plus the timed-out run itself)
    assert elapsed < 0.8, elapsed
    assert slow_calls["n"] < 12, slow_calls["n"]


def test_limit_also_stops_a_scalar_method_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    # nelder/powell/bfgs/... minimise a wrapped SSE objective that calls the same
    # guarded_residuals choke point, so the deadline stops them identically.
    original = ModelSpec.evaluate

    def slow(self: ModelSpec, x: FloatArray, **params: float) -> FloatArray:
        time.sleep(SLEEP)
        return original(self, x, **params)

    monkeypatch.setattr(ModelSpec, "evaluate", slow)
    res = fit(EXP_DECAY, prd_data(), FitConfig(method="nelder", time_limit=0.05))
    assert res.status is FitStatus.NOT_CONVERGED
    assert "time limit" in res.termination_reason
    assert res.method == "nelder"


def test_limit_validation() -> None:
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="time_limit"):
            FitConfig(time_limit=bad)
    assert FitConfig().time_limit is None
    assert FitConfig(time_limit=0.05).time_limit == 0.05


def test_no_limit_changes_nothing() -> None:
    # An unset, an explicit None and a never-reached finite limit must give byte-identical
    # results: the deadline check only reads the clock, it never touches the numerics.
    base = {"n_starts": 4, "seed": 3}
    without_field = fit(EXP_DECAY, prd_data(), FitConfig(**base)).to_json()
    explicit_none = fit(EXP_DECAY, prd_data(), FitConfig(**base, time_limit=None)).to_json()
    generous = fit(EXP_DECAY, prd_data(), FitConfig(**base, time_limit=60.0)).to_json()
    assert _without_created_at(without_field) == _without_created_at(explicit_none)
    assert _without_created_at(without_field) == _without_created_at(generous)


def test_last_start_cut_short_still_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two starts: start 1 sits on the truth and converges at once; start 2 (the LAST one)
    # is slow and crosses the deadline. The converged start is kept, and the user must
    # still be told the multistart was truncated.
    original = ModelSpec.evaluate

    def slow_away_from_truth(self: ModelSpec, x: FloatArray, **params: float) -> FloatArray:
        if abs(params["b"] - 0.4) > 1e-9:
            time.sleep(0.05)
        return original(self, x, **params)

    monkeypatch.setattr(ModelSpec, "evaluate", slow_away_from_truth)
    x = np.linspace(0.0, 10.0, 100)
    data = make_xy(x, 5.0 * np.exp(-0.4 * x) + 2.0)
    res = fit(
        EXP_DECAY,
        data,
        FitConfig(initial={"a": 5.0, "b": 0.4, "c": 2.0}, n_starts=2, seed=0, time_limit=0.25),
    )
    assert res.status is FitStatus.CONVERGED, res.termination_reason
    assert any("cut short" in w for w in res.warnings), res.warnings
