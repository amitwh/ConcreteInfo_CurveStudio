"""Tests for turning one Auto-fit row into a full ``FitBundle``."""

import functools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import curvestudio.workflow.autofit as autofit_module
from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    CandidateOutcome,
    PolynomialRef,
    autofit,
    fit_candidate,
    subsample,
)
from curvestudio.discovery.engine import LARGE_N
from curvestudio.models import CustomModelDefinition
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.workflow import FitBundle, WorkflowError, bundle_for_candidate


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()


def make_data() -> XY:
    x = np.linspace(0.0, 6.0, 40)
    rng = np.random.default_rng(5)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0.0, 0.05, 40)
    return XY(x=x, y=y, weights=None, x_name="t", y_name="signal", fingerprint="bundle-test")


def run(xy: XY) -> AutoFitResult:
    mine = CustomModelDefinition(id="my_decay", name="My decay", expression="a*exp(-b*x) + c")
    cfg = AutoFitConfig(
        workers=1,
        n_starts=3,
        cv_folds=4,
        families=("exponential", "polynomial"),
        max_params=3,
        custom_models=(mine,),
    )
    return autofit(xy, cfg)


def params(bundle: FitBundle) -> dict[str, float]:
    return {p.name: p.value for p in bundle.result.parameters}


@pytest.mark.parametrize("candidate_id", ["exp_decay_offset", "poly2", "poly1_origin", "my_decay"])
def test_bundle_reproduces_the_row_it_came_from(candidate_id: str) -> None:
    xy = make_data()
    result = run(xy)
    bundle = bundle_for_candidate(xy, result, candidate_id)
    assert isinstance(bundle, FitBundle)
    row = next(o for o in result.outcomes if o.candidate_id == candidate_id)
    assert row.fit is not None
    assert bundle.result.model_id == row.fit.model_id
    stored = {p.name: p.value for p in row.fit.parameters}
    assert params(bundle) == pytest.approx(stored, rel=1e-9)
    assert bundle.result.stats is not None and row.metrics is not None
    assert bundle.result.stats.r2 == pytest.approx(row.metrics.r2)
    assert bundle.curve is not None and len(bundle.curve.x) == 200
    assert bundle.confidence_band is not None
    assert bundle.data.x_name == "t"


def test_bundle_needs_the_same_data() -> None:
    xy = make_data()
    result = run(xy)
    other = XY(xy.x, xy.y, None, "t", "signal", "another-dataset")
    with pytest.raises(WorkflowError) as info:
        bundle_for_candidate(other, result, "poly2")
    assert info.value.code == "DATA_MISMATCH"


def test_unknown_candidate_is_an_error_naming_it() -> None:
    xy = make_data()
    result = run(xy)
    with pytest.raises(WorkflowError) as info:
        bundle_for_candidate(xy, result, "no_such_candidate")
    assert info.value.code == "UNKNOWN_MODEL"
    assert "no_such_candidate" in str(info.value)


def test_bundle_of_a_slim_result_still_refits() -> None:
    xy = make_data()
    slim: Any = AutoFitResult.from_json(run(xy).to_json(include_fits=False))
    bundle = bundle_for_candidate(xy, slim, "exp_decay_offset")
    assert bundle.result.status.value == "converged"


def shell_result(
    xy: XY, outcomes: tuple[CandidateOutcome, ...], *, n_starts: int = 5
) -> AutoFitResult:
    """A result carrying only what ``bundle_for_candidate`` reads: config, fingerprint, outcomes."""
    config = AutoFitConfig(
        workers=1,
        n_starts=n_starts,
        cv_folds=None,
        families=("polynomial",),
        max_params=2,
    )
    return AutoFitResult(
        config=config,
        dataset_fingerprint=xy.fingerprint,
        n_active=int(xy.x.size),
        ranked=(),
        outcomes=outcomes,
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


def spy_on_starts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the effective ``n_starts`` each refit asks for, delegating to the real fit."""
    seen: list[int] = []
    real = autofit_module.fit_candidate

    @functools.wraps(real)
    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["n_starts"])
        return real(*args, **kwargs)

    monkeypatch.setattr(autofit_module, "fit_candidate", spy)
    return seen


def test_a_subsampled_row_is_reproduced_on_the_subsample() -> None:
    x = np.linspace(0.0, 60.0, LARGE_N + 500)
    xy = XY(x=x, y=2.0 * x + 1.0 + np.sin(x), weights=None, x_name="t", y_name="signal",
            fingerprint="large-run")  # fmt: skip
    sub = subsample(xy)
    fit = fit_candidate(sub, PolynomialRef(degree=1), None, seed=1, n_starts=1, time_limit=None)
    outcome = CandidateOutcome(
        candidate_id="poly1", outcome="converged", fit=fit, on_subsample=True, stage="screen"
    )
    bundle = bundle_for_candidate(xy, shell_result(xy, (outcome,)), "poly1")
    assert bundle.data.n_total == len(sub.x) == len(subsample(xy).x)
    assert bundle.data.n_total < len(xy.x)


def test_an_explicit_n_starts_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    xy = make_data()
    seen = spy_on_starts(monkeypatch)
    result = shell_result(xy, ())
    bundle_for_candidate(xy, result, "poly2", n_starts=2)
    assert seen[-1] == 2
    bundle_for_candidate(xy, result, "poly2")
    assert seen[-1] == 5  # no override and no stored outcome: the configuration's count


def test_a_stored_screening_row_keeps_its_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    xy = make_data()
    result = run(xy)
    row = next(o for o in result.outcomes if o.candidate_id == "exp_decay_offset")
    assert row.fit is not None
    staged = row.model_copy(
        update={"stage": "screen", "fit": row.fit.model_copy(update={"n_starts": 2})}
    )
    patched = result.model_copy(
        update={
            "config": result.config.model_copy(update={"n_starts": 5}),
            "outcomes": (staged,),
        }
    )
    seen = spy_on_starts(monkeypatch)
    bundle_for_candidate(xy, patched, "exp_decay_offset")
    assert seen[-1] == 2
