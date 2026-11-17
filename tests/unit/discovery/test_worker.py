"""Tests for the Auto-fit worker job: one candidate in, one outcome out."""

import hashlib
import pickle
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.discovery import (
    CandidateJob,
    CandidateOutcome,
    CustomRef,
    PolynomialRef,
    RegisteredRef,
    candidate_seed,
    run_candidate,
)
from curvestudio.models import CustomModelDefinition, ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()


def make_job(ref: Any, *, y: Any = None, x: Any = None, **overrides: Any) -> CandidateJob:
    xs = np.linspace(0.0, 6.0, 40) if x is None else np.asarray(x, dtype=float)
    if y is None:
        rng = np.random.default_rng(3)
        ys = 3.0 * np.exp(-0.8 * xs) + 1.0 + rng.normal(0.0, 0.05, xs.size)
    else:
        ys = np.asarray(y, dtype=float)
    fields: dict[str, Any] = {
        "x": xs,
        "y": ys,
        "weights": None,
        "x_name": "x",
        "y_name": "y",
        "fingerprint": "worker-test",
        "ref": ref,
        "seed": 0,
        "n_starts": 3,
        "time_limit": 20.0,
        "cv_folds": 4,
    }
    fields.update(overrides)
    return CandidateJob(**fields)


def test_candidate_seed_is_the_documented_hash_of_seed_and_id() -> None:
    digest = hashlib.sha256(b"5:exp_offset").digest()
    assert candidate_seed(5, "exp_offset") == int.from_bytes(digest[:8]) % 2**32
    assert candidate_seed(5, "exp_offset") != candidate_seed(6, "exp_offset")
    assert candidate_seed(5, "exp_offset") != candidate_seed(5, "exp_growth")
    assert 0 <= candidate_seed(0, "x") < 2**32


def test_candidate_seed_independent_of_order() -> None:
    refs = [RegisteredRef(model_id=m) for m in ("exp_offset", "exp_decay_offset", "exp_growth")]
    forward = {r.id: run_candidate(make_job(r)).metrics for r in refs}
    backward = {r.id: run_candidate(make_job(r)).metrics for r in reversed(refs)}
    assert forward == backward


def test_converged_registered_model_has_every_metric() -> None:
    outcome = run_candidate(make_job(RegisteredRef(model_id="exp_decay_offset")))
    assert isinstance(outcome, CandidateOutcome)
    assert outcome.outcome == "converged"
    assert outcome.candidate_id == "exp_decay_offset"
    assert outcome.name == "Exponential decay with offset"
    assert outcome.category == "exponential"
    assert outcome.fit is not None
    m = outcome.metrics
    assert m is not None
    assert (m.k, m.n) == (3, 40)
    for value in (m.r2, m.adj_r2, m.rmse, m.aicc, m.bic, m.cv_rmse):
        assert value is not None
    assert m.cv_rmse is not None and m.rmse is not None
    assert m.cv_rmse >= m.rmse * 0.5  # held-out error is of the order of the in-sample error
    assert outcome.elapsed_seconds > 0.0


def test_polynomial_candidate_uses_the_closed_form_fit() -> None:
    outcome = run_candidate(make_job(PolynomialRef(degree=2)))
    assert outcome.outcome == "converged"
    assert outcome.candidate_id == "poly2"
    assert outcome.category == "polynomial"
    assert outcome.fit is not None and outcome.fit.model_id == "poly2"
    assert outcome.metrics is not None and outcome.metrics.k == 3
    assert outcome.metrics.cv_rmse is not None


def test_polynomial_through_origin_counts_its_parameters() -> None:
    outcome = run_candidate(make_job(PolynomialRef(degree=2, through_origin=True)))
    assert outcome.candidate_id == "poly2_origin"
    assert outcome.metrics is not None and outcome.metrics.k == 2
    assert outcome.metrics.cv_rmse is not None


def test_custom_candidate_is_built_from_its_definition() -> None:
    ref = CustomRef(
        definition=CustomModelDefinition(id="mine", name="Mine", expression="a*exp(-b*x) + c")
    )
    outcome = run_candidate(make_job(ref))
    assert outcome.outcome == "converged"
    assert outcome.category == "custom"
    assert outcome.metrics is not None and outcome.metrics.k == 3


def test_exact_fit_has_no_information_criteria_but_keeps_cv() -> None:
    x = np.arange(15.0)
    outcome = run_candidate(make_job(PolynomialRef(degree=1), x=x, y=2.0 * x + 1.0))
    assert outcome.outcome == "exact_fit"
    assert outcome.metrics is not None
    assert outcome.metrics.aicc is None and outcome.metrics.bic is None
    assert outcome.metrics.r2 == pytest.approx(1.0)
    assert outcome.metrics.cv_rmse is not None
    assert "exact" in outcome.reason


def test_cv_disabled_leaves_cv_rmse_none_without_a_note() -> None:
    outcome = run_candidate(make_job(RegisteredRef(model_id="exp_offset"), cv_folds=None))
    assert outcome.outcome == "converged"
    assert outcome.metrics is not None and outcome.metrics.cv_rmse is None
    assert outcome.reason == ""


def test_cv_failure_keeps_the_fit_and_explains_in_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    import curvestudio.discovery.worker as worker

    def broken_cv(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("folds exploded")

    monkeypatch.setattr(worker, "k_fold_cv", broken_cv)
    outcome = run_candidate(make_job(RegisteredRef(model_id="exp_offset")))
    assert outcome.outcome == "converged"
    assert outcome.metrics is not None
    assert outcome.metrics.aicc is not None and outcome.metrics.cv_rmse is None
    assert "cross-validation" in outcome.reason and "folds exploded" in outcome.reason


def test_non_convergence_is_reported_with_the_fit_reason() -> None:
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
    outcome = run_candidate(
        make_job(RegisteredRef(model_id="degenerate_pair"), x=x, y=2.0 * x + np.cos(7 * x))
    )
    assert outcome.outcome == "not_converged"
    assert outcome.reason
    assert outcome.metrics is None


def test_model_whose_evaluate_raises_becomes_a_failed_outcome() -> None:
    spec = ModelSpec(
        id="explodes",
        name="Explodes",
        category="kinetics",
        expression="a*x + b",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.0, "b": 1.0},
    )
    register(spec)

    def raising(x: Any, **params: float) -> Any:
        raise RuntimeError("evaluate blew up")

    object.__setattr__(spec, "evaluate", raising)
    outcome = run_candidate(make_job(RegisteredRef(model_id="explodes")))
    assert outcome.outcome == "failed"
    assert "evaluate blew up" in outcome.reason
    assert outcome.metrics is None


def test_exception_outside_the_fit_becomes_a_failed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import curvestudio.discovery.worker as worker

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise ZeroDivisionError("division by zero in statistics")

    monkeypatch.setattr(worker, "information_criteria", boom)
    outcome = run_candidate(make_job(RegisteredRef(model_id="exp_offset")))
    assert outcome.outcome == "failed"
    assert "ZeroDivisionError" in outcome.reason
    assert outcome.candidate_id == "exp_offset"
    assert outcome.name == "Exponential with offset (asymptotic regression)"


def test_unknown_model_id_is_a_failed_outcome_not_an_exception() -> None:
    outcome = run_candidate(make_job(RegisteredRef(model_id="no_such_model")))
    assert outcome.outcome == "failed"
    assert "no_such_model" in outcome.reason


def test_job_and_outcome_are_picklable() -> None:
    job = make_job(RegisteredRef(model_id="exp_offset"))
    # the pool pickles jobs and outcomes; this round-trips our own objects only
    clone = pickle.loads(pickle.dumps(job))
    assert clone.ref == job.ref and np.array_equal(clone.x, job.x)
    outcome = run_candidate(job)
    assert pickle.loads(pickle.dumps(outcome)) == outcome


def test_weighted_data_is_fitted_weighted() -> None:
    weights = np.ones(40)
    weights[::2] = 0.0  # half the points are inactive
    outcome = run_candidate(make_job(RegisteredRef(model_id="exp_offset"), weights=weights))
    assert outcome.metrics is not None and outcome.metrics.n == 20
