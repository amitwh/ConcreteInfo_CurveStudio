"""Tests for the nonlinear least-squares engine ``fit``."""

import json
import math
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitStatus, fit, fit_polynomial
from curvestudio.fitting.metrics import fit_statistics
from curvestudio.fitting.nonlinear import (
    _better_unconverged_warning,
    _pick_winner,
    _Run,
    _uncertainties,
    generate_starts,
)
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    _clear_registry()
    yield
    _clear_registry()


def make_xy(x: Any, y: Any, weights: Any = None) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="test-fingerprint",
    )


def _exp_decay_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    c = float(np.min(y))
    return {"a": float(y[0] - c), "b": 0.1, "c": c}


EXP_DECAY = ModelSpec(
    id="exp_decay",
    name="Exponential decay with offset",
    category="test",
    expression="a*exp(-b*x) + c",
    parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    initial_guess=_exp_decay_guess,
)

EXP_GROWTH = ModelSpec(
    id="exp_growth",
    name="Exponential growth",
    category="test",
    expression="a*exp(b*x)",
    parameters=(Parameter("a"), Parameter("b")),
    initial_guess=lambda x, y: {"a": 1.0, "b": 1.0},
)

LINE = ModelSpec(
    id="line",
    name="Straight line",
    category="test",
    expression="b0 + b1*x",
    parameters=(Parameter("b0"), Parameter("b1")),
    initial_guess=lambda x, y: {"b0": 0.0, "b1": 0.0},
)


def prd_data() -> XY:
    """PRD section 20 example: 5*exp(-0.4x) + 2 with N(0, 0.1) noise, default_rng(42)."""
    x = np.linspace(0.0, 10.0, 100)
    y = 5.0 * np.exp(-0.4 * x) + 2.0 + np.random.default_rng(42).normal(0.0, 0.1, x.size)
    return make_xy(x, y)


def values(result: Any) -> dict[str, float]:
    return {p.name: p.value for p in result.parameters}


# -- (a) the PRD example ---------------------------------------------------------------


def test_prd_example_recovers_parameters() -> None:
    res = fit(EXP_DECAY, prd_data())
    assert res.status is FitStatus.CONVERGED
    v = values(res)
    assert v["a"] == pytest.approx(5.0, rel=0.05)
    assert v["b"] == pytest.approx(0.4, rel=0.05)
    assert v["c"] == pytest.approx(2.0, rel=0.05)
    assert res.method == "lm"
    assert res.loss == "linear"
    assert res.model_id == "exp_decay"
    assert res.equation == "y = a*exp(-b*x) + c"
    assert res.equation_fitted == EXP_DECAY.equation_text(v)
    guess = _exp_decay_guess(prd_data().x, prd_data().y)
    assert res.initial_params == guess
    assert res.stats is not None and res.stats.n == 100 and res.stats.df == 97
    assert len(res.fitted) == 100 and len(res.residuals) == 100
    assert res.covariance is not None and res.correlation is not None
    assert all(p.std_err is not None and p.std_err > 0 for p in res.parameters)
    for p in res.parameters:
        assert p.ci_low is not None and p.ci_high is not None
        assert p.ci_low < p.value < p.ci_high
    corr = np.array(res.correlation)
    assert np.all(np.diag(corr) == 1.0)
    assert res.warnings == ()
    assert res.provenance.seed == 0
    assert res.provenance.dataset_fingerprint == "test-fingerprint"


# -- (b) determinism ---------------------------------------------------------------------


def _without_created_at(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    del payload["provenance"]["created_at"]
    return payload


@pytest.mark.parametrize("n_starts", [1, 4])
def test_same_seed_gives_identical_json(n_starts: int) -> None:
    cfg = FitConfig(n_starts=n_starts, seed=7)
    first = fit(EXP_DECAY, prd_data(), cfg).to_json()
    second = fit(EXP_DECAY, prd_data(), cfg).to_json()
    assert _without_created_at(first) == _without_created_at(second)


# -- (c) incompatible method -------------------------------------------------------------


def test_lm_with_bounds_raises_before_fitting() -> None:
    cfg = FitConfig(method="lm", bounds={"b": (0.0, 10.0)})
    with pytest.raises(ValueError, match="bounds"):
        fit(EXP_DECAY, prd_data(), cfg)


def test_lm_with_model_bounds_raises() -> None:
    bounded = ModelSpec(
        id="bounded_line",
        name="Bounded line",
        category="test",
        expression="b0 + b1*x",
        parameters=(Parameter("b0", lower=0.0), Parameter("b1")),
        initial_guess=lambda x, y: {"b0": 1.0, "b1": 1.0},
    )
    with pytest.raises(ValueError, match="bounds"):
        fit(bounded, make_xy([1, 2, 3, 4], [1, 2, 3, 5]), FitConfig(method="lm"))


def test_auto_with_bounds_resolves_to_trf() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(bounds={"b": (0.0, 10.0)}))
    assert res.status is FitStatus.CONVERGED
    assert res.method == "trf"
    assert values(res)["b"] == pytest.approx(0.4, rel=0.05)


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_bounded_methods_agree_with_lm(method: str) -> None:
    ref = values(fit(EXP_DECAY, prd_data(), FitConfig(method="lm")))
    res = fit(EXP_DECAY, prd_data(), FitConfig(method=method))
    assert res.status is FitStatus.CONVERGED
    assert res.method == method
    for name, value in values(res).items():
        assert value == pytest.approx(ref[name], rel=1e-6)


def test_robust_loss_is_passed_through() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(loss="soft_l1"))
    assert res.status is FitStatus.CONVERGED
    assert (res.method, res.loss) == ("trf", "soft_l1")


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (FitConfig(initial={"z": 1.0}), "unknown parameter"),
        (FitConfig(bounds={"z": (0.0, 1.0)}), "unknown parameter"),
        (FitConfig(method="simplex"), "unknown method"),
        (FitConfig(loss="l1"), "unknown loss"),
    ],
)
def test_configuration_errors_raise(cfg: FitConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit(EXP_DECAY, prd_data(), cfg)


# -- (d) multi-start rescues a bad start -------------------------------------------------

# b = -5 (the lower end of the box b in [-5, 5]) turns the decay into a steep growth,
# exp(50) at x = 10. From there a single trf run drifts along a degenerate valley
# (a -> 0) and exhausts its 200-evaluation budget: NOT_CONVERGED. With n_starts=5 and
# seed 0 the restarts draw b uniformly from the box; the one starting at b ~ 2.29 reaches
# the global minimum. Recorded with numpy 2.5.5 / scipy 1.18.1; both outcomes are
# deterministic for a fixed seed.
BAD_START = {"a": 1.0, "b": -5.0, "c": 0.0}
BAD_START_CONFIG = {"initial": BAD_START, "bounds": {"b": (-5.0, 5.0)}, "max_nfev": 200}


def test_multistart_rescues_a_bad_start() -> None:
    single = fit(EXP_DECAY, prd_data(), FitConfig(**BAD_START_CONFIG, n_starts=1))  # type: ignore[arg-type]
    assert single.status is FitStatus.NOT_CONVERGED
    assert "maximum number of function evaluations" in single.termination_reason
    assert single.initial_params == BAD_START
    multi = fit(EXP_DECAY, prd_data(), FitConfig(**BAD_START_CONFIG, n_starts=5))  # type: ignore[arg-type]
    assert multi.status is FitStatus.CONVERGED
    assert multi.n_starts == 5
    assert multi.method == "trf"
    v = values(multi)
    assert v["a"] == pytest.approx(5.0, rel=0.05)
    assert v["b"] == pytest.approx(0.4, rel=0.05)
    assert v["c"] == pytest.approx(2.0, rel=0.05)
    assert multi.initial_params != BAD_START  # the winner was one of the restarts
    assert multi.initial_params["b"] > 0.0


def test_generate_starts_scheme() -> None:
    base = np.array([2.0, 0.0, -3.0, 0.5])
    lower = np.array([-np.inf, -np.inf, -np.inf, 0.0])
    upper = np.array([np.inf, np.inf, np.inf, 1.0])
    starts = generate_starts(base, lower, upper, n_starts=50, seed=3)
    assert len(starts) == 50
    np.testing.assert_array_equal(starts[0], base)
    extra = np.array(starts[1:])
    ratio = extra[:, 0] / 2.0
    assert np.all((ratio >= 0.1) & (ratio <= 10.0))  # log-uniform multiplier, sign kept
    assert np.all((extra[:, 1] >= -1.0) & (extra[:, 1] <= 1.0))  # zero: uniform in [-1, 1]
    assert np.all(extra[:, 2] < 0.0)  # sign of a negative start is preserved
    assert np.all((extra[:, 3] >= 0.0) & (extra[:, 3] <= 1.0))  # both bounds: uniform inside
    again = generate_starts(base, lower, upper, n_starts=50, seed=3)
    np.testing.assert_array_equal(np.array(starts), np.array(again))


def test_generate_starts_clips_into_bounds() -> None:
    starts = generate_starts(
        np.array([5.0]), np.array([0.0]), np.array([np.inf]), n_starts=20, seed=1
    )
    assert all(s[0] >= 0.0 for s in starts)
    starts = generate_starts(np.array([5.0]), np.array([-np.inf]), np.array([6.0]), 20, 1)
    assert all(s[0] <= 6.0 for s in starts)


def test_n_starts_below_one_raises() -> None:
    with pytest.raises(ValueError, match="n_starts"):
        fit(EXP_DECAY, prd_data(), FitConfig(n_starts=0))


def test_multistart_disagreement_is_reported() -> None:
    # sin(b*x) has many local minima in b. The restarts (log-uniform multiples of the
    # start 1.2) mostly land in other valleys, so the converged SSEs differ widely; the
    # lowest one (the true b = 1) must win and the spread must be reported.
    model = ModelSpec(
        id="sine",
        name="Sine",
        category="test",
        expression="sin(b*x)",
        parameters=(Parameter("b"),),
        initial_guess=lambda x, y: {"b": 1.2},
    )
    x = np.linspace(0.0, 20.0, 200)
    y = np.sin(1.0 * x) + np.random.default_rng(1).normal(0.0, 0.05, x.size)
    res = fit(model, make_xy(x, y), FitConfig(n_starts=8, seed=0))
    assert res.status is FitStatus.CONVERGED
    assert values(res)["b"] == pytest.approx(1.0, rel=1e-2)
    disagree = [w for w in res.warnings if w.startswith("MULTISTART_DISAGREE: sse=[")]
    assert len(disagree) == 1


# -- (e) too few points ------------------------------------------------------------------


def test_fewer_points_than_parameters_is_invalid_input() -> None:
    res = fit(EXP_DECAY, make_xy([0.0, 1.0], [3.0, 2.0]))
    assert res.status is FitStatus.INVALID_INPUT
    assert res.parameters == ()
    assert res.stats is None
    assert res.covariance is None
    assert "2 active observation(s) for 3 parameter(s)" in res.termination_reason
    FitResult_round_trip(res)


def test_zero_weights_count_as_missing_for_n_le_k() -> None:
    res = fit(EXP_DECAY, make_xy([0, 1, 2, 3], [5, 4, 3, 2.5], weights=[1, 1, 1, 0]))
    assert res.status is FitStatus.INVALID_INPUT


def FitResult_round_trip(res: Any) -> None:
    assert type(res).from_json(res.to_json()) == res


# -- (f) cannot fit ----------------------------------------------------------------------


def test_unfittable_data_is_not_converged() -> None:
    x = np.linspace(0.0, 5.0, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit(EXP_GROWTH, make_xy(x, np.full_like(x, 3.0)), FitConfig(max_nfev=3))
    assert res.status is FitStatus.NOT_CONVERGED
    assert res.termination_reason
    assert res.covariance is None
    assert all(p.std_err is None for p in res.parameters)
    FitResult_round_trip(res)


# -- (g) noiseless data ------------------------------------------------------------------


def test_noiseless_data_fits_exactly_without_warnings() -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = 5.0 * np.exp(-0.4 * x) + 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit(EXP_DECAY, make_xy(x, y))
    assert res.status is FitStatus.CONVERGED
    assert res.stats is not None
    assert res.stats.sse < 1e-20
    assert res.stats.r2 == 1.0
    FitResult_round_trip(res)


def test_exact_zero_residual_gives_zero_covariance() -> None:
    # Starting exactly on the solution leaves every residual exactly 0.0, so s2 = 0: the
    # covariance must be zeros (not NaN), standard errors 0 and the correlation undefined.
    x = np.arange(6.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit(LINE, make_xy(x, 1.0 + 2.0 * x), FitConfig(initial={"b0": 1.0, "b1": 2.0}))
    assert res.status is FitStatus.CONVERGED
    assert res.stats is not None and res.stats.sse == 0.0 and res.stats.r2 == 1.0
    assert res.covariance == [[0.0, 0.0], [0.0, 0.0]]
    assert res.correlation is None
    assert all(p.std_err == 0.0 and p.ci_low == p.value == p.ci_high for p in res.parameters)
    FitResult_round_trip(res)


# -- (h) applicability -------------------------------------------------------------------


def test_inapplicable_model_is_invalid_input_naming_the_reason() -> None:
    def positive_x(x: FloatArray, y: FloatArray) -> list[str]:
        return [] if bool(np.all(x > 0)) else ["x > 0 required"]

    power = ModelSpec(
        id="power",
        name="Power law",
        category="test",
        expression="a*x**b",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.0, "b": 1.0},
        domain_check=positive_x,
    )
    res = fit(power, make_xy([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0]))
    assert res.status is FitStatus.INVALID_INPUT
    assert "x > 0 required" in res.termination_reason
    assert res.parameters == ()


# -- covariance, weights and the Jacobian ------------------------------------------------


def _noisy_line() -> tuple[FloatArray, FloatArray]:
    x = np.linspace(0.0, 10.0, 30)
    y = 1.5 + 0.7 * x + np.random.default_rng(5).normal(0.0, 0.3, x.size)
    return x, y


@pytest.mark.parametrize("weighted", [False, True])
def test_linear_model_matches_fit_polynomial(weighted: bool) -> None:
    x, y = _noisy_line()
    w = np.linspace(0.5, 2.0, x.size) if weighted else None
    if w is not None:
        w[3] = 0.0  # an excluded point
    data = make_xy(x, y, None if w is None else w)
    ref = fit_polynomial(data, 1)
    res = fit(LINE, data)
    assert res.status is FitStatus.CONVERGED
    assert res.stats is not None and ref.stats is not None
    assert res.stats.sse == pytest.approx(ref.stats.sse, rel=1e-9)
    assert res.stats.df == ref.stats.df
    for got, want in zip(res.parameters, ref.parameters, strict=True):
        assert got.value == pytest.approx(want.value, rel=1e-8)
        assert got.std_err == pytest.approx(want.std_err, rel=1e-6)
        assert got.ci_low == pytest.approx(want.ci_low, rel=1e-6)
        assert got.ci_high == pytest.approx(want.ci_high, rel=1e-6)
    assert np.allclose(res.covariance, ref.covariance, rtol=1e-6)  # type: ignore[arg-type]
    assert np.allclose(res.fitted, ref.fitted) and len(res.fitted) == x.size
    assert np.allclose(res.residuals, ref.residuals)


def test_zero_weight_point_does_not_influence_the_fit() -> None:
    x, y = _noisy_line()
    y_outlier = y.copy()
    y_outlier[4] = 1e3
    w = np.ones_like(x)
    w[4] = 0.0
    with_outlier = fit(LINE, make_xy(x, y_outlier, w))
    without = fit(LINE, make_xy(np.delete(x, 4), np.delete(y, 4)))
    for a, b in zip(with_outlier.parameters, without.parameters, strict=True):
        assert a.value == pytest.approx(b.value, rel=1e-9)
        assert a.std_err == pytest.approx(b.std_err, rel=1e-6)
    assert len(with_outlier.fitted) == x.size


def test_analytic_jacobian_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    original = ModelSpec.jacobian

    def counting(self: ModelSpec, x: FloatArray, **params: float) -> FloatArray:
        calls["n"] += 1
        return original(self, x, **params)

    monkeypatch.setattr(ModelSpec, "jacobian", counting)
    for method in ("lm", "trf", "dogbox"):
        calls["n"] = 0
        res = fit(EXP_DECAY, prd_data(), FitConfig(method=method))
        assert res.status is FitStatus.CONVERGED
        # at least one call inside the optimizer plus one for the covariance
        assert calls["n"] >= 3, method


def test_redundant_parameters_give_singular_covariance() -> None:
    model = ModelSpec(
        id="redundant",
        name="Redundant product",
        category="test",
        expression="a*b*x",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.0, "b": 1.0},
    )
    x, y = _noisy_line()
    res = fit(model, make_xy(x, y - 1.5))
    assert "SINGULAR_COVARIANCE" in res.warnings
    assert res.covariance is None and res.correlation is None
    assert all(p.std_err is None and p.ci_low is None for p in res.parameters)
    assert res.status is FitStatus.CONVERGED
    v = values(res)
    assert v["a"] * v["b"] == pytest.approx(0.7, rel=0.05)


def test_parameter_at_bound_is_flagged() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(bounds={"c": (2.5, 10.0)}))
    assert res.status is FitStatus.CONVERGED
    assert values(res)["c"] == pytest.approx(2.5)
    assert "PARAMETER_AT_BOUND: c" in res.warnings


def test_initial_guess_is_clipped_into_bounds() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(bounds={"b": (0.2, 1.0)}))
    assert res.initial_params["b"] == 0.2  # the model guesses 0.1
    assert res.status is FitStatus.CONVERGED


def test_config_initial_overrides_model_guess() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(initial={"b": 0.3}))
    assert res.initial_params["b"] == 0.3
    assert res.initial_params["c"] == pytest.approx(float(np.min(prd_data().y)))


def test_non_finite_model_at_start_is_failed() -> None:
    model = ModelSpec(
        id="log_model",
        name="Log",
        category="test",
        expression="a*log(b*x)",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.0, "b": -1.0},
    )
    res = fit(model, make_xy([1.0, 2.0, 3.0, 4.0], [0.0, 0.7, 1.1, 1.4]))
    assert res.status is FitStatus.FAILED
    assert "model not finite at starting values" in res.termination_reason
    assert res.parameters == ()
    FitResult_round_trip(res)


def test_exception_inside_optimizer_is_failed_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ModelSpec.jacobian
    calls = {"n": 0}

    def exploding(self: ModelSpec, x: FloatArray, **params: float) -> FloatArray:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom from the model")
        return original(self, x, **params)

    monkeypatch.setattr(ModelSpec, "jacobian", exploding)
    res = fit(EXP_DECAY, prd_data())
    assert res.status is FitStatus.FAILED
    assert "boom from the model" in res.termination_reason
    FitResult_round_trip(res)


def test_bad_initial_guess_from_model_is_failed() -> None:
    model = ModelSpec(
        id="nan_guess",
        name="NaN guess",
        category="test",
        expression="a*x",
        parameters=(Parameter("a"),),
        initial_guess=lambda x, y: {"a": math.nan},
    )
    res = fit(model, make_xy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    assert res.status is FitStatus.FAILED
    assert "initial" in res.termination_reason


def test_model_lookup_by_id() -> None:
    register(EXP_DECAY)
    res = fit("exp_decay", prd_data())
    assert res.status is FitStatus.CONVERGED
    assert res.model_id == "exp_decay"


def test_unknown_model_id_raises_key_error() -> None:
    with pytest.raises(KeyError, match="nope"):
        fit("nope", prd_data())


def test_confidence_level_changes_interval_width() -> None:
    narrow = fit(EXP_DECAY, prd_data(), FitConfig(confidence_level=0.5))
    wide = fit(EXP_DECAY, prd_data(), FitConfig(confidence_level=0.99))
    for n, w in zip(narrow.parameters, wide.parameters, strict=True):
        assert n.ci_low is not None and w.ci_low is not None
        assert w.ci_low < n.ci_low


def test_steps_into_an_undefined_region_are_survived_without_warnings() -> None:
    # sqrt(x - b) is NaN for b > min(x) = 1. Starting at b = 0 with a poor amplitude, trial
    # steps that cross b = 1 get the finite penalty and are rejected; the fit must still
    # converge to b = 0.5 and no RuntimeWarning may escape.
    model = ModelSpec(
        id="sqrt_shift",
        name="Shifted square root",
        category="test",
        expression="a*sqrt(x - b)",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 0.1, "b": 0.0},
    )
    x = np.linspace(1.0, 10.0, 25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit(model, make_xy(x, 2.0 * np.sqrt(x - 0.5)))
    assert res.status is FitStatus.CONVERGED
    assert values(res)["a"] == pytest.approx(2.0, rel=1e-6)
    assert values(res)["b"] == pytest.approx(0.5, rel=1e-6)


# -- review fixes: better unconverged start, guarded final evaluation ---------------------


def _run(status: FitStatus, sse: float) -> _Run:
    return _Run(start=np.array([1.0]), status=status, message="", theta=np.array([1.0]), sse=sse)


def test_better_unconverged_run_is_warned_about_but_converged_still_wins() -> None:
    runs = [
        _run(FitStatus.NOT_CONVERGED, 32.0),
        _run(FitStatus.CONVERGED, 4e64),
        _run(FitStatus.FAILED, math.inf),
    ]
    winner = _pick_winner(runs)
    assert winner is runs[1]
    assert _better_unconverged_warning(winner, runs, scale=100.0) == (
        "MULTISTART_BETTER_UNCONVERGED: winner_sse=4e+64 best_unconverged_sse=32"
    )


@pytest.mark.parametrize(
    "runs",
    [
        [_run(FitStatus.CONVERGED, 1.0), _run(FitStatus.NOT_CONVERGED, 2.0)],
        [_run(FitStatus.CONVERGED, 1.0), _run(FitStatus.NOT_CONVERGED, 1.0 - 1e-9)],  # within rtol
        [_run(FitStatus.CONVERGED, 1e-30), _run(FitStatus.NOT_CONVERGED, 1e-32)],  # abs floor
        [_run(FitStatus.CONVERGED, 1.0), _run(FitStatus.CONVERGED, 0.5)],
        [_run(FitStatus.CONVERGED, 5.0)],
    ],
)
def test_no_better_unconverged_warning_when_winner_is_best(runs: list[_Run]) -> None:
    assert _better_unconverged_warning(_pick_winner(runs), runs, scale=1.0) is None


def test_no_better_unconverged_warning_when_nothing_converged() -> None:
    runs = [_run(FitStatus.NOT_CONVERGED, 3.0), _run(FitStatus.NOT_CONVERGED, 1.0)]
    winner = _pick_winner(runs)
    assert winner is runs[1]
    assert _better_unconverged_warning(winner, runs, scale=1.0) is None


def test_garbage_converged_winner_is_flagged_end_to_end() -> None:
    # Unbounded, b = -5, seed 0, 5 starts: one restart "converges" (lm, xtol) at
    # b ~ -14.4 with SSE ~ 4e64 while the unconverged base run reaches SSE ~ 32. The
    # converged run still wins (ruling) but the warning must say a better start exists.
    res = fit(EXP_DECAY, prd_data(), FitConfig(initial=BAD_START, n_starts=5))
    assert res.status is FitStatus.CONVERGED
    flagged = [w for w in res.warnings if w.startswith("MULTISTART_BETTER_UNCONVERGED: ")]
    assert len(flagged) == 1
    assert res.stats is not None and res.stats.sse > 1e60


def test_good_multistart_fit_has_no_better_unconverged_warning() -> None:
    res = fit(EXP_DECAY, prd_data(), FitConfig(**BAD_START_CONFIG, n_starts=5))  # type: ignore[arg-type]
    assert res.status is FitStatus.CONVERGED
    assert not any(w.startswith("MULTISTART_BETTER_UNCONVERGED") for w in res.warnings)


def test_final_evaluation_error_is_failed_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    # x = 99 is a zero-weight point, so only the final full-data evaluation sees it.
    original = ModelSpec.evaluate

    def picky(self: ModelSpec, x: FloatArray, **params: float) -> FloatArray:
        if bool(np.any(np.asarray(x) == 99.0)):
            raise RuntimeError("cannot evaluate at x = 99")
        return original(self, x, **params)

    monkeypatch.setattr(ModelSpec, "evaluate", picky)
    x, y = _noisy_line()
    w = np.ones_like(x)
    x[2], w[2] = 99.0, 0.0
    res = fit(LINE, make_xy(x, y, w))
    assert res.status is FitStatus.FAILED
    assert "cannot evaluate at x = 99" in res.termination_reason
    assert res.parameters == ()
    FitResult_round_trip(res)


RATIONAL = ModelSpec(
    id="rational_2_2",
    name="Rational (ill-conditioned)",
    category="test",
    expression="b1*(x**2 + x*b2) / (x**2 + x*b3 + b4)",
    parameters=(Parameter("b1"), Parameter("b2"), Parameter("b3"), Parameter("b4")),
    initial_guess=lambda x, y: {"b1": 0.25, "b2": 0.39, "b3": 0.415, "b4": 0.39},
)


def test_default_tolerances_resolve_parameters_of_an_ill_conditioned_fit() -> None:
    # Kowalik-Osborne enzyme data (the NIST MGH09 problem): the SSE valley is flat and
    # curved, so an ftol of 1e-10 stops once the SSE has settled to ~10 digits while b2..b4
    # are only right to ~5. The default must keep going until the parameters match a
    # reference run with tolerances at the floor of double precision.
    x = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625])
    y = [0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    data = make_xy(x, y)
    tight = FitConfig(ftol=1e-15, xtol=1e-15, gtol=1e-15)
    reference = {p.name: p.value for p in fit(RATIONAL, data, tight).parameters}

    res = fit(RATIONAL, data)

    assert res.status is FitStatus.CONVERGED
    for p in res.parameters:
        assert p.value == pytest.approx(reference[p.name], rel=1e-7), p.name


def test_tiny_jacobian_withholds_covariance_without_numpy_warnings() -> None:
    # On a plateau where the model has vanished the Jacobian can be ~1e-240: well-conditioned
    # relative to itself (passes the SVD cut-off) but its squared singular values underflow.
    jac = np.array([[1e-240, 0.0], [0.0, 2e-240], [1e-240, 1e-240], [3e-240, -1e-240]])
    stats = fit_statistics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 1.9, 3.2, 3.9]), 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se, covariance, corr = _uncertainties(jac, stats)
    assert (se, covariance, corr) == (None, None, None)


def test_huge_jacobian_withholds_covariance_instead_of_zero_standard_errors() -> None:
    # Singular values ~1e160 pass the relative cut-off but their squares overflow to inf,
    # and V'/inf would silently give an all-zero covariance (SE = 0) that looks finite.
    jac = np.array([[1e160, 0.0], [0.0, 2e160], [1e160, 1e160], [3e160, -1e160]])
    stats = fit_statistics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 1.9, 3.2, 3.9]), 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se, covariance, corr = _uncertainties(jac, stats)
    assert (se, covariance, corr) == (None, None, None)


HUGE_SCALE = ModelSpec(
    id="huge_scale",
    name="Line with a 1e160 scale factor",
    category="test",
    expression="a*1e160*x + b*1e160",
    parameters=(Parameter("a"), Parameter("b")),
    initial_guess=lambda x, y: {"a": 1e-160, "b": 0.0},
)


def test_fit_with_a_huge_jacobian_reports_singular_covariance_not_zero_errors() -> None:
    x = np.linspace(1.0, 10.0, 20)
    y = 2.0 * x + 1.0 + np.random.default_rng(0).normal(0.0, 0.1, x.size)
    res = fit(HUGE_SCALE, make_xy(x, y), FitConfig(method="lm"))
    assert res.status is FitStatus.CONVERGED, res.termination_reason
    assert all(math.isfinite(r) for r in res.residuals)
    assert "SINGULAR_COVARIANCE" in res.warnings
    assert res.covariance is None
    assert all(p.std_err is None for p in res.parameters)
