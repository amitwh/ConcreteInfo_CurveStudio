"""Tests for ``fit_dataset`` / ``predict``: specs, bundles, failure reporting, JSON safety."""

import json
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from curvestudio.data import XY
from curvestudio.fitting import FitStatus, fit_polynomial
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.statistics import StatisticsError
from curvestudio.workflow import (
    MAX_PREDICT_POINTS,
    CustomRef,
    FitBundle,
    FitSpec,
    PolynomialRef,
    RegisteredRef,
    WorkflowError,
    fit_dataset,
    predict,
)


@pytest.fixture(autouse=True)
def starter_registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()


def make_xy(x: Any, y: Any, weights: Any = None, tag: str = "wf") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=f"fp-{tag}",
    )


def quadratic_xy(n: int = 40, seed: int = 11) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)
    return make_xy(x, 1.0 + 0.5 * x - 0.03 * x**2 + rng.normal(0.0, 0.1, n), tag="quad")


def exponential_xy(n: int = 40, seed: int = 3) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 5.0, n)
    return make_xy(x, 2.0 * np.exp(0.3 * x) + rng.normal(0.0, 0.05, n), tag="exp")


def poly_spec(degree: int = 2, **kwargs: Any) -> FitSpec:
    return FitSpec(model=PolynomialRef(degree=degree), **kwargs)


def assert_json_safe(bundle: FitBundle) -> None:
    """The bundle serialises to standard JSON (no NaN/Infinity) and parses back equal."""
    dumped = bundle.model_dump(mode="json")
    text = json.dumps(dumped, allow_nan=False)

    def reject(token: str) -> Any:
        raise AssertionError(f"non-finite JSON constant {token}")

    assert FitBundle.model_validate(json.loads(text, parse_constant=reject)) == bundle
    assert FitBundle.model_validate_json(bundle.model_dump_json()) == bundle


# -- FitSpec ------------------------------------------------------------------------------------


def test_spec_json_round_trip_and_discriminated_union() -> None:
    spec = FitSpec.model_validate(
        {
            "model": {"kind": "custom", "equation": "a*x+b", "parameters": ["a", "b"]},
            "bounds": {"a": [0.0, None]},
            "initial": {"a": 1.0},
            "use_weights": False,
            "seed": 7,
            "time_limit": 2.5,
        }
    )
    assert isinstance(spec.model, CustomRef)
    assert spec.model.variable == "x"
    assert FitSpec.model_validate_json(spec.model_dump_json()) == spec
    assert isinstance(
        FitSpec.model_validate({"model": {"kind": "polynomial", "degree": 3}}).model, PolynomialRef
    )
    assert isinstance(
        FitSpec.model_validate({"model": {"kind": "registered", "id": "gaussian"}}).model,
        RegisteredRef,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"kind": "polynomial", "degree": 0}},
        {"model": {"kind": "polynomial", "degree": 11}},
        {"model": {"kind": "unknown"}},
        {"model": {"kind": "polynomial", "degree": 2}, "time_limit": 0},
        {"model": {"kind": "polynomial", "degree": 2}, "seed": -1},
        {"model": {"kind": "polynomial", "degree": 2}, "extra": 1},
    ],
)
def test_spec_validation_rejects_bad_payloads(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FitSpec.model_validate(payload)


# -- polynomial / registered / custom on known truth --------------------------------------------


def test_polynomial_fit_recovers_truth_and_matches_the_engine() -> None:
    xy = quadratic_xy()
    bundle = fit_dataset(xy, poly_spec(2))
    assert bundle.result.status is FitStatus.CONVERGED
    direct = fit_polynomial(xy, 2)
    assert [p.value for p in bundle.result.parameters] == [p.value for p in direct.parameters]
    truth = {"b0": 1.0, "b1": 0.5, "b2": -0.03}
    for estimate in bundle.result.parameters:
        assert estimate.ci_low is not None
        assert estimate.ci_high is not None
        assert estimate.ci_low <= truth[estimate.name] <= estimate.ci_high
    assert bundle.equation_text.startswith("y = ")
    assert bundle.summary is not None
    assert bundle.summary.criteria.aic is not None
    assert_json_safe(bundle)


def test_curve_has_200_points_over_the_data_range_extended_five_percent() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    assert bundle.curve is not None
    assert bundle.curve_reason is None
    assert len(bundle.curve.x) == len(bundle.curve.y) == 200
    assert bundle.curve.x[0] == pytest.approx(-0.5)
    assert bundle.curve.x[-1] == pytest.approx(10.5)
    assert all(math.isfinite(v) for v in bundle.curve.y)


def test_bands_surround_the_curve_and_prediction_is_wider() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    assert bundle.confidence_band is not None
    assert bundle.prediction_band is not None
    assert bundle.bands_reason is None
    assert bundle.curve is not None
    conf, pred = bundle.confidence_band, bundle.prediction_band
    assert conf.x == bundle.curve.x
    assert conf.fit == pytest.approx(bundle.curve.y)
    for i in range(200):
        assert conf.lower[i] <= conf.fit[i] <= conf.upper[i]
        assert pred.lower[i] <= conf.lower[i]
        assert pred.upper[i] >= conf.upper[i]


def test_registered_model_fit_recovers_truth() -> None:
    bundle = fit_dataset(exponential_xy(), FitSpec(model=RegisteredRef(id="exp_growth")))
    assert bundle.result.status is FitStatus.CONVERGED
    a, b = bundle.result.parameters
    assert a.value == pytest.approx(2.0, rel=0.05)
    assert b.value == pytest.approx(0.3, rel=0.05)
    assert bundle.result.model_id == "exp_growth"
    assert bundle.curve is not None
    assert_json_safe(bundle)


def test_custom_equation_matches_the_registered_model() -> None:
    xy = exponential_xy()
    registered = fit_dataset(xy, FitSpec(model=RegisteredRef(id="exp_growth")))
    custom = fit_dataset(
        xy,
        FitSpec(
            model=CustomRef(equation="a*exp(b*x)", parameters=["a", "b"]),
            initial={"a": 1.0, "b": 0.1},
        ),
    )
    assert custom.result.status is FitStatus.CONVERGED
    for c, r in zip(custom.result.parameters, registered.result.parameters, strict=True):
        assert c.value == pytest.approx(r.value, rel=1e-6)
    assert "exp" in custom.equation_text
    assert_json_safe(custom)


def test_custom_equation_accepts_a_leading_y_equals_and_another_variable_name() -> None:
    xy = quadratic_xy()
    bundle = fit_dataset(
        xy, FitSpec(model=CustomRef(equation="y = p + q*t", parameters=["p", "q"], variable="t"))
    )
    assert bundle.result.status is FitStatus.CONVERGED
    assert [p.name for p in bundle.result.parameters] == ["p", "q"]


@pytest.mark.parametrize(
    ("equation", "hint"),
    [
        ("a*x^2", "**"),
        ("2x", "implicit multiplication"),
        ("a*Exp(x)", "exp"),
    ],
)
def test_custom_equation_typos_are_invalid_equation_with_the_hint(equation: str, hint: str) -> None:
    spec = FitSpec(model=CustomRef(equation=equation, parameters=["a"]))
    with pytest.raises(WorkflowError) as caught:
        fit_dataset(quadratic_xy(), spec)
    assert caught.value.code == "INVALID_EQUATION"
    assert hint in str(caught.value)


@pytest.mark.parametrize("parameters", [[], ["a", "a"], ["1a"], ["exp"]])
def test_custom_equation_with_bad_parameter_names_is_invalid_equation(
    parameters: list[str],
) -> None:
    spec = FitSpec(model=CustomRef(equation="2*x", parameters=parameters))
    with pytest.raises(WorkflowError) as caught:
        fit_dataset(quadratic_xy(), spec)
    assert caught.value.code == "INVALID_EQUATION"


def test_unknown_registered_model() -> None:
    with pytest.raises(WorkflowError) as caught:
        fit_dataset(quadratic_xy(), FitSpec(model=RegisteredRef(id="exp_growht")))
    assert caught.value.code == "UNKNOWN_MODEL"
    assert "exp_growth" in str(caught.value)


# -- spec/data combinations ---------------------------------------------------------------------


def test_weights_are_used_only_when_asked() -> None:
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 10.0, 30)
    y = 3.0 + 2.0 * x + rng.normal(0.0, 0.5, 30) * (1 + x)
    weights = 1.0 / (1 + x) ** 2
    xy = make_xy(x, y, weights, tag="w")
    weighted = fit_dataset(xy, poly_spec(1, use_weights=True))
    plain = fit_dataset(xy, poly_spec(1, use_weights=False))
    direct = fit_polynomial(xy, 1)
    assert [p.value for p in weighted.result.parameters] == [p.value for p in direct.parameters]
    assert [p.value for p in weighted.result.parameters] != [
        p.value for p in plain.result.parameters
    ]
    assert any("weights" in w for w in plain.warnings)
    assert (
        plain.result.provenance.dataset_fingerprint
        != weighted.result.provenance.dataset_fingerprint
    )
    assert not any("weights" in w for w in weighted.warnings)


def test_use_weights_without_weights_is_invalid_spec() -> None:
    with pytest.raises(WorkflowError) as caught:
        fit_dataset(quadratic_xy(), poly_spec(2, use_weights=True))
    assert caught.value.code == "INVALID_SPEC"


def test_polynomial_rejects_nonlinear_only_options() -> None:
    for kwargs in (
        {"method": "trf"},
        {"loss": "huber"},
        {"bounds": {"b0": (0.0, 1.0)}},
        {"initial": {"b0": 1.0}},
    ):
        with pytest.raises(WorkflowError) as caught:
            fit_dataset(quadratic_xy(), poly_spec(2, **kwargs))
        assert caught.value.code == "INVALID_SPEC", kwargs


def test_bad_bounds_initial_and_method_are_invalid_spec() -> None:
    xy = exponential_xy()
    cases = [
        {"bounds": {"zzz": (0.0, 1.0)}},
        {"bounds": {"a": (2.0, 1.0)}},
        {"initial": {"zzz": 1.0}},
        {"method": "nonsense"},
        {"loss": "huber", "method": "lm"},
    ]
    for kwargs in cases:
        spec = FitSpec(model=RegisteredRef(id="exp_growth"), **kwargs)
        with pytest.raises(WorkflowError) as caught:
            fit_dataset(xy, spec)
        assert caught.value.code == "INVALID_SPEC", kwargs


def test_open_ended_bounds_are_supported() -> None:
    spec = FitSpec(
        model=RegisteredRef(id="exp_growth"),
        bounds={"a": (0.0, None), "b": (None, 1.0)},
        method="trf",
    )
    bundle = fit_dataset(exponential_xy(), spec)
    assert bundle.result.status is FitStatus.CONVERGED


def test_fit_is_deterministic_for_a_seed() -> None:
    spec = FitSpec(model=RegisteredRef(id="exp_growth"), seed=5)
    first = fit_dataset(exponential_xy(), spec)
    second = fit_dataset(exponential_xy(), spec)
    assert [p.value for p in first.result.parameters] == [p.value for p in second.result.parameters]
    assert first.result.provenance.seed == 5


# -- failures are reported, never disguised -----------------------------------------------------


def test_time_limit_exceeded_gives_a_failed_bundle_without_a_curve() -> None:
    spec = FitSpec(model=RegisteredRef(id="exp_growth"), time_limit=1e-9)
    bundle = fit_dataset(exponential_xy(), spec)
    assert bundle.result.status is FitStatus.FAILED
    assert "time limit" in bundle.result.termination_reason
    assert bundle.curve is None
    assert bundle.curve_reason is not None
    assert bundle.confidence_band is None
    assert bundle.prediction_band is None
    assert bundle.bands_reason is not None
    assert bundle.summary is None
    assert bundle.summary_reason is not None
    assert any("time limit" in w for w in bundle.warnings)
    assert_json_safe(bundle)


def test_generous_time_limit_does_not_change_the_answer() -> None:
    xy = exponential_xy()
    free = fit_dataset(xy, FitSpec(model=RegisteredRef(id="exp_growth")))
    limited = fit_dataset(xy, FitSpec(model=RegisteredRef(id="exp_growth"), time_limit=60.0))
    assert [p.value for p in free.result.parameters] == [p.value for p in limited.result.parameters]


def test_unidentifiable_parameters_do_not_converge_and_have_no_curve() -> None:
    spec = FitSpec(
        model=CustomRef(equation="a*x + b*x", parameters=["a", "b"]),
        initial={"a": 1.0, "b": 2.0},
    )
    bundle = fit_dataset(quadratic_xy(), spec)
    assert bundle.result.status is not FitStatus.CONVERGED
    assert bundle.curve is None
    assert bundle.curve_reason is not None
    assert bundle.confidence_band is None
    assert bundle.summary is None
    assert_json_safe(bundle)


def test_too_few_points_is_invalid_input_without_a_curve() -> None:
    bundle = fit_dataset(make_xy([0.0, 1.0], [1.0, 2.0]), poly_spec(3))
    assert bundle.result.status is FitStatus.INVALID_INPUT
    assert bundle.curve is None
    assert bundle.summary is None
    assert_json_safe(bundle)


def _withhold_covariance(*_args: Any, **_kwargs: Any) -> Any:
    raise StatisticsError("NO_COVARIANCE", "the fit has no covariance matrix (test double)")


def test_undefined_bands_keep_the_curve_and_say_why(monkeypatch: pytest.MonkeyPatch) -> None:
    # A converged fit without a usable covariance is not easy to provoke from real data, so
    # the statistics layer's refusal is injected to check how the bundle reports it.
    monkeypatch.setattr("curvestudio.workflow.service.confidence_band", _withhold_covariance)
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    assert bundle.result.status is FitStatus.CONVERGED
    assert bundle.curve is not None
    assert bundle.confidence_band is None
    assert bundle.prediction_band is None
    assert bundle.bands_reason is not None
    assert "NO_COVARIANCE" in bundle.bands_reason
    assert_json_safe(bundle)


def test_constant_y_dataset_is_handled_without_nan() -> None:
    x = np.linspace(0.0, 9.0, 10)
    bundle = fit_dataset(make_xy(x, np.full(10, 4.0), tag="const"), poly_spec(1))
    assert bundle.result.status is FitStatus.CONVERGED
    assert bundle.result.stats is not None
    assert bundle.result.stats.r2 is None  # undefined, never NaN or 0
    assert bundle.curve is not None
    assert bundle.curve.y == pytest.approx([4.0] * 200)
    assert_json_safe(bundle)


def test_constant_x_dataset_does_not_crash() -> None:
    bundle = fit_dataset(make_xy([2.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]), poly_spec(1))
    assert bundle.result.status is FitStatus.FAILED
    assert bundle.curve is None
    assert bundle.curve_reason is not None
    assert_json_safe(bundle)


def _log_xy(lo: float = 0.1, hi: float = 10.0) -> XY:
    x = np.linspace(lo, hi, 30)
    rng = np.random.default_rng(8)
    return make_xy(x, 3.0 * np.log(x) + 1.0 + rng.normal(0.0, 0.02, 30), tag="logx")


def assert_curve_and_bands_trimmed_left(
    bundle: FitBundle, data_min: float, data_max: float
) -> None:
    assert bundle.result.status is FitStatus.CONVERGED
    assert bundle.curve is not None
    assert bundle.curve_reason is None
    assert len(bundle.curve.x) == 200
    assert bundle.curve.x[0] == pytest.approx(data_min)  # no extension into the undefined side
    assert bundle.curve.x[-1] > data_max  # the defined side keeps its 5 % margin
    assert bundle.confidence_band is not None
    assert bundle.prediction_band is not None
    assert any("not extended" in w for w in bundle.warnings)
    assert_json_safe(bundle)


def test_log_model_keeps_curve_and_bands_when_the_left_margin_leaves_its_domain() -> None:
    bundle = fit_dataset(_log_xy(), FitSpec(model=RegisteredRef(id="log_natural")))
    assert_curve_and_bands_trimmed_left(bundle, 0.1, 10.0)


def test_power_model_keeps_curve_and_bands_when_the_left_margin_leaves_its_domain() -> None:
    x = np.linspace(0.1, 10.0, 30)
    y = 2.0 * x**1.5 + np.random.default_rng(2).normal(0.0, 0.05, 30)
    bundle = fit_dataset(make_xy(x, y, tag="pow"), FitSpec(model=RegisteredRef(id="power")))
    assert_curve_and_bands_trimmed_left(bundle, 0.1, 10.0)


def test_custom_log_equation_keeps_curve_and_bands() -> None:
    spec = FitSpec(
        model=CustomRef(equation="a*ln(x) + b", parameters=["a", "b"]),
        initial={"a": 1.0, "b": 0.0},
    )
    assert_curve_and_bands_trimmed_left(fit_dataset(_log_xy(), spec), 0.1, 10.0)


def test_sqrt_model_starting_at_zero_is_trimmed_not_dropped() -> None:
    x = np.linspace(0.0, 10.0, 20)
    y = 2.0 * np.sqrt(x) + 1.0 + np.random.default_rng(1).normal(0, 0.01, 20)
    spec = FitSpec(
        model=CustomRef(equation="a*sqrt(x) + b", parameters=["a", "b"]),
        initial={"a": 1.0, "b": 0.0},
    )
    bundle = fit_dataset(make_xy(x, y, tag="sqrt"), spec)
    assert_curve_and_bands_trimmed_left(bundle, 0.0, 10.0)


def test_untrimmed_curve_has_no_trim_warning() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    assert not any("not extended" in w for w in bundle.warnings)


# -- predict ------------------------------------------------------------------------------------


def test_predict_matches_the_bundle_curve_and_bands() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    assert bundle.curve is not None
    assert bundle.confidence_band is not None
    assert bundle.prediction_band is not None
    xs = bundle.curve.x[10:20]
    plain = predict(bundle, poly_spec(2), xs, None)
    assert plain.y == pytest.approx(bundle.curve.y[10:20])
    assert plain.lower is None
    assert plain.upper is None
    conf = predict(bundle, poly_spec(2), xs, "confidence")
    assert conf.lower == pytest.approx(bundle.confidence_band.lower[10:20])
    assert conf.upper == pytest.approx(bundle.confidence_band.upper[10:20])
    pred = predict(bundle, poly_spec(2), xs, "prediction")
    assert pred.lower == pytest.approx(bundle.prediction_band.lower[10:20])
    assert pred.band == "prediction"
    assert pred.level == 0.95


def test_predict_flags_extrapolation() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    result = predict(bundle, poly_spec(2), [5.0, 12.0], "confidence")
    assert result.extrapolated == [False, True]


def test_predict_registered_and_custom_models() -> None:
    xy = exponential_xy()
    for spec in (
        FitSpec(model=RegisteredRef(id="exp_growth")),
        FitSpec(
            model=CustomRef(equation="a*exp(b*x)", parameters=["a", "b"]),
            initial={"a": 1.0, "b": 0.1},
        ),
    ):
        bundle = fit_dataset(xy, spec)
        out = predict(bundle, spec, [0.0, 1.0, 2.0], "prediction")
        assert out.y[1] == pytest.approx(2.0 * math.exp(0.3), rel=0.05)
        assert out.lower is not None
        assert out.lower[1] is not None


def test_predict_gives_none_rows_where_the_model_is_undefined() -> None:
    xy = make_xy(
        np.linspace(1.0, 10.0, 20),
        3.0 * np.log(np.linspace(1.0, 10.0, 20))
        + 1.0
        + np.random.default_rng(4).normal(0, 0.02, 20),
        tag="log",
    )
    spec = FitSpec(
        model=CustomRef(equation="a*ln(x) + b", parameters=["a", "b"]), initial={"a": 1.0, "b": 0.0}
    )
    bundle = fit_dataset(xy, spec)
    out = predict(bundle, spec, [-1.0, 2.0, 0.0, 5.0], "confidence")
    assert out.y[0] is None
    assert out.y[2] is None
    assert out.y[1] == pytest.approx(3.0 * math.log(2.0) + 1.0, abs=0.1)
    assert out.lower is not None
    assert out.lower[0] is None
    assert out.lower[1] is not None
    assert out.upper is not None
    assert out.upper[3] is not None
    json.dumps(out.model_dump(mode="json"), allow_nan=False)


def test_predict_with_undefined_bands_returns_none_rows_and_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = poly_spec(2)
    bundle = fit_dataset(quadratic_xy(), spec)
    monkeypatch.setattr("curvestudio.workflow.service.confidence_band", _withhold_covariance)
    out = predict(bundle, spec, [0.5, 1.5], "confidence")
    assert all(v is not None for v in out.y)
    assert out.lower == [None, None]
    assert out.upper == [None, None]
    assert out.band_reason is not None
    assert "NO_COVARIANCE" in out.band_reason


def test_predict_empty_x_and_non_finite_x() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    empty = predict(bundle, poly_spec(2), [], "confidence")
    assert empty.x == []
    assert empty.y == []
    out = predict(bundle, poly_spec(2), [1.0, float("nan"), float("inf")], "confidence")
    assert out.y[0] is not None
    assert out.y[1:] == [None, None]
    json.dumps(out.model_dump(mode="json"), allow_nan=False)


def test_predict_refuses_a_fit_that_did_not_converge() -> None:
    spec = FitSpec(model=RegisteredRef(id="exp_growth"), time_limit=1e-9)
    bundle = fit_dataset(exponential_xy(), spec)
    with pytest.raises(WorkflowError) as caught:
        predict(bundle, spec, [1.0], None)
    assert caught.value.code == "FIT_NOT_CONVERGED"


def test_predict_with_the_wrong_spec_is_a_model_mismatch() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    with pytest.raises(WorkflowError) as caught:
        predict(bundle, FitSpec(model=RegisteredRef(id="exp_growth")), [1.0], None)
    assert caught.value.code == "MODEL_MISMATCH"


def test_predict_from_a_bare_result_needs_the_data() -> None:
    xy = quadratic_xy()
    bundle = fit_dataset(xy, poly_spec(2))
    with pytest.raises(WorkflowError) as caught:
        predict(bundle.result, poly_spec(2), [1.0], None)
    assert caught.value.code == "DATA_REQUIRED"
    out = predict(bundle.result, poly_spec(2), [1.0], "confidence", data=xy)
    assert out.y[0] == pytest.approx(predict(bundle, poly_spec(2), [1.0], None).y[0])


def test_predict_survives_a_json_round_trip_of_the_bundle() -> None:
    spec = poly_spec(2)
    bundle = fit_dataset(quadratic_xy(), spec)
    restored = FitBundle.model_validate_json(bundle.model_dump_json())
    assert predict(restored, spec, [3.0, 4.0], "prediction") == predict(
        bundle, spec, [3.0, 4.0], "prediction"
    )


def test_predict_with_a_different_equation_is_a_model_mismatch() -> None:
    xy = exponential_xy()
    fitted = FitSpec(
        model=CustomRef(equation="a*exp(b*x)", parameters=["a", "b"]), initial={"a": 1.0, "b": 0.1}
    )
    bundle = fit_dataset(xy, fitted)
    other = FitSpec(model=CustomRef(equation="a*x + b", parameters=["a", "b"]))
    with pytest.raises(WorkflowError) as caught:
        predict(bundle, other, [1.0], None)
    assert caught.value.code == "MODEL_MISMATCH"
    spaced = FitSpec(model=CustomRef(equation="y =  a * exp( b*x )", parameters=["a", "b"]))
    assert predict(bundle, spaced, [1.0], None).y[0] is not None


# -- custom equation pre-checks ---------------------------------------------------------


def test_unused_parameter_is_invalid_equation() -> None:
    spec = FitSpec(model=CustomRef(equation="a*x", parameters=["a", "b"]))
    with pytest.raises(WorkflowError) as caught:
        fit_dataset(quadratic_xy(), spec)
    assert caught.value.code == "INVALID_EQUATION"
    assert "parameter 'b' is not used" in caught.value.message


def test_empty_parameter_list_asks_for_the_names() -> None:
    spec = FitSpec(model=CustomRef(equation="a*exp(b*x)", parameters=[]))
    with pytest.raises(WorkflowError) as caught:
        fit_dataset(quadratic_xy(), spec)
    assert caught.value.code == "INVALID_EQUATION"
    assert "list the parameter names" in caught.value.message
    assert "['a','b']" in caught.value.message
    assert "a, b" in caught.value.message


def test_empty_parameter_list_with_an_invalid_text_reports_the_grammar_problem_first() -> None:
    spec = FitSpec(model=CustomRef(equation="a*x^2", parameters=[]))
    with pytest.raises(WorkflowError) as caught:
        fit_dataset(quadratic_xy(), spec)
    assert caught.value.code == "INVALID_EQUATION"
    assert "**" in caught.value.message


def test_seed_is_capped_to_a_json_safe_integer() -> None:
    FitSpec(model=PolynomialRef(degree=1), seed=2**53 - 1)
    with pytest.raises(ValidationError):
        FitSpec(model=PolynomialRef(degree=1), seed=2**53)


# -- compact bundles / prediction size ----------------------------------------------------------


def big_linear_xy(n: int) -> XY:
    x = np.linspace(0.0, 100.0, n)
    y = 4.0 + 0.5 * x + np.random.default_rng(6).normal(0.0, 1.0, n)
    return make_xy(x, y, tag=f"big{n}")


def test_snapshot_records_the_total_and_is_full_by_default() -> None:
    bundle = fit_dataset(big_linear_xy(6000), poly_spec(1))
    assert bundle.data.n_total == 6000
    assert bundle.data.subsampled is False
    assert len(bundle.data.x) == 6000


def test_compact_subsamples_the_snapshot_deterministically() -> None:
    xy = big_linear_xy(6000)
    first = fit_dataset(xy, poly_spec(1), compact=True)
    second = fit_dataset(xy, poly_spec(1), compact=True)
    assert first.data == second.data
    assert first.data.subsampled is True
    assert first.data.n_total == 6000
    assert len(first.data.x) == len(first.data.y) == 5000
    assert first.data.x[0] == xy.x[0]
    assert first.data.x[-1] == xy.x[-1]
    assert first.data.x == sorted(first.data.x)
    # the fit itself is computed on all rows
    # (the provenance timestamp has one-second resolution, so a slow runner can cross a second)
    full = fit_dataset(xy, poly_spec(1)).result
    assert full.model_dump(exclude={"provenance"}) == first.result.model_dump(
        exclude={"provenance"}
    )
    assert_json_safe(first)


def test_compact_leaves_small_data_alone() -> None:
    bundle = fit_dataset(big_linear_xy(300), poly_spec(1), compact=True)
    assert bundle.data.subsampled is False
    assert len(bundle.data.x) == 300


def test_predict_from_a_subsampled_bundle_needs_the_full_data() -> None:
    xy = big_linear_xy(6000)
    bundle = fit_dataset(xy, poly_spec(1), compact=True)
    with pytest.raises(WorkflowError) as caught:
        predict(bundle, poly_spec(1), [1.0], "confidence")
    assert caught.value.code == "DATA_REQUIRED"
    assert "subsample" in caught.value.message
    out = predict(bundle, poly_spec(1), [1.0], "confidence", data=xy)
    assert out.lower is not None
    assert out.lower[0] is not None


def test_predict_refuses_an_excessive_number_of_points() -> None:
    bundle = fit_dataset(quadratic_xy(), poly_spec(2))
    assert MAX_PREDICT_POINTS == 100_000
    with pytest.raises(WorkflowError) as caught:
        predict(bundle, poly_spec(2), [0.0] * (MAX_PREDICT_POINTS + 1), None)
    assert caught.value.code == "TOO_LARGE"
    assert len(predict(bundle, poly_spec(2), [1.0] * MAX_PREDICT_POINTS, None).y) == 100_000
