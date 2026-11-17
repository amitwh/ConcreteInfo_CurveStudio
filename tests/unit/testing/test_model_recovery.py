"""The noiseless parameter-recovery harness."""

import math

import numpy as np
import pytest

from curvestudio.fitting import FitConfig
from curvestudio.models import ModelExample, ModelSpec, Parameter, get_model
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import STARTER_MODELS, register_starter_models
from curvestudio.testing import model_recovery as recovery
from curvestudio.testing.model_recovery import RecoveryReport, check_model_recovery


@pytest.fixture(autouse=True)
def starters() -> None:
    _clear_registry()
    register_starter_models()


def _spec(expression: str, params: dict[str, float], x_range: tuple[float, float], **kw: object):
    parameters = tuple(Parameter(name) for name in params)
    return ModelSpec(
        id="probe",
        name="Probe",
        category="exponential",
        expression=expression,
        parameters=parameters,
        initial_guess=lambda x, y: dict.fromkeys(params, 1.0),
        example=ModelExample(params, x_range),
        **kw,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("model_id", ["michaelis_menten", "gaussian"])
def test_recovery_passes_for_hand_written_examples(model_id: str) -> None:
    spec = get_model(model_id)
    assert spec.example is not None
    report = check_model_recovery(spec)
    assert isinstance(report, RecoveryReport)
    assert report.converged
    assert report.max_relative_error <= 1e-6
    assert math.isfinite(report.jacobian_condition)
    assert 1.0 <= report.jacobian_condition <= 1e8


@pytest.mark.parametrize("spec", STARTER_MODELS, ids=lambda s: s.id)
def test_every_starter_recovers_its_example(spec: ModelSpec) -> None:
    report = check_model_recovery(spec)
    assert report.max_relative_error <= 1e-6
    assert report.single_start_recovered


def test_recovery_is_reproducible_for_a_seed() -> None:
    spec = get_model("gaussian")
    a = check_model_recovery(spec, n_starts=4, seed=7)
    b = check_model_recovery(spec, n_starts=4, seed=7)
    assert a == b


def test_ambiguous_model_fails_naming_the_parameters() -> None:
    spec = _spec("a*b*x", {"a": 2.0, "b": 3.0}, (0.0, 5.0))
    with pytest.raises(AssertionError) as info:
        check_model_recovery(spec)
    message = str(info.value)
    assert "'a'" in message and "'b'" in message
    assert "not separately identifiable" in message


def test_ill_conditioned_example_fails() -> None:
    # Two columns that differ only by a 1e-9 relative wiggle: identifiable in exact
    # arithmetic, hopeless in practice.
    spec = _spec("a*x + b*(x + 1e-9*x**2)", {"a": 1.0, "b": 1.0}, (0.0, 1.0))
    with pytest.raises(AssertionError, match=r"condition number"):
        check_model_recovery(spec)


def test_unconstrained_parameter_is_named() -> None:
    spec = _spec("a*x + 0*b", {"a": 1.0, "b": 1.0}, (0.0, 1.0))
    with pytest.raises(AssertionError, match=r"\['b'\]"):
        check_model_recovery(spec)


def test_missing_example_fails() -> None:
    spec = ModelSpec(
        id="noex",
        name="No example",
        category="exponential",
        expression="a*x",
        parameters=(Parameter("a"),),
        initial_guess=lambda x, y: {"a": 1.0},
    )
    with pytest.raises(AssertionError, match="no example"):
        check_model_recovery(spec)


def test_truth_outside_bounds_is_a_recovery_failure_naming_a_parameter() -> None:
    spec = ModelSpec(
        id="bounded",
        name="Bounded",
        category="exponential",
        expression="a*x + b",
        parameters=(Parameter("a", lower=5.0), Parameter("b")),
        initial_guess=lambda x, y: {"a": 6.0, "b": 0.0},
        example=ModelExample({"a": 2.0, "b": 1.0}, (0.0, 3.0)),
    )
    with pytest.raises(AssertionError, match=r"parameter '[ab]' recovered as"):
        check_model_recovery(spec)


def test_non_finite_example_curve_fails() -> None:
    spec = _spec("a*log(x)", {"a": 1.0}, (-1.0, 1.0))
    with pytest.raises(AssertionError, match="not finite"):
        check_model_recovery(spec)


def test_example_data_matches_numpy() -> None:
    spec = get_model("michaelis_menten")
    x, y = spec.example_data()
    p = spec.example.params  # type: ignore[union-attr]
    np.testing.assert_allclose(y, p["vmax"] * x / (p["km"] + x))


def test_report_separates_single_start_status_from_recovery() -> None:
    report = check_model_recovery(get_model("gaussian"), require_single_start=True)
    assert report.single_start_converged
    assert report.single_start_recovered


def test_require_single_start_fails_when_the_single_start_misses_the_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A single start that "converges" to a wrong answer (another local minimum): stand it in
    # by shifting one fitted parameter of the single-start result only.
    real_fit = recovery.fit

    def fit_with_wrong_single_start(spec: ModelSpec, data: object, config: FitConfig):  # type: ignore[no-untyped-def]
        result = real_fit(spec, data, config)  # type: ignore[arg-type]
        if config.n_starts != 1:
            return result
        moved = tuple(
            p.model_copy(update={"value": p.value + 0.01}) if p.name == "mu" else p
            for p in result.parameters
        )
        return result.model_copy(update={"parameters": moved})

    monkeypatch.setattr(recovery, "fit", fit_with_wrong_single_start)
    spec = get_model("gaussian")
    report = check_model_recovery(spec)
    assert report.converged
    assert report.single_start_converged
    assert not report.single_start_recovered
    with pytest.raises(AssertionError, match=r"single start.*'mu'"):
        check_model_recovery(spec, require_single_start=True)
