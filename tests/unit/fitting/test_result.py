"""Tests for FitResult, its parts and lossless JSON serialization."""

import json
import re
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from curvestudio.fitting import (
    FitResult,
    FitStatistics,
    FitStatus,
    ParameterEstimate,
    Provenance,
    collect_provenance,
    fit_statistics,
)


def make_result(**overrides: Any) -> FitResult:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    fitted = np.array([1.1, 1.9, 3.2, 3.8])
    kwargs: dict[str, Any] = {
        "model_id": "linear",
        "equation": "a + b*x",
        "equation_fitted": "0.1 + 0.97*x",
        "parameters": (
            ParameterEstimate(name="a", value=0.1, std_err=0.05, ci_low=-0.1, ci_high=0.3),
            ParameterEstimate(name="b", value=0.97, std_err=None, ci_low=None, ci_high=None),
        ),
        "covariance": [[0.0025, -0.001], [-0.001, 0.0004]],
        "correlation": [[1.0, -1.0], [-1.0, 1.0]],
        "stats": fit_statistics(y, fitted, n_params=2),
        "fitted": fitted.tolist(),
        "residuals": (y - fitted).tolist(),
        "status": FitStatus.CONVERGED,
        "termination_reason": "converged",
        "method": "least_squares",
        "loss": "linear",
        "initial_params": {"a": 0.0, "b": 1.0},
        "n_starts": 1,
        "warnings": ("ill-conditioned design matrix",),
        "provenance": collect_provenance("abc123", seed=7),
    }
    kwargs.update(overrides)
    return FitResult(**kwargs)


def test_json_round_trip_is_exact() -> None:
    r = make_result()
    assert FitResult.from_json(r.to_json()) == r


def test_json_round_trip_preserves_awkward_floats() -> None:
    rng = np.random.default_rng(0)
    values = [0.1 + 0.2, 1e-300, 5e-324, 1.7976931348623157e308, -0.0, 1 / 3]
    values += rng.normal(scale=1e5, size=200).tolist()
    r = make_result(fitted=values, residuals=values)
    assert FitResult.from_json(r.to_json()) == r


def test_optional_blocks_none_round_trip() -> None:
    r = make_result(covariance=None, correlation=None, stats=None, equation_fitted=None)
    back = FitResult.from_json(r.to_json())
    assert back == r
    assert back.covariance is None


def test_json_has_no_nan_or_infinity_tokens() -> None:
    text = make_result().to_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant {c}"))


def test_status_serialises_as_plain_string() -> None:
    data = json.loads(make_result(status=FitStatus.NOT_CONVERGED).to_json())
    assert data["status"] == "not_converged"


def test_created_at_is_ist_iso_8601() -> None:
    p = collect_provenance("fp", seed=1)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+05:30", p.created_at)


def test_collect_provenance_records_versions() -> None:
    p = collect_provenance("fp", seed=3)
    assert p.dataset_fingerprint == "fp"
    assert p.seed == 3
    assert set(p.software_versions) == {"curvestudio", "numpy", "scipy", "lmfit", "python"}
    assert p.software_versions["numpy"] == np.__version__
    assert p.software_versions["python"].startswith("3.14")


def test_collect_provenance_never_raises_on_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as md

    real = md.version

    def fake(name: str) -> str:
        if name == "lmfit":
            raise md.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr("curvestudio.fitting.result.metadata.version", fake)
    assert collect_provenance("fp", seed=0).software_versions["lmfit"] == "unknown"


def test_provenance_rejects_non_ist_timestamp() -> None:
    with pytest.raises(ValidationError):
        Provenance(
            software_versions={}, dataset_fingerprint="x", created_at="2026-11-16T10:00:00Z", seed=0
        )
    with pytest.raises(ValidationError):
        Provenance(
            software_versions={}, dataset_fingerprint="x", created_at="2026-11-16T10:00:00", seed=0
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
class TestNonFiniteRejected:
    def test_fitted(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            make_result(fitted=[1.0, bad])

    def test_residuals(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            make_result(residuals=[bad])

    def test_covariance(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            make_result(covariance=[[bad, 0.0], [0.0, 1.0]])

    def test_correlation(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            make_result(correlation=[[1.0, bad], [bad, 1.0]])

    def test_parameter_value(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            ParameterEstimate(name="a", value=bad)

    def test_parameter_std_err_and_bounds(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            ParameterEstimate(name="a", value=1.0, std_err=bad)
        with pytest.raises(ValidationError):
            ParameterEstimate(name="a", value=1.0, ci_low=bad)
        with pytest.raises(ValidationError):
            ParameterEstimate(name="a", value=1.0, ci_high=bad)

    def test_initial_params(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            make_result(initial_params={"a": bad})

    @pytest.mark.parametrize("field", ["sse", "mse", "rmse", "mae", "r2", "adj_r2", "residual_se"])
    def test_stats(self, bad: float, field: str) -> None:
        good = fit_statistics(
            np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 1.9, 3.2, 3.8]), n_params=2
        )
        data = good.model_dump()
        data[field] = bad
        with pytest.raises(ValidationError):
            FitStatistics(**data)


def test_to_json_refuses_non_finite_smuggled_past_frozen_lists() -> None:
    r = make_result()
    r.fitted.append(float("nan"))  # list contents are not deep-frozen
    with pytest.raises(ValueError):
        r.to_json()


def test_from_json_rejects_nan_tokens() -> None:
    text = make_result().to_json()
    assert '"fitted": [' in text
    text = text.replace('"fitted": [', '"fitted": [NaN, ', 1)
    with pytest.raises(ValueError):
        FitResult.from_json(text)


def test_from_json_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        FitResult.from_json("{not json")
    with pytest.raises(ValueError):
        FitResult.from_json("{}")


def test_result_is_frozen() -> None:
    r = make_result()
    with pytest.raises(ValidationError):
        r.method = "other"  # type: ignore[misc]


def test_unknown_status_rejected() -> None:
    with pytest.raises(ValidationError):
        make_result(status="exploded")
