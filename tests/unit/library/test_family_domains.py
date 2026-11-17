"""Domain, derivative and size checks for the hand-written growth, sigmoid and dose families.

The catalogue contract checks a model only at its example points. This module sweeps the whole
example range for a finite value and a finite analytic Jacobian, asserts the per-family
minima the plan requires, and checks that models whose formula needs a positive (or
non-negative) x say so in their applicability, so Auto-fit skips them with a reason instead
of fitting data they cannot take.
"""

from collections import Counter

import numpy as np
import pytest

from curvestudio.models import ModelSpec
from curvestudio_models import catalogue

pytestmark = pytest.mark.real_plugins

FAMILIES = ("growth", "sigmoidal", "dose_response")
MODELS: dict[str, ModelSpec] = {m.id: m for m in catalogue() if m.category in FAMILIES}


@pytest.mark.parametrize("spec", MODELS.values(), ids=lambda s: s.id)
def test_values_and_jacobian_are_finite_across_the_example_range(spec: ModelSpec) -> None:
    assert spec.example is not None
    low, high = spec.example.x_range
    x = np.linspace(low, high, 201)
    params = dict(spec.example.params)
    assert np.isfinite(spec.evaluate(x, **params)).all()
    assert np.isfinite(spec.jacobian(x, **params)).all()


def test_each_family_meets_its_minimum_size() -> None:
    counts = Counter(m.category for m in catalogue())
    assert counts["growth"] >= 14
    assert counts["sigmoidal"] >= 14
    assert counts["dose_response"] >= 10


POSITIVE_X_MODELS = (
    # growth
    "chapman_richards",
    "weibull_growth",
    "morgan_mercer_flodin",
    "allometric_offset",
    "exponential_power_growth",
    # sigmoidal
    "weibull_cdf3",
    "hill_sigmoid",
    "frechet_cdf",
    # dose response
    "dose_response_ll2",
    "dose_response_ll3",
    "dose_response_4pl",
    "dose_response_5pl",
    "dose_response_weibull1_3",
    "dose_response_brain_cousens",
    "dose_response_cedergreen",
    "dose_response_lognormal",
)
"""Models whose formula takes a power or a logarithm of x and therefore needs x > 0."""


@pytest.mark.parametrize("model_id", POSITIVE_X_MODELS)
def test_positive_x_models_reject_non_positive_data(model_id: str) -> None:
    reasons = MODELS[model_id].applicability(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert any("strictly positive" in reason for reason in reasons), (model_id, reasons)


@pytest.mark.parametrize("model_id", ["dose_response_emax", "dose_response_inhibitory_emax"])
def test_non_negative_dose_models_reject_negative_data(model_id: str) -> None:
    reasons = MODELS[model_id].applicability(np.array([-1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert any("non-negative" in reason for reason in reasons), (model_id, reasons)
