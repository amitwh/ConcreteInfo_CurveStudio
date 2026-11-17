"""Applicability rules and the size of the distribution family."""

import numpy as np
import pytest

from curvestudio.models import ModelSpec
from curvestudio_models import catalogue

pytestmark = pytest.mark.real_plugins

BY_ID: dict[str, ModelSpec] = {m.id: m for m in catalogue()}
DISTRIBUTION_IDS = {
    "distribution_lognormal",
    "distribution_gamma",
    "distribution_weibull",
    "distribution_cauchy",
    "distribution_levy",
    "distribution_rayleigh",
    "distribution_maxwell_boltzmann",
    "distribution_beta",
    "distribution_logistic",
    "distribution_gumbel",
    "distribution_laplace",
    "distribution_inverse_gaussian",
    "distribution_normal",
    "distribution_log_logistic",
}
POSITIVE_X = (
    "distribution_lognormal",
    "distribution_gamma",
    "distribution_weibull",
    "distribution_levy",
    "distribution_inverse_gaussian",
    "distribution_log_logistic",
)
NON_NEGATIVE_X = ("distribution_rayleigh", "distribution_maxwell_boltzmann")
X = np.linspace(-2.0, 4.0, 5)


def test_the_distribution_family_has_the_models_of_the_plan() -> None:
    distributions = {m.id for m in catalogue() if m.category == "distribution"}
    assert distributions >= DISTRIBUTION_IDS
    assert len(distributions) >= 12


def test_every_distribution_is_in_the_distribution_category() -> None:
    assert {m.id for m in catalogue() if m.category == "distribution"} >= DISTRIBUTION_IDS


@pytest.mark.parametrize("model_id", sorted(POSITIVE_X))
def test_distributions_that_need_a_positive_x_reject_the_others(model_id: str) -> None:
    reasons = BY_ID[model_id].applicability(np.array([0.0, 1.0, 2.0]), np.ones(3))
    assert any("strictly positive" in reason for reason in reasons), reasons
    assert BY_ID[model_id].applicability(np.array([0.5, 1.0, 2.0]), np.ones(3)) == []


@pytest.mark.parametrize("model_id", sorted(NON_NEGATIVE_X))
def test_distributions_that_need_a_non_negative_x_reject_the_others(model_id: str) -> None:
    reasons = BY_ID[model_id].applicability(np.array([-1.0, 1.0, 2.0]), np.ones(3))
    assert any("must not be negative" in reason for reason in reasons), reasons
    assert BY_ID[model_id].applicability(np.array([0.0, 1.0, 2.0]), np.ones(3)) == []


def test_the_beta_needs_x_between_zero_and_one() -> None:
    model = BY_ID["distribution_beta"]
    reasons = model.applicability(np.array([0.0, 0.5, 1.0]), np.ones(3))
    assert any("strictly positive" in reason for reason in reasons), reasons
    assert any("below 1" in reason for reason in reasons), reasons
    assert model.applicability(np.array([0.1, 0.5, 0.9]), np.ones(3)) == []


@pytest.mark.parametrize(
    "model_id",
    sorted(DISTRIBUTION_IDS - set(POSITIVE_X) - set(NON_NEGATIVE_X) - {"distribution_beta"}),
)
def test_the_other_distributions_accept_any_x(model_id: str) -> None:
    assert BY_ID[model_id].applicability(X, np.linspace(0.0, 1.0, 5)) == []


@pytest.mark.parametrize("model_id", sorted(DISTRIBUTION_IDS))
def test_distributions_reject_data_that_is_not_finite(model_id: str) -> None:
    reasons = BY_ID[model_id].applicability(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]))
    assert any("y must be finite" in reason for reason in reasons), reasons


def test_the_amplitude_of_a_distribution_is_not_negative() -> None:
    for model_id in sorted(DISTRIBUTION_IDS):
        amplitude = BY_ID[model_id].parameters[0]
        assert amplitude.name == "a", model_id
        assert amplitude.lower >= 0.0, model_id


def test_the_logistic_density_underflows_to_zero_in_the_far_tails() -> None:
    """The symmetric form pairs no underflowed 0 with an inf denominator, so 1e3 scale
    widths out the density is a clean 0 rather than NaN."""
    values = BY_ID["distribution_logistic"].evaluate(np.array([-600.0, 600.0]), a=2.0, m=0.5, s=0.8)
    assert np.isfinite(values).all(), values
    np.testing.assert_array_equal(values, 0.0)


def test_the_gamma_density_survives_a_huge_x() -> None:
    """Where x**(k-1) overflows to inf and exp(-x/t) underflows to 0, the exact rewrite in
    the exponent returns the true value 0 rather than the NaN of inf*0."""
    gamma = BY_ID["distribution_gamma"].evaluate(np.array([1e130]), a=2.0, k=3.5, t=3.0)
    assert np.isfinite(gamma).all(), gamma
    np.testing.assert_array_equal(gamma, 0.0)
