"""Applicability rules and the size of the peak family."""

import numpy as np
import pytest

from curvestudio.models import ModelSpec
from curvestudio_models import catalogue

pytestmark = pytest.mark.real_plugins

BY_ID: dict[str, ModelSpec] = {m.id: m for m in catalogue()}
PEAK_IDS = {
    "peak_gaussian_offset",
    "lorentzian_centered",
    "peak_lorentzian",
    "peak_lorentzian_offset",
    "peak_pearson_vii",
    "peak_pseudo_voigt",
    "peak_voigt",
    "peak_emg",
    "peak_skew_normal",
    "peak_split_gaussian",
    "peak_power_tail",
    "peak_two_gaussian",
    "peak_three_gaussian",
}
X = np.linspace(-2.0, 4.0, 5)


def test_the_peak_family_has_the_models_of_the_plan() -> None:
    peaks = {m.id for m in catalogue() if m.category == "peak"}
    assert peaks >= PEAK_IDS
    assert len(peaks) >= 12


def test_every_peak_is_in_the_peak_category() -> None:
    assert {m.id for m in catalogue() if m.category == "peak"} >= PEAK_IDS


@pytest.mark.parametrize("model_id", sorted(PEAK_IDS))
def test_peaks_accept_any_finite_data(model_id: str) -> None:
    assert BY_ID[model_id].applicability(X, np.linspace(0.0, 1.0, 5)) == []


@pytest.mark.parametrize("model_id", sorted(PEAK_IDS))
def test_peaks_reject_data_that_is_not_finite(model_id: str) -> None:
    reasons = BY_ID[model_id].applicability(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert any("x must be finite" in reason for reason in reasons), reasons
    reasons = BY_ID[model_id].applicability(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.inf, 3.0]))
    assert any("y must be finite" in reason for reason in reasons), reasons


def test_the_widths_of_the_peaks_are_bounded_above_zero() -> None:
    widths = {"s", "g", "w", "sigma", "gamma", "tau", "s1", "s2", "b", "nu", "eta"}
    for model_id in sorted(PEAK_IDS):
        for parameter in BY_ID[model_id].parameters:
            if parameter.name not in widths:
                continue
            assert parameter.lower >= 0.0, (model_id, parameter.name)


def test_the_two_peak_sums_keep_their_centres_in_order() -> None:
    for model_id in ("peak_two_gaussian", "peak_three_gaussian"):
        gaps = [p for p in BY_ID[model_id].parameters if p.name.startswith("d")]
        assert gaps, model_id
        for gap in gaps:
            assert gap.lower >= 0.0, (model_id, gap.name)


def test_a_peak_model_has_a_bounded_number_of_parameters() -> None:
    for model_id in sorted(PEAK_IDS):
        assert len(BY_ID[model_id].parameters) <= 9, model_id


def test_the_exponential_tail_of_the_emg_is_bounded_above_zero() -> None:
    """The EMG divides by tau, so tau = 0 must lie outside its bounds, not on them."""
    tau = next(p for p in BY_ID["peak_emg"].parameters if p.name == "tau")
    assert tau.lower > 0.0


def test_the_emg_evaluates_without_an_exception_at_its_tau_corner() -> None:
    """tau sits strictly inside its bounds and the expression divides by tau, so a direct
    evaluation at the declared corner must not raise (a degenerate corner may still return
    non-finite values, as in every sibling model)."""
    model = BY_ID["peak_emg"]
    corner = next(p for p in model.parameters if p.name == "tau").lower
    with np.errstate(all="ignore"):
        values = model.evaluate(X, a=2.0, mu=0.5, sigma=0.7, tau=corner)
    assert values.shape == X.shape
