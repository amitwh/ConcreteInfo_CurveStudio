"""Confidence and prediction bands equal Statsmodels' ``get_prediction`` bounds."""

from typing import Any

import numpy as np
import pytest
from crosscheck import fixture_xy, load_fixtures

from curvestudio.fitting import fit_polynomial
from curvestudio.statistics import FitView, build_view, confidence_band, prediction_band

FIXTURES = load_fixtures()
RTOL = 1e-9


def close(actual: object, expected: object) -> None:
    reference = np.asarray(expected, dtype=float)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        reference,
        rtol=RTOL,
        atol=1e-12 * float(np.max(np.abs(reference))),
    )


def view_and_reference(name: str) -> tuple[FitView, dict[str, Any], np.ndarray[Any, Any]]:
    entry = FIXTURES[name]
    data = fixture_xy(name, entry)
    view = build_view(data, fit_polynomial(data, entry["degree"]))
    reference = entry["bands"]
    return view, reference, np.asarray(reference["x"], dtype=float)


def test_every_dataset_has_a_grid_with_three_points_outside_the_data() -> None:
    for name, entry in FIXTURES.items():
        grid = np.asarray(entry["bands"]["x"])
        outside = (grid < min(entry["x"])) | (grid > max(entry["x"]))
        assert grid.size == 15 and int(outside.sum()) == 3, name


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_confidence_band_equals_statsmodels(name: str) -> None:
    view, ref, grid = view_and_reference(name)
    band = confidence_band(view, grid, level=0.95)
    close(band.fit, ref["mean"])
    close(band.lower, ref["mean_ci_lower"])
    close(band.upper, ref["mean_ci_upper"])
    outside = (grid < view.x.min()) | (grid > view.x.max())
    assert band.extrapolated == outside.tolist()


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_prediction_band_equals_statsmodels(name: str) -> None:
    view, ref, grid = view_and_reference(name)
    new_weight = 1.0 if ref["new_weight"] is None else np.asarray(ref["new_weight"], dtype=float)
    band = prediction_band(view, grid, level=0.95, new_weight=new_weight)
    close(band.fit, ref["mean"])
    close(band.lower, ref["obs_ci_lower"])
    close(band.upper, ref["obs_ci_upper"])


def test_the_weighted_dataset_uses_inverse_square_prediction_weights() -> None:
    ref = FIXTURES["wls35"]["bands"]
    np.testing.assert_allclose(ref["new_weight"], 1.0 / np.asarray(ref["x"]) ** 2)
    assert all(FIXTURES[n]["bands"]["new_weight"] is None for n in FIXTURES if n != "wls35")
