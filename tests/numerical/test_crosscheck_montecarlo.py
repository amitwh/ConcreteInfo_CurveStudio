"""Monte-Carlo sampling and bands agree with the stored Statsmodels fixtures.

The fixtures hold the Statsmodels covariance of every dataset and its ``get_prediction``
confidence band. The Monte-Carlo sampler draws from the covariance of our fit, so the
covariance of the draws must be the Statsmodels covariance, and the Monte-Carlo band must
be the Statsmodels band with its Student-t quantile replaced by the normal one (the
Monte-Carlo band is a percentile band of normal draws and carries no t correction).

Two tolerances are looser than the 1e-9 of the deterministic cross-checks, because a
Monte-Carlo estimate carries a sampling error: the covariance of 200 000 draws is good to
about 1%, and a percentile of 100 000 draws to about 0.4% of the half width.
"""

from typing import Any

import numpy as np
import pytest
from crosscheck import fixture_xy, load_fixtures
from scipy import stats as sps

from curvestudio.fitting import fit_polynomial
from curvestudio.statistics import FitView, build_view, monte_carlo_band, sample_parameters

FIXTURES = load_fixtures()
DRAWS = 200_000
BAND_DRAWS = 100_000
COV_RTOL = 0.02
BAND_RTOL = 0.03


def view_and_entry(name: str) -> tuple[FitView, dict[str, Any]]:
    entry = FIXTURES[name]
    data = fixture_xy(name, entry)
    return build_view(data, fit_polynomial(data, entry["degree"])), entry


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_sampled_parameters_reproduce_the_statsmodels_covariance(name: str) -> None:
    view, entry = view_and_entry(name)
    reference = np.asarray(entry["fit"]["cov"], dtype=float)
    parameters = np.asarray(entry["fit"]["params"], dtype=float)

    draws = sample_parameters(view, n=DRAWS, seed=0)

    standard_error = np.sqrt(np.diag(reference) / DRAWS)
    assert np.all(np.abs(draws.mean(axis=0) - parameters) <= 3.0 * standard_error)

    # The sample covariance of the draws is an estimate: an off-diagonal entry carries a
    # sampling error of about sqrt(s_ii s_jj / n) even where the true entry is zero (the
    # polynomial design is orthogonal on these grids), so every entry is compared with
    # 2% relative tolerance or four of its own standard errors, whichever is wider.
    sample = np.cov(draws, rowvar=False)
    diagonal = np.diag(reference)
    entry_error = np.sqrt(np.outer(diagonal, diagonal) / DRAWS)
    np.fill_diagonal(entry_error, np.sqrt(2.0) * entry_error.diagonal())
    tolerance = COV_RTOL * np.abs(reference) + 4.0 * entry_error
    assert np.all(np.abs(sample - reference) <= tolerance)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_monte_carlo_band_matches_the_statsmodels_band(name: str) -> None:
    view, entry = view_and_entry(name)
    grid = np.asarray(entry["bands"]["x"], dtype=float)
    reference_fit = np.asarray(entry["bands"]["mean"], dtype=float)
    reference_half = np.asarray(entry["bands"]["mean_ci_upper"], dtype=float) - reference_fit

    band = monte_carlo_band(view, grid, n=BAND_DRAWS, seed=0)

    np.testing.assert_allclose(band.fit, reference_fit, rtol=1e-9, atol=1e-12)
    z = float(sps.norm.ppf(0.975))
    t = float(sps.t.ppf(0.975, view.df))
    expected = (z / t) * reference_half
    got = np.asarray(band.upper, dtype=float) - np.asarray(band.fit, dtype=float)
    np.testing.assert_allclose(
        got,
        expected,
        rtol=BAND_RTOL,
        atol=1e-12 * float(np.max(np.abs(reference_half))),
    )


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_monte_carlo_band_is_narrower_than_the_statsmodels_band(name: str) -> None:
    view, entry = view_and_entry(name)
    grid = np.asarray(entry["bands"]["x"], dtype=float)
    reference_fit = np.asarray(entry["bands"]["mean"], dtype=float)
    reference_half = np.asarray(entry["bands"]["mean_ci_upper"], dtype=float) - reference_fit

    band = monte_carlo_band(view, grid, n=BAND_DRAWS, seed=0)

    got = np.asarray(band.upper, dtype=float) - np.asarray(band.fit, dtype=float)
    assert np.all(got < reference_half)
