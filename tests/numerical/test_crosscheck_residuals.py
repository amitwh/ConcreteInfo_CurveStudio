"""Leverage, Cook's distance, studentized residuals and PRESS equal Statsmodels (OLS and WLS).

Reference: ``statsmodels.stats.outliers_influence.OLSInfluence`` on the stored datasets (see
``reference_influence`` in ``scripts/make_crosscheck_fixtures.py``; the weighted dataset goes
through the equivalent OLS fit of the whitened data because ``OLSInfluence`` mishandles WLS).
"""

import numpy as np
import pytest
from crosscheck import fixture_xy, load_fixtures

from curvestudio.diagnostics.residuals import press_rmse, residual_set
from curvestudio.fitting import fit_polynomial
from curvestudio.statistics import build_view

FIXTURES = load_fixtures()
RTOL = 1e-9


def close(actual: object, expected: object) -> None:
    """Relative 1e-9; the tiny absolute floor only matters for entries that are ~0."""
    reference = np.asarray(expected, dtype=float)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        reference,
        rtol=RTOL,
        atol=1e-12 * float(np.max(np.abs(reference))),
    )


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_influence_measures_equal_statsmodels(name: str) -> None:
    entry = FIXTURES[name]
    ref = entry["influence"]
    data = fixture_xy(name, entry)
    view = build_view(data, fit_polynomial(data, entry["degree"]))
    result = residual_set(view)

    close(result.leverage, ref["hat_matrix_diag"])
    close(result.cooks_distance, ref["cooks_distance"])
    close(result.standardized, ref["resid_studentized_internal"])
    close(result.studentized_external, ref["resid_studentized_external"])

    press = np.asarray(result.residuals) / (1.0 - np.asarray(result.leverage))
    close(press, ref["resid_press"])
    reference_rmse = float(np.sqrt(np.mean(np.square(ref["resid_press"]))))
    assert press_rmse(view) == pytest.approx(reference_rmse, rel=RTOL)
