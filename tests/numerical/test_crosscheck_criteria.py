"""Cross-check information criteria against Statsmodels on the stored fixtures.

Statsmodels charges ``df_model + k_constant`` parameters for OLS and does **not** count the
error variance, while our convention uses ``k' = k + 1`` and drops the constants common to
all models. Absolute AIC/BIC levels therefore differ by a known offset; criterion
*differences* between models of the same dataset must agree directly.
"""

import math
from itertools import combinations

import numpy as np
import pytest
from crosscheck import fixture_xy, load_fixtures

from curvestudio.fitting import FitStatus, fit_polynomial
from curvestudio.statistics import InformationCriteria, information_criteria

FIXTURES = load_fixtures()
DATASETS = ("heavy60", "linear30", "quad40")
DEGREES = (1, 2, 3, 4)
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


def our_criteria(name: str, degree: int) -> InformationCriteria:
    entry = FIXTURES[name]
    result = fit_polynomial(fixture_xy(name, entry), degree)
    assert result.status is FitStatus.CONVERGED
    return information_criteria(result)


@pytest.mark.parametrize("name", DATASETS)
def test_statsmodels_references_are_internally_consistent(name: str) -> None:
    entry = FIXTURES[name]
    for degree in DEGREES:
        reference = entry["criteria"][str(degree)]
        charged = reference["df_model"] + 1  # slopes plus the intercept, no variance term
        assert reference["nobs"] == len(entry["x"])
        assert reference["df_model"] == degree
        assert reference["aic"] == pytest.approx(-2.0 * reference["llf"] + 2.0 * charged, rel=1e-12)
        assert reference["bic"] == pytest.approx(
            -2.0 * reference["llf"] + math.log(reference["nobs"]) * charged, rel=1e-12
        )


@pytest.mark.parametrize("name", DATASETS)
def test_information_criteria_differences_match_statsmodels(name: str) -> None:
    ours = {degree: our_criteria(name, degree) for degree in DEGREES}
    reference = FIXTURES[name]["criteria"]
    for first, second in combinations(DEGREES, 2):
        close(
            ours[first].aic - ours[second].aic,
            reference[str(first)]["aic"] - reference[str(second)]["aic"],
        )
        close(
            ours[first].bic - ours[second].bic,
            reference[str(first)]["bic"] - reference[str(second)]["bic"],
        )


@pytest.mark.parametrize("name", DATASETS)
def test_absolute_offset_is_the_dropped_gaussian_constant(name: str) -> None:
    entry = FIXTURES[name]
    n = len(entry["x"])
    dropped = n * math.log(2.0 * math.pi) + n
    for degree in DEGREES:
        ours = our_criteria(name, degree)
        reference = entry["criteria"][str(degree)]
        assert ours.n == n
        assert ours.k_total == degree + 2  # degree + 1 coefficients, plus the variance
        close(reference["aic"] - ours.aic, dropped - 2.0)
        close(reference["bic"] - ours.bic, dropped - math.log(n))
