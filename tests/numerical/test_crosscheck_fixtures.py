"""The stored Statsmodels fixtures agree with our polynomial engine (validates the fixtures).

If this fails, either the fixtures were generated wrongly or ``fit_polynomial`` is off; the
later cross-checks of derived statistics are only meaningful while this passes.
"""

import numpy as np
import pytest
from crosscheck import fixture_xy, load_fixtures

from curvestudio.fitting import FitStatus, fit_polynomial

FIXTURES = load_fixtures()
EXPECTED = ("ar50", "cubic8", "heavy60", "linear30", "quad40", "wls35")
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


def test_all_six_datasets_are_present() -> None:
    assert tuple(sorted(FIXTURES)) == EXPECTED
    assert {name: len(entry["x"]) for name, entry in FIXTURES.items()} == {
        "linear30": 30,
        "quad40": 40,
        "wls35": 35,
        "ar50": 50,
        "heavy60": 60,
        "cubic8": 8,
    }
    assert [name for name, entry in FIXTURES.items() if entry["weighted"]] == ["wls35"]


@pytest.mark.parametrize("name", EXPECTED)
def test_polynomial_fit_equals_statsmodels(name: str) -> None:
    entry = FIXTURES[name]
    ref = entry["fit"]
    result = fit_polynomial(fixture_xy(name, entry), entry["degree"], confidence_level=0.95)

    assert result.status is FitStatus.CONVERGED
    assert result.stats is not None and result.covariance is not None
    assert result.stats.df == ref["df_resid"]
    close(result.stats.sse, ref["ssr"])
    close(result.stats.sse / result.stats.df, ref["scale"])
    close([p.value for p in result.parameters], ref["params"])
    close([p.std_err for p in result.parameters], ref["bse"])
    close(result.covariance, ref["cov"])
    close([[p.ci_low, p.ci_high] for p in result.parameters], ref["conf_int"])
