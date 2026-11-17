"""Tests for the Statsmodels cross-check report generator (``scripts/crosscheck_report.py``).

``build_report`` re-runs every cross-checked quantity against the stored fixtures, so the
committed ``tests/numerical/CROSSCHECK.md`` must equal what the builder renders today; the
renderer itself is a pure function and is checked character for character on a hand-built
report.
"""

import math
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from crosscheck_report import (  # noqa: E402
    DEFAULT_OUT,
    CrosscheckReport,
    Row,
    absolute_deviation,
    build_report,
    format_deviation,
    main,
    relative_deviation,
    render_markdown,
)

REQUIRED = (
    "Parameter standard errors",
    "Parameter covariance matrix",
    "Mean (confidence) band",
    "Prediction band",
    "AIC differences",
    "BIC differences",
    "Leverage",
    "Cook's distance",
    "Internally studentized",
    "Externally studentized",
    "PRESS",
    "Shapiro-Wilk W",
    "Anderson-Darling A",
    "Breusch-Pagan LM",
    "Ljung-Box Q",
    "Durbin-Watson",
)

EXPECTED_TINY = """# Statsmodels cross-check report

Reference: Statsmodels 9.9.9 (and SciPy for Shapiro-Wilk) on 2 stored datasets (`tests/data/crosscheck/statsmodels.json`).
Regenerate with `uv run python scripts/crosscheck_report.py`; `tests/numerical/test_crosscheck_report.py` fails when this file drifts.

Each row re-runs our implementation on every dataset that carries the quantity and reports the maximum observed deviation from the stored Statsmodels value. *Relative* deviation is `max |ours - ref| / |ref|`, with matrix and vector quantities judged as `assert_allclose(rtol, atol=1e-12 * max|ref|)` does, so that entries that are numerically zero next to the rest of the matrix are not compared by ratio; *absolute* is `max |ours - ref|` (p-values). Deviations are rounded up to the next even power of ten (nothing is shown below `1e-10`) so that floating-point round-off differences between machines cannot change this file.

## Cross-checked quantities

| Quantity | Statsmodels reference | Datasets | Kind | Max deviation | Tolerance | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Standard errors | `bse` | a, b | relative | ≤ 1e-10 | 1e-9 | pass |
| Shapiro-Wilk p-value | `shapiro` | a | absolute | ≤ 1e-4 | 1e-6 | **FAIL** |
| Durbin-Watson statistic | `durbin_watson` | b | relative | NOT COMPUTED | 1e-8 | **FAIL** |

## Summary

- Quantities cross-checked: 3
- Within tolerance: 1/3
- **FAIL**: Shapiro-Wilk p-value
- **FAIL**: Durbin-Watson statistic

## Notes

- Information criteria use our convention (`k' = k + 1`, constants common to all models dropped), while Statsmodels charges `df_model + 1` parameters and keeps the constants; only criterion *differences* are comparable, and only those are checked here.
- The weighted dataset (`wls35`) is fitted with `fit_polynomial` weights; its influence measures are compared with the equivalent OLS fit of the whitened data because `OLSInfluence` mishandles WLS (see `scripts/make_crosscheck_fixtures.py`).
- Tests are run on the weighted residuals in ascending-x order; Ljung-Box is checked at lags 1 and 5, and the larger of the two deviations is reported.
"""  # noqa: E501 — the rendered prose lines are wider than 100 columns


def tiny_report() -> CrosscheckReport:
    return CrosscheckReport(
        statsmodels_version="9.9.9",
        n_datasets=2,
        rows=(
            Row("Standard errors", "`bse`", ("a", "b"), "relative", 3e-13, 1e-9),
            Row("Shapiro-Wilk p-value", "`shapiro`", ("a",), "absolute", 4e-5, 1e-6),
            Row("Durbin-Watson statistic", "`durbin_watson`", ("b",), "relative", math.inf, 1e-8),
        ),
    )


def test_render_markdown_is_character_exact() -> None:
    assert render_markdown(tiny_report()) == EXPECTED_TINY


def test_the_failing_rows_are_the_ones_outside_their_tolerance() -> None:
    report = tiny_report()
    assert [row.quantity for row in report.failures] == [
        "Shapiro-Wilk p-value",
        "Durbin-Watson statistic",
    ]


@pytest.mark.parametrize(
    ("deviation", "text"),
    [
        (0.0, "≤ 1e-10"),
        (3e-16, "≤ 1e-10"),
        (1e-10, "≤ 1e-10"),
        (1.1e-10, "≤ 1e-8"),
        (4e-9, "≤ 1e-8"),
        (2e-8, "≤ 1e-6"),
        (math.inf, "NOT COMPUTED"),
    ],
)
def test_format_deviation_rounds_up_to_even_decades(deviation: float, text: str) -> None:
    assert format_deviation(deviation) == text


def test_relative_deviation_is_a_plain_ratio_for_scalars() -> None:
    assert relative_deviation(1.0 + 3e-9, 1.0, floored=False) == pytest.approx(3e-9, rel=1e-6)


def test_relative_deviation_does_not_compare_a_near_zero_entry_by_ratio() -> None:
    reference = [1.0, 1e-20]
    ours = [1.0, 1e-18]  # a 100x ratio, but 1e-18 absolute next to a matrix scale of 1
    assert relative_deviation(ours, reference) < 1e-9


def test_a_missing_or_non_finite_value_is_infinite_deviation_not_zero() -> None:
    assert relative_deviation([math.nan], [1.0]) == math.inf
    assert relative_deviation([1.0], [1.0, 2.0]) == math.inf
    assert absolute_deviation(math.nan, 0.5) == math.inf


def test_every_cross_checked_quantity_is_in_the_report_and_within_tolerance() -> None:
    report = build_report()
    quantities = [row.quantity for row in report.rows]
    for needle in REQUIRED:
        assert any(needle in quantity for quantity in quantities), needle
    for row in report.rows:
        assert row.datasets, row.quantity
        assert math.isfinite(row.deviation), row.quantity
        assert row.passed, f"{row.quantity}: {row.deviation:.3e} > {row.tolerance:.0e}"
    assert report.failures == ()


def test_crosscheck_report_regenerates_identically() -> None:
    assert DEFAULT_OUT.read_text(encoding="utf-8") == render_markdown(build_report())


def test_main_writes_the_report_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "out" / "CROSSCHECK.md"
    assert main(["--out", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == DEFAULT_OUT.read_text(encoding="utf-8")
    assert "0 quantity(ies) outside tolerance" in capsys.readouterr().out
