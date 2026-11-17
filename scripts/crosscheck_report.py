"""Generate the Statsmodels cross-check report (tests/numerical/CROSSCHECK.md).

For every quantity that Phase 3 cross-checks against the stored Statsmodels fixtures
(``tests/data/crosscheck/statsmodels.json``) the report re-runs our implementation on every
dataset that carries the quantity and records the **maximum observed deviation**, next to the
tolerance the cross-check tests enforce. Nothing is copied from the tests' tolerances: the
deviations are measured.

Deviation measures
* ``relative``: ``max |ours - reference| / |reference|`` over all entries of the quantity;
  for matrix/vector quantities the denominator is floored exactly as the tests'
  ``assert_allclose(rtol, atol=1e-12 * max|reference|)`` floors it (see
  :func:`relative_deviation`).
* ``absolute``: ``max |ours - reference|`` (p-values).

The committed Markdown is deterministic: no timestamps, no platform or version strings of the
machine running it. Deviations are printed rounded **up** to the next even power of ten
(``1e-10``, ``1e-8``, ...; nothing is shown below ``1e-10``) so that floating-point round-off
differences between BLAS builds cannot change the file; the exact maxima are printed to the
console when the script runs.

Exit status: 0 when every quantity is within its tolerance, 1 otherwise.
"""

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _REPO_ROOT / "tests" / "numerical"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from crosscheck import FIXTURE_FILE, fixture_xy, load_fixtures  # noqa: E402

from curvestudio.diagnostics import (  # noqa: E402
    DiagnosticsReport,
    DiagnosticTest,
    press_rmse,
    residual_set,
    run_diagnostics,
)
from curvestudio.fitting import fit_polynomial  # noqa: E402
from curvestudio.statistics import (  # noqa: E402
    FitView,
    build_view,
    confidence_band,
    information_criteria,
    prediction_band,
)

DEFAULT_OUT = _HARNESS_DIR / "CROSSCHECK.md"

FloatArray = NDArray[np.float64]

RTOL = 1e-9
"""Tolerance for SE, covariance, bands, influence measures and criterion differences."""

STAT_RTOL = 1e-8
"""Relative tolerance on the test statistics."""

P_ATOL = 1e-6
"""Absolute tolerance on the p-values."""

ZERO_FLOOR = 1e-12
"""Fraction of ``max |reference|`` below which a reference entry counts as zero."""

DISPLAY_FLOOR_EXPONENT = -10
"""Deviations at or below ``10 ** -10`` are displayed as ``<= 1e-10`` (round-off level)."""

CRITERIA_DATASETS = ("heavy60", "linear30", "quad40")
CRITERIA_DEGREES = (1, 2, 3, 4)
LJUNG_BOX_LAGS = (1, 5)


@dataclass(frozen=True)
class Row:
    """One cross-checked quantity: where it was checked and how far off we were."""

    quantity: str
    reference: str
    datasets: tuple[str, ...]
    kind: str  # "relative" | "absolute"
    deviation: float  # math.inf when our value was missing
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.deviation <= self.tolerance


@dataclass(frozen=True)
class CrosscheckReport:
    statsmodels_version: str
    n_datasets: int
    rows: tuple[Row, ...]

    @property
    def failures(self) -> tuple[Row, ...]:
        return tuple(row for row in self.rows if not row.passed)


# --------------------------------------------------------------------------- deviations


def relative_deviation(actual: object, expected: object, *, floored: bool = True) -> float:
    """Largest ``|a - e|`` relative to ``|e|``; ``math.inf`` if ``a`` is missing or non-finite.

    With ``floored`` (matrices and vectors, as in the tests' ``assert_allclose``) the
    reference entries are floored at ``ZERO_FLOOR * max|e| / RTOL``: an entry that is
    numerically zero next to its matrix (an off-diagonal covariance term, say) is judged on
    ``1e-12 * max|e|`` absolute error instead of on a meaningless ratio. The result is then
    the smallest ``rtol`` for which ``assert_allclose(rtol, atol=1e-12 * max|e|)`` passes at
    the 1e-9 level. Scalar statistics use ``floored=False``: a plain relative error.
    """
    reference = np.asarray(expected, dtype=np.float64).ravel()
    ours = np.asarray(actual, dtype=np.float64).ravel()
    if ours.shape != reference.shape or not np.all(np.isfinite(ours)):
        return math.inf
    floor = ZERO_FLOOR * float(np.max(np.abs(reference))) / RTOL if floored else 0.0
    scale = np.abs(reference) + max(floor, float(np.finfo(np.float64).tiny))
    return float(np.max(np.abs(ours - reference) / scale))


def absolute_deviation(actual: object, expected: object) -> float:
    reference = np.asarray(expected, dtype=np.float64).ravel()
    ours = np.asarray(actual, dtype=np.float64).ravel()
    if ours.shape != reference.shape or not np.all(np.isfinite(ours)):
        return math.inf
    return float(np.max(np.abs(ours - reference)))


def _scalar(value: float | None) -> float:
    """A missing statistic is a failed cross-check, never a silent zero."""
    return math.nan if value is None else float(value)


# --------------------------------------------------------------------------- datasets


@dataclass(frozen=True)
class _Case:
    name: str
    entry: dict[str, Any]
    view: FitView


def _cases(fixtures: dict[str, dict[str, Any]]) -> list[_Case]:
    cases = []
    for name in sorted(fixtures):
        entry = fixtures[name]
        data = fixture_xy(name, entry)
        cases.append(_Case(name, entry, build_view(data, fit_polynomial(data, entry["degree"]))))
    return cases


def _worst(pairs: Iterable[tuple[str, float]]) -> tuple[tuple[str, ...], float]:
    names: list[str] = []
    worst = 0.0
    for name, deviation in pairs:
        names.append(name)
        worst = max(worst, deviation)
    return tuple(names), worst


def _row(
    quantity: str,
    reference: str,
    pairs: Iterable[tuple[str, float]],
    kind: str,
    tolerance: float,
) -> Row:
    datasets, worst = _worst(pairs)
    return Row(quantity, reference, datasets, kind, worst, tolerance)


# --------------------------------------------------------------------------- collectors


def _fit_rows(cases: Sequence[_Case]) -> list[Row]:
    def standard_errors(case: _Case) -> float:
        ours = [p.std_err for p in case.view.result.parameters]
        return relative_deviation(
            [math.nan if v is None else v for v in ours], case.entry["fit"]["bse"]
        )

    def covariance(case: _Case) -> float:
        covariance_matrix = case.view.result.covariance
        if covariance_matrix is None:
            return math.inf
        return relative_deviation(covariance_matrix, case.entry["fit"]["cov"])

    return [
        _row(
            "Parameter standard errors",
            "OLS/WLS `bse`",
            ((c.name, standard_errors(c)) for c in cases),
            "relative",
            RTOL,
        ),
        _row(
            "Parameter covariance matrix",
            "OLS/WLS `cov_params()`",
            ((c.name, covariance(c)) for c in cases),
            "relative",
            RTOL,
        ),
    ]


def _band_rows(cases: Sequence[_Case]) -> list[Row]:
    def confidence(case: _Case) -> float:
        ref = case.entry["bands"]
        grid = np.asarray(ref["x"], dtype=np.float64)
        band = confidence_band(case.view, grid, level=0.95)
        return max(
            relative_deviation(band.fit, ref["mean"]),
            relative_deviation(band.lower, ref["mean_ci_lower"]),
            relative_deviation(band.upper, ref["mean_ci_upper"]),
        )

    def prediction(case: _Case) -> float:
        ref = case.entry["bands"]
        grid = np.asarray(ref["x"], dtype=np.float64)
        new_weight = 1.0 if ref["new_weight"] is None else np.asarray(ref["new_weight"], float)
        band = prediction_band(case.view, grid, level=0.95, new_weight=new_weight)
        return max(
            relative_deviation(band.lower, ref["obs_ci_lower"]),
            relative_deviation(band.upper, ref["obs_ci_upper"]),
        )

    return [
        _row(
            "Mean (confidence) band: fit, lower, upper",
            "`get_prediction().summary_frame()` mean, mean_ci_lower, mean_ci_upper",
            ((c.name, confidence(c)) for c in cases),
            "relative",
            RTOL,
        ),
        _row(
            "Prediction band: lower, upper",
            "`get_prediction().summary_frame()` obs_ci_lower, obs_ci_upper",
            ((c.name, prediction(c)) for c in cases),
            "relative",
            RTOL,
        ),
    ]


def _criteria_rows(fixtures: dict[str, dict[str, Any]]) -> list[Row]:
    def differences(name: str, key: str) -> float:
        entry = fixtures[name]
        data = fixture_xy(name, entry)
        ours = {
            degree: information_criteria(fit_polynomial(data, degree))
            for degree in CRITERIA_DEGREES
        }
        ours_diffs: list[float] = []
        reference_diffs: list[float] = []
        for first, second in combinations(CRITERIA_DEGREES, 2):
            reference = entry["criteria"]
            ours_value = getattr(ours[first], key) - getattr(ours[second], key)
            ours_diffs.append(ours_value)
            reference_diffs.append(reference[str(first)][key] - reference[str(second)][key])
        return relative_deviation(ours_diffs, reference_diffs)

    return [
        _row(
            "AIC differences between polynomial degrees 1-4",
            "OLS `aic` (differences only; absolute levels differ by the dropped constants)",
            ((name, differences(name, "aic")) for name in CRITERIA_DATASETS),
            "relative",
            RTOL,
        ),
        _row(
            "BIC differences between polynomial degrees 1-4",
            "OLS `bic` (differences only; absolute levels differ by the dropped constants)",
            ((name, differences(name, "bic")) for name in CRITERIA_DATASETS),
            "relative",
            RTOL,
        ),
    ]


def _influence_rows(cases: Sequence[_Case]) -> list[Row]:
    sets = {c.name: residual_set(c.view) for c in cases}

    def optional(values: Sequence[float | None]) -> list[float]:
        return [math.nan if v is None else v for v in values]

    def leverage(case: _Case) -> float:
        return relative_deviation(
            sets[case.name].leverage, case.entry["influence"]["hat_matrix_diag"]
        )

    def cooks(case: _Case) -> float:
        return relative_deviation(
            optional(sets[case.name].cooks_distance), case.entry["influence"]["cooks_distance"]
        )

    def internal(case: _Case) -> float:
        return relative_deviation(
            optional(sets[case.name].standardized),
            case.entry["influence"]["resid_studentized_internal"],
        )

    def external(case: _Case) -> float:
        return relative_deviation(
            optional(sets[case.name].studentized_external),
            case.entry["influence"]["resid_studentized_external"],
        )

    def press(case: _Case) -> float:
        ref = case.entry["influence"]["resid_press"]
        current = sets[case.name]
        ours = np.asarray(current.residuals) / (1.0 - np.asarray(current.leverage))
        reference_rmse = float(np.sqrt(np.mean(np.square(ref))))
        rmse = press_rmse(case.view)
        return max(
            relative_deviation(ours, ref),
            math.inf if rmse is None else relative_deviation(rmse, reference_rmse, floored=False),
        )

    spec: list[tuple[str, str, Callable[[_Case], float]]] = [
        ("Leverage (hat-matrix diagonal)", "`OLSInfluence.hat_matrix_diag`", leverage),
        ("Cook's distance", "`OLSInfluence.cooks_distance`", cooks),
        (
            "Internally studentized (standardized) residuals",
            "`OLSInfluence.resid_studentized_internal`",
            internal,
        ),
        (
            "Externally studentized residuals",
            "`OLSInfluence.resid_studentized_external`",
            external,
        ),
        ("PRESS residuals and PRESS RMSE", "`OLSInfluence.resid_press`", press),
    ]
    return [
        _row(label, ref, ((c.name, measure(c)) for c in cases), "relative", RTOL)
        for label, ref, measure in spec
    ]


def _test_named(report: DiagnosticsReport, name: str) -> DiagnosticTest:
    return next(test for test in report.tests if test.name == name)


def _test_rows(cases: Sequence[_Case]) -> list[Row]:
    reports = {c.name: run_diagnostics(c.view) for c in cases}
    by_lags = {
        lags: {c.name: run_diagnostics(c.view, lags=lags) for c in cases} for lags in LJUNG_BOX_LAGS
    }

    def statistic(test: DiagnosticTest) -> float:
        return _scalar(test.statistic)

    def p_value(test: DiagnosticTest) -> float:
        return _scalar(test.p_value)

    def ref(case: _Case, key: str) -> float:
        return float(case.entry["tests"][key])

    def shapiro(case: _Case) -> DiagnosticTest:
        return _test_named(reports[case.name], "shapiro_wilk")

    def anderson(case: _Case) -> DiagnosticTest:
        return _test_named(reports[case.name], "anderson_darling")

    def breusch(case: _Case) -> DiagnosticTest:
        return _test_named(reports[case.name], "breusch_pagan")

    def durbin(case: _Case) -> DiagnosticTest:
        return _test_named(reports[case.name], "durbin_watson")

    def ljung_q(case: _Case) -> float:
        return max(
            relative_deviation(
                statistic(_test_named(by_lags[lags][case.name], "ljung_box")),
                ref(case, f"ljung_box_q{lags}"),
                floored=False,
            )
            for lags in LJUNG_BOX_LAGS
        )

    def ljung_p(case: _Case) -> float:
        return max(
            absolute_deviation(
                p_value(_test_named(by_lags[lags][case.name], "ljung_box")),
                ref(case, f"ljung_box_p{lags}"),
            )
            for lags in LJUNG_BOX_LAGS
        )

    spec: list[tuple[str, str, str, float, Callable[[_Case], float]]] = [
        (
            "Shapiro-Wilk W",
            "`scipy.stats.shapiro`",
            "relative",
            STAT_RTOL,
            lambda c: relative_deviation(statistic(shapiro(c)), ref(c, "shapiro_w"), floored=False),
        ),
        (
            "Shapiro-Wilk p-value",
            "`scipy.stats.shapiro`",
            "absolute",
            P_ATOL,
            lambda c: absolute_deviation(p_value(shapiro(c)), ref(c, "shapiro_p")),
        ),
        (
            "Anderson-Darling A²* (small-sample adjusted)",
            "`statsmodels.stats.diagnostic.normal_ad`",
            "relative",
            STAT_RTOL,
            lambda c: relative_deviation(statistic(anderson(c)), ref(c, "ad_a2a"), floored=False),
        ),
        (
            "Anderson-Darling p-value",
            "`statsmodels.stats.diagnostic.normal_ad`",
            "absolute",
            P_ATOL,
            lambda c: absolute_deviation(p_value(anderson(c)), ref(c, "ad_p")),
        ),
        (
            "Breusch-Pagan LM statistic",
            "`statsmodels.stats.diagnostic.het_breuschpagan`",
            "relative",
            STAT_RTOL,
            lambda c: relative_deviation(statistic(breusch(c)), ref(c, "bp_lm"), floored=False),
        ),
        (
            "Breusch-Pagan p-value",
            "`statsmodels.stats.diagnostic.het_breuschpagan`",
            "absolute",
            P_ATOL,
            lambda c: absolute_deviation(p_value(breusch(c)), ref(c, "bp_p")),
        ),
        (
            "Ljung-Box Q (lags 1 and 5)",
            "`statsmodels.stats.diagnostic.acorr_ljungbox`",
            "relative",
            STAT_RTOL,
            ljung_q,
        ),
        (
            "Ljung-Box p-value (lags 1 and 5)",
            "`statsmodels.stats.diagnostic.acorr_ljungbox`",
            "absolute",
            P_ATOL,
            ljung_p,
        ),
        (
            "Durbin-Watson statistic",
            "`statsmodels.stats.stattools.durbin_watson`",
            "relative",
            STAT_RTOL,
            lambda c: relative_deviation(
                statistic(durbin(c)), ref(c, "durbin_watson"), floored=False
            ),
        ),
    ]
    return [
        _row(label, reference, ((c.name, measure(c)) for c in cases), kind, tolerance)
        for label, reference, kind, tolerance, measure in spec
    ]


# --------------------------------------------------------------------------- build / render


def build_report() -> CrosscheckReport:
    """Re-run every cross-checked quantity against the stored fixtures."""
    fixtures = load_fixtures()
    cases = _cases(fixtures)
    rows = (
        _fit_rows(cases)
        + _band_rows(cases)
        + _criteria_rows(fixtures)
        + _influence_rows(cases)
        + _test_rows(cases)
    )
    return CrosscheckReport(
        statsmodels_version=_statsmodels_version(), n_datasets=len(cases), rows=tuple(rows)
    )


def _statsmodels_version() -> str:
    with FIXTURE_FILE.open(encoding="utf-8") as handle:
        return str(json.load(handle)["meta"]["statsmodels"])


def format_deviation(deviation: float) -> str:
    """The deviation rounded up to the next even power of ten, or ``NOT COMPUTED`` if missing."""
    if not math.isfinite(deviation):
        return "NOT COMPUTED"
    if deviation <= 10.0**DISPLAY_FLOOR_EXPONENT:
        return f"≤ 1e{DISPLAY_FLOOR_EXPONENT}"
    exponent = 2 * math.ceil(math.log10(deviation) / 2.0)
    return f"≤ 1e{exponent}"


def _tolerance(row: Row) -> str:
    return f"{row.tolerance:.0e}".replace("e-0", "e-")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(report: CrosscheckReport) -> str:
    """The report as Markdown (pure function of ``report``; fully deterministic)."""
    lines = [
        "# Statsmodels cross-check report",
        "",
        f"Reference: Statsmodels {report.statsmodels_version} (and SciPy for Shapiro-Wilk) "
        f"on {report.n_datasets} stored datasets (`tests/data/crosscheck/statsmodels.json`).",
        "Regenerate with `uv run python scripts/crosscheck_report.py`; "
        "`tests/numerical/test_crosscheck_report.py` fails when this file drifts.",
        "",
        "Each row re-runs our implementation on every dataset that carries the quantity and "
        "reports the maximum observed deviation from the stored Statsmodels value. "
        "*Relative* deviation is `max |ours - ref| / |ref|`, with matrix and vector quantities "
        "judged as `assert_allclose(rtol, atol=1e-12 * max|ref|)` does, so that entries that are "
        "numerically zero next to the rest of the matrix are not compared by ratio; "
        "*absolute* is `max |ours - ref|` (p-values). "
        "Deviations are rounded up to the next even power of ten (nothing is shown below "
        "`1e-10`) so that floating-point round-off differences between machines cannot "
        "change this file.",
        "",
        "## Cross-checked quantities",
        "",
    ]
    lines.extend(
        _table(
            (
                "Quantity",
                "Statsmodels reference",
                "Datasets",
                "Kind",
                "Max deviation",
                "Tolerance",
                "Status",
            ),
            [
                (
                    row.quantity,
                    row.reference,
                    ", ".join(row.datasets),
                    row.kind,
                    format_deviation(row.deviation),
                    _tolerance(row),
                    "pass" if row.passed else "**FAIL**",
                )
                for row in report.rows
            ],
        )
    )
    failed = report.failures
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Quantities cross-checked: {len(report.rows)}",
            f"- Within tolerance: {len(report.rows) - len(failed)}/{len(report.rows)}",
        ]
    )
    lines.extend(f"- **FAIL**: {row.quantity}" for row in failed)
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Information criteria use our convention (`k' = k + 1`, constants common to all "
            "models dropped), while Statsmodels charges `df_model + 1` parameters and keeps the "
            "constants; only criterion *differences* are comparable, and only those are checked "
            "here.",
            "- The weighted dataset (`wls35`) is fitted with `fit_polynomial` weights; its "
            "influence measures are compared with the equivalent OLS fit of the whitened data "
            "because `OLSInfluence` mishandles WLS (see `scripts/make_crosscheck_fixtures.py`).",
            "- Tests are run on the weighted residuals in ascending-x order; Ljung-Box is checked "
            "at lags 1 and 5, and the larger of the two deviations is reported.",
        ]
    )
    return "\n".join(lines) + "\n"


def _console_summary(report: CrosscheckReport) -> list[str]:
    return [
        f"{row.deviation:.3e}  (tol {row.tolerance:.0e}, {row.kind})  {row.quantity}"
        for row in report.rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output Markdown path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print("\n".join(_console_summary(report)))
    print(f"wrote {args.out}: {len(report.failures)} quantity(ies) outside tolerance")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
