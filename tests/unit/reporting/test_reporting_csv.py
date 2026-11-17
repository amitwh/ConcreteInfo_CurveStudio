"""CSV reports: parameters, observations and statistics as safe, exact CSV."""

import csv
import io
from typing import TYPE_CHECKING

from curvestudio.workflow import FitBundle
from curvestudio_reporting import observations_csv, parameters_csv, statistics_csv

if TYPE_CHECKING:
    from collections.abc import Callable

    FitBundleFactory = Callable[..., FitBundle]


def rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def test_parameters_csv_carries_every_estimate_exactly(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle()
    table = rows(parameters_csv(bundle))
    assert table[0] == ["parameter", "value", "std_err", "ci_low", "ci_high"]
    assert len(table) == 1 + len(bundle.result.parameters)
    for row, estimate in zip(table[1:], bundle.result.parameters, strict=True):
        assert row[0] == estimate.name
        assert float(row[1]) == estimate.value  # repr round trip: no precision lost
        assert float(row[2]) == estimate.std_err
        assert float(row[3]) == estimate.ci_low
        assert float(row[4]) == estimate.ci_high


def test_parameters_csv_uses_repr_floats(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle()
    table = rows(parameters_csv(bundle))
    for row, estimate in zip(table[1:], bundle.result.parameters, strict=True):
        assert row[1] == repr(estimate.value), "floats must serialise as their repr"


def test_observations_csv_lists_every_row(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle()
    assert bundle.data.weights is None
    table = rows(observations_csv(bundle))
    assert table[0] == ["x", "y", "weight", "fitted", "residual"]
    assert len(table) == 1 + len(bundle.data.x)
    for row, x, y, fitted, residual in zip(
        table[1:],
        bundle.data.x,
        bundle.data.y,
        bundle.result.fitted,
        bundle.result.residuals,
        strict=True,
    ):
        assert float(row[0]) == x
        assert float(row[1]) == y
        assert row[2] == ""
        assert float(row[3]) == fitted
        assert float(row[4]) == residual


def test_observations_csv_uses_the_data_column_names(poly_bundle: FitBundleFactory) -> None:
    table = rows(observations_csv(poly_bundle(x_name="day", y_name="load")))
    assert table[0] == ["day", "load", "weight", "fitted", "residual"]


def test_observations_csv_neutralises_formula_injection(poly_bundle: FitBundleFactory) -> None:
    table = rows(observations_csv(poly_bundle(x_name="=1+1", y_name="+SUM(A1:A9)")))
    assert table[0] == ["'=1+1", "'+SUM(A1:A9)", "weight", "fitted", "residual"]


def test_observations_csv_for_a_failed_fit_leaves_fitted_and_residual_empty(
    failed_bundle: FitBundle,
) -> None:
    table = rows(observations_csv(failed_bundle))
    assert len(table) == 1 + len(failed_bundle.data.x)
    for row in table[1:]:
        assert row[3] == "" and row[4] == ""


def test_statistics_csv_lists_defined_values(poly_bundle: FitBundleFactory) -> None:
    bundle = poly_bundle()
    assert bundle.result.stats is not None
    table = rows(statistics_csv(bundle))
    assert table[0] == ["statistic", "value", "note"]
    names = [row[0] for row in table[1:]]
    for expected in ("status", "n", "k", "df", "rmse", "r2", "adj_r2", "aic", "aicc", "bic"):
        assert expected in names, f"{expected} missing from statistics"
    rmse_row = table[1:][names.index("rmse")]
    assert float(rmse_row[1]) == bundle.result.stats.rmse


def test_statistics_csv_marks_undefined_statistics_not_as_zero(
    constant_bundle: FitBundle,
) -> None:
    assert constant_bundle.result.stats is not None
    assert constant_bundle.result.stats.r2 is None
    table = rows(statistics_csv(constant_bundle))
    names = [row[0] for row in table[1:]]
    r2_row = table[1:][names.index("r2")]
    assert r2_row[1] == "", "an undefined R^2 must be empty, never 0"
    assert r2_row[2].startswith("not defined: ")


def test_statistics_csv_for_a_non_converged_fit_is_honest(
    non_converged_bundle: FitBundle,
) -> None:
    table = rows(statistics_csv(non_converged_bundle))
    names = [row[0] for row in table[1:]]
    status_row = table[1:][names.index("status")]
    assert status_row[1] == "not_converged"
    assert non_converged_bundle.result.termination_reason in status_row[2]
    aic_row = table[1:][names.index("aic")]
    assert aic_row[1] == ""
    assert "not defined" in aic_row[2]


def test_neutralisation_covers_every_dangerous_prefix() -> None:
    from curvestudio_reporting.csv_report import neutralise

    for prefix in ("=", "+", "-", "@", "\t", "\r", "\n"):
        text = f"{prefix}cmd()"
        assert neutralise(text) == f"'{text}", prefix
    assert neutralise("day") == "day"
    assert neutralise("n (MPa)") == "n (MPa)"
    assert neutralise("1e-3 x") == "1e-3 x"


def test_neutralisation_covers_leading_whitespace_before_a_formula() -> None:
    from curvestudio_reporting.csv_report import neutralise

    # whitespace (spaces, newlines, tabs) before a dangerous character is the
    # classic CSV-injection shape too: spreadsheets skip the leading blanks
    for text in (" =1+1", "\n=1", "  @SUM(A1)", " \t-1+1", "\n\t+cmd"):
        assert neutralise(text) == f"'{text}", text
    # plain indented text stays untouched: no apostrophe on harmless labels
    # (a literal leading tab/newline is dangerous by itself, so only spaces are safe)
    for text in ("  day", "   note", "n (MPa)"):
        assert neutralise(text) == text, text


def test_control_characters_are_stripped_from_csv_cells(poly_bundle: FitBundleFactory) -> None:
    table = rows(observations_csv(poly_bundle(x_name="a\x1bb\x00c", y_name="d\ne")))
    assert table[0][0] == "abc", "XML-illegal control characters must not reach the CSV"
    assert table[0][1] == "d\ne", "legal characters (newline) are kept"


def test_observations_csv_flags_a_subsampled_export(compact_bundle: FitBundle) -> None:
    table = rows(observations_csv(compact_bundle))
    assert compact_bundle.data.subsampled
    note = (
        f"showing {len(compact_bundle.data.x)} "
        f"of {compact_bundle.data.n_total} observations (subsampled)"
    )
    assert table[-1] == [note, "", "", "", ""], "the note must be the last CSV row"
    assert len(table) == 2 + len(compact_bundle.data.x)  # header + rows + note
