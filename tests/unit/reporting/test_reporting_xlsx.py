"""``fit_to_xlsx``: a readable workbook that stays honest about undefined values."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from openpyxl import load_workbook

from curvestudio.workflow import FitBundle
from curvestudio_reporting import fit_to_xlsx

if TYPE_CHECKING:
    from collections.abc import Callable

    FitBundleFactory = Callable[..., FitBundle]

SHEETS = ["Summary", "Parameters", "Observations", "Statistics", "Plot"]


def load(path: Path):
    return load_workbook(path, data_only=True)


def cell_values(sheet, column: str) -> list:
    return [sheet[f"{column}{r}"].value for r in range(1, sheet.max_row + 1)]


def summary_flat(tmp_path: Path) -> dict[str, object]:
    rows = sheet_rows(load(tmp_path / "fit.xlsx")["Summary"])
    return {str(row[0]): row[1] for row in rows if row[0]}


def sheet_rows(sheet) -> list[list]:
    return [list(row) for row in sheet.iter_rows(values_only=True)]


def test_workbook_has_the_five_sheets(poly_bundle: FitBundleFactory, tmp_path: Path) -> None:
    target = tmp_path / "fit.xlsx"
    fit_to_xlsx(poly_bundle(), target)
    wb = load(target)
    assert wb.sheetnames == SHEETS


def test_parameters_sheet_matches_the_bundle(poly_bundle: FitBundleFactory, tmp_path: Path) -> None:
    bundle = poly_bundle()
    fit_to_xlsx(bundle, tmp_path / "fit.xlsx")
    sheet = load(tmp_path / "fit.xlsx")["Parameters"]
    table = sheet_rows(sheet)
    assert table[0] == ["parameter", "value", "std_err", "ci_low", "ci_high"]
    for row, estimate in zip(table[1:], bundle.result.parameters, strict=True):
        assert row[0] == estimate.name
        # xlsx stores at most 16 significant digits (openpyxl writes %.16g), so the
        # last digit of a 17-digit repr can differ; CSV/JSON are the exact formats.
        assert row[1] == pytest.approx(estimate.value, rel=1e-14)
        assert row[2] == pytest.approx(estimate.std_err, rel=1e-14)
        assert row[3] == pytest.approx(estimate.ci_low, rel=1e-14)
        assert row[4] == pytest.approx(estimate.ci_high, rel=1e-14)


def test_observations_sheet_lists_every_row_with_numbers(
    poly_bundle: FitBundleFactory, tmp_path: Path
) -> None:
    bundle = poly_bundle()
    fit_to_xlsx(bundle, tmp_path / "fit.xlsx")
    sheet = load(tmp_path / "fit.xlsx")["Observations"]
    table = sheet_rows(sheet)
    assert table[0][:2] == ["x", "y"]
    assert len(table) == 1 + len(bundle.data.x)
    assert table[1][0] == bundle.data.x[0]
    assert table[-1][4] == bundle.result.residuals[-1]


def test_statistics_sheet_undefined_is_empty_never_zero(
    constant_bundle: FitBundle, tmp_path: Path
) -> None:
    fit_to_xlsx(constant_bundle, tmp_path / "fit.xlsx")
    table = sheet_rows(load(tmp_path / "fit.xlsx")["Statistics"])
    names = [row[0] for row in table[1:]]
    r2 = table[1:][names.index("r2")]
    assert r2[1] is None, "undefined R^2 must be an empty cell, never 0"
    assert str(r2[2]).startswith("not defined: ")


def test_summary_sheet_identifies_the_fit(poly_bundle: FitBundleFactory, tmp_path: Path) -> None:
    bundle = poly_bundle()
    fit_to_xlsx(bundle, tmp_path / "fit.xlsx")
    flat = summary_flat(tmp_path)
    assert flat["status"] == "converged"
    assert flat["dataset fingerprint"] == bundle.result.provenance.dataset_fingerprint
    assert flat["seed"] == bundle.result.provenance.seed
    assert bundle.equation_text in str(flat["equation (fitted)"])
    assert flat["equation (fitted)"] == bundle.equation_text
    assert str(flat["generated_at"]).endswith("+05:30") or "+05:30" in str(flat["generated_at"])


def test_summary_sheet_flags_a_non_converged_fit(
    non_converged_bundle: FitBundle,
    tmp_path: Path,
) -> None:
    fit_to_xlsx(non_converged_bundle, tmp_path / "fit.xlsx")
    flat = summary_flat(tmp_path)
    assert flat["status"] == "not_converged"
    assert non_converged_bundle.result.termination_reason in str(flat["termination reason"])


def test_formula_injection_is_neutralised_in_cells(
    poly_bundle: FitBundleFactory,
    tmp_path: Path,
) -> None:
    fit_to_xlsx(poly_bundle(x_name="=1+1"), tmp_path / "fit.xlsx")
    wb = load(tmp_path / "fit.xlsx")
    assert wb["Observations"]["A1"].value == "'=1+1"
    assert wb["Summary"]["A1"].value is not None


def test_formula_injection_behind_leading_whitespace_is_neutralised(
    poly_bundle: FitBundleFactory,
    tmp_path: Path,
) -> None:
    fit_to_xlsx(poly_bundle(x_name=" =1+1", y_name="\n=HYPERLINK"), tmp_path / "fit.xlsx")
    sheet = load(tmp_path / "fit.xlsx")["Observations"]
    assert sheet["A1"].value == "' =1+1"
    assert sheet["B1"].value == "'\n=HYPERLINK"


def test_control_characters_do_not_crash_the_workbook(
    poly_bundle: FitBundleFactory,
    tmp_path: Path,
) -> None:
    # openpyxl raises IllegalCharacterError for XML-illegal characters; the
    # cell writer must strip them instead
    fit_to_xlsx(poly_bundle(x_name="a\x1bb\x00c"), tmp_path / "fit.xlsx")
    assert load(tmp_path / "fit.xlsx")["Observations"]["A1"].value == "abc"


def test_subsampled_export_is_flagged_in_observations_and_summary(
    compact_bundle: FitBundle,
    tmp_path: Path,
) -> None:
    assert compact_bundle.data.subsampled
    fit_to_xlsx(compact_bundle, tmp_path / "fit.xlsx")
    note = (
        f"showing {len(compact_bundle.data.x)} "
        f"of {compact_bundle.data.n_total} observations (subsampled)"
    )
    observations = sheet_rows(load(tmp_path / "fit.xlsx")["Observations"])
    assert observations[-1][0] == note, "the Observations sheet must carry the note"
    assert len(observations) == 2 + len(compact_bundle.data.x)  # header + rows + note
    flat = summary_flat(tmp_path)
    assert flat["observations"] == compact_bundle.data.n_total
    assert flat["observation export"] == note, "the Summary must not claim all rows are shown"


def test_plot_sheet_embeds_the_fit_figure(poly_bundle: FitBundleFactory, tmp_path: Path) -> None:
    fit_to_xlsx(poly_bundle(), tmp_path / "fit.xlsx")
    plot = load(tmp_path / "fit.xlsx")["Plot"]
    assert len(plot._images) == 1, "the fit figure must be embedded as a PNG"


def test_plot_sheet_for_a_failed_fit_shows_data_and_banner_only(
    failed_bundle: FitBundle,
    tmp_path: Path,
) -> None:
    fit_to_xlsx(failed_bundle, tmp_path / "fit.xlsx")
    plot = load(tmp_path / "fit.xlsx")["Plot"]
    assert len(plot._images) == 1, "the figure is the honest data-only plot with its banner"
    flat = summary_flat(tmp_path)
    assert flat["status"] == "failed"


def test_workbook_properties_are_fixed_for_reproducibility(
    poly_bundle: FitBundleFactory, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    fit_to_xlsx(poly_bundle(), tmp_path / "fit.xlsx")
    props = load(tmp_path / "fit.xlsx").properties
    assert props.creator == "CurveStudio"
    # openpyxl reads the pinned epoch back as a naive datetime (no timezone kept)
    assert props.created is not None
    assert props.created.replace(tzinfo=UTC) == datetime(1970, 1, 1, tzinfo=UTC)
