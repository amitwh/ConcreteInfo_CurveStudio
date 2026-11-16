"""Tests for the file and clipboard importers."""

import datetime
import json
import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from curvestudio.data import DataError, ImportLimits, read_dataset, read_text, validate

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "import"

EXPECTED = pd.DataFrame(
    {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.1, 3.9, 6.2, 7.8, 10.1]},
)


def code_of(excinfo: pytest.ExceptionInfo[DataError]) -> str:
    return excinfo.value.code


@pytest.fixture
def sample_xlsx(tmp_path: Path) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Data"
    sheet.append(["x", "y"])
    for row in zip(EXPECTED["x"], EXPECTED["y"], strict=True):
        sheet.append(list(row))
    other = workbook.create_sheet("Other")
    other.append(["a", "b"])
    other.append([10, 20])
    path = tmp_path / "sample.xlsx"
    workbook.save(path)
    return path


# --- the same table through every format -----------------------------------------------


def test_basic_csv_fixture_loads_expected_table() -> None:
    ds = read_dataset(DATA_DIR / "basic.csv")
    pd.testing.assert_frame_equal(ds.original, EXPECTED)
    assert ds.name == "basic"
    assert ds.import_notes == ()


def test_tsv_loads_expected_table(tmp_path: Path) -> None:
    path = tmp_path / "t.tsv"
    path.write_text("x\ty\n1\t2.1\n2\t3.9\n3\t6.2\n4\t7.8\n5\t10.1\n")
    pd.testing.assert_frame_equal(read_dataset(path).original, EXPECTED)


@pytest.mark.parametrize(
    "text",
    [
        "x\ty\n1\t2.1\n2\t3.9\n3\t6.2\n4\t7.8\n5\t10.1\n",
        "x;y\n1;2.1\n2;3.9\n3;6.2\n4;7.8\n5;10.1\n",
        "x,y\n1,2.1\n2,3.9\n3,6.2\n4,7.8\n5,10.1\n",
        "x    y\n1    2.1\n2  3.9\n3 6.2\n4   7.8\n5     10.1\n",
    ],
    ids=["tab", "semicolon", "comma", "whitespace"],
)
def test_txt_sniffs_delimiter(tmp_path: Path, text: str) -> None:
    path = tmp_path / "t.txt"
    path.write_text(text)
    pd.testing.assert_frame_equal(read_dataset(path).original, EXPECTED)


def test_json_records_orientation(tmp_path: Path) -> None:
    records = [{"x": x, "y": y} for x, y in zip(EXPECTED["x"], EXPECTED["y"], strict=True)]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(records))
    pd.testing.assert_frame_equal(read_dataset(path).original, EXPECTED)


def test_json_columns_orientation(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"x": EXPECTED["x"].tolist(), "y": EXPECTED["y"].tolist()}))
    pd.testing.assert_frame_equal(read_dataset(path).original, EXPECTED)


def test_json_null_becomes_nan_and_missing_keys_are_blank(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text('[{"x": 1, "y": 2}, {"x": 2}, {"x": 3, "y": null}, {"x": 4, "y": 5}]')
    original = read_dataset(path).original
    assert list(original.columns) == ["x", "y"]
    assert original["y"].isna().tolist() == [False, True, True, False]


def test_xlsx_loads_expected_table(sample_xlsx: Path) -> None:
    ds = read_dataset(sample_xlsx)
    pd.testing.assert_frame_equal(ds.original, EXPECTED)
    assert ds.name == "sample"


def test_xlsx_selects_sheet_by_name_and_index(sample_xlsx: Path) -> None:
    expected = pd.DataFrame({"a": [10.0], "b": [20.0]})
    pd.testing.assert_frame_equal(read_dataset(sample_xlsx, sheet="Other").original, expected)
    pd.testing.assert_frame_equal(read_dataset(sample_xlsx, sheet=1).original, expected)


@pytest.mark.parametrize("sheet", ["Nope", 5, -1])
def test_xlsx_unknown_sheet(sample_xlsx: Path, sheet: str | int) -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(sample_xlsx, sheet=sheet)
    assert code_of(excinfo) == "UNKNOWN_SHEET"


def test_xlsx_dates_and_blank_cells(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["when", "v"])
    sheet.append([datetime.datetime(2026, 1, 1), 1.5])
    sheet.append([datetime.datetime(2026, 1, 2), None])
    sheet.append([None, None])
    sheet.append([datetime.datetime(2026, 1, 4), "7"])
    path = tmp_path / "d.xlsx"
    workbook.save(path)
    ds = read_dataset(path)
    original = ds.original
    assert str(original["when"].dtype).startswith("datetime64")
    assert original["v"].isna().tolist() == [False, True, False]
    assert original["v"].iloc[2] == 7.0
    assert ds.import_notes == ("dropped 1 blank row(s)",)


def test_xlsx_uncached_formula_is_blank_not_formula_text(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["x", "y"])
    sheet.append([1, 2])
    sheet.append([2, "=A3*10"])  # no cached value: openpyxl never evaluates formulas
    path = tmp_path / "f.xlsx"
    workbook.save(path)
    original = read_dataset(path).original
    assert math.isnan(original["y"].iloc[1])  # reported as blank, not as the formula text


# --- Review Focus 5: BOM, blank rows/cols, header names, thousands ----------------------


def test_bom_and_trailing_blank_rows_and_columns() -> None:
    ds = read_dataset(DATA_DIR / "bom_blank.csv")
    assert ds.original.shape == (3, 2)
    assert list(ds.original.columns) == ["x", "y"]
    assert ds.original["y"].tolist() == [2.1, 3.9, 6.2]
    assert ds.import_notes == (
        "stripped UTF-8 byte order mark",
        "dropped 2 blank row(s)",
        "dropped 2 blank column(s)",
    )


def test_interior_blank_row_is_dropped_and_noted() -> None:
    ds = read_text("x,y\n1,2\n\n2,3\n3,4\n")
    assert ds.original["x"].tolist() == [1.0, 2.0, 3.0]
    assert ds.import_notes == ("dropped 1 blank row(s)",)


def test_blank_and_duplicate_headers_are_renamed_and_noted() -> None:
    ds = read_text("x,x,,x\n1,2,3,4\n5,6,7,8\n")
    assert list(ds.original.columns) == ["x", "x_2", "col_3", "x_3"]
    assert ds.import_notes == (
        "duplicate header 'x' in column 2 renamed to 'x_2'",
        "blank header in column 3 named 'col_3'",
        "duplicate header 'x' in column 4 renamed to 'x_3'",
    )


def test_headerless_numeric_table_gets_generated_names() -> None:
    ds = read_text("1,2\n3,4\n5,6\n")
    assert list(ds.original.columns) == ["col_1", "col_2"]
    assert ds.original.shape == (3, 2)
    assert ds.import_notes == ("no header row found; columns named col_1..col_2",)


def test_thousands_separators_parsed_when_every_cell_fits() -> None:
    ds = read_dataset(DATA_DIR / "thousands.csv")
    assert ds.original["y"].tolist() == [1234.5, 2345.25, 10000.0, 999.0, 1000000.5]
    assert ds.import_notes == ()


def test_thousands_not_parsed_when_a_cell_does_not_fit() -> None:
    ds = read_text("x\ty\n1\t1,234\n2\tabc\n3\t5\n")
    assert ds.original["y"].dtype == object
    assert ds.original["y"].tolist() == ["1,234", "abc", "5"]
    assert ds.import_notes == ()


def test_malformed_grouping_stays_text() -> None:
    ds = read_text("x\ty\n1\t1,23\n2\tab\n3\t4\n")
    assert ds.original["y"].tolist() == ["1,23", "ab", "4"]


def test_decimal_comma_is_not_supported_and_stays_text() -> None:
    ds = read_text("x;y\n1;2,5\n2;3,5\n")
    assert ds.original["y"].tolist() == ["2,5", "3,5"]


# --- coercion --------------------------------------------------------------------------


def test_non_numeric_cell_in_numeric_column_becomes_nan_with_note() -> None:
    ds = read_text("x,y\n1,2\n2,3\n3,abc\n4,5\n5,6\n")
    assert ds.original["y"].dtype == "float64"
    assert math.isnan(ds.original["y"].iloc[2])
    assert ds.import_notes == ("coerced:y:2",)
    assert len(ds.original) == 5  # never dropped


def test_coerced_row_indexes_the_final_table_after_blank_rows_are_dropped() -> None:
    ds = read_text("x,y\n1,2\n\n2,3\n3,abc\n4,5\n")
    assert ds.import_notes == ("dropped 1 blank row(s)", "coerced:y:2")
    assert math.isnan(ds.original["y"].iloc[2])


def test_validate_reports_coerced_cell_at_the_noted_row() -> None:
    ds = read_text("x,y\n1,2\n\n2,3\n3,abc\n4,5\n")
    issues = {issue.code: issue for issue in validate(ds, "x", "y")}
    assert issues["NON_NUMERIC"].rows == (2,)


def test_blank_cells_are_nan_without_notes() -> None:
    ds = read_text("x,y\n1,2\n2,\n3,4\n")
    assert ds.original["y"].isna().tolist() == [False, True, False]
    assert ds.import_notes == ()


def test_half_numeric_column_is_numeric_with_coercion_notes() -> None:
    ds = read_text("x,y\n1,2\n2,a\n3,4\n4,b\n")
    assert ds.original["y"].dtype == "float64"
    assert ds.import_notes == ("coerced:y:1", "coerced:y:3")


def test_mostly_text_column_stays_text_without_notes() -> None:
    ds = read_text("x,label\n1,a\n2,b\n3,7\n4,d\n")
    assert ds.original["label"].tolist() == ["a", "b", "7", "d"]
    assert ds.original["x"].dtype == "float64"
    assert ds.import_notes == ()


def test_nan_and_inf_literals_are_numbers() -> None:
    ds = read_text("x,y\n1,nan\n2,inf\n3,-Infinity\n")
    values = ds.original["y"].tolist()
    assert math.isnan(values[0])
    assert values[1:] == [math.inf, -math.inf]
    assert ds.import_notes == ()


def test_python_only_number_spellings_are_text() -> None:
    ds = read_text("x\ty\n1\t1_000\n2\t0x10\n")
    assert ds.original["y"].tolist() == ["1_000", "0x10"]


# --- read_text -------------------------------------------------------------------------


def test_read_text_pasted_tab_separated_block() -> None:
    ds = read_text("Time\tTemp\n0\t20.5\n1\t21.0\n2\t22.5\n")
    assert ds.name == "pasted"
    expected = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Temp": [20.5, 21.0, 22.5]})
    pd.testing.assert_frame_equal(ds.original, expected)


def test_read_text_name_argument() -> None:
    assert read_text("x,y\n1,2\n", name="clip").name == "clip"


def test_read_text_tab_wins_when_cells_contain_commas_and_spaces() -> None:
    ds = read_text("peak load\tarea, mm2\n1,5\t2\n3\t4,5\n")
    assert list(ds.original.columns) == ["peak load", "area, mm2"]


def test_read_text_tie_prefers_semicolon_over_comma() -> None:
    ds = read_text("a;b\n1,5;2,5\n3,5;4,5\n")
    assert list(ds.original.columns) == ["a", "b"]


def test_read_text_single_column() -> None:
    ds = read_text("v\n1\n2\n3\n")
    assert ds.original["v"].tolist() == [1.0, 2.0, 3.0]


def test_read_text_quoted_cells_and_bom() -> None:
    ds = read_text('﻿"a b","c,d"\n1,2\n')
    assert list(ds.original.columns) == ["a b", "c,d"]
    assert ds.import_notes == ("stripped UTF-8 byte order mark",)


def test_read_text_crlf_line_endings() -> None:
    ds = read_text("x,y\r\n1,2\r\n3,4\r\n")
    assert ds.original["x"].tolist() == [1.0, 3.0]


def test_read_text_empty_input() -> None:
    for text in ("", "\n\n  \n", ",,\n,,\n"):
        with pytest.raises(DataError) as excinfo:
            read_text(text)
        assert code_of(excinfo) == "EMPTY_DATA"


def test_header_only_is_empty_data() -> None:
    with pytest.raises(DataError) as excinfo:
        read_text("x,y\n")
    assert code_of(excinfo) == "EMPTY_DATA"


def test_ragged_rows_are_padded() -> None:
    ds = read_text("x,y,z\n1,2,3\n4,5\n")
    assert ds.original["z"].isna().tolist() == [False, True]


# --- safety limits and errors ----------------------------------------------------------


def test_file_over_max_bytes_is_rejected() -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(DATA_DIR / "basic.csv", limits=ImportLimits(max_bytes=10))
    assert code_of(excinfo) == "FILE_TOO_LARGE"


def test_read_text_over_max_bytes_is_rejected() -> None:
    with pytest.raises(DataError) as excinfo:
        read_text("x,y\n1,2\n3,4\n", limits=ImportLimits(max_bytes=5))
    assert code_of(excinfo) == "FILE_TOO_LARGE"


def test_too_many_rows() -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(DATA_DIR / "basic.csv", limits=ImportLimits(max_rows=4))
    assert code_of(excinfo) == "TOO_MANY_ROWS"
    read_dataset(DATA_DIR / "basic.csv", limits=ImportLimits(max_rows=5))  # exactly at the limit


def test_too_many_rows_without_header() -> None:
    with pytest.raises(DataError) as excinfo:
        read_text("1,2\n3,4\n5,6\n", limits=ImportLimits(max_rows=2))
    assert code_of(excinfo) == "TOO_MANY_ROWS"


def test_too_many_cols() -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(DATA_DIR / "basic.csv", limits=ImportLimits(max_cols=1))
    assert code_of(excinfo) == "TOO_MANY_COLS"


def test_trailing_blank_cells_do_not_count_toward_max_cols() -> None:
    ds = read_dataset(DATA_DIR / "bom_blank.csv", limits=ImportLimits(max_cols=2))
    assert ds.original.shape == (3, 2)


def test_json_limits(tmp_path: Path) -> None:
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"x": [1, 2, 3], "y": [4, 5, 6]}))
    with pytest.raises(DataError) as excinfo:
        read_dataset(rows, limits=ImportLimits(max_rows=2))
    assert code_of(excinfo) == "TOO_MANY_ROWS"
    with pytest.raises(DataError) as excinfo:
        read_dataset(rows, limits=ImportLimits(max_cols=1))
    assert code_of(excinfo) == "TOO_MANY_COLS"


def test_xlsx_row_limit(sample_xlsx: Path) -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(sample_xlsx, limits=ImportLimits(max_rows=3))
    assert code_of(excinfo) == "TOO_MANY_ROWS"


def test_xlsx_zip_bomb_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"\0" * 5_000_000)
    assert path.stat().st_size < 100_000
    with pytest.raises(DataError) as excinfo:
        read_dataset(path, limits=ImportLimits(max_bytes=100_000))
    assert code_of(excinfo) == "FILE_TOO_LARGE"


def test_xlsx_that_is_not_a_zip(tmp_path: Path) -> None:
    path = tmp_path / "bad.xlsx"
    path.write_text("not a workbook")
    with pytest.raises(DataError) as excinfo:
        read_dataset(path)
    assert code_of(excinfo) == "PARSE_ERROR"


def test_xls_is_unsupported_even_if_the_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(tmp_path / "old.xls")
    assert code_of(excinfo) == "UNSUPPORTED_FORMAT"
    assert "xlsx" in str(excinfo.value)


def test_unknown_extension_needs_fmt(tmp_path: Path) -> None:
    path = tmp_path / "data.dat"
    path.write_text("x,y\n1,2\n3,4\n")
    with pytest.raises(DataError) as excinfo:
        read_dataset(path)
    assert code_of(excinfo) == "UNSUPPORTED_FORMAT"
    assert read_dataset(path, fmt="csv").original.shape == (2, 2)
    assert read_dataset(path, fmt=".CSV").original.shape == (2, 2)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(tmp_path / "nope.csv")
    assert code_of(excinfo) == "FILE_NOT_FOUND"


def test_directory_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(DataError) as excinfo:
        read_dataset(tmp_path)
    assert code_of(excinfo) in {"FILE_NOT_FOUND", "UNSUPPORTED_FORMAT"}


@pytest.mark.parametrize(
    "content",
    ["3", '"text"', "[1, 2, 3]", '{"a": 1}', '[{"x": [1]}]', '{"x": [1, 2], "y": [1]}'],
)
def test_json_unsupported_shapes(tmp_path: Path, content: str) -> None:
    path = tmp_path / "s.json"
    path.write_text(content)
    with pytest.raises(DataError) as excinfo:
        read_dataset(path)
    assert code_of(excinfo) == "UNSUPPORTED_FORMAT"


def test_json_malformed(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text("{not json")
    with pytest.raises(DataError) as excinfo:
        read_dataset(path)
    assert code_of(excinfo) == "PARSE_ERROR"


def test_json_empty_containers(tmp_path: Path) -> None:
    for content in ("[]", "{}"):
        path = tmp_path / "e.json"
        path.write_text(content)
        with pytest.raises(DataError) as excinfo:
            read_dataset(path)
        assert code_of(excinfo) == "EMPTY_DATA"


def test_invalid_utf8_is_a_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "latin.csv"
    path.write_bytes(b"x,y\n1,caf\xe9\n")
    with pytest.raises(DataError) as excinfo:
        read_dataset(path)
    assert code_of(excinfo) == "PARSE_ERROR"


def test_csv_field_over_parser_limit_is_a_parse_error() -> None:
    with pytest.raises(DataError) as excinfo:
        read_text("x,y\n1," + "a" * 200_000 + "\n")
    assert code_of(excinfo) == "PARSE_ERROR"


# --- csv delimiter policy --------------------------------------------------------------


def test_semicolon_csv_fixture() -> None:
    pd.testing.assert_frame_equal(read_dataset(DATA_DIR / "semicolon.csv").original, EXPECTED)


def test_csv_prefers_comma_on_a_tie(tmp_path: Path) -> None:
    path = tmp_path / "tie.csv"
    path.write_text("a,b;c\n1,2;3\n")
    assert list(read_dataset(path).original.columns) == ["a", "b;c"]


def test_tsv_ignores_commas(tmp_path: Path) -> None:
    path = tmp_path / "t.tsv"
    path.write_text("a,b\tc\n1\t2\n")
    assert list(read_dataset(path).original.columns) == ["a,b", "c"]


def test_imported_dataset_is_usable_for_fitting() -> None:
    ds = read_dataset(DATA_DIR / "basic.csv")
    xy = ds.xy("x", "y")
    assert xy.x.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
