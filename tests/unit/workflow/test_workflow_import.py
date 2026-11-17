"""Tests for ``curvestudio.workflow.import_dataset``: sources, limits and data-quality issues."""

import io
import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from curvestudio.workflow import ImportLimits, WorkflowError, import_dataset

CSV = b"x,y\n1,2.1\n2,3.9\n3,6.2\n4,7.8\n"


def issue_codes(outcome_issues: list) -> set[str]:
    return {issue.code for issue in outcome_issues}


def xlsx_bytes(rows: list[list[object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# -- formats ------------------------------------------------------------------------------


def test_csv_bytes_import_picks_the_first_two_numeric_columns() -> None:
    outcome = import_dataset(CSV, "calibration.csv")
    assert outcome.dataset.name == "calibration"
    assert outcome.n_rows == 4
    assert outcome.columns == ("x", "y")
    assert outcome.xy is not None
    assert outcome.xy.x_name == "x"
    assert outcome.xy.y_name == "y"
    assert outcome.xy.x.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert outcome.issues == []


def test_explicit_columns_and_weights() -> None:
    text = b"a,b,w,c\n1,2,1,9\n2,4,2,9\n3,5,1,9\n4,9,3,9\n"
    outcome = import_dataset(text, "t.csv", x_column="a", y_column="b", weight_column="w")
    assert outcome.xy is not None
    assert outcome.xy.x.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert outcome.xy.y.tolist() == [2.0, 4.0, 5.0, 9.0]
    assert outcome.xy.weights is not None
    assert outcome.xy.weights.tolist() == [1.0, 2.0, 1.0, 3.0]
    assert outcome.issues == []


def test_tsv_and_txt_and_json_and_xlsx() -> None:
    tsv = import_dataset(b"x\ty\n1\t2\n2\t4\n3\t7\n", "d.tsv")
    txt = import_dataset(b"x y\n1 2\n2 4\n3 7\n", "d.txt")
    records = json.dumps([{"x": 1, "y": 2}, {"x": 2, "y": 4}, {"x": 3, "y": 7}]).encode()
    js = import_dataset(records, "d.json")
    xl = import_dataset(xlsx_bytes([["x", "y"], [1, 2], [2, 4], [3, 7]]), "d.xlsx")
    for outcome in (tsv, txt, js, xl):
        assert outcome.xy is not None
        assert outcome.xy.y.tolist() == [2.0, 4.0, 7.0]
        assert outcome.issues == []


def test_path_source(tmp_path: Path) -> None:
    path = tmp_path / "from_disk.csv"
    path.write_bytes(CSV)
    outcome = import_dataset(path, "from_disk.csv")
    assert outcome.dataset.name == "from_disk"
    assert outcome.xy is not None


def test_filename_is_never_used_as_a_path(tmp_path: Path) -> None:
    outcome = import_dataset(CSV, "../../etc/evil\\name.csv")
    assert outcome.dataset.name == "name"
    assert outcome.xy is not None


# -- unreadable input -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "filename"),
    [
        (b"", "empty.csv"),
        (b"x,y\n", "header_only.csv"),
        (b"\n\n\n", "blank.csv"),
        (bytes(range(256)) * 20, "garbage.csv"),
        (b"not a zip", "garbage.xlsx"),
        (b"\x00\x01", "garbage.json"),
    ],
)
def test_unreadable_files_raise_unreadable(payload: bytes, filename: str) -> None:
    with pytest.raises(WorkflowError) as caught:
        import_dataset(payload, filename)
    assert caught.value.code == "UNREADABLE"
    assert str(caught.value)  # carries a message


@pytest.mark.parametrize("filename", ["notes.pdf", "table.xls", "noextension", ""])
def test_unknown_extension_is_unsupported_format(filename: str) -> None:
    with pytest.raises(WorkflowError) as caught:
        import_dataset(CSV, filename)
    assert caught.value.code == "UNSUPPORTED_FORMAT"


def test_missing_path_is_unreadable(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError) as caught:
        import_dataset(tmp_path / "nope.csv", "nope.csv")
    assert caught.value.code == "UNREADABLE"


# -- limits ---------------------------------------------------------------------------------


def test_default_limits() -> None:
    assert ImportLimits() == ImportLimits(max_bytes=20_000_000, max_rows=1_000_000)


def test_too_many_bytes() -> None:
    with pytest.raises(WorkflowError) as caught:
        import_dataset(CSV, "a.csv", limits=ImportLimits(max_bytes=10))
    assert caught.value.code == "TOO_LARGE"


def test_too_many_rows() -> None:
    with pytest.raises(WorkflowError) as caught:
        import_dataset(CSV, "a.csv", limits=ImportLimits(max_rows=3))
    assert caught.value.code == "TOO_LARGE"


def test_too_large_path_is_rejected_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    path.write_bytes(CSV)
    with pytest.raises(WorkflowError) as caught:
        import_dataset(path, "big.csv", limits=ImportLimits(max_bytes=5))
    assert caught.value.code == "TOO_LARGE"


# -- data-quality issues never raise --------------------------------------------------------


def test_missing_value_is_an_issue_not_an_exception() -> None:
    outcome = import_dataset(b"x,y\n1,2\n2,\n3,5\n4,6\n", "gaps.csv")
    assert outcome.xy is None
    errors = [i for i in outcome.issues if i.severity == "error"]
    assert "MISSING_VALUE" in {i.code for i in errors}
    missing = next(i for i in errors if i.code == "MISSING_VALUE")
    assert missing.rows == (1,)
    assert missing.model_dump(mode="json")["rows"] == [1]


def test_text_in_a_numeric_column_is_an_issue() -> None:
    outcome = import_dataset(b"x,y\n1,2\n2,oops\n3,5\n4,6\n5,7\n", "text.csv")
    assert outcome.xy is None
    assert "NON_NUMERIC" in issue_codes(outcome.issues)


def test_too_few_rows_is_an_issue() -> None:
    outcome = import_dataset(b"x,y\n1,2\n2,3\n", "tiny.csv")
    assert outcome.xy is None
    assert "INSUFFICIENT_DATA" in issue_codes(outcome.issues)


def test_unknown_column_is_an_issue() -> None:
    outcome = import_dataset(CSV, "a.csv", x_column="x", y_column="nope")
    assert outcome.xy is None
    assert "UNKNOWN_COLUMN" in issue_codes(outcome.issues)


def test_warnings_keep_the_xy() -> None:
    outcome = import_dataset(b"x,y\n1,2\n1,2\n2,3\n3,5\n", "dup.csv")
    assert outcome.xy is not None
    assert [i.severity for i in outcome.issues] == ["warning"]
    assert outcome.issues[0].code == "DUPLICATE_ROW"


def test_one_column_given_without_the_other_is_an_issue() -> None:
    outcome = import_dataset(CSV, "a.csv", x_column="x")
    assert outcome.xy is None
    assert "COLUMNS_REQUIRED" in issue_codes(outcome.issues)


def test_fewer_than_two_numeric_columns_is_an_issue() -> None:
    outcome = import_dataset(b"name,v\na,1\nb,2\nc,3\n", "one.csv")
    assert outcome.xy is None
    assert "COLUMNS_REQUIRED" in issue_codes(outcome.issues)
    assert outcome.columns == ("name", "v")


def test_a_dotfile_name_is_a_stem_without_a_format() -> None:
    with pytest.raises(WorkflowError) as caught:
        import_dataset(CSV, ".csv")
    assert caught.value.code == "UNSUPPORTED_FORMAT"
    assert "extension" in caught.value.message
    assert import_dataset(CSV, "report.final.csv").dataset.name == "report.final"


def test_non_utf8_text_gets_an_actionable_message() -> None:
    with pytest.raises(WorkflowError) as caught:
        import_dataset("x,y\n1,é\n".encode("latin-1"), "latin.csv")
    assert caught.value.code == "UNREADABLE"
    assert "UTF-8" in caught.value.message
    assert "re-save the file as UTF-8" in caught.value.message
    bom = import_dataset(b"\xef\xbb\xbf" + CSV, "bom.csv")
    assert bom.xy is not None
