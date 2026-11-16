"""Tests for fitting-readiness validation."""

import math

import pandas as pd
import pytest

from curvestudio.data import DataError, Dataset, ExcludeRows, Issue, validate

NAN = math.nan
INF = math.inf


def make(*, notes: tuple[str, ...] = (), **columns: list[object]) -> Dataset:
    return Dataset(name="t", original=pd.DataFrame(columns), import_notes=notes)


def by_code(issues: list[Issue]) -> dict[str, Issue]:
    return {issue.code: issue for issue in issues}


def test_validate_clean_dataset_returns_empty_list() -> None:
    ds = make(x=[1.0, 2.0, 3.0, 4.0], y=[2.0, 4.1, 5.9, 8.0], w=[1.0, 2.0, 1.0, 2.0])
    assert validate(ds, "x", "y") == []
    assert validate(ds, "x", "y", weights="w", positive_required=("x", "y")) == []


def test_non_numeric_from_import_notes() -> None:
    ds = make(
        notes=("coerced:x:3", "coerced:y:1", "coerced:other:0"),
        x=[1.0, 2.0, 3.0, NAN, 5.0],
        y=[1.0, NAN, 3.0, 4.0, 5.0],
        other=[NAN, 1.0, 1.0, 1.0, 1.0],
    )
    issue = by_code(validate(ds, "x", "y"))["NON_NUMERIC"]
    assert issue.severity == "error"
    assert issue.rows == (1, 3)
    assert "x" in issue.message
    assert "y" in issue.message
    assert "other" not in issue.message


def test_non_numeric_from_unparseable_text_cells() -> None:
    ds = make(x=["1", "two", "3", "4"], y=[1.0, 2.0, 3.0, 4.0])
    issues = by_code(validate(ds, "x", "y"))
    assert issues["NON_NUMERIC"].severity == "error"
    assert issues["NON_NUMERIC"].rows == (1,)
    assert "MISSING_VALUE" not in issues  # unparseable text is not a blank cell


def test_non_numeric_notes_for_excluded_or_unknown_rows_ignored() -> None:
    ds = make(
        notes=("coerced:x:2", "coerced:x:99", "coerced:x:abc", "unrelated note"),
        x=[1.0, 2.0, 3.0, 4.0],
        y=[1.0, 2.0, 3.0, 4.0],
    ).with_op(ExcludeRows((2,)))
    assert "NON_NUMERIC" not in by_code(validate(ds, "x", "y"))


def test_missing_value_rows() -> None:
    ds = make(x=[1.0, NAN, 3.0, 4.0], y=[1.0, 2.0, None, 4.0])
    issue = by_code(validate(ds, "x", "y"))["MISSING_VALUE"]
    assert issue.severity == "error"
    assert issue.rows == (1, 2)


def test_non_finite_rows() -> None:
    ds = make(x=[1.0, 2.0, INF, 4.0], y=[1.0, -INF, 3.0, 4.0])
    issue = by_code(validate(ds, "x", "y"))["NON_FINITE"]
    assert issue.severity == "error"
    assert issue.rows == (1, 2)


def test_duplicate_row_reports_second_and_later_occurrences() -> None:
    ds = make(x=[1.0, 2.0, 1.0, 3.0, 1.0], y=[5.0, 6.0, 5.0, 7.0, 5.0])
    issue = by_code(validate(ds, "x", "y"))["DUPLICATE_ROW"]
    assert issue.severity == "warning"
    assert issue.rows == (2, 4)


def test_duplicate_row_considers_weights() -> None:
    ds = make(x=[1.0, 1.0, 1.0], y=[5.0, 5.0, 5.0], w=[1.0, 2.0, 1.0])
    assert by_code(validate(ds, "x", "y", weights="w"))["DUPLICATE_ROW"].rows == (2,)
    assert by_code(validate(ds, "x", "y"))["DUPLICATE_ROW"].rows == (1, 2)


def test_insufficient_data_with_fewer_than_three_active_rows() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0]).with_op(ExcludeRows((0,)))
    issue = by_code(validate(ds, "x", "y"))["INSUFFICIENT_DATA"]
    assert issue.severity == "error"
    assert issue.rows == ()


def test_three_active_rows_are_sufficient() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0])
    assert "INSUFFICIENT_DATA" not in by_code(validate(ds, "x", "y"))


def test_constant_x_is_error() -> None:
    ds = make(x=[2.0, 2.0, 2.0, 2.0], y=[1.0, 2.0, 3.0, 4.0])
    issue = by_code(validate(ds, "x", "y"))["CONSTANT_X"]
    assert issue.severity == "error"
    assert issue.rows == ()


def test_constant_x_ignores_blank_cells_and_needs_two_finite_values() -> None:
    constant = make(x=[2.0, NAN, 2.0, 2.0], y=[1.0, 2.0, 3.0, 4.0])
    assert "CONSTANT_X" in by_code(validate(constant, "x", "y"))
    single = make(x=[2.0, NAN, NAN], y=[1.0, 2.0, 3.0])
    assert "CONSTANT_X" not in by_code(validate(single, "x", "y"))


def test_nonpositive_for_log_only_for_required_columns() -> None:
    ds = make(x=[1.0, 0.0, 3.0, 4.0], y=[-1.0, 2.0, 3.0, 4.0])
    assert "NONPOSITIVE_FOR_LOG" not in by_code(validate(ds, "x", "y"))
    issue = by_code(validate(ds, "x", "y", positive_required=("x",)))["NONPOSITIVE_FOR_LOG"]
    assert issue.severity == "error"
    assert issue.rows == (1,)
    both = by_code(validate(ds, "x", "y", positive_required=("x", "y")))["NONPOSITIVE_FOR_LOG"]
    assert both.rows == (0, 1)
    assert "x" in both.message
    assert "y" in both.message


def test_invalid_weight_covers_zero_negative_and_infinite() -> None:
    ds = make(x=[1.0, 2.0, 3.0, 4.0, 5.0], y=[1.0] * 5, w=[1.0, 0.0, -2.0, INF, 1.0])
    issue = by_code(validate(ds, "x", "y", weights="w"))["INVALID_WEIGHT"]
    assert issue.severity == "error"
    assert issue.rows == (1, 2, 3)


def test_nan_weight_is_missing_value_not_invalid_weight() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], w=[1.0, NAN, 1.0])
    issues = by_code(validate(ds, "x", "y", weights="w"))
    assert issues["MISSING_VALUE"].rows == (1,)
    assert "INVALID_WEIGHT" not in issues


def test_x_range_1e_minus9_to_1e9_warns() -> None:
    ds = make(x=[1e-9, 1.0, 1e9, 2e9], y=[1.0, 2.0, 3.0, 4.0])
    issue = by_code(validate(ds, "x", "y"))["WIDE_DYNAMIC_RANGE"]
    assert issue.severity == "warning"
    assert issue.rows == ()


def test_wide_range_ignores_zero_and_stays_quiet_at_threshold() -> None:
    zero = make(x=[0.0, 1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0, 4.0])
    assert "WIDE_DYNAMIC_RANGE" not in by_code(validate(zero, "x", "y"))
    at_limit = make(x=[1.0, 2.0, 1e12], y=[1.0, 2.0, 3.0])
    assert "WIDE_DYNAMIC_RANGE" not in by_code(validate(at_limit, "x", "y"))
    over = make(x=[1.0, 2.0, 1.1e12], y=[1.0, 2.0, 3.0])
    assert "WIDE_DYNAMIC_RANGE" in by_code(validate(over, "x", "y"))


def test_excluded_rows_not_reported() -> None:
    ds = make(
        notes=("coerced:x:1",),
        x=[1.0, NAN, INF, 4.0, 5.0, 1.0],
        y=[1.0, 2.0, 3.0, NAN, 5.0, 1.0],
        w=[1.0, 1.0, 1.0, 1.0, -1.0, 1.0],
    ).with_op(ExcludeRows((1, 2, 3, 4, 5)))
    issues = validate(ds, "x", "y", weights="w", positive_required=("x",))
    assert [i.code for i in issues] == ["INSUFFICIENT_DATA"]


def test_row_indices_are_original_positions_after_exclusion() -> None:
    ds = make(x=[1.0, 2.0, NAN, 4.0], y=[1.0, 2.0, 3.0, 4.0]).with_op(ExcludeRows((0,)))
    assert by_code(validate(ds, "x", "y"))["MISSING_VALUE"].rows == (2,)


def test_issues_sorted_errors_first_then_warnings_then_code() -> None:
    ds = make(
        x=[1e-9, 1e9, 1e9, NAN, INF],
        y=[1.0, 1.0, 1.0, 2.0, 3.0],
    )
    issues = validate(ds, "x", "y")
    assert [(i.severity, i.code) for i in issues] == [
        ("error", "MISSING_VALUE"),
        ("error", "NON_FINITE"),
        ("warning", "DUPLICATE_ROW"),
        ("warning", "WIDE_DYNAMIC_RANGE"),
    ]
    for issue in issues:
        assert list(issue.rows) == sorted(set(issue.rows))


def test_validate_does_not_raise_for_bad_data() -> None:
    ds = make(
        x=["a", None, "3"],
        y=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.NaT],
    )
    issues = validate(ds, "x", "y")
    assert by_code(issues)["NON_NUMERIC"].rows == (0, 1, 2)


def test_date_column_is_non_numeric_for_every_active_row() -> None:
    ds = make(
        x=[1.0, 2.0, 3.0],
        y=[pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")],
    )
    assert by_code(validate(ds, "x", "y"))["NON_NUMERIC"].rows == (0, 1, 2)


@pytest.mark.parametrize(
    ("x", "y", "weights", "positive"),
    [
        ("nope", "y", None, ()),
        ("x", "nope", None, ()),
        ("x", "y", "nope", ()),
        ("x", "y", None, ("nope",)),
    ],
)
def test_unknown_column_raises(
    x: str, y: str, weights: str | None, positive: tuple[str, ...]
) -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0])
    with pytest.raises(DataError) as excinfo:
        validate(ds, x, y, weights=weights, positive_required=positive)
    assert excinfo.value.code == "UNKNOWN_COLUMN"
