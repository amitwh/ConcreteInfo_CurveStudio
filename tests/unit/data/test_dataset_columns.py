"""Tests for the missing-tolerant column accessors that feed the statistical tools."""

import math

import numpy as np
import pandas as pd
import pytest

from curvestudio.data import (
    Column,
    DataError,
    Dataset,
    ExcludeRows,
    GroupColumn,
    RenameColumn,
    read_text,
)


def make(**columns: object) -> Dataset:
    """A dataset built by hand; ``columns`` are passed to ``DataFrame`` as given."""
    return Dataset(name="t", original=pd.DataFrame(columns))


def test_a_numeric_column_comes_as_values_with_their_rows() -> None:
    n = 6
    ds = make(a=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5], b=[1.0] * n, g=["u", "v"] * 3)
    column = ds.column("a")
    assert isinstance(column, Column)
    assert column.name == "a"
    assert column.values.dtype == np.float64
    assert column.values.tolist() == [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    assert np.issubdtype(column.rows.dtype, np.integer)
    assert column.rows.tolist() == list(range(n))
    assert column.present == n
    assert column.missing.tolist() == [False] * n


def test_a_blank_and_a_text_cell_are_both_missing_and_counted() -> None:
    cells = pd.Series([1, "", 3, "x", 5], dtype=object)
    column = make(a=cells).column("a")
    assert column.values.tolist() == pytest.approx([1.0, math.nan, 3.0, math.nan, 5.0], nan_ok=True)
    assert column.present == 3
    assert column.missing.tolist() == [False, True, False, True, False]
    assert column.rows.tolist() == [0, 1, 2, 3, 4]  # the rows stay, only the values are NaN


def test_a_text_column_is_all_missing_and_not_an_error() -> None:
    column = make(t=["red", "green", "blue"]).column("t")
    assert np.isnan(column.values).all()
    assert column.present == 0
    assert column.missing.tolist() == [True, True, True]
    assert column.rows.tolist() == [0, 1, 2]


def test_an_unknown_column_is_refused() -> None:
    ds = make(a=[1.0], g=["u"])
    for accessor in (ds.column, ds.group):
        with pytest.raises(DataError) as info:
            accessor("nope")
        assert info.value.code == "UNKNOWN_COLUMN"
        assert info.value.rows == ()


@pytest.mark.parametrize(
    "dates",
    [pd.to_datetime(["2026-01-01", "2026-01-02"]), pd.to_timedelta(["1 day", "2 days"])],
)
def test_a_date_column_is_refused_as_non_numeric(dates: pd.Index) -> None:
    ds = Dataset("t", pd.DataFrame({"a": [1.0, 2.0], "d": dates}))
    with pytest.raises(DataError) as info:
        ds.column("d")
    assert info.value.code == "NON_NUMERIC"
    assert "dates" in str(info.value)


def test_excluded_rows_take_no_part_and_are_listed() -> None:
    ds = make(a=[10.0, 20.0, 30.0, 40.0, 50.0], g=["u", "v", "u", "v", "u"]).with_op(
        ExcludeRows((1, 3))
    )
    assert ds.column("a").rows.tolist() == [0, 2, 4]
    assert ds.column("a").values.tolist() == [10.0, 30.0, 50.0]
    assert ds.group("g").rows.tolist() == [0, 2, 4]
    assert ds.group("g").labels.tolist() == ["u", "u", "u"]
    assert ds.excluded_rows().tolist() == [1, 3]
    assert make(a=[1.0]).excluded_rows().tolist() == []


def test_group_labels_are_strings_with_blanks_left_out() -> None:
    group = make(g=["low", "high", "", " low ", None]).group("g")
    assert isinstance(group, GroupColumn)
    assert group.name == "g"
    assert group.labels.dtype.kind == "U"
    assert group.labels.tolist() == ["low", "high", "low"]
    assert group.rows.tolist() == [0, 1, 3]
    assert group.present == 3
    blank = make(g=["", "   ", None]).group("g")
    assert blank.labels.size == 0
    assert blank.rows.size == 0
    assert blank.present == 0


@pytest.mark.parametrize("blank", [None, math.nan])
def test_a_numeric_group_column_gives_its_numbers_as_text(blank: float | None) -> None:
    group = make(g=[1.0, 2.0, 0.1 + 0.2, 0.3, 2.5, blank]).group("g")
    # whole numbers lose their ".0" and arithmetic noise is not shown
    assert group.labels.tolist() == ["1", "2", "0.3", "0.3", "2.5"]
    assert group.rows.tolist() == [0, 1, 2, 3, 4]


def test_a_date_column_serves_as_a_group() -> None:
    dates = pd.to_datetime(["2026-01-01", None, "2026-01-03"])
    group = Dataset("t", pd.DataFrame({"d": dates})).group("d")
    assert group.labels.tolist() == ["2026-01-01 00:00:00", "2026-01-03 00:00:00"]
    assert group.rows.tolist() == [0, 2]


def test_the_arrays_are_read_only_and_detached() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "g": ["u", "v"]})
    ds = Dataset("t", frame).with_op(ExcludeRows((1,)))
    column = ds.column("a")
    group = ds.group("g")
    excluded = ds.excluded_rows()
    for array, value in (
        (column.values, 9.0),
        (column.rows, 5),
        (column.missing, True),
        (group.labels, "x"),
        (group.rows, 5),
        (excluded, 0),
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = value

    # nothing the caller can reach changes them: the frame the dataset was built from,
    # the copies the dataset hands out, and the arrays a model was built from
    frame.loc[0, "a"] = 99.0
    frame.loc[0, "g"] = "changed"
    leaked = ds.original
    leaked.loc[0, "a"] = 77.0
    shown = ds.table()
    shown.loc[0, "g"] = "changed again"
    assert column.values.tolist() == [1.0]
    assert group.labels.tolist() == ["u"]
    assert not np.shares_memory(column.values, ds.original["a"].to_numpy())
    values, labels, rows = np.array([3.0]), np.array(["z"]), np.array([7])
    built = Column(name="a", values=values, rows=rows)
    grouped = GroupColumn(name="g", labels=labels, rows=rows)
    values[0], labels[0], rows[0] = 8.0, "y", 1
    assert built.values.tolist() == [3.0]
    assert built.rows.tolist() == [7]
    assert grouped.labels.tolist() == ["z"]
    assert grouped.rows.tolist() == [7]


def test_a_rename_is_followed() -> None:
    ds = make(a=[1.0, 2.0], g=["u", "v"]).with_op(RenameColumn("a", "b"))
    assert ds.column("b").values.tolist() == [1.0, 2.0]
    with pytest.raises(DataError) as info:
        ds.column("a")
    assert info.value.code == "UNKNOWN_COLUMN"


def test_an_imported_table_reads_without_errors() -> None:
    # a text cell in a numeric column (NaN plus an import note), blanks, and a padded label
    ds = read_text("a,b,g\n1,10,low\n,20,high\n3,, x \nx,40,low\n5,50,\n")
    assert "coerced:a:3" in ds.import_notes
    assert ds.column("a").values.tolist() == pytest.approx(
        [1.0, math.nan, 3.0, math.nan, 5.0], nan_ok=True
    )
    assert ds.column("a").present == 3
    assert ds.column("b").missing.tolist() == [False, False, True, False, False]
    assert ds.group("g").labels.tolist() == ["low", "high", "x", "low"]
    assert ds.group("g").rows.tolist() == [0, 1, 2, 3]


def test_a_column_refuses_mismatched_arrays() -> None:
    fields: dict[str, object] = {"name": "a", "values": np.array([1.0, 2.0]), "rows": np.array([0])}
    with pytest.raises(ValueError, match="equal lengths"):
        Column(**fields)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one-dimensional"):
        GroupColumn(name="g", labels=np.array([["u"]]), rows=np.array([0]))
    with pytest.raises(ValueError, match="extra"):
        Column(**{**fields, "scale": 2})  # type: ignore[arg-type]


def test_xy_is_unchanged() -> None:
    with pytest.raises(DataError) as info:
        make(x=[1.0, 2.0], y=[3.0, math.nan]).xy("x", "y")
    assert info.value.code == "MISSING_VALUE"
