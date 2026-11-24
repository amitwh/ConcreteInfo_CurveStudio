"""The table stored beside a dataset's fit input: what :mod:`curvestudio_api.storage` writes
for the other columns of a dataset and what it reads back, on core ``Dataset`` objects."""

import csv

import numpy as np
import pandas as pd
import pytest

from curvestudio.data import Dataset, ExcludeRows, group_labels
from curvestudio_api.storage import (
    decode_table,
    encode_table,
    table_column_list,
    table_columns,
)

FIT = {"x_column": "x", "y_column": "y", "weight_column": None}
WEIGHTED = {"x_column": "x", "y_column": "y", "weight_column": "w"}


def test_excluded_rows_are_left_out() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0], "g": ["a", "b", "c", "d"]}
    )
    dataset = Dataset("t", frame)
    assert encode_table(dataset, n_rows=4, **FIT) == b"g\na\nb\nc\nd\n"

    kept = dataset.with_op(ExcludeRows((0, 2)))
    assert kept.xy("x", "y").x.size == 2
    assert encode_table(kept, n_rows=2, **FIT) == b"g\nb\nd\n"  # the rows that take part

    with pytest.raises(ValueError, match="2 rows take part but the fit's input has 4"):
        encode_table(kept, n_rows=4, **FIT)
    with pytest.raises(ValueError, match="4 rows take part but the fit's input has 3"):
        encode_table(dataset, n_rows=3, **FIT)


def test_a_table_of_the_fit_columns_alone_stores_nothing() -> None:
    bare = Dataset("t", pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}))
    assert encode_table(bare, n_rows=2, **FIT) is None
    weighted = Dataset("t", pd.DataFrame({"w": [1.0], "x": [2.0], "y": [3.0]}))
    assert encode_table(weighted, n_rows=1, **WEIGHTED) is None
    # the rows are checked even when there is nothing to store
    with pytest.raises(ValueError, match="2 rows take part but the fit's input has 1"):
        encode_table(bare, n_rows=1, **FIT)
    assert table_column_list(["x", "y"], table_payload=b"", **FIT) == [
        {"name": "x", "kind": "number", "n_missing": 0},
        {"name": "y", "kind": "number", "n_missing": 0},
    ]
    assert table_column_list(["w", "x", "y"], table_payload=b"", **WEIGHTED) == [
        {"name": "w", "kind": "number", "n_missing": 0},
        {"name": "x", "kind": "number", "n_missing": 0},
        {"name": "y", "kind": "number", "n_missing": 0},
    ]


def test_a_fit_column_that_is_not_in_the_table_is_refused() -> None:
    dataset = Dataset("t", pd.DataFrame({"x": [1.0], "y": [2.0], "g": ["a"]}))
    with pytest.raises(ValueError, match="'nope'"):
        encode_table(dataset, n_rows=1, **{**FIT, "x_column": "nope"})
    with pytest.raises(ValueError, match="'w'"):
        encode_table(dataset, n_rows=1, **WEIGHTED)


def test_the_other_columns_are_stored_in_the_tables_order_under_their_own_names() -> None:
    frame = pd.DataFrame(
        {
            "y": [3.0, 4.0],
            "a,b": ["u,v", "w"],
            'q"x': ["u,v", "w"],
            "größe": ["u,v", "w"],
            " lead": ["u,v", "w"],
            "w": [1.0, 1.0],
            "x": [1.0, 2.0],
        }
    )
    payload = encode_table(Dataset("t", frame), n_rows=2, **WEIGHTED)
    assert payload is not None
    header, first, second = payload.split(b"\n")[:3]
    assert header == '"a,b","q""x",größe, lead'.encode()
    assert (first, second) == (b'"u,v","u,v","u,v","u,v"', b"w,w,w,w")
    assert list(decode_table(payload)) == ["a,b", 'q"x', "größe", " lead"]
    assert [entry["name"] for entry in table_columns(payload)] == ["a,b", 'q"x', "größe", " lead"]


def test_text_cells_round_trip_exactly() -> None:
    written = [
        "a,b",
        'say "hi"',
        "line one\nline two",
        "  padded  ",
        "NA",
        "null",
        None,
        "   ",
        "cr\r\nlf",
        " 12 ",
        "1_000",
        'lead " and trail',
    ]
    cells = len(written)
    frame = pd.DataFrame(
        {
            "x": np.arange(cells, dtype=float),
            "y": np.arange(cells, dtype=float),
            "text": pd.Series(written, dtype=object),
            "mixed": ["a", "1", "b", "2.5", "007", "NA", "null", "x", "", "y", "z", "w"],
            "blanks": pd.Series([None] * cells, dtype=object),
        }
    )
    payload = encode_table(Dataset("t", frame), n_rows=cells, **FIT)
    assert payload is not None

    decoded = decode_table(payload)
    assert list(decoded) == ["text", "mixed", "blanks"]
    text = decoded["text"]
    assert text.dtype.kind == "O"  # an object array of Python strings
    assert text.tolist() == [cell or "" for cell in written]  # a missing cell is empty
    mixed = decoded["mixed"]
    assert mixed.dtype.kind == "O"  # text beside digit-like cells stays text
    assert mixed.tolist() == ["a", "1", "b", "2.5", "007", "NA", "null", "x", "", "y", "z", "w"]
    blanks = decoded["blanks"]
    assert blanks.dtype == np.float64 and blanks.shape == (cells,)
    assert np.isnan(blanks).all()

    assert table_columns(payload) == [
        {"name": "text", "kind": "text", "n_missing": 2},  # None and the spaces-only cell
        {"name": "mixed", "kind": "text", "n_missing": 1},
        {"name": "blanks", "kind": "number", "n_missing": cells},
    ]


def test_a_text_column_is_an_array_of_python_strings() -> None:
    """Not fixed-width cells: those reserve four bytes for every character of the longest
    cell, in every row, so one long note in a column would ask for gigabytes."""
    rows = 5_000
    cells = ["short"] * rows
    cells[-1] = "x" * 200_000
    frame = pd.DataFrame(
        {
            "x": np.arange(rows, dtype=float),
            "y": np.arange(rows, dtype=float),
            "note": pd.Series(cells, dtype=object),
        }
    )
    payload = encode_table(Dataset("t", frame), n_rows=rows, **FIT)
    note = decode_table(payload)["note"]
    assert note.dtype == np.dtype(object)
    assert note.nbytes == rows * note.itemsize
    assert all(isinstance(cell, str) for cell in note)
    assert (note[0], len(note[-1])) == ("short", 200_000)
    assert table_columns(payload) == [{"name": "note", "kind": "text", "n_missing": 0}]


def test_a_date_column_is_written_as_its_group_label_reads() -> None:
    """A date column gives the same groups on the web as on the desktop: it is stored as
    ``group_labels`` reads it. No other column is touched, blanks included."""
    days = pd.Series(
        [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02 09:30:00"),
            pd.NaT,
            pd.Timestamp("2026-01-01"),
        ]
    )
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
            "day": days,
            "note": pd.Series([" a ", "", "b", None], dtype=object),
        }
    )
    assert frame["day"].dtype.kind == "M"
    dataset = Dataset("t", frame)
    payload = encode_table(dataset, n_rows=4, **FIT)
    assert payload == b"day,note\n2026-01-01, a \n2026-01-02 09:30:00,\n,b\n2026-01-01,\n"
    stored = decode_table(payload)
    assert stored["day"].tolist() == ["2026-01-01", "2026-01-02 09:30:00", "", "2026-01-01"]
    assert stored["note"].tolist() == [" a ", "", "b", ""]
    labels, rows = group_labels(stored["day"])
    on_the_desktop = dataset.group("day")
    assert labels.tolist() == on_the_desktop.labels.tolist()
    assert rows.tolist() == on_the_desktop.rows.tolist() == [0, 1, 3]
    assert table_columns(payload) == [
        {"name": "day", "kind": "text", "n_missing": 1},
        {"name": "note", "kind": "text", "n_missing": 2},
    ]

    # over the rows that take part: the stored positions count those rows alone
    kept = dataset.with_op(ExcludeRows((1,)))
    payload = encode_table(kept, n_rows=3, **FIT)
    assert payload == b"day,note\n2026-01-01, a \n,b\n2026-01-01,\n"
    labels, rows = group_labels(decode_table(payload)["day"])
    on_the_desktop = kept.group("day")
    assert labels.tolist() == on_the_desktop.labels.tolist() == ["2026-01-01", "2026-01-01"]
    taking_part = np.flatnonzero(kept.active_mask())
    assert taking_part[rows].tolist() == on_the_desktop.rows.tolist() == [0, 3]


def test_a_text_cell_longer_than_the_csv_field_limit_round_trips() -> None:
    """The csv module refuses a field of more than 128 KiB unless the limit is lifted for the
    read; the limit of the process is what it was once the read is over."""
    limit = csv.field_size_limit()
    long_cell = "x" * (limit + 1)
    frame = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [3.0, 4.0], "note": pd.Series(["short", long_cell], dtype=object)}
    )
    payload = encode_table(Dataset("t", frame), n_rows=2, **FIT)
    assert payload is not None
    note = decode_table(payload)["note"]
    assert note.tolist() == ["short", long_cell]
    assert table_columns(payload) == [{"name": "note", "kind": "text", "n_missing": 0}]
    assert csv.field_size_limit() == limit


def test_number_cells_round_trip_exactly() -> None:
    values = np.array(
        [
            0.1 + 0.2,
            1e-300,
            1e21,
            1e-7,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            5.0,
            123456789012345678.0,
            1e16,
            2.5e-10,
        ]
    )
    rows = values.size
    other = np.arange(rows, dtype=float)
    other[3] = np.nan
    frame = pd.DataFrame(
        {
            "x": np.arange(rows, dtype=float),
            "y": other.copy(),
            "w": np.arange(rows),
            "m": values,
            "k": other,
        }
    )
    payload = encode_table(Dataset("t", frame), n_rows=rows, **WEIGHTED)
    assert payload is not None
    lines = payload.split(b"\n")
    assert lines[0] == b"m,k"
    assert lines[1 + 7] == b",7.0"  # a missing number cell is empty
    assert lines[1 + 3] == b"1e-07,"
    assert lines[1 + 5 : 1 + 7] == [b"inf,5.0", b"-inf,6.0"]

    again = decode_table(payload)
    assert again["m"].dtype == np.float64
    assert np.array_equal(again["m"], values, equal_nan=True)
    assert np.signbit(again["m"][4])  # the sign of a zero is kept
    assert np.array_equal(again["k"], other, equal_nan=True)
    assert table_columns(payload) == [
        {"name": "m", "kind": "number", "n_missing": 1},
        {"name": "k", "kind": "number", "n_missing": 1},
    ]

    # A table of one column: pandas writes its empty cells as "" (quoted); they read back
    # as missing all the same.
    single = encode_table(Dataset("t", frame[["x", "y", "m"]]), n_rows=rows, **FIT)
    assert single is not None
    assert single.split(b"\n")[1 + 7] == b'""'
    alone = decode_table(single)
    assert list(alone) == ["m"] and np.array_equal(alone["m"], values, equal_nan=True)


def test_an_empty_table_gives_a_header_and_empty_columns() -> None:
    frame = pd.DataFrame(
        {
            "x": pd.Series([], dtype=float),
            "y": pd.Series([], dtype=float),
            "g": pd.Series([], dtype=object),
        }
    )
    payload = encode_table(Dataset("t", frame), n_rows=0, **FIT)
    assert payload == b"g\n"
    columns = decode_table(payload)
    assert list(columns) == ["g"]
    assert columns["g"].shape == (0,)
    assert table_columns(payload) == [{"name": "g", "kind": "number", "n_missing": 0}]


def test_no_payload_holds_nothing() -> None:
    assert decode_table(b"") == {}
    assert table_columns(b"") == []


def test_table_column_list_covers_the_whole_table_in_the_order_given() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, 2.0], "w": [1.0, 1.0], "g": ["a", "b"], "a": [1.5, None], "y": [3.0, 4.0]}
    )
    table = encode_table(Dataset("t", frame), n_rows=2, **WEIGHTED)
    assert table == b"g,a\na,1.5\nb,\n"
    listed = table_column_list(["x", "w", "g", "a", "y"], table_payload=table, **WEIGHTED)
    assert listed == [
        {"name": "x", "kind": "number", "n_missing": 0},
        {"name": "w", "kind": "number", "n_missing": 0},
        {"name": "g", "kind": "text", "n_missing": 0},
        {"name": "a", "kind": "number", "n_missing": 1},
        {"name": "y", "kind": "number", "n_missing": 0},
    ]
    assert [
        entry["name"]
        for entry in table_column_list(["y", "g", "a", "w", "x"], table_payload=table, **WEIGHTED)
    ] == ["y", "g", "a", "w", "x"]


def test_table_column_list_refuses_a_column_it_cannot_account_for() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "g": ["a", "b"], "a": [1.5, None]})
    table = encode_table(Dataset("t", frame), n_rows=2, **FIT)
    assert table is not None
    # a name that is neither a fit column nor a stored one
    with pytest.raises(ValueError, match="'extra'"):
        table_column_list(["x", "g", "a", "y", "extra"], table_payload=table, **FIT)
    # a stored column the list does not name
    with pytest.raises(ValueError, match="'a'"):
        table_column_list(["x", "g", "y"], table_payload=table, **FIT)
    # a fit column the list does not name
    with pytest.raises(ValueError, match="'y'"):
        table_column_list(["x", "g", "a"], table_payload=table, **FIT)


def test_a_malformed_payload_is_refused() -> None:
    with pytest.raises(ValueError, match="row 2 of the stored table has 1 cell"):
        decode_table(b"a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="names a column twice"):
        decode_table(b"a,a\n1,2\n")
    with pytest.raises(ValueError, match="not well-formed"):
        decode_table(b'a\n"unterminated\n')
    with pytest.raises(UnicodeDecodeError):
        decode_table(b"a\n\xff\n")
