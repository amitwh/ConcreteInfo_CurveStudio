"""Tests for the immutable Dataset and its reversible operations log."""

import math

import numpy as np
import pandas as pd
import pytest

from curvestudio.data import (
    XY,
    DataError,
    Dataset,
    ExcludeRows,
    RenameColumn,
    TransformColumn,
)


def make(**columns: list[float]) -> Dataset:
    return Dataset(name="t", original=pd.DataFrame(columns))


def test_with_op_exclude_rows_leaves_original_untouched() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
    before = ds.original.copy()
    ds2 = ds.with_op(ExcludeRows((1,)))
    assert ds.original.equals(before)
    assert ds2.original.equals(before)
    assert ds2.fingerprint() != ds.fingerprint()
    assert bool(ds2.active_mask()[1]) is False
    assert ds2.active_mask().tolist() == [True, False, True]
    assert len(ds2.table()) == 3  # excluded rows are kept in the table
    assert ds.active_mask().all()


def test_undo_restores_previous_fingerprint() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
    changed = ds.with_op(ExcludeRows((0,))).with_op(RenameColumn("y", "z"))
    undone = changed.undo()
    assert undone.ops == (ExcludeRows((0,)),)
    assert undone.undo().fingerprint() == ds.fingerprint()
    assert undone.undo().ops == ()


def test_transform_log_of_e() -> None:
    ds = make(x=[1.0, math.e], y=[1.0, 2.0]).with_op(TransformColumn("x", "lx", "log"))
    assert np.allclose(ds.table()["lx"].to_numpy(), [0.0, 1.0], rtol=0, atol=1e-12)
    assert "lx" not in ds.original.columns


@pytest.mark.parametrize(
    ("func", "scale", "offset", "expected"),
    [
        ("log10", 1.0, 0.0, [0.0, 1.0, 2.0]),
        ("sqrt", 1.0, 0.0, [1.0, math.sqrt(10.0), 10.0]),
        ("reciprocal", 1.0, 0.0, [1.0, 0.1, 0.01]),
        ("square", 1.0, 0.0, [1.0, 100.0, 10000.0]),
        ("affine", 2.0, 3.0, [5.0, 23.0, 203.0]),
    ],
)
def test_transform_functions(func: str, scale: float, offset: float, expected: list[float]) -> None:
    ds = make(x=[1.0, 10.0, 100.0]).with_op(
        TransformColumn("x", "t", func, scale=scale, offset=offset)  # type: ignore[arg-type]
    )
    assert np.allclose(ds.table()["t"].to_numpy(), expected, rtol=1e-12, atol=0)


def test_xy_missing_value_names_row_and_exclusion_fixes_it() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[4.0, math.nan, 6.0])
    with pytest.raises(DataError) as info:
        ds.xy("x", "y")
    assert info.value.code == "MISSING_VALUE"
    assert info.value.rows == (1,)
    xy = ds.with_op(ExcludeRows((1,))).xy("x", "y")
    assert xy.x.tolist() == [1.0, 3.0]
    assert xy.y.tolist() == [4.0, 6.0]
    assert xy.weights is None


def test_xy_missing_rows_are_original_positions_and_ignore_excluded() -> None:
    ds = make(x=[1.0, math.nan, 3.0, 4.0], y=[1.0, 2.0, math.nan, math.nan])
    ds = ds.with_op(ExcludeRows((3,)))
    with pytest.raises(DataError) as info:
        ds.xy("x", "y")
    assert info.value.rows == (1, 2)


def test_xy_rejects_inf() -> None:
    ds = make(x=[1.0, 2.0], y=[math.inf, 1.0])
    with pytest.raises(DataError) as info:
        ds.xy("x", "y")
    assert info.value.code == "NON_FINITE"
    assert info.value.rows == (0,)


def test_xy_weights_validated_and_returned() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], w=[1.0, math.nan, 2.0])
    with pytest.raises(DataError) as info:
        ds.xy("x", "y", weights="w")
    assert info.value.code == "MISSING_VALUE"
    assert info.value.rows == (1,)
    xy = ds.with_op(ExcludeRows((1,))).xy("x", "y", weights="w")
    assert xy.weights is not None
    assert xy.weights.tolist() == [1.0, 2.0]
    assert xy.x_name == "x"
    assert xy.y_name == "y"


def test_xy_unknown_column() -> None:
    with pytest.raises(DataError) as info:
        make(x=[1.0], y=[1.0]).xy("x", "nope")
    assert info.value.code == "UNKNOWN_COLUMN"


def test_xy_non_numeric_cell() -> None:
    ds = Dataset("t", pd.DataFrame({"x": [1.0, 2.0], "y": ["1.5", "abc"]}))
    with pytest.raises(DataError) as info:
        ds.xy("x", "y")
    assert info.value.code == "NON_NUMERIC"
    assert info.value.rows == (1,)


def test_xy_rejects_negative_weights_with_original_rows() -> None:
    ds = make(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], w=[1.0, -0.5, 2.0])
    with pytest.raises(DataError) as info:
        ds.xy("x", "y", weights="w")
    assert info.value.code == "INVALID_WEIGHT"
    assert info.value.rows == (1,)


def test_xy_negative_weight_rows_ignore_excluded_rows() -> None:
    ds = make(x=[1.0, 2.0, 3.0, 4.0], y=[1.0, 2.0, 3.0, 4.0], w=[1.0, -1.0, -2.0, 3.0])
    ds = ds.with_op(ExcludeRows((3,)))
    with pytest.raises(DataError) as info:
        ds.xy("x", "y", weights="w")
    assert info.value.code == "INVALID_WEIGHT"
    assert info.value.rows == (1, 2)


# --- XY defends itself (public type, built directly by library callers) ------------------


def direct_xy(**overrides: object) -> XY:
    fields: dict[str, object] = {
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
        "weights": None,
        "x_name": "x",
        "y_name": "y",
        "fingerprint": "f",
    }
    return XY(**{**fields, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fields", "code", "rows"),
    [
        ({"x": [[1.0], [2.0]]}, "INVALID_SHAPE", ()),
        ({"y": np.ones((2, 2))}, "INVALID_SHAPE", ()),
        ({"weights": np.ones((3, 1))}, "INVALID_SHAPE", ()),
        ({"y": [1.0, 2.0]}, "LENGTH_MISMATCH", ()),
        ({"weights": [1.0, 1.0, 1.0, 1.0]}, "LENGTH_MISMATCH", ()),
        ({"x": [1.0, math.nan, 3.0]}, "MISSING_VALUE", (1,)),
        ({"y": [1.0, 2.0, math.inf]}, "NON_FINITE", (2,)),
        ({"weights": [1.0, math.nan, 1.0]}, "MISSING_VALUE", (1,)),
        ({"weights": [1.0, 2.0, -1.0]}, "INVALID_WEIGHT", (2,)),
        ({"weights": [1.0, 2.0, -math.inf]}, "NON_FINITE", (2,)),
    ],
)
def test_xy_constructed_directly_validates_itself(
    fields: dict[str, object], code: str, rows: tuple[int, ...]
) -> None:
    with pytest.raises(DataError) as info:
        direct_xy(**fields)
    assert info.value.code == code
    assert info.value.rows == rows


def test_xy_constructed_directly_accepts_zero_weights() -> None:
    xy = direct_xy(weights=[1.0, 0.0, 2.0])
    assert xy.weights is not None
    assert xy.weights.tolist() == [1.0, 0.0, 2.0]


def test_fingerprint_deterministic_and_ulp_sensitive() -> None:
    a = make(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
    b = make(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 64
    bumped = np.nextafter(5.0, math.inf)
    c = make(x=[1.0, 2.0, 3.0], y=[4.0, float(bumped), 6.0])
    assert c.fingerprint() != a.fingerprint()
    renamed = make(x=[1.0, 2.0, 3.0], z=[4.0, 5.0, 6.0])
    assert renamed.fingerprint() != a.fingerprint()


def test_xy_fingerprint_deterministic_and_data_sensitive() -> None:
    a = make(x=[1.0, 2.0], y=[3.0, 4.0]).xy("x", "y")
    b = make(x=[1.0, 2.0], y=[3.0, 4.0]).xy("x", "y")
    c = make(x=[1.0, 2.0], y=[3.0, 4.5]).xy("x", "y")
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint


def test_xy_arrays_are_read_only_float64() -> None:
    xy = make(x=[1.0, 2.0], y=[3.0, 4.0], w=[1.0, 1.0]).xy("x", "y", weights="w")
    assert xy.x.dtype == np.float64
    assert xy.x.ndim == 1
    for arr in (xy.x, xy.y, xy.weights):
        assert arr is not None
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 99.0


def test_original_is_copied_on_construction() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    ds = Dataset("t", frame)
    frame.loc[0, "x"] = 99.0
    assert ds.original.loc[0, "x"] == 1.0


def test_ops_apply_in_order_against_a_copy() -> None:
    ds = make(x=[1.0, 2.0], y=[3.0, 4.0])
    out = ds.with_op(RenameColumn("x", "t")).with_op(TransformColumn("t", "t2", "square")).table()
    assert list(out.columns) == ["t", "y", "t2"]
    assert list(ds.original.columns) == ["x", "y"]


@pytest.mark.parametrize(
    ("func", "values", "bad_rows"),
    [
        ("log", [1.0, 0.0, -1.0, 2.0], (1, 2)),
        ("log10", [0.0, 1.0], (0,)),
        ("sqrt", [4.0, -0.5], (1,)),
        ("reciprocal", [0.0, 2.0, 0.0], (0, 2)),
    ],
)
def test_domain_error_lists_rows(func: str, values: list[float], bad_rows: tuple[int, ...]) -> None:
    ds = make(x=values)
    with pytest.raises(DataError) as info:
        ds.with_op(TransformColumn("x", "t", func))  # type: ignore[arg-type]
    assert info.value.code == "DOMAIN_ERROR"
    assert info.value.rows == bad_rows


def test_domain_error_ignores_excluded_rows() -> None:
    ds = make(x=[1.0, -1.0, 4.0]).with_op(ExcludeRows((1,)))
    out = ds.with_op(TransformColumn("x", "r", "sqrt")).table()
    assert out["r"].iloc[0] == 1.0
    assert out["r"].iloc[2] == 2.0


def test_rename_errors() -> None:
    ds = make(x=[1.0], y=[2.0])
    with pytest.raises(DataError) as missing:
        ds.with_op(RenameColumn("nope", "z"))
    assert missing.value.code == "UNKNOWN_COLUMN"
    with pytest.raises(DataError) as dup:
        ds.with_op(RenameColumn("x", "y"))
    assert dup.value.code == "DUPLICATE_COLUMN"


def test_transform_unknown_source() -> None:
    with pytest.raises(DataError) as info:
        make(x=[1.0]).with_op(TransformColumn("nope", "t", "log"))
    assert info.value.code == "UNKNOWN_COLUMN"


def test_exclude_rows_out_of_range() -> None:
    with pytest.raises(DataError) as info:
        make(x=[1.0, 2.0]).with_op(ExcludeRows((5,)))
    assert info.value.code == "ROW_OUT_OF_RANGE"
    assert info.value.rows == (5,)


def test_undo_with_no_ops() -> None:
    with pytest.raises(DataError) as info:
        make(x=[1.0]).undo()
    assert info.value.code == "NOTHING_TO_UNDO"


def test_failed_with_op_leaves_dataset_unchanged() -> None:
    ds = make(x=[0.0])
    with pytest.raises(DataError):
        ds.with_op(TransformColumn("x", "t", "log"))
    assert ds.ops == ()


def test_dataset_is_frozen_and_equality_is_value_based() -> None:
    a = make(x=[1.0], y=[2.0])
    b = make(x=[1.0], y=[2.0])
    assert a == b
    assert a.with_op(ExcludeRows((0,))) != a
    with pytest.raises(AttributeError):
        a.name = "other"  # type: ignore[misc]


def test_mutating_original_does_not_affect_dataset() -> None:
    ds = make(x=[1.0, 2.0], y=[3.0, 4.0])
    fingerprint = ds.fingerprint()
    before = ds.xy("x", "y")
    leaked = ds.original
    leaked.iloc[0, 0] = 77.0
    leaked.loc[:, "y"] = -1.0
    assert ds.xy("x", "y").x.tolist() == [1.0, 2.0]
    assert ds.xy("x", "y").fingerprint == before.fingerprint
    assert ds.fingerprint() == fingerprint
    assert ds.original.loc[0, "x"] == 1.0


def test_mutating_table_does_not_affect_dataset() -> None:
    ds = make(x=[1.0, 2.0], y=[3.0, 4.0]).with_op(TransformColumn("x", "t", "square"))
    fingerprint = ds.fingerprint()
    table = ds.table()
    table.iloc[0, 0] = 77.0
    table.loc[:, "t"] = 0.0
    table["extra"] = 1
    assert ds.xy("x", "t").x.tolist() == [1.0, 2.0]
    assert ds.xy("x", "t").y.tolist() == [1.0, 4.0]
    assert list(ds.table().columns) == ["x", "y", "t"]
    assert ds.fingerprint() == fingerprint


def test_derived_dataset_unaffected_by_parent_original_mutation() -> None:
    ds = make(x=[1.0, 2.0], y=[3.0, 4.0])
    child = ds.with_op(ExcludeRows((0,)))
    leaked = ds.original
    leaked.iloc[1, 0] = 55.0
    assert child.xy("x", "y").x.tolist() == [2.0]


def test_constructor_accepts_positional_ops_and_notes() -> None:
    ds = Dataset("n", pd.DataFrame({"x": [1.0]}), (ExcludeRows(()),), ("note",))
    assert ds.ops == (ExcludeRows(()),)
    assert ds.import_notes == ("note",)


def test_unknown_transform_function() -> None:
    with pytest.raises(DataError) as info:
        make(x=[1.0]).with_op(TransformColumn("x", "t", "cube"))  # type: ignore[arg-type]
    assert info.value.code == "UNKNOWN_FUNCTION"


def test_constructor_rejects_duplicate_column_names() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
    with pytest.raises(DataError) as info:
        Dataset("t", frame)
    assert info.value.code == "DUPLICATE_COLUMN"


@pytest.mark.parametrize(
    "column",
    [
        pd.to_datetime(["2026-01-01", "2026-01-02"]),
        pd.to_timedelta(["1 day", "2 days"]),
    ],
)
def test_xy_rejects_datetime_and_timedelta_columns(column: pd.Index) -> None:
    ds = Dataset("t", pd.DataFrame({"x": [1.0, 2.0], "y": column}))
    with pytest.raises(DataError) as info:
        ds.xy("x", "y")
    assert info.value.code == "NON_NUMERIC"


def test_transform_onto_existing_column_is_rejected() -> None:
    ds = make(x=[1.0, 2.0], y=[3.0, 4.0])
    for target in ("y", "x"):
        with pytest.raises(DataError) as info:
            ds.with_op(TransformColumn("x", target, "square"))
        assert info.value.code == "DUPLICATE_COLUMN"


def test_fingerprint_canonicalises_nan_but_keeps_signed_zero() -> None:
    inf = float("inf")
    other_nan = inf - inf
    assert np.float64(other_nan).tobytes() != np.float64(np.nan).tobytes()  # distinct bit patterns
    a = make(x=[1.0, math.nan])
    b = make(x=[1.0, other_nan])
    assert a.fingerprint() == b.fingerprint()
    assert make(x=[0.0]).fingerprint() != make(x=[-0.0]).fingerprint()


def test_exclude_rows_duplicates_are_idempotent() -> None:
    ds = make(x=[1.0, 2.0, 3.0]).with_op(ExcludeRows((1, 1, 1)))
    assert ds.active_mask().tolist() == [True, False, True]


def test_exclude_rows_empty_tuple_is_a_noop_on_the_mask() -> None:
    ds = make(x=[1.0, 2.0]).with_op(ExcludeRows(()))
    assert ds.active_mask().all()
    assert ds.xy("x", "x").x.tolist() == [1.0, 2.0]


def test_exclude_rows_negative_index_is_out_of_range() -> None:
    with pytest.raises(DataError) as info:
        make(x=[1.0, 2.0]).with_op(ExcludeRows((-1,)))
    assert info.value.code == "ROW_OUT_OF_RANGE"
    assert info.value.rows == (-1,)
