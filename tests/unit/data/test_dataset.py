"""Tests for the immutable Dataset and its reversible operations log."""

import math

import numpy as np
import pandas as pd
import pytest

from curvestudio.data import (
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
