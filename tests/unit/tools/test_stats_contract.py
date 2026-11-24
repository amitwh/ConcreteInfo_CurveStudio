"""The contract of the statistical tools: one result model, built from blocks.

Every tool returns a :class:`ToolResult`, and one renderer shows it on every surface, so
these tests hold the contract itself to what the renderers rely on: a number that is not
finite never gets in, a p-value reads the same everywhere (the web app pins its own copy of
the rule to the same cases), the blocks come back from JSON as their own classes, and the
schema the web app's types are generated from is plain JSON. The tools themselves are
tested where they live, family by family.
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic.json_schema import PydanticJsonSchemaWarning

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ExcludedRow, ToolError, Verdict
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    BoxPlot,
    Counts,
    Estimate,
    HistogramPlot,
    HypothesisTest,
    MultiLinePlot,
    QQPlot,
    ScatterPlot,
    Table,
    TableColumn,
    ToolResult,
    aligned,
    assumption,
    check_level,
    levels,
    list_excluded,
    p_text,
    pairs,
    stride,
    table,
)
from curvestudio.tools.stats._display import (
    MAX_PLOT_POINTS,
    Band,
    BoxGroup,
    Line,
    Series,
    XYSeries,
)


def _column(name: str, values: Sequence[float], rows: Sequence[int] | None = None) -> Column:
    """A column over ``rows`` (every position when none are given), as the data layer builds it."""
    positions = list(range(len(values))) if rows is None else list(rows)
    return Column(
        name=name,
        values=np.asarray(values, dtype=np.float64),
        rows=np.asarray(positions, dtype=np.intp),
    )


def _group(name: str, labels: Sequence[str], rows: Sequence[int]) -> GroupColumn:
    """A group column, its labels an object array of ``str`` as the data layer builds them."""
    return GroupColumn(
        name=name,
        labels=np.asarray(labels, dtype=object),
        rows=np.asarray(rows, dtype=np.intp),
    )


def _result(**blocks: object) -> ToolResult:
    """A result holding ``blocks``, on the plain counts of a toy that used everything."""
    base: dict[str, object] = {
        "tool": "toy",
        "title": "A toy comparison",
        "level": 0.95,
        "interpretation": "The two groups differ.",
        "methods_fragment": "A toy test was run.",
        "counts": Counts(n_offered=4, n_used=4, n_excluded=0),
    }
    return ToolResult(**{**base, **blocks})  # type: ignore[arg-type]


def _test(p: float | None) -> HypothesisTest:
    return HypothesisTest(
        name="Toy",
        method="A toy test, two-sided",
        null_hypothesis="the two means are equal",
        statistic=None if p is None else 1.5,
        p_value=p,
        p_value_text=p_text(p),
        reason=None if p is not None else "the toy had too few rows",
    )


# --------------------------------------------------------------------------- numbers


def test_a_nan_or_infinite_number_is_refused_wherever_it_sits() -> None:
    with pytest.raises(ValueError, match="value"):
        Estimate(label="mean", value=float("nan"))
    with pytest.raises(ValueError, match="se"):
        _result(estimates=(Estimate(label="mean", value=1.0, se=float("inf")),))
    with pytest.raises(ValueError, match="p_value"):
        _result(tests=(_test(float("nan")),))
    with pytest.raises(ValueError, match="statistic"):
        _result(
            tests=(
                HypothesisTest(
                    name="t",
                    method="m",
                    null_hypothesis="h",
                    statistic=float("-inf"),
                    p_value_text="p = 0.5",
                ),
            )
        )
    with pytest.raises(ValueError, match="level"):
        _result(level=float("nan"))
    with pytest.raises(ValueError, match="row 0, column 0"):
        _result(
            tables=(
                Table(
                    title="T", columns=(TableColumn(key="a", label="A"),), rows=((float("nan"),),)
                ),
            )
        )
    with pytest.raises(ValueError, match="x"):
        ScatterPlot(title="S", x_label="x", y_label="y", x=(1.0, float("nan")), y=(1.0, 2.0))
    with pytest.raises(ValueError, match="lower"):
        _result(
            plots=(
                ScatterPlot(
                    title="S",
                    x_label="x",
                    y_label="y",
                    x=(1.0, 2.0),
                    y=(1.0, 2.0),
                    band=Band(x=(1.0, 2.0), lower=(0.0, float("inf")), upper=(2.0, 3.0)),
                ),
            )
        )
    with pytest.raises(ValueError, match="outliers"):
        BoxPlot(
            title="B",
            y_label="y",
            boxes=(
                BoxGroup(
                    label="a",
                    q1=1.0,
                    median=2.0,
                    q3=3.0,
                    lower_whisker=0.0,
                    upper_whisker=4.0,
                    outliers=(float("nan"),),
                ),
            ),
        )


def test_a_block_refuses_a_field_it_does_not_have_and_cannot_be_written_to() -> None:
    with pytest.raises(ValueError, match="extra"):
        Estimate(label="mean", value=1.0, extra=2.0)
    with pytest.raises(ValueError, match="surprise"):
        _result(surprise="nothing a tool may add")
    result = _result()
    with pytest.raises(ValueError):
        result.interpretation = "Something else"  # type: ignore[misc]


def test_a_zero_p_value_reads_as_below_one_e_minus_300() -> None:
    test = _test(0.0)
    assert test.p_value == 0.0
    assert test.p_value_text == "p < 1e-300"


# One list, in ascending order of p, that pins the whole rule: the web app keeps the same
# list against its TypeScript copy of it (apps/web/src/... : pText), so a change here is a
# change there. ``None`` first, since it is not a number.
@pytest.mark.parametrize(
    ("p", "text"),
    [
        (None, "not computed"),
        (0.0, "p < 1e-300"),
        (3.2e-300, "p < 0.0001"),
        (0.00009, "p < 0.0001"),
        # The two cut-offs themselves fall on the side that still shows the number.
        (0.0001, "p = 0.0001"),
        (0.00012, "p = 0.00012"),
        (0.0324, "p = 0.0324"),
        # Below 0.1 three significant digits are kept, and their trailing zeros dropped.
        (0.05, "p = 0.05"),
        (0.09999, "p = 0.1"),
        # From 0.1 three decimals are shown, trailing zeros and all.
        (0.1, "p = 0.100"),
        (0.3171, "p = 0.317"),
        (0.5, "p = 0.500"),
        (0.9996, "p = 1.000"),
        (1.0, "p = 1.000"),
    ],
)
def test_p_text(p: float | None, text: str) -> None:
    assert p_text(p) == text


# --------------------------------------------------------------------------- alignment


def test_pairs_leaves_a_position_out_of_every_column() -> None:
    a = _column("a", [1.0, 2.0, float("nan"), 4.0, float("inf")], rows=[0, 1, 2, 3, 4])
    b = _column("b", [10.0, 20.0, 30.0, float("nan"), 60.0], rows=[1, 2, 3, 4, 5])

    aligned_ = pairs(a, b)
    assert type(aligned_).__name__ == "Pairs"

    assert [column.name for column in aligned_.columns] == ["a", "b"]
    first, second = aligned_.columns
    assert first.values.tolist() == [2.0, 4.0] and first.rows.tolist() == [1, 3]
    assert second.values.tolist() == [10.0, 30.0] and second.rows.tolist() == [1, 3]
    assert aligned_.excluded == (
        ExcludedRow(index=0, reason="missing in b"),
        ExcludedRow(index=2, reason="missing in a"),
        ExcludedRow(index=4, reason="not a finite number in a"),
        ExcludedRow(index=5, reason="missing in a"),
    )
    # The columns handed back are the data layer's: frozen, with arrays that cannot be written to.
    assert not first.values.flags.writeable
    with pytest.raises(ValueError):
        first.name = "other"  # type: ignore[misc]


def test_pairs_names_the_first_column_at_fault_in_the_order_given() -> None:
    a = _column("a", [float("inf")], rows=[0])
    b = _column("b", [float("nan")], rows=[0])

    assert pairs(a, b).excluded == (ExcludedRow(index=0, reason="not a finite number in a"),)
    assert pairs(b, a).excluded == (ExcludedRow(index=0, reason="missing in b"),)


def test_pairs_of_three_columns_needs_the_position_in_all_of_them() -> None:
    a = _column("a", [1.0, 2.0, 3.0, 4.0], rows=[0, 1, 2, 3])
    b = _column("b", [10.0, 20.0, 30.0, 40.0], rows=[0, 1, 2, 3])
    c = _column("c", [100.0, 200.0, 300.0, float("inf")], rows=[1, 2, 3, 4])

    aligned_ = pairs(a, b, c)

    # Positions 1 to 3 are held by all three; 0 only by a and b, 4 only by c (and not finitely).
    assert [column.values.tolist() for column in aligned_.columns] == [
        [2.0, 3.0, 4.0],
        [20.0, 30.0, 40.0],
        [100.0, 200.0, 300.0],
    ]
    assert [column.rows.tolist() for column in aligned_.columns] == [[1, 2, 3]] * 3
    assert aligned_.excluded == (
        ExcludedRow(index=0, reason="missing in c"),
        ExcludedRow(index=4, reason="missing in a"),
    )
    # The other way round, the same positions are kept, and c names why 4 is not.
    other = pairs(c, b, a)
    assert [column.rows.tolist() for column in other.columns] == [[1, 2, 3]] * 3
    assert other.excluded == (
        ExcludedRow(index=0, reason="missing in c"),
        ExcludedRow(index=4, reason="not a finite number in c"),
    )


def test_pairs_of_one_column_keeps_its_finite_values() -> None:
    a = _column("a", [1.0, float("nan"), float("inf"), 4.0], rows=[7, 8, 9, 10])

    aligned_ = pairs(a)
    assert pairs().columns == () and pairs().excluded == ()

    assert len(aligned_.columns) == 1
    assert aligned_.columns[0].values.tolist() == [1.0, 4.0]
    assert aligned_.columns[0].rows.tolist() == [7, 10]
    assert aligned_.excluded == (
        ExcludedRow(index=8, reason="missing in a"),
        ExcludedRow(index=9, reason="not a finite number in a"),
    )


def _walk_through(*columns: Column) -> tuple[list[NDArray[np.float64]], list[ExcludedRow]]:
    """The same alignment one position at a time: what the vectorised reader is held to."""
    held = [
        {int(row): float(value) for row, value in zip(c.rows, c.values, strict=True)}
        for c in columns
    ]
    kept: list[int] = []
    excluded: list[ExcludedRow] = []
    for position in sorted({row for entry in held for row in entry}):
        reason = None
        for column, entry in zip(columns, held, strict=True):
            if position not in entry or math.isnan(entry[position]):
                reason = f"missing in {column.name}"
            elif not math.isfinite(entry[position]):
                reason = f"not a finite number in {column.name}"
            else:
                continue
            break
        if reason is None:
            kept.append(position)
        else:
            excluded.append(ExcludedRow(index=position, reason=reason))
    return [np.array([entry[position] for position in kept]) for entry in held], excluded


def test_pairs_agrees_with_a_walk_through_every_position() -> None:
    rng = np.random.default_rng(20261124)
    for _ in range(60):
        size = int(rng.integers(1, 40))
        columns = []
        for index in range(int(rng.integers(1, 5))):
            rows = np.sort(rng.choice(size, size=int(rng.integers(0, size + 1)), replace=False))
            values = rng.normal(size=rows.size)
            holes = rng.choice(
                values.size, size=int(rng.integers(0, values.size + 1)), replace=False
            )
            values[holes] = rng.choice([np.nan, np.inf, -np.inf], size=holes.size)
            columns.append(_column(f"c{index}", values.tolist(), rows=rows.tolist()))
        expected, excluded = _walk_through(*columns)
        found = pairs(*columns)
        assert [column.values.tolist() for column in found.columns] == [
            v.tolist() for v in expected
        ]
        assert found.excluded == tuple(excluded)
        assert all(
            np.array_equal(column.rows, found.columns[0].rows) for column in found.columns[1:]
        )


def test_aligned_leaves_out_a_row_without_a_label_in_either_group() -> None:
    values = _column("y", [1.0, 2.0, float("nan"), float("inf"), 5.0, 6.0])
    first = _group("mix", ["a", "b", "a", "b"], rows=[0, 1, 3, 5])
    second = _group("lab", ["x", "y", "x", "y"], rows=[1, 2, 3, 5])

    aligned_ = aligned(values, first, second)
    assert type(aligned_).__name__ == "Aligned"

    assert aligned_.values.name == "y"
    assert aligned_.values.values.tolist() == [2.0, 6.0] and aligned_.values.rows.tolist() == [1, 5]
    assert [group.name for group in aligned_.groups] == ["mix", "lab"]
    assert aligned_.groups[0].labels.tolist() == ["b", "b"] and aligned_.groups[
        0
    ].rows.tolist() == [1, 5]
    assert aligned_.groups[1].labels.tolist() == ["x", "y"] and aligned_.groups[
        1
    ].rows.tolist() == [1, 5]
    assert aligned_.excluded == (
        ExcludedRow(index=0, reason="no group label in lab"),
        ExcludedRow(index=2, reason="missing in y"),
        ExcludedRow(index=3, reason="not a finite number in y"),
        ExcludedRow(index=4, reason="no group label in mix"),
    )
    # The other way round the same rows are kept. Position 4 has a label in neither group, so
    # the first one given names why it went; position 0 has one in "mix" and lacks it only in
    # "lab", whose name is given in either order.
    other = aligned(values, second, first)
    assert other.values.rows.tolist() == [1, 5]
    assert other.excluded[0] == ExcludedRow(index=0, reason="no group label in lab")
    assert other.excluded[3] == ExcludedRow(index=4, reason="no group label in lab")


def test_levels_keep_the_order_of_first_appearance_and_refuse_a_single_one() -> None:
    mixed = _group("mix", ["b", "a", "c", "b", "a"], rows=[0, 1, 2, 3, 4])
    assert levels(mixed) == ("b", "a", "c")

    single = _group("mix", ["a", "a"], rows=[0, 1])
    with pytest.raises(ToolError) as refused:
        levels(single)
    assert refused.value.code == "SINGLE_GROUP"
    assert "1" in refused.value.message and "2" in refused.value.message

    with pytest.raises(ToolError) as three_needed:
        levels(mixed, minimum=4)
    assert three_needed.value.code == "SINGLE_GROUP"
    assert "3" in three_needed.value.message

    with pytest.raises(ToolError) as none_found:
        levels(_group("mix", [], rows=[]))
    assert none_found.value.code == "SINGLE_GROUP"


def test_levels_keep_the_order_of_first_appearance_over_many_groups() -> None:
    # Thousands of rows over dozens of groups, each repeated far from its first appearance:
    # a level still comes where it first appears. An ordered dictionary gives that order
    # independently of NumPy's sort.
    rng = np.random.default_rng(20261124)
    drawn = [f"mix {value}" for value in rng.integers(0, 60, size=5_000)]
    crowd = _group("mix", drawn, rows=range(len(drawn)))

    assert len(set(drawn)) > 30
    assert levels(crowd) == tuple(dict.fromkeys(drawn))


def test_list_excluded_lists_two_hundred_and_counts_the_rest() -> None:
    rows = tuple(ExcludedRow(index=i, reason="missing in y") for i in range(250))
    listed, unlisted = list_excluded(rows)
    assert len(listed) == 200 and listed[0].index == 0 and listed[-1].index == 199
    assert unlisted == 50

    few = rows[:3]
    assert list_excluded(few) == (few, 0)
    assert list_excluded(()) == ((), 0)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_check_level_refuses_a_level_outside_zero_to_one(level: float) -> None:
    with pytest.raises(ToolError) as refused:
        check_level(level)
    assert refused.value.code == "INVALID_OPTION"
    assert "level" in refused.value.message


@pytest.mark.parametrize("level", [0.5, 0.9, 0.95, 0.99, 0.999, 1e-9, 1 - 1e-12])
def test_check_level_takes_a_level_inside_zero_to_one(level: float) -> None:
    assert check_level(level) is None


# --------------------------------------------------------------------------- blocks


def _every_block() -> tuple[ToolResult, tuple[type, ...]]:
    """A result with one table and one plot of every kind, beside the class each plot is."""
    estimates = table(
        "Estimates",
        [
            ("term", "Term", "text"),
            ("estimate", "Estimate", "number"),
            ("p", "p", "p_value"),
            ("n", "n", "int"),
        ],
        [("a", 1.5, 0.0324, 12), ("b", -2, None, 3)],
        note="Toys, all of them.",
    )
    plots = (
        ScatterPlot(
            title="y against x",
            x_label="x",
            y_label="y",
            x=(1.0, 2.0, 3.0),
            y=(2.0, 3.0, 5.0),
            lines=(Line(x0=0.0, y0=0.0, x1=1.0, y1=1.0, label="y = x"),),
            band=Band(x=(0.0, 1.0), lower=(0.0, 1.0), upper=(1.0, 2.0)),
        ),
        HistogramPlot(
            title="Histogram of y",
            x_label="y",
            edges=(0.0, 1.0, 2.0, 3.0),
            counts=(2.0, 1.0, 0.0),
            overlays=(Series(label="expected", values=(1.5, 1.5, 0.0)),),
        ),
        QQPlot(
            title="Normal Q-Q plot of the residuals",
            theoretical=(-1.0, 0.0, 1.0),
            ordered=(0.5, 1.0, 2.0),
            line=Line(x0=-1.0, y0=-1.0, x1=1.0, y1=1.0),
        ),
        BoxPlot(
            title="y by mix",
            y_label="y",
            boxes=(
                BoxGroup(
                    label="a",
                    q1=1.0,
                    median=2.0,
                    q3=3.0,
                    lower_whisker=0.0,
                    upper_whisker=4.0,
                    outliers=(9.0, 12.0),
                ),
            ),
            note="Whiskers reach 1.5 times the interquartile range.",
        ),
        MultiLinePlot(
            title="Mean by dose and mix",
            x_label="Dose",
            y_label="Mean of y",
            series=(
                XYSeries(label="mix a", x=(1.0, 2.0), y=(1.0, 2.0)),
                XYSeries(label="mix b", x=(1.0, 2.0), y=(1.5, 2.5)),
            ),
            x_labels=("low", "high"),
        ),
    )
    result = _result(
        estimates=(Estimate(label="Mean of y", value=2.0, se=0.5, lower=0.9, upper=3.1),),
        tests=(_test(0.0324),),
        assumptions=(
            assumption(
                "equal_spread",
                "The two groups have similar spread",
                Verdict.PASS,
                "Levene's test p = 0.41",
            ),
        ),
        warnings=("Two rows were left out.",),
        counts=Counts(n_offered=6, n_used=4, n_excluded=2),
        excluded=(
            ExcludedRow(index=3, reason="missing in y"),
            ExcludedRow(index=5, reason="missing in y"),
        ),
        tables=(estimates,),
        plots=plots,
    )
    return result, (ScatterPlot, HistogramPlot, QQPlot, BoxPlot, MultiLinePlot)


def test_a_result_survives_a_json_round_trip_as_its_own_classes() -> None:
    result, classes = _every_block()

    back = ToolResult.model_validate_json(result.model_dump_json())

    assert back == result
    assert [type(plot) for plot in back.plots] == list(classes)
    assert [plot.kind for plot in back.plots] == ["scatter", "histogram", "qq", "box", "lines"]
    assert back.tables[0].rows == (("a", 1.5, 0.0324, 12), ("b", -2, None, 3))
    assert back.tests[0].p_value_text == "p = 0.0324"
    assert back.assumptions[0].verdict is Verdict.PASS
    assert back.plots[0].band is not None and back.plots[0].band.lower == (0.0, 1.0)


def test_the_schema_is_plain_json_with_every_point_array_bounded() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = ToolResult.model_json_schema()
    offending = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, PydanticJsonSchemaWarning)
    ]
    assert offending == []

    scatter = schema["$defs"]["ScatterPlot"]["properties"]
    assert scatter["x"]["type"] == "array"
    assert scatter["x"]["items"] == {"type": "number"}
    assert scatter["x"]["maxItems"] == 2000
    assert scatter["kind"]["const"] == "scatter"
    table = schema["$defs"]["Table"]["properties"]
    assert table["rows"]["type"] == "array"
    assert table["rows"]["items"]["items"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
        {"type": "number"},
        {"type": "null"},
    ]
    assert "ndarray" not in repr(schema)


def test_no_plot_holds_more_than_two_thousand_points() -> None:
    assert MAX_PLOT_POINTS == 2_000
    points = tuple(float(value) for value in range(MAX_PLOT_POINTS))
    one_over = (*points, 1.0)
    line = Line(x0=0.0, y0=0.0, x1=1.0, y1=1.0)

    ScatterPlot(title="S", x_label="x", y_label="y", x=points, y=points)
    with pytest.raises(ValueError, match="x"):
        ScatterPlot(title="S", x_label="x", y_label="y", x=one_over, y=one_over)
    with pytest.raises(ValueError, match="edges"):
        HistogramPlot(title="H", x_label="x", edges=(*points, 1.0, 2.0), counts=one_over)
    with pytest.raises(ValueError, match="theoretical"):
        QQPlot(title="Q", theoretical=one_over, ordered=one_over, line=line)
    with pytest.raises(ValueError, match="outliers"):
        BoxPlot(
            title="B",
            y_label="y",
            boxes=(
                BoxGroup(
                    label="a",
                    q1=1.0,
                    median=2.0,
                    q3=3.0,
                    lower_whisker=0.0,
                    upper_whisker=4.0,
                    outliers=one_over,
                ),
            ),
        )
    with pytest.raises(ValueError, match="x"):
        MultiLinePlot(
            title="M",
            x_label="x",
            y_label="y",
            series=(XYSeries(label="a", x=one_over, y=one_over),),
        )


def test_a_plot_takes_any_sequence_of_numbers_and_stores_a_tuple() -> None:
    scatter = ScatterPlot(title="S", x_label="x", y_label="y", x=np.arange(3.0) ** 2, y=[1, 2, 5])
    assert scatter.x == (0.0, 1.0, 4.0) and type(scatter.x) is tuple
    assert scatter.y == (1.0, 2.0, 5.0)
    assert all(type(value) is float for value in scatter.x + scatter.y)


def test_a_plot_refuses_arrays_that_do_not_fit_together() -> None:
    with pytest.raises(ValueError, match="y has 1 point for 2 in x"):
        ScatterPlot(title="S", x_label="x", y_label="y", x=(1.0, 2.0), y=(1.0,))
    with pytest.raises(ValueError, match="lower has 1 point for 2 in x"):
        Band(x=(1.0, 2.0), lower=(0.0,), upper=(2.0, 3.0))
    with pytest.raises(ValueError, match="edges"):
        HistogramPlot(title="H", x_label="x", edges=(0.0, 1.0, 2.0), counts=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="overlays"):
        HistogramPlot(
            title="H",
            x_label="x",
            edges=(0.0, 1.0, 2.0),
            counts=(1.0, 1.0),
            overlays=(Series(label="expected", values=(1.0,)),),
        )
    with pytest.raises(ValueError, match="ordered has 2 points for 3 in theoretical"):
        QQPlot(
            title="Q",
            theoretical=(-1.0, 0.0, 1.0),
            ordered=(0.5, 1.0),
            line=Line(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        )
    with pytest.raises(ValueError, match="y has 1 point for 2 in x"):
        XYSeries(label="a", x=(1.0, 2.0), y=(1.0,))


def test_a_plot_union_takes_only_the_five_kinds() -> None:
    with pytest.raises(ValueError, match="kind"):
        _result(plots=({"kind": "pie", "title": "S"},))  # type: ignore[arg-type]


def test_a_table_cell_fits_its_column() -> None:
    columns = (
        TableColumn(key="group", label="Group", kind="text"),
        TableColumn(key="n", label="n", kind="int"),
        TableColumn(key="mean", label="Mean"),
        TableColumn(key="p", label="p", kind="p_value"),
    )
    built = Table(title="T", columns=columns, rows=(("a", 3, 1.5, 0.0324), ("b", None, None, None)))
    assert built.columns[2].kind == "number"
    assert built.rows == (("a", 3, 1.5, 0.0324), ("b", None, None, None))

    with pytest.raises(ValueError, match="row 1"):
        Table(title="T", columns=columns, rows=(("a", 3, 1.5, 0.0324), ("b", 3, 1.5)))
    with pytest.raises(ValueError, match="row 0"):
        Table(title="T", columns=columns, rows=(("a", 3, 1.5, 0.0324, 1.0),))
    with pytest.raises(ValueError, match="a bool"):
        Table(title="T", columns=columns, rows=(("a", True, 1.5, 0.0324),))
    with pytest.raises(ValueError, match="a bool"):
        Table(title="T", columns=columns, rows=(("a", 3, False, 0.0324),))
    with pytest.raises(ValueError, match="whole number"):
        Table(title="T", columns=columns, rows=(("a", 3.0, 1.5, 0.0324),))
    with pytest.raises(ValueError, match="not a number"):
        Table(title="T", columns=columns, rows=(("a", 3, "1.5", 0.0324),))
    with pytest.raises(ValueError, match="not text"):
        Table(title="T", columns=columns, rows=((1, 3, 1.5, 0.0324),))


def test_a_malformed_table_is_refused_by_the_model_itself() -> None:
    columns = (TableColumn(key="n", label="n", kind="int"),)
    with pytest.raises(ValueError, match="rows"):
        Table(title="T", columns=columns, rows=(3,))
    with pytest.raises(ValueError, match="columns"):
        Table(title="T", columns=5, rows=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        Table(title="T", columns=({"key": "n", "label": "n", "kind": "hex"}), rows=())


def test_a_table_keeps_a_numpy_count_whole_and_its_kinds_from_the_helper() -> None:
    built = table(
        "T",
        [("group", "Group", "text"), ("n", "n", "int"), ("mean", "Mean", "number")],
        [("a", np.int64(3), np.float32(1.5))],
    )
    group, count, mean = built.rows[0]
    assert (group, count, mean) == ("a", 3, 1.5)
    assert type(count) is int and type(mean) is float
    assert tuple((column.key, column.label, column.kind) for column in built.columns) == (
        ("group", "Group", "text"),
        ("n", "n", "int"),
        ("mean", "Mean", "number"),
    )
    assert built.note is None


# --------------------------------------------------------------------------- words


def toy_tool() -> ToolResult:
    """The shape of a result, with no statistic worth the name behind it."""
    p = 0.0324
    return ToolResult(
        tool="toy",
        title="A toy comparison",
        level=0.95,
        estimates=(Estimate(label="Mean of y", value=2.0, se=0.5, lower=0.9, upper=3.1),),
        tests=(
            HypothesisTest(
                name="Welch",
                method="Welch's t test, two-sided",
                null_hypothesis="the two groups have the same mean",
                statistic=2.3,
                df=18.0,
                p_value=p,
                p_value_text=p_text(p),
                effect=Estimate(label="Difference of means", value=1.0, lower=0.1, upper=1.9),
            ),
        ),
        assumptions=(
            assumption(
                "equal_spread",
                "The two groups have similar spread",
                Verdict.PASS,
                "Levene's test p = 0.41",
                statistic="F = 0.7",
                threshold="p < 0.05 caution; p < 0.001 fail",
            ),
        ),
        warnings=("Two rows were left out.",),
        interpretation=(
            "The mean of y is higher in group b than in group a. The data are consistent "
            "with a difference between 0.1 and 1.9."
        ),
        methods_fragment=(
            "Welch's t test, two-sided, gave t(18) = 2.3, p = 0.0324. The 95 % interval for "
            "the difference of the means is 0.1 to 1.9."
        ),
        counts=Counts(n_offered=22, n_used=20, n_excluded=2),
        excluded=(
            ExcludedRow(index=3, reason="missing in y"),
            ExcludedRow(index=9, reason="missing in y"),
        ),
    )


def test_a_toy_tools_words_state_evidence_and_never_claim_more() -> None:
    assert claims_proof("This proves the difference")
    assert claims_proof("The model is validated")
    assert not claims_proof("Cross-validation was not run; the cross-validated error is 0.4")

    result = toy_tool()
    for check in result.assumptions:
        for text in (check.title, check.evidence, check.explanation, check.next_step or ""):
            assert not claims_proof(text), (check.id, text)
    assert not claims_proof(result.interpretation)
    assert not claims_proof(result.methods_fragment)
    assert not claims_proof(" ".join(result.warnings))
    assert not claims_proof(result.title)


def test_an_assumption_states_what_was_found_and_names_no_test_in_its_title() -> None:
    check = assumption(
        "equal_spread",
        "The two groups have similar spread",
        Verdict.CAUTION,
        "Levene's test p = 0.03",
    )
    assert check.id == "equal_spread"
    assert check.title == "The two groups have similar spread"
    assert check.verdict is Verdict.CAUTION
    assert check.evidence == "Levene's test p = 0.03"
    assert check.explanation and not claims_proof(check.explanation)
    assert (check.statistic, check.threshold, check.next_step) == (None, None, None)

    full = assumption(
        "normality",
        "The residuals look normally distributed",
        Verdict.FAIL,
        "Shapiro-Wilk W = 0.80",
        statistic="W = 0.80",
        threshold="W below 0.90",
        next_step="Use a rank-based comparison.",
    )
    assert (full.statistic, full.threshold, full.next_step) == (
        "W = 0.80",
        "W below 0.90",
        "Use a rank-based comparison.",
    )


def test_stride_hands_back_an_array_a_plot_can_hold() -> None:
    taken, note = stride(np.arange(5.0, 53_215.0))
    assert len(taken) <= MAX_PLOT_POINTS
    assert note == "2,000 of 53,210 points shown, taken at an even stride"
    points = tuple(float(value) for value in taken)
    scatter = ScatterPlot(title="S", x_label="x", y_label="y", x=points, y=points, note=note)
    assert scatter.note == note and len(scatter.x) == len(taken)
    under, quiet = stride(np.arange(2_000.0))
    assert len(under) == 2_000 and quiet is None
