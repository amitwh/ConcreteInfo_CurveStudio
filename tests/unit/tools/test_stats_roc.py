"""The receiver operating characteristic curve tool: the area, its DeLong interval and the cut.

The area is checked three ways: against the Mann-Whitney U written out in this file, against
``scipy.stats.mannwhitneyu`` on the same rows, and against the trapezoid area under the curve
the tool plots. DeLong's interval is checked against an implementation of its variance written
here, with the placement values shown in a comment so the arithmetic can be followed by hand.

The worked example is the aSAH example of the pROC package. Its own expected output,
``tests/testthat/print_output/r.ndka.formula.ci`` in pROC 1.19.1, fetched from the CRAN source
tarball at ``cran.r-project.org/src/contrib/pROC_1.19.1.tar.gz`` on 2026-11-25, prints
``Area under the curve: 0.612`` and ``95% CI: 0.5012-0.7227 (DeLong)`` for ``outcome ~ ndka``,
with 72 controls and 41 cases. The 113 rows are the package's ``data/aSAH.RData`` from the same
tarball, reproduced here only so the test runs without the network.

The tool's properties are checked too: the area is invariant to a strictly increasing
transformation of the score, reversing the score's direction or swapping the labels turns it
into one minus itself, an ideal score gives one and an uninformative one a half. The edge
cases are the rows a class is too short of, a score with too few distinct values, a class left
empty by the complete pairs, a single large tie block, and a score read in the lower direction.
"""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import BaseModel
from scipy import stats

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ToolResult,
    Verdict,
    build_request,
    catalogue,
    run_tool,
)
from curvestudio.tools.stats.registry import FAMILIES, TOOL_MODULES, TOOLS
from curvestudio.tools.stats.roc import RocRequest, run_roc

AUC_RTOL = 1e-12
P_ATOL = 1e-10

_ASAH = """\
Good,3.01
Good,8.54
Good,8.09
Good,10.42
Poor,17.4
Poor,12.75
Good,6
Poor,13.2
Good,15.54
Good,6.01
Poor,15.96
Good,17.86
Poor,5.18
Poor,8.9
Good,13.41
Good,20.75
Poor,11.6
Good,16.11
Poor,32.37
Poor,54.82
Poor,32.41
Good,49.94
Poor,40.34
Good,9.47
Good,6.29
Good,12.53
Good,6.54
Good,6.3
Good,80.3
Good,12.8
Good,9.8
Good,9.81
Poor,9.85
Good,18.21
Good,5.03
Poor,14.04
Poor,21.93
Good,8.02
Poor,7.42
Poor,8.38
Good,6.59
Good,9.63
Good,13.12
Good,7.96
Good,14.34
Good,41.43
Good,7.63
Good,7.06
Good,12.59
Poor,13.56
Poor,3.87
Good,9.44
Good,7.66
Good,12.98
Poor,419.19
Good,27.19
Good,22.27
Good,9.95
Poor,21.22
Good,11.73
Poor,10.4
Good,58.83
Good,6.39
Poor,11.09
Poor,12.22
Poor,13.67
Good,17.21
Poor,22.63
Poor,12.9
Good,9.7
Poor,21.57
Poor,8.23
Poor,72.57
Poor,15.54
Good,10.51
Good,10.6
Good,14.57
Poor,5.19
Poor,9.63
Good,4.61
Poor,21.48
Good,17.3
Good,12.71
Good,9.44
Good,11.07
Good,19.46
Good,10.83
Poor,5.37
Good,11.97
Good,7.75
Good,9.83
Good,10.55
Good,28.49
Good,8.53
Good,12.57
Good,12.9
Good,46.83
Good,11.68
Good,12.67
Good,9.01
Poor,9.57
Poor,34.06
Poor,11.72
Poor,14.26
Poor,47.61
Poor,11.67
Good,24.58
Good,10.33
Good,13.87
Poor,15.89
Good,22.43
Good,6.79
Good,13.45
"""


def _column(
    values: Sequence[float], name: str = "score", rows: Sequence[int] | None = None
) -> Column:
    positions = list(range(len(values))) if rows is None else list(rows)
    return Column(
        name=name,
        values=np.asarray(values, dtype=np.float64),
        rows=np.asarray(positions, dtype=np.intp),
    )


def _group(
    labels: Sequence[str], name: str = "outcome", rows: Sequence[int] | None = None
) -> GroupColumn:
    positions = list(range(len(labels))) if rows is None else list(rows)
    return GroupColumn(
        name=name,
        labels=np.asarray(labels, dtype=object),
        rows=np.asarray(positions, dtype=np.intp),
    )


def _two_classes(
    negative: Sequence[float], positive: Sequence[float]
) -> tuple[Column, GroupColumn]:
    """One score column and the labels that split it, the positive class appearing second."""
    values = list(negative) + list(positive)
    labels = ["neg"] * len(negative) + ["pos"] * len(positive)
    return _column(values), _group(labels)


def _estimate(result: ToolResult, label: str):
    for found in result.estimates:
        if found.label == label:
            return found
    raise AssertionError(f"no estimate labelled {label!r}: {[e.label for e in result.estimates]}")


def _written_u(positive: Sequence[float], negative: Sequence[float]) -> float:
    """The Mann-Whitney U by its definition: wins plus one half for each tie."""
    return sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in positive for y in negative)


def _asah_case() -> tuple[Column, GroupColumn]:
    rows = [line.split(",") for line in _ASAH.strip().splitlines()]
    labels = [row[0] for row in rows]
    values = [float(row[1]) for row in rows]
    return _column(values, "ndka"), _group(labels, "outcome")


# --------------------------------------------------------------------------- the contract


def test_the_result_is_the_contract_with_a_threshold_table_and_a_curve() -> None:
    column, groups = _two_classes([1.0, 4.0, 5.0, 7.0, 9.0, 11.0], [2.0, 4.0, 4.0, 6.0, 8.0, 10.0])

    result = run_roc(RocRequest(score=column, outcome=groups))

    assert type(result) is ToolResult
    assert result.tool == "roc"
    assert result.title == "Receiver operating characteristic curve of score by outcome"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=12, n_used=12, n_excluded=0)
    assert [estimate.label for estimate in result.estimates] == [
        "Area under the curve",
        "Best cut-off (Youden)",
        "Sensitivity at the cut",
        "Specificity at the cut",
    ]
    assert len(result.tests) == 1
    test = result.tests[0]
    assert test.name == "Mann-Whitney U test of the scores"
    assert test.null_hypothesis == (
        "the scores do not rank the positive class above the negative class"
    )
    assert test.effect == result.estimates[0]
    assert test.statistic == pytest.approx(16.0)
    assert [check.id for check in result.assumptions] == ["independence"]
    assert result.assumptions[0].verdict is Verdict.NOT_APPLICABLE
    assert [found.title for found in result.tables] == ["Thresholds"]
    assert [column.label for column in result.tables[0].columns] == [
        "Threshold",
        "Sensitivity",
        "Specificity",
        "Youden index",
        "Classified positive",
        "Classified negative",
    ]
    assert [column.kind for column in result.tables[0].columns] == [
        "number",
        "number",
        "number",
        "number",
        "int",
        "int",
    ]
    assert len(result.tables[0].rows) == 10
    assert result.plots[0].kind == "lines"
    assert result.plots[0].title == "ROC curve"
    assert [series.label for series in result.plots[0].series] == ["ROC curve", "Chance"]
    assert result.plots[0].series[1].x == (0.0, 1.0)
    assert result.plots[0].series[1].y == (0.0, 1.0)
    assert result.plots[0].series[0].x[0] == 0.0 and result.plots[0].series[0].y[0] == 0.0
    assert result.plots[0].series[0].x[-1] == 1.0 and result.plots[0].series[0].y[-1] == 1.0
    assert "6 positive rows ('pos') and 6 negative rows ('neg')" in result.tables[0].note
    assert len(result.warnings) == 2
    assert "does not establish a diagnosis" in result.interpretation


# --------------------------------------------------------------------------- references


def test_the_auc_is_the_mann_whitney_probability_and_matches_scipy() -> None:
    positive = [1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 6.0, 8.0]
    negative = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 7.0]
    column, groups = _two_classes(negative, positive)

    result = run_roc(RocRequest(score=column, outcome=groups))

    written = _written_u(positive, negative)
    expected = written / (len(positive) * len(negative))
    assert _estimate(result, "Area under the curve").value == pytest.approx(expected, rel=AUC_RTOL)

    reference = stats.mannwhitneyu(
        positive, negative, alternative="two-sided", method="asymptotic", use_continuity=True
    )
    assert reference.statistic / (len(positive) * len(negative)) == pytest.approx(
        expected, rel=AUC_RTOL
    )
    assert result.tests[0].statistic == pytest.approx(float(reference.statistic), rel=AUC_RTOL)
    assert result.tests[0].p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)


def test_the_trapezoid_area_under_the_plotted_curve_is_the_reported_area() -> None:
    positive = [1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 6.0, 8.0]
    negative = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 7.0]
    column, groups = _two_classes(negative, positive)

    result = run_roc(RocRequest(score=column, outcome=groups))
    line = result.plots[0].series[0]
    x = np.asarray(line.x, dtype=np.float64)
    y = np.asarray(line.y, dtype=np.float64)
    trapezoid = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))

    assert trapezoid == pytest.approx(_estimate(result, "Area under the curve").value, rel=AUC_RTOL)


def test_the_delong_interval_is_the_placement_values_variance() -> None:
    """A six against six dataset whose placement values are read by hand.

    The positive scores are 2, 4, 4, 6, 8, 10 and the negative scores 1, 4, 5, 7, 9, 11.
    V10 = (1/6, 1.5/6, 1.5/6, 3/6, 4/6, 5/6) and V01 = (1, 4/6, 3/6, 2/6, 1/6, 0); both
    means are 4/9. Their sample variances are 462/6480 and 840/6480, so the DeLong standard
    error is the square root of (462 + 840) / 6480 / 6 = 0.18299632.
    """
    positive = np.asarray([2.0, 4.0, 4.0, 6.0, 8.0, 10.0])
    negative = np.asarray([1.0, 4.0, 5.0, 7.0, 9.0, 11.0])
    column, groups = _two_classes(negative.tolist(), positive.tolist())

    result = run_roc(RocRequest(score=column, outcome=groups))
    area = _estimate(result, "Area under the curve")

    sorted_negative = np.sort(negative)
    below = np.searchsorted(sorted_negative, positive, side="left")
    through = np.searchsorted(sorted_negative, positive, side="right")
    v10 = (below + 0.5 * (through - below)) / negative.size
    sorted_positive = np.sort(positive)
    below = np.searchsorted(sorted_positive, negative, side="left")
    through = np.searchsorted(sorted_positive, negative, side="right")
    v01 = ((positive.size - through) + 0.5 * (through - below)) / positive.size

    assert v10 == pytest.approx([1 / 6, 1.5 / 6, 1.5 / 6, 3 / 6, 4 / 6, 5 / 6])
    assert v01 == pytest.approx([1.0, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0.0])
    assert float(np.mean(v10)) == pytest.approx(4 / 9) == pytest.approx(float(np.mean(v01)))
    assert float(np.var(v10, ddof=1)) == pytest.approx(462 / 6480)
    assert float(np.var(v01, ddof=1)) == pytest.approx(840 / 6480)
    se = math.sqrt((462 / 6480) / 6 + (840 / 6480) / 6)
    z = float(stats.norm.ppf(0.975))

    assert area.value == pytest.approx(4 / 9, rel=AUC_RTOL)
    assert area.se == pytest.approx(se, rel=1e-12)
    assert area.lower == pytest.approx(4 / 9 - z * se, rel=1e-12)
    assert area.upper == pytest.approx(4 / 9 + z * se, rel=1e-12)


def test_the_published_asah_example_reproduces_its_area_and_delong_interval() -> None:
    column, groups = _asah_case()

    result = run_roc(RocRequest(score=column, outcome=groups))
    area = _estimate(result, "Area under the curve")

    assert result.counts == Counts(n_offered=113, n_used=113, n_excluded=0)
    assert round(area.value, 3) == 0.612
    assert area.lower == pytest.approx(0.5012, abs=5e-5)
    assert area.upper == pytest.approx(0.7227, abs=5e-5)
    assert "41 positive rows ('Poor') and 72 negative rows ('Good')" in result.tables[0].note
    assert result.warnings == ()


# --------------------------------------------------------------------------- properties


def test_a_strictly_increasing_transformation_leaves_the_area_alone() -> None:
    positive = [1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 6.0, 8.0]
    negative = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 7.0]
    base_column, base_groups = _two_classes(negative, positive)
    base = run_roc(RocRequest(score=base_column, outcome=base_groups))
    raised_column, raised_groups = _two_classes(
        [math.exp(value) for value in negative], [math.exp(value) for value in positive]
    )

    raised = run_roc(RocRequest(score=raised_column, outcome=raised_groups))

    assert _estimate(raised, "Area under the curve").value == pytest.approx(
        _estimate(base, "Area under the curve").value, rel=AUC_RTOL
    )
    assert raised.tests[0].statistic == pytest.approx(base.tests[0].statistic, rel=AUC_RTOL)
    assert raised.tests[0].p_value == pytest.approx(base.tests[0].p_value, abs=P_ATOL)


def test_reversing_the_direction_gives_one_minus_the_area() -> None:
    column, groups = _two_classes([1.0, 4.0, 5.0, 7.0, 9.0, 11.0], [2.0, 4.0, 4.0, 6.0, 8.0, 10.0])
    forward = run_roc(RocRequest(score=column, outcome=groups))
    backward = run_roc(RocRequest(score=column, outcome=groups, higher_is_positive=False))

    assert _estimate(backward, "Area under the curve").value == pytest.approx(
        1.0 - _estimate(forward, "Area under the curve").value, rel=AUC_RTOL
    )
    assert backward.tests[0].statistic == pytest.approx(
        36.0 - forward.tests[0].statistic, rel=AUC_RTOL
    )
    assert backward.tests[0].effect == _estimate(backward, "Area under the curve")
    assert "negated" in backward.methods_fragment
    assert "negated" in backward.interpretation


def test_swapping_the_labels_gives_one_minus_the_area() -> None:
    # The class read as positive is the level that appears second, so swapping which class
    # that is means putting the other block of rows first, with its label.
    negative = [1.0, 4.0, 5.0, 7.0, 9.0, 11.0]
    positive = [2.0, 4.0, 4.0, 6.0, 8.0, 10.0]
    column, groups = _two_classes(negative, positive)
    swapped = run_roc(
        RocRequest(
            score=_column(positive + negative),
            outcome=_group(["pos"] * 6 + ["neg"] * 6),
        )
    )
    forward = run_roc(RocRequest(score=column, outcome=groups))

    assert _estimate(swapped, "Area under the curve").value == pytest.approx(
        1.0 - _estimate(forward, "Area under the curve").value, rel=AUC_RTOL
    )


def test_the_ideal_score_has_area_one_and_an_uninformative_score_one_half() -> None:
    ideal_column, ideal_groups = _two_classes(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    flat_column, flat_groups = _two_classes(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )

    ideal = run_roc(RocRequest(score=ideal_column, outcome=ideal_groups))
    flat = run_roc(RocRequest(score=flat_column, outcome=flat_groups))

    assert _estimate(ideal, "Area under the curve").value == pytest.approx(1.0, rel=AUC_RTOL)
    assert _estimate(flat, "Area under the curve").value == pytest.approx(0.5, rel=AUC_RTOL)


# --------------------------------------------------------------------------- edge cases


def test_five_rows_in_a_class_are_refused() -> None:
    column, groups = _two_classes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0])

    with pytest.raises(ToolError) as refused:
        run_roc(RocRequest(score=column, outcome=groups))
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "at least 6" in refused.value.message
    assert "5" in refused.value.message


def test_one_distinct_score_is_refused() -> None:
    column = _column([5.0] * 12)
    groups = _group(["neg"] * 6 + ["pos"] * 6)

    with pytest.raises(ToolError) as refused:
        run_roc(RocRequest(score=column, outcome=groups))
    assert refused.value.code == "CONSTANT_COLUMN"
    assert "1 distinct value" in refused.value.message


def test_a_class_left_empty_by_the_complete_pairs_is_refused() -> None:
    values = [float(value) for value in range(1, 13)] + [float("nan")] * 6
    labels = ["neg"] * 12 + ["pos"] * 6

    with pytest.raises(ToolError) as refused:
        run_roc(RocRequest(score=_column(values), outcome=_group(labels)))
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "'pos' holds 0" in refused.value.message


def test_a_single_large_tie_block_is_handled_by_its_mid_ranks() -> None:
    positive = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    negative = [2.0, 2.0, 2.0, 2.0, 2.0, 3.0]
    column, groups = _two_classes(negative, positive)

    result = run_roc(RocRequest(score=column, outcome=groups))

    written = _written_u(positive, negative)
    assert _estimate(result, "Area under the curve").value == pytest.approx(
        written / (len(positive) * len(negative)), rel=AUC_RTOL
    )
    assert len(result.tables[0].rows) == 3
    assert any("many are tied" in warning for warning in result.warnings)


def test_a_lower_score_can_be_read_as_the_positive_direction() -> None:
    # The positives hold the low scores, so a lower score means positive.
    column, groups = _two_classes([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = run_roc(RocRequest(score=column, outcome=groups, higher_is_positive=False))

    assert _estimate(result, "Area under the curve").value == pytest.approx(1.0, rel=AUC_RTOL)
    assert _estimate(result, "Best cut-off (Youden)").value == pytest.approx(6.0)
    assert _estimate(result, "Sensitivity at the cut").value == pytest.approx(1.0)
    assert _estimate(result, "Specificity at the cut").value == pytest.approx(1.0)
    assert "negated" in result.methods_fragment
    assert "at or below the cut-off" in result.interpretation
    assert "at or below it is classified positive" in result.tables[0].note


def test_a_level_outside_zero_to_one_is_refused() -> None:
    column, groups = _two_classes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    with pytest.raises(ToolError) as refused:
        run_roc(RocRequest(score=column, outcome=groups, level=1.0))
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_that_is_not_its_own_is_refused() -> None:
    class Other(BaseModel):
        x: int = 1

    with pytest.raises(ToolError) as refused:
        run_roc(Other())
    assert refused.value.code == "INVALID_OPTION"
    assert "RocRequest" in refused.value.message


# --------------------------------------------------------------------------- excluded rows


def test_the_excluded_rows_are_listed_with_their_reason() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, float("nan"), 99.0]
    labels = ["neg"] * 6 + ["pos"] * 6 + ["pos"]

    result = run_roc(
        RocRequest(
            score=_column(values, "s"),
            outcome=_group(labels, "g", rows=list(range(13))),
        )
    )

    assert result.counts == Counts(n_offered=14, n_used=12, n_excluded=2)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (12, "missing in s"),
        (13, "no group label in g"),
    ]
    assert result.excluded_unlisted == 0


# --------------------------------------------------------------------------- bounded tables


def test_a_long_curve_is_sampled_in_the_table_at_an_even_stride() -> None:
    negative = [float(value) for value in range(150)]
    positive = [float(value) for value in range(150, 300)]
    column, groups = _two_classes(negative, positive)

    result = run_roc(RocRequest(score=column, outcome=groups))

    assert len(result.tables[0].rows) == 200
    assert "200 of 300 thresholds shown, taken at an even stride" in result.tables[0].note


def test_a_very_long_curve_is_strided_in_the_plot() -> None:
    negative = [float(value) for value in range(1_250)]
    positive = [float(value) for value in range(1_250, 2_500)]
    column, groups = _two_classes(negative, positive)

    result = run_roc(RocRequest(score=column, outcome=groups))

    assert len(result.plots[0].series[0].x) == 2_000
    assert "2,000 of 2,501 points shown" in result.plots[0].note


# --------------------------------------------------------------------------- words and JSON


def test_the_words_name_the_methods_and_claim_no_proof() -> None:
    column, groups = _asah_case()
    result = run_roc(RocRequest(score=column, outcome=groups))

    texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
    for text in texts:
        assert not claims_proof(text), text

    for phrase in ("Receiver operating characteristic", "trapezoid", "DeLong", "mid-rank"):
        assert phrase in result.methods_fragment, phrase
    assert "does not establish a diagnosis" in result.interpretation
    assert "chance that a randomly chosen positive row scores above" in result.interpretation


def test_the_result_survives_a_json_round_trip_without_a_nan() -> None:
    column, groups = _asah_case()

    result = run_roc(RocRequest(score=column, outcome=groups))
    payload = result.model_dump_json()

    assert ToolResult.model_validate_json(payload) == result
    assert "NaN" not in payload


# --------------------------------------------------------------------------- registration


def test_the_tool_is_registered_under_measurement_with_described_fields() -> None:
    assert "curvestudio.tools.stats.roc" in TOOL_MODULES
    assert ("measurement", "Measurement and classification") in FAMILIES
    assert [key for key, _ in FAMILIES].index("measurement") == 6
    assert "roc" in TOOLS
    spec = TOOLS["roc"]
    assert spec.family == "measurement"
    assert spec.title == "Receiver operating characteristic curve"

    info = TOOLS.info("roc")
    assert info.needs_data
    assert [field.name for field in info.fields] == [
        "score",
        "outcome",
        "higher_is_positive",
        "level",
    ]
    assert [field.kind for field in info.fields] == ["column", "group", "option", "option"]
    assert [field.label for field in info.fields] == [
        "Score",
        "Outcome",
        "A higher score means positive",
        "Level",
    ]
    assert all(field.description for field in info.fields)
    assert info.fields[-1].option_schema is not None
    assert {entry.name for entry in catalogue() if entry.family == "measurement"} == {"roc"}


def test_the_registry_builds_and_runs_the_curve_from_column_names() -> None:
    columns = {"s": _column([1.0, 4.0, 5.0, 7.0, 9.0, 11.0, 2.0, 4.0, 4.0, 6.0, 8.0, 10.0])}
    groups = {"o": _group(["neg"] * 6 + ["pos"] * 6)}

    request = build_request(
        "roc",
        {"score": "s", "outcome": "o", "higher_is_positive": True, "level": 0.9},
        column=lambda name: columns[name],
        group=lambda name: groups[name],
    )
    assert isinstance(request, RocRequest) and request.level == 0.9
    result = run_tool("roc", request)
    assert result.tool == "roc" and result.level == 0.9
