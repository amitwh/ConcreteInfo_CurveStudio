"""The reliability tool: Cronbach's alpha, its Feldt interval and the item statistics.

The worked example is the binary one on the Real Statistics Using Excel page "Cronbach's
Alpha: Basic Concepts", fetched and read on 2026-11-25:

- https://real-statistics.com/reliability/internal-consistency-reliability/cronbachs-alpha/
  cronbachs-alpha-basic-concepts/
- Its Figure 1 ("Cronbach's Alpha for Example 1") is a worksheet with 12 rows and 11 binary
  items, item totals 10, 9, 9, 9, 8, 6, 5, 5, 4, 3, 1, a total variance of 6.52083, a sum of
  item variances of 2.14583, and alpha in cell B22 of 0.73802. The data below are read from
  that figure. Its sentence that cell B22 reads ".73082" disagrees with the figure itself and
  with the page's later "Cronbach's alpha is .73802"; the computation here gives 0.738019, so
  the figure and the later sentence are the values used.

Every reference number the tool reports is also written out in the tests below: the raw alpha
formula, the standardised alpha from the mean inter-item correlation, Feldt's F test and his
interval from independently taken F quantiles. The properties a reliability coefficient has are
checked too: adding a constant to every item or multiplying every item by a positive constant
leaves it unchanged, two items reproduce the two-item correlation formula, and reversing a
negatively keyed item raises it.
"""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    Estimate,
    ExcludedRow,
    ToolResult,
    Verdict,
    build_request,
    catalogue,
    run_tool,
)
from curvestudio.tools.stats.registry import FAMILIES, TOOL_MODULES, TOOLS
from curvestudio.tools.stats.reliability import ReliabilityRequest, run_reliability

STAT_RTOL = 1e-9
"""The tolerance on a number the test computes by the same formula on the same data."""


def _column(name: str, values: object) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    return Column(name=name, values=array, rows=np.arange(array.size, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_reliability(ReliabilityRequest(**options))  # type: ignore[arg-type]


def _published() -> tuple[Column, ...]:
    """The 12 x 11 binary worksheet of Figure 1 on the page cited in the module docstring."""
    items = [
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return tuple(_column(f"Q{index + 1}", values) for index, values in enumerate(items))


def _scale(rows: int = 30, seed: int = 0) -> tuple[Column, ...]:
    """Three positively keyed items and a fourth keyed in the opposite direction."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=rows)
    positive = [latent + rng.normal(scale=0.6, size=rows) for _ in range(3)]
    negative = 6.0 - (latent + rng.normal(scale=0.6, size=rows))
    return tuple(
        _column(f"item{index + 1}", values) for index, values in enumerate([*positive, negative])
    )


def _matrix(result: ToolResult) -> list[tuple[object, ...]]:
    table = next(found for found in result.tables if found.title == "Item statistics")
    return list(table.rows)


def _estimate(result: ToolResult, label: str) -> Estimate:
    return next(found for found in result.estimates if found.label == label)


# --------------------------------------------------------------------------- the formula


def test_alpha_follows_the_formula_written_out_here() -> None:
    rng = np.random.default_rng(1)
    latent = rng.normal(size=25)
    columns = tuple(
        _column(f"item{index}", latent + rng.normal(scale=0.7, size=25)) for index in range(5)
    )
    values = np.column_stack([column.values for column in columns])
    size = values.shape[1]

    variances = np.var(values, axis=0, ddof=1)
    total = np.var(np.sum(values, axis=1), ddof=1)
    expected = size / (size - 1) * (1.0 - float(np.sum(variances)) / float(total))

    result = _run(items=columns)

    assert _estimate(result, "Cronbach's alpha").value == pytest.approx(expected, rel=STAT_RTOL)


def test_the_published_binary_example_reproduces_alpha_and_its_item_statistics() -> None:
    result = _run(items=_published())

    alpha = _estimate(result, "Cronbach's alpha")
    assert alpha.value == pytest.approx(0.73802, abs=1e-5)
    assert alpha.lower is not None and alpha.upper is not None
    assert alpha.lower < alpha.value < alpha.upper
    assert _estimate(result, "Standardised alpha").value == pytest.approx(0.73272, abs=1e-5)
    assert _estimate(result, "Mean inter-item correlation").value == pytest.approx(
        0.19950, abs=1e-5
    )

    rows = {row[0]: row for row in _matrix(result)}
    corrected = rows["Q1"][3]
    deleted = rows["Q4"][4]
    assert isinstance(corrected, float) and isinstance(deleted, float)
    assert corrected == pytest.approx(-0.014720, abs=1e-6)
    assert deleted == pytest.approx(0.776353, abs=1e-6)
    assert any("Q1" in warning and "negatively" in warning for warning in result.warnings)
    assert "Q4" in result.interpretation
    assert "0.776" in result.interpretation


def test_standardised_alpha_follows_the_mean_inter_item_correlation() -> None:
    columns = _scale(rows=28, seed=2)
    values = np.column_stack([column.values for column in columns])
    size = values.shape[1]
    correlations = np.corrcoef(values, rowvar=False)
    mean = float(np.mean(correlations[np.triu_indices(size, 1)]))
    expected = size * mean / (1.0 + (size - 1.0) * mean)

    result = _run(items=columns)

    assert _estimate(result, "Mean inter-item correlation").value == pytest.approx(
        mean, rel=STAT_RTOL
    )
    assert _estimate(result, "Standardised alpha").value == pytest.approx(expected, rel=STAT_RTOL)


def test_feldts_interval_is_built_from_independent_f_quantiles() -> None:
    columns = _scale(rows=26, seed=3)
    result = _run(items=columns, level=0.9)
    alpha = float(_estimate(result, "Cronbach's alpha").value)
    rows, items = 26, 4
    lower_quantile = float(scipy_stats.f.ppf(0.05, rows - 1, (rows - 1) * (items - 1)))
    upper_quantile = float(scipy_stats.f.ppf(0.95, rows - 1, (rows - 1) * (items - 1)))
    expected_lower = 1.0 - (1.0 - alpha) * upper_quantile
    expected_upper = 1.0 - (1.0 - alpha) * lower_quantile

    estimate = _estimate(result, "Cronbach's alpha")
    assert estimate.lower is not None and estimate.upper is not None
    assert estimate.lower == pytest.approx(expected_lower, rel=STAT_RTOL)
    assert estimate.upper == pytest.approx(expected_upper, rel=STAT_RTOL)
    assert estimate.lower < alpha < estimate.upper


def test_feldts_test_uses_one_over_one_minus_alpha_on_the_published_degrees_of_freedom() -> None:
    result = _run(items=_published())
    alpha = float(_estimate(result, "Cronbach's alpha").value)

    test = result.tests[0]
    expected_statistic = 1.0 / (1.0 - alpha)
    expected_p = float(scipy_stats.f.sf(expected_statistic, 11.0, 110.0))

    assert test.name == "F test of the reliability coefficient (Feldt)"
    assert test.statistic == pytest.approx(expected_statistic, rel=STAT_RTOL)
    assert test.df == 11.0 and test.df_denominator == 110.0
    assert test.p_value == pytest.approx(expected_p, rel=1e-8)
    assert test.effect is not None and test.effect.value == alpha
    assert test.null_hypothesis.startswith("the reliability coefficient is zero")


# --------------------------------------------------------------------------- the properties


def test_adding_a_constant_to_every_item_leaves_alpha_unchanged() -> None:
    columns = _scale(rows=24, seed=4)
    shifted = tuple(_column(column.name, column.values + 10.0) for column in columns)

    plain = _run(items=columns)
    moved = _run(items=shifted)

    for label in ("Cronbach's alpha", "Standardised alpha"):
        assert _estimate(moved, label).value == pytest.approx(
            _estimate(plain, label).value, rel=STAT_RTOL
        )


def test_multiplying_every_item_by_a_positive_constant_leaves_alpha_unchanged() -> None:
    columns = _scale(rows=22, seed=5)
    scaled = tuple(_column(column.name, 3.0 * column.values) for column in columns)

    plain = _run(items=columns)
    larger = _run(items=scaled)

    assert _estimate(larger, "Cronbach's alpha").value == pytest.approx(
        _estimate(plain, "Cronbach's alpha").value, rel=STAT_RTOL
    )


def test_two_items_reproduce_the_correlation_formula() -> None:
    """With k = 2, alpha is the two-item correlation formula, 2r / (1 + r).

    The two items are placed on the same standard deviation first, which is the condition
    under which the raw coefficient reduces to that formula; the standardised coefficient has
    no such condition. Solving the formula for r gives alpha / (2 - alpha) = r, which is
    checked as well.
    """
    rng = np.random.default_rng(6)
    first = rng.normal(size=30)
    second = 0.6 * first + rng.normal(scale=0.8, size=30)
    standardised = [(values - values.mean()) / values.std(ddof=1) for values in (first, second)]
    columns = (_column("a", standardised[0]), _column("b", standardised[1]))
    r = float(np.corrcoef(standardised[0], standardised[1])[0, 1])

    result = _run(items=columns)
    alpha = float(_estimate(result, "Cronbach's alpha").value)

    assert alpha == pytest.approx(2.0 * r / (1.0 + r), rel=STAT_RTOL)
    assert alpha / (2.0 - alpha) == pytest.approx(r, rel=STAT_RTOL)
    assert _estimate(result, "Standardised alpha").value == pytest.approx(
        2.0 * r / (1.0 + r), rel=STAT_RTOL
    )


def test_reversing_a_negatively_keyed_item_raises_alpha() -> None:
    columns = _scale(rows=30, seed=7)
    plain = _run(items=columns, names=("a", "b", "c", "d"))
    recoded = _run(items=columns, names=("a", "b", "c", "d"), reverse=("d",))

    assert _estimate(recoded, "Cronbach's alpha").value > _estimate(plain, "Cronbach's alpha").value
    assert any("item d" in warning for warning in plain.warnings)
    assert "d (range" in recoded.methods_fragment
    assert "max + min - value" in recoded.methods_fragment


# --------------------------------------------------------------------------- the blocks


def test_the_item_table_reports_every_item() -> None:
    columns = _scale(rows=20, seed=8)

    result = _run(items=columns)
    table = next(found for found in result.tables if found.title == "Item statistics")

    assert [column.label for column in table.columns] == [
        "Item",
        "Mean",
        "SD",
        "Corrected item-total correlation",
        "Alpha if deleted",
    ]
    assert [row[0] for row in table.rows] == ["item1", "item2", "item3", "item4"]
    values = np.column_stack([column.values for column in columns])
    means = np.mean(values, axis=0)
    sds = np.std(values, axis=0, ddof=1)
    for index, row in enumerate(table.rows):
        mean, sd = row[1], row[2]
        assert isinstance(mean, float) and isinstance(sd, float)
        assert mean == pytest.approx(float(means[index]), rel=STAT_RTOL)
        assert sd == pytest.approx(float(sds[index]), rel=STAT_RTOL)
        assert row[3] is not None and row[4] is not None


def test_alpha_if_deleted_is_empty_with_two_items() -> None:
    rng = np.random.default_rng(9)
    columns = (_column("a", rng.normal(size=12)), _column("b", rng.normal(size=12)))

    result = _run(items=columns)
    rows = _matrix(result)
    note = result.tables[0].note

    assert [row[4] for row in rows] == [None, None]
    assert note is not None and "two items" in note


def test_fewer_than_five_rows_warns() -> None:
    columns = (
        _column("a", [1.0, 2.0, 4.0, 3.0]),
        _column("b", [2.0, 3.0, 5.0, 4.0]),
        _column("c", [1.0, 3.0, 5.0, 2.0]),
    )

    result = _run(items=columns)

    assert any("fewer than 5" in warning for warning in result.warnings)


def test_the_assumptions_state_tau_equivalence_and_the_single_direction() -> None:
    columns = _scale(rows=15, seed=10)
    result = _run(items=columns)

    assert [check.id for check in result.assumptions] == ["tau_equivalence", "one_direction"]
    assert all(check.verdict is Verdict.NOT_APPLICABLE for check in result.assumptions)
    assert "lower bound" in result.assumptions[0].evidence
    assert "one direction" in result.assumptions[1].title


def test_a_row_missing_in_one_item_is_left_out_and_listed() -> None:
    columns = (
        _column("a", [1.0, 2.0, 3.0, float("nan"), 5.0]),
        _column("b", [2.0, 2.0, 4.0, 4.0, 6.0]),
        _column("c", [3.0, 1.0, 5.0, 2.0, 7.0]),
    )

    result = _run(items=columns)

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=3, reason="missing in a"),)


def test_the_result_is_the_contract() -> None:
    result = _run(items=_scale(rows=18, seed=11))

    assert type(result) is ToolResult
    assert result.tool == "reliability"
    assert result.title == "Reliability (Cronbach's alpha)"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=18, n_used=18, n_excluded=0)
    assert [estimate.label for estimate in result.estimates] == [
        "Cronbach's alpha",
        "Standardised alpha",
        "Mean inter-item correlation",
    ]
    assert len(result.tests) == 1
    assert [table.title for table in result.tables] == ["Item statistics"]
    assert result.plots == ()


def test_two_runs_give_the_same_json() -> None:
    columns = _scale(rows=16, seed=12)

    first = _run(items=columns).model_dump_json()
    second = _run(items=columns).model_dump_json()

    assert first == second


def test_the_result_survives_a_json_round_trip_without_a_nan() -> None:
    result = _run(items=_published())
    payload = result.model_dump_json()

    assert ToolResult.model_validate_json(payload) == result
    assert "NaN" not in payload


def test_the_words_state_the_guides_and_claim_no_proof() -> None:
    result = _run(items=_scale(rows=20, seed=13))

    texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
    for text in texts:
        assert not claims_proof(text), text

    assert "0.70 is a convention, not a law" in result.interpretation
    assert "does not establish that the items measure one construct" in result.interpretation
    for phrase in ("k / (k - 1)", "Feldt's", "corrected item-total correlation", "max + min"):
        assert phrase in result.methods_fragment, phrase


# --------------------------------------------------------------------------- the refusals


def test_two_rows_are_too_few() -> None:
    columns = (
        _column("a", [1.0, 2.0]),
        _column("b", [2.0, 4.0]),
        _column("c", [3.0, 6.0]),
    )

    with pytest.raises(ToolError) as refused:
        _run(items=columns)
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "at least 3" in refused.value.message


def test_one_item_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(items=(_column("a", [1.0, 2.0, 3.0]),))
    assert refused.value.code == "INVALID_OPTION"
    assert "between 2 and 50" in refused.value.message


def test_fifty_one_items_are_refused_and_fifty_are_taken() -> None:
    rng = np.random.default_rng(14)
    many = tuple(_column(f"item{index}", rng.normal(size=6)) for index in range(50))
    assert _run(items=many).counts.n_used == 6

    too_many = (*many, _column("item50", rng.normal(size=6)))
    with pytest.raises(ToolError) as refused:
        _run(items=too_many)
    assert refused.value.code == "INVALID_OPTION"


def test_a_constant_item_is_refused() -> None:
    columns = (
        _column("a", [1.0, 2.0, 3.0, 4.0]),
        _column("b", [2.0, 4.0, 3.0, 5.0]),
        _column("constant", [7.0, 7.0, 7.0, 7.0]),
    )

    with pytest.raises(ToolError) as refused:
        _run(items=columns)
    assert refused.value.code == "CONSTANT_COLUMN"
    assert "constant" in refused.value.message


def test_a_reversed_item_whose_range_has_no_width_is_refused_as_constant() -> None:
    columns = (
        _column("a", [1.0, 2.0, 3.0, 4.0]),
        _column("b", [2.0, 4.0, 3.0, 5.0]),
        _column("constant", [7.0, 7.0, 7.0, 7.0]),
    )

    with pytest.raises(ToolError) as refused:
        _run(items=columns, reverse=("constant",))
    assert refused.value.code == "CONSTANT_COLUMN"


def test_a_reverse_name_that_is_not_an_item_is_unknown() -> None:
    columns = (_column("a", [1.0, 2.0, 3.0]), _column("b", [2.0, 4.0, 5.0]))

    with pytest.raises(ToolError) as refused:
        _run(items=columns, reverse=("c",))
    assert refused.value.code == "UNKNOWN_COLUMN"
    assert "c" in refused.value.message


def test_names_must_name_one_item_each() -> None:
    columns = (_column("a", [1.0, 2.0, 3.0]), _column("b", [2.0, 4.0, 5.0]))

    with pytest.raises(ToolError) as refused:
        _run(items=columns, names=("only one",))
    assert refused.value.code == "INVALID_OPTION"


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    columns = (_column("a", [1.0, 2.0, 3.0]), _column("b", [2.0, 4.0, 5.0]))

    with pytest.raises(ToolError) as refused:
        _run(items=columns, level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_reliability(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"


# --------------------------------------------------------------------------- registration


def test_the_registry_lists_and_runs_the_tool() -> None:
    assert "curvestudio.tools.stats.reliability" in TOOL_MODULES
    assert ("measurement", "Measurement and classification") in FAMILIES
    assert "reliability" in TOOLS
    spec = TOOLS["reliability"]
    assert spec.family == "measurement"
    assert spec.title == "Reliability (Cronbach's alpha)"

    info = TOOLS.info("reliability")
    assert info.needs_data
    assert [field.name for field in info.fields] == ["items", "names", "reverse", "level"]
    assert [field.kind for field in info.fields] == ["columns", "option", "option", "option"]
    assert [field.label for field in info.fields] == [
        "Items",
        "Item names",
        "Reverse-keyed items",
        "Level",
    ]
    assert all(field.description for field in info.fields)
    assert info.fields[-1].option_schema is not None
    assert "reliability" in {entry.name for entry in catalogue() if entry.family == "measurement"}

    columns = dict(zip("abc", _scale(rows=10, seed=15)[:3], strict=True))

    def group(name: str) -> object:
        raise AssertionError("reliability takes no group column")

    request = build_request(
        "reliability",
        {"items": ["a", "b", "c"], "names": ["a", "b", "c"], "reverse": ["c"], "level": 0.9},
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    result = run_tool("reliability", request)

    assert result.tool == "reliability"
    assert result.level == 0.9
    assert math.isfinite(result.estimates[0].value)
