"""The meta-analysis: the two pools, the heterogeneity numbers and the funnel's asymmetry.

The reference numbers for the pools, tau squared, I squared, Q and its p-value, and every
weight come from ``statsmodels.stats.meta_analysis.combine_effects`` with the DerSimonian
and Laird estimator. The published example is Normand's stroke-care meta-analysis, read from
the metafor analysis page fetched on 2026-11-25:

- https://www.metafor-project.org/doku.php/analyses:normand1999
- Its second dataset is Table VIII (p. 351): nine studies of the length of hospital stay,
  with mean differences -20, -2, -55, -71, -4, 1, 11, -10 and 7 days and sampling variances
  40.59, 2.05, 15.28, 150.22, 20.19, 1.22, 95.38, 8.03 and 20.69. The equal-effects fit on
  the page reports -3.49 (SE 0.78, interval -5.03 to -1.96), Q(8) = 241.06 and I squared
  96.68 %. Its DerSimonian-Laird fit reports tau squared 218.72, -14.10 (SE 5.28, interval
  -24.45 to -3.75) and the same Q and I squared; Table X (p. 352) rounds these to tau
  squared 219 and -14 (-24 to -4). The variances and the reported statistics are rounded to
  two decimals on the page, so the test compares against them with a small absolute
  tolerance and pins the unrounded values against statsmodels separately.

Egger's weighted regression, the prediction interval and the ratio standard errors are
written out independently here from their definitions.
"""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats
from statsmodels.stats.meta_analysis import combine_effects

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    Estimate,
    ExcludedRow,
    Table,
    ToolResult,
    Verdict,
    build_request,
    catalogue,
    run_tool,
)
from curvestudio.tools.stats.meta import MetaRequest, run_meta
from curvestudio.tools.stats.registry import FAMILIES, TOOL_MODULES, TOOLS

STAT_RTOL = 1e-8
"""The tolerance on a number the reference implementation computes from the same data."""

EFFECTS = np.array([0.10, 0.25, -0.05, 0.40, 0.15, 0.30, 0.05])
ERRORS = np.array([0.10, 0.20, 0.05, 0.25, 0.12, 0.18, 0.08])

NORMAND_EFFECTS = np.array([-20.0, -2.0, -55.0, -71.0, -4.0, 1.0, 11.0, -10.0, 7.0])
NORMAND_VARIANCES = np.array([40.59, 2.05, 15.28, 150.22, 20.19, 1.22, 95.38, 8.03, 20.69])

EGGER_EFFECTS = np.array([0.50, 0.42, 0.35, 0.30, 0.28, 0.20, 0.18, 0.15, 0.10, 0.05])
EGGER_ERRORS = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30])


def _column(name: str, values: object) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    return Column(name=name, values=array, rows=np.arange(array.size, dtype=np.intp))


def _group(name: str, labels: object) -> GroupColumn:
    """A group column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(labels, dtype=object)
    return GroupColumn(name=name, labels=array, rows=np.arange(array.size, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_meta(MetaRequest(**options))  # type: ignore[arg-type]


def _plain(effects: object = EFFECTS, errors: object = ERRORS) -> ToolResult:
    return _run(
        effects=(_column("effect", effects),),
        standard_errors=(_column("se", errors),),
    )


def _estimate(result: ToolResult, label: str) -> Estimate:
    return next(found for found in result.estimates if found.label == label)


def _table(result: ToolResult, title: str) -> Table:
    return next(found for found in result.tables if found.title == title)


def _ratio(effect: float, se: float, level: float = 0.95) -> tuple[float, float]:
    """The normal interval of one effect at ``level``."""
    z = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    return effect - z * se, effect + z * se


def _egger_by_hand(effects: np.ndarray, errors: np.ndarray) -> tuple[float, float, float, float]:
    """Egger's intercept, its standard error, its t and its p, from the weighted regression."""
    weights = 1.0 / errors**2
    design = np.column_stack([np.ones(effects.size), errors])
    normal = design.T @ (weights[:, np.newaxis] * design)
    beta = np.linalg.inv(normal) @ (design.T @ (weights * effects))
    residual = effects - design @ beta
    sigma2 = float(np.sum(weights * residual**2)) / float(effects.size - 2)
    intercept_se = math.sqrt(float((sigma2 * np.linalg.inv(normal))[0, 0]))
    statistic = float(beta[0]) / intercept_se
    p_value = float(2.0 * scipy_stats.t.sf(abs(statistic), effects.size - 2))
    return float(beta[0]), intercept_se, statistic, p_value


# --------------------------------------------------------------------------- the reference


def test_the_pools_and_heterogeneity_agree_with_statsmodels() -> None:
    reference = combine_effects(EFFECTS, ERRORS**2, method_re="dl")
    result = _plain()

    fixed = _estimate(result, "Pooled effect (fixed)")
    random = _estimate(result, "Pooled effect (random)")
    assert fixed.value == pytest.approx(float(reference.mean_effect_fe), rel=STAT_RTOL)
    assert fixed.se is not None
    assert fixed.se == pytest.approx(math.sqrt(float(reference.var_eff_w_fe)), rel=STAT_RTOL)
    assert random.value == pytest.approx(float(reference.mean_effect_re), rel=STAT_RTOL)
    assert random.se is not None
    assert random.se == pytest.approx(math.sqrt(float(reference.var_eff_w_re)), rel=STAT_RTOL)
    assert _estimate(result, "Tau squared").value == pytest.approx(
        float(reference.tau2), rel=STAT_RTOL
    )
    assert _estimate(result, "Heterogeneity I squared %").value == pytest.approx(
        float(reference.i2) * 100.0, rel=STAT_RTOL
    )

    heterogeneity = result.tests[0]
    assert heterogeneity.name == "Test of heterogeneity (Q)"
    assert heterogeneity.statistic == pytest.approx(float(reference.q), rel=STAT_RTOL)
    assert heterogeneity.df == float(EFFECTS.size - 1)
    assert heterogeneity.p_value == pytest.approx(
        float(reference.test_homogeneity().pvalue), rel=STAT_RTOL
    )


def test_the_study_weights_agree_with_statsmodels() -> None:
    reference = combine_effects(EFFECTS, ERRORS**2, method_re="dl")
    table = _table(_plain(), "Studies")

    expected_fixed = reference.weights_fe / np.sum(reference.weights_fe) * 100.0
    expected_random = reference.weights_re / np.sum(reference.weights_re) * 100.0
    assert [row[0] for row in table.rows] == [f"Study {index + 1}" for index in range(7)]
    for index, row in enumerate(table.rows):
        assert row[1] == pytest.approx(float(EFFECTS[index]), rel=STAT_RTOL)
        assert row[2] == pytest.approx(float(ERRORS[index]), rel=STAT_RTOL)
        assert row[3] == pytest.approx(float(expected_fixed[index]), rel=STAT_RTOL)
        assert row[4] == pytest.approx(float(expected_random[index]), rel=STAT_RTOL)


def test_the_pools_match_a_second_reference_fixture() -> None:
    effects = np.array([-0.30, 0.10, 0.45, 0.05, 0.60, -0.15])
    errors = np.array([0.30, 0.15, 0.20, 0.10, 0.25, 0.40])
    reference = combine_effects(effects, errors**2, method_re="dl")

    result = _plain(effects, errors)

    assert _estimate(result, "Pooled effect (fixed)").value == pytest.approx(
        float(reference.mean_effect_fe), rel=STAT_RTOL
    )
    assert _estimate(result, "Pooled effect (random)").value == pytest.approx(
        float(reference.mean_effect_re), rel=STAT_RTOL
    )
    assert _estimate(result, "Tau squared").value == pytest.approx(
        float(reference.tau2), rel=STAT_RTOL
    )
    assert float(reference.tau2) > 0.0


# --------------------------------------------------------------------------- the publication


def test_the_published_normand_example_reproduces_the_der_simonian_laird_pool() -> None:
    errors = np.sqrt(NORMAND_VARIANCES)
    result = _plain(NORMAND_EFFECTS, errors)

    assert result.counts == Counts(n_offered=9, n_used=9, n_excluded=0)
    fixed = _estimate(result, "Pooled effect (fixed)")
    random = _estimate(result, "Pooled effect (random)")
    assert fixed.value == pytest.approx(-3.49, abs=0.02)
    assert fixed.lower == pytest.approx(-5.03, abs=0.03)
    assert fixed.upper == pytest.approx(-1.96, abs=0.02)
    assert random.value == pytest.approx(-14.10, abs=0.02)
    assert random.se == pytest.approx(5.28, abs=0.02)
    assert random.lower == pytest.approx(-24.45, abs=0.03)
    assert random.upper == pytest.approx(-3.75, abs=0.03)
    assert _estimate(result, "Tau squared").value == pytest.approx(218.72, abs=0.05)
    assert _estimate(result, "Heterogeneity I squared %").value == pytest.approx(96.68, abs=0.02)

    heterogeneity = result.tests[0]
    assert heterogeneity.statistic == pytest.approx(241.06, abs=0.10)
    assert heterogeneity.df == 8.0
    assert heterogeneity.p_value is not None and heterogeneity.p_value < 1e-30


# --------------------------------------------------------------------------- the properties


def test_the_two_pools_coincide_when_tau_squared_is_zero() -> None:
    effects = np.array([0.10, 0.12, 0.09, 0.11, 0.10])
    errors = np.array([0.10, 0.10, 0.10, 0.10, 0.10])

    result = _plain(effects, errors)

    fixed = _estimate(result, "Pooled effect (fixed)")
    random = _estimate(result, "Pooled effect (random)")
    assert _estimate(result, "Tau squared").value == 0.0
    assert _estimate(result, "Heterogeneity I squared %").value == 0.0
    assert random.value == pytest.approx(fixed.value, rel=STAT_RTOL)
    assert random.lower == pytest.approx(fixed.lower, rel=STAT_RTOL)
    assert random.upper == pytest.approx(fixed.upper, rel=STAT_RTOL)


def test_a_study_with_a_huge_standard_error_moves_the_pool_negligibly() -> None:
    """A hundred and fifty balanced studies make both pools sit at zero to rounding; the
    added study's weight is essentially zero, and any move it makes is far below 1e-9. (The
    DerSimonian-Laird variance still gains one degree of freedom, which is why the random
    pool is not required to be unchanged to the bit.)"""
    size = 150
    effects = np.where(np.arange(size) % 2 == 0, 0.5, -0.5) + np.linspace(-0.2, 0.2, size)
    errors = np.full(size, 0.05)

    base = _plain(effects, errors)
    grown = _plain(np.append(effects, 100.0), np.append(errors, 1e6))

    assert (
        abs(
            _estimate(grown, "Pooled effect (fixed)").value
            - _estimate(base, "Pooled effect (fixed)").value
        )
        < 1e-9
    )
    assert (
        abs(
            _estimate(grown, "Pooled effect (random)").value
            - _estimate(base, "Pooled effect (random)").value
        )
        < 1e-9
    )
    last = _table(grown, "Studies").rows[-1]
    assert last[3] < 1e-6
    assert last[4] < 1e-6


def test_a_study_equal_to_the_pool_leaves_the_fixed_pool_unchanged() -> None:
    base = _plain()
    pooled = _estimate(base, "Pooled effect (fixed)").value
    effects = np.append(EFFECTS, pooled)
    errors = np.append(ERRORS, 0.5)

    grown = _plain(effects, errors)

    assert _estimate(grown, "Pooled effect (fixed)").value == pytest.approx(pooled, rel=1e-12)


# --------------------------------------------------------------------------- the blocks


def test_the_pooled_table_reports_both_models_and_the_prediction_interval() -> None:
    result = _plain()
    table = _table(result, "Pooled")

    assert [column.label for column in table.columns] == [
        "Model",
        "Estimate",
        "SE",
        "Lower",
        "Upper",
        "Tau squared",
    ]
    assert [row[0] for row in table.rows] == [
        "Fixed effect",
        "Random effects (DerSimonian-Laird)",
        "Prediction interval (random)",
    ]
    assert table.rows[0][5] is None
    assert table.note is not None and "prediction interval" in table.note


def test_the_prediction_interval_is_a_t_interval_from_three_studies() -> None:
    effects = np.array([0.10, 0.20, 0.30])
    errors = np.array([0.10, 0.20, 0.30])
    result = _plain(effects, errors)

    random = _estimate(result, "Pooled effect (random)")
    tau2 = float(_estimate(result, "Tau squared").value)
    spread = math.sqrt(tau2 + float(random.se) ** 2)
    t = float(scipy_stats.t.ppf(0.975, 1))
    table = _table(result, "Pooled")
    prediction = table.rows[2]

    assert prediction[0] == "Prediction interval (random)"
    assert prediction[3] == pytest.approx(random.value - t * spread, rel=STAT_RTOL)
    assert prediction[4] == pytest.approx(random.value + t * spread, rel=STAT_RTOL)


def test_two_studies_report_the_prediction_interval_as_not_applicable() -> None:
    result = _plain(np.array([0.0, 1.0]), np.array([0.1, 0.1]))

    pooled = _table(result, "Pooled")

    assert len(pooled.rows) == 2
    assert pooled.note is not None and "not applicable" in pooled.note


def test_the_forest_plot_draws_one_line_per_study_and_the_pool() -> None:
    result = _plain()
    forest = result.plots[0]

    assert forest.kind == "lines"
    assert forest.x_label == "Effect"
    assert len(forest.series) == EFFECTS.size + 1
    for index, series in enumerate(forest.series[:-1]):
        lower, upper = _ratio(float(EFFECTS[index]), float(ERRORS[index]))
        assert series.label == f"Study {index + 1}"
        assert series.x == (pytest.approx(lower), pytest.approx(upper))
        assert series.y == (float(index), float(index))
    pooled = forest.series[-1]
    assert pooled.label == "Pooled (random)"
    assert pooled.y == (float(EFFECTS.size), float(EFFECTS.size))
    assert forest.note is not None and "diamond is drawn as a line" in forest.note


def test_the_funnel_plot_arrives_from_three_studies_with_an_inverted_axis() -> None:
    result = _plain()

    assert [plot.kind for plot in result.plots] == ["lines", "scatter"]
    funnel = result.plots[1]
    assert funnel.kind == "scatter"
    assert "inverted" in funnel.y_label
    assert funnel.x == tuple(float(value) for value in EFFECTS)
    assert funnel.y == tuple(float(value) for value in ERRORS)
    assert len(funnel.lines) == 1
    assert funnel.lines[0].x0 == pytest.approx(_estimate(result, "Pooled effect (random)").value)


def test_two_studies_have_no_funnel_plot() -> None:
    result = _plain(np.array([0.0, 1.0]), np.array([0.1, 0.1]))

    assert [plot.kind for plot in result.plots] == ["lines"]


def test_eggers_test_runs_from_ten_studies() -> None:
    intercept, intercept_se, statistic, p_value = _egger_by_hand(EGGER_EFFECTS, EGGER_ERRORS)

    result = _plain(EGGER_EFFECTS, EGGER_ERRORS)
    egger = result.tests[1]

    assert egger.name == "Egger's test of funnel asymmetry"
    assert egger.df == 8.0
    assert egger.statistic == pytest.approx(statistic, rel=STAT_RTOL)
    assert egger.p_value == pytest.approx(p_value, rel=STAT_RTOL)
    assert egger.effect is not None
    assert egger.effect.label == "Egger intercept"
    assert egger.effect.value == pytest.approx(intercept, rel=STAT_RTOL)
    assert egger.effect.se == pytest.approx(intercept_se, rel=STAT_RTOL)
    assert p_value < 0.05


def test_below_ten_studies_eggers_test_is_an_assumption_row() -> None:
    result = _plain()

    assert len(result.tests) == 1
    egger = next(check for check in result.assumptions if check.id == "egger_min_studies")
    assert egger.verdict is Verdict.NOT_APPLICABLE
    assert "10" in egger.title
    assert str(EFFECTS.size) in egger.evidence


def test_egger_is_empty_when_every_standard_error_is_equal() -> None:
    effects = np.linspace(0.0, 1.0, 10)
    errors = np.full(10, 0.1)

    result = _plain(effects, errors)
    egger = result.tests[1]

    assert egger.statistic is None
    assert egger.p_value is None
    assert egger.reason is not None and "without a slope" in egger.reason


def test_the_warnings_name_few_studies_zero_tau_and_high_heterogeneity() -> None:
    few = _plain(np.array([0.0, 0.5, 1.0, 0.2]), np.array([0.1, 0.2, 0.15, 0.1]))
    assert any("fewer than 5" in warning for warning in few.warnings)

    homogeneous = _plain(np.array([0.10, 0.12, 0.09, 0.11, 0.10]), np.full(5, 0.1))
    assert any("Tau squared is estimated at zero" in warning for warning in homogeneous.warnings)

    high = _plain(NORMAND_EFFECTS, np.sqrt(NORMAND_VARIANCES))
    assert any("above 75" in warning for warning in high.warnings)


def test_the_interpretation_gives_both_pools_and_i_squared_in_words() -> None:
    result = _plain()

    assert "fixed-effect pool" in result.interpretation
    assert "random-effects pool" in result.interpretation
    assert "I squared" in result.interpretation
    assert "heterogeneity" in result.interpretation.lower()
    assert "38.2" in result.interpretation


def test_the_methods_fragment_names_the_pooling_the_estimator_and_the_measure() -> None:
    result = _plain()

    for phrase in (
        "inverse-variance",
        "DerSimonian-Laird",
        "chi-square",
        "prediction interval",
        "the scale they were given",
    ):
        assert phrase in result.methods_fragment, phrase


def test_the_words_claim_no_proof() -> None:
    result = _plain()

    texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
    for text in texts:
        assert not claims_proof(text), text


def test_the_assumptions_state_independence_and_publication_bias() -> None:
    result = _plain(EGGER_EFFECTS, EGGER_ERRORS)

    assert [check.id for check in result.assumptions] == ["independence", "publication_bias"]
    assert all(check.verdict is Verdict.NOT_APPLICABLE for check in result.assumptions)
    assert "independence" in result.assumptions[0].title.lower()
    assert "publication bias" in result.assumptions[1].title.lower()


# --------------------------------------------------------------------------- the rows


def test_a_row_missing_a_value_is_left_out_and_listed() -> None:
    effects = np.array([0.10, 0.25, np.nan, 0.40, 0.15])

    result = _plain(effects, np.array([0.10, 0.20, 0.05, 0.25, 0.12]))

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="missing in effect"),)


def test_a_shorter_standard_error_column_is_aligned_and_listed() -> None:
    effects = _column("effect", [0.10, 0.25, 0.15, 0.30, 0.20])
    errors = _column("se", [0.10, 0.20, 0.15, 0.30])

    result = _run(effects=(effects,), standard_errors=(errors,))

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=4, reason="missing in se"),)


def test_labels_name_the_studies_and_align_by_position() -> None:
    effects = _column("effect", [0.10, 0.25, 0.15, 0.30, 0.20])
    errors = _column("se", [0.10, 0.20, 0.15, 0.30, 0.20])
    labels = _group("trial", ["A", "B", "C", "D"])

    result = _run(effects=(effects,), standard_errors=(errors,), labels=labels)

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=4, reason="no group label in trial"),)
    assert [row[0] for row in _table(result, "Studies").rows] == ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- the ratios


def test_the_odds_ratio_is_analysed_as_its_logarithm() -> None:
    events_a = np.array([10.0, 20.0])
    n_a = np.array([100.0, 200.0])
    events_b = np.array([20.0, 30.0])
    n_b = np.array([100.0, 200.0])

    result = _run(
        measure="odds_ratio",
        events_a=_column("events_a", events_a),
        n_a=_column("n_a", n_a),
        events_b=_column("events_b", events_b),
        n_b=_column("n_b", n_b),
    )

    logs = np.log(events_a * (n_b - events_b) / ((n_a - events_a) * events_b))
    errors = np.sqrt(
        1.0 / events_a + 1.0 / (n_a - events_a) + 1.0 / events_b + 1.0 / (n_b - events_b)
    )
    weights = 1.0 / errors**2
    pooled = float(np.sum(weights * logs) / np.sum(weights))

    assert _table(result, "Studies").rows[0][1] == pytest.approx(float(logs[0]), rel=STAT_RTOL)
    assert _table(result, "Studies").rows[0][2] == pytest.approx(float(errors[0]), rel=STAT_RTOL)
    assert _estimate(result, "Pooled effect (fixed)").value == pytest.approx(pooled, rel=STAT_RTOL)
    assert result.plots[0].x_label == "Log odds ratio"
    assert "natural logarithm" in result.methods_fragment
    assert "odds ratio of" in result.interpretation


def test_the_risk_ratio_is_analysed_as_its_logarithm() -> None:
    events_a = np.array([10.0, 20.0])
    n_a = np.array([100.0, 200.0])
    events_b = np.array([20.0, 30.0])
    n_b = np.array([100.0, 200.0])

    result = _run(
        measure="risk_ratio",
        events_a=_column("events_a", events_a),
        n_a=_column("n_a", n_a),
        events_b=_column("events_b", events_b),
        n_b=_column("n_b", n_b),
    )

    logs = np.log((events_a / n_a) / (events_b / n_b))
    errors = np.sqrt(1.0 / events_a - 1.0 / n_a + 1.0 / events_b - 1.0 / n_b)
    weights = 1.0 / errors**2
    pooled = float(np.sum(weights * logs) / np.sum(weights))

    assert _table(result, "Studies").rows[1][1] == pytest.approx(float(logs[1]), rel=STAT_RTOL)
    assert _table(result, "Studies").rows[1][2] == pytest.approx(float(errors[1]), rel=STAT_RTOL)
    assert _estimate(result, "Pooled effect (fixed)").value == pytest.approx(pooled, rel=STAT_RTOL)
    assert result.plots[0].x_label == "Log risk ratio"


@pytest.mark.parametrize("events", [0.0, 100.0])
def test_a_zero_or_complete_cell_is_refused_naming_the_study(events: float) -> None:
    labels = _group("trial", ["Trial A", "Trial B"])

    with pytest.raises(ToolError) as refused:
        _run(
            measure="odds_ratio",
            events_a=_column("events_a", [10.0, events]),
            n_a=_column("n_a", [100.0, 100.0]),
            events_b=_column("events_b", [20.0, 20.0]),
            n_b=_column("n_b", [100.0, 100.0]),
            labels=labels,
        )

    assert refused.value.code == "INVALID_OPTION"
    assert "Trial B" in refused.value.message
    assert "refused rather than corrected" in refused.value.message


def test_a_ratio_measure_refuses_effect_columns() -> None:
    with pytest.raises(ToolError) as refused:
        _run(
            measure="odds_ratio",
            effects=(_column("effect", [0.1, 0.2]),),
            standard_errors=(_column("se", [0.1, 0.2]),),
            events_a=_column("events_a", [10.0, 20.0]),
            n_a=_column("n_a", [100.0, 100.0]),
            events_b=_column("events_b", [20.0, 20.0]),
            n_b=_column("n_b", [100.0, 100.0]),
        )
    assert refused.value.code == "INVALID_OPTION"
    assert "leave effects" in refused.value.message


def test_an_effect_measure_refuses_ratio_columns() -> None:
    with pytest.raises(ToolError) as refused:
        _run(
            effects=(_column("effect", [0.1, 0.2]),),
            standard_errors=(_column("se", [0.1, 0.2]),),
            events_a=_column("events_a", [10.0, 20.0]),
        )
    assert refused.value.code == "INVALID_OPTION"
    assert "leave the event" in refused.value.message


# --------------------------------------------------------------------------- the refusals


def test_one_study_is_too_few() -> None:
    with pytest.raises(ToolError) as refused:
        _plain(np.array([0.5]), np.array([0.1]))

    assert refused.value.code == "TOO_FEW_ROWS"
    assert "at least 2" in refused.value.message


def test_more_than_two_hundred_studies_are_refused() -> None:
    effects = np.linspace(0.0, 1.0, 201)
    errors = np.full(201, 0.1)

    with pytest.raises(ToolError) as refused:
        _plain(effects, errors)

    assert refused.value.code == "INVALID_OPTION"
    assert "between 2 and 200" in refused.value.message


@pytest.mark.parametrize("error", [0.0, -0.1])
def test_a_zero_or_negative_standard_error_is_refused_naming_the_study(error: float) -> None:
    labels = _group("trial", ["Trial A", "Trial B"])

    with pytest.raises(ToolError) as refused:
        _run(
            effects=(_column("effect", [0.1, 0.2]),),
            standard_errors=(_column("se", [0.1, error]),),
            labels=labels,
        )

    assert refused.value.code == "INVALID_OPTION"
    assert "Trial B" in refused.value.message
    assert "positive standard error" in refused.value.message


def test_a_ratio_measure_needs_all_four_columns() -> None:
    with pytest.raises(ToolError) as refused:
        _run(
            measure="odds_ratio",
            events_a=_column("events_a", [10.0, 20.0]),
            n_a=_column("n_a", [100.0, 100.0]),
        )

    assert refused.value.code == "INVALID_OPTION"
    assert "events_b" in refused.value.message
    assert "n_b" in refused.value.message


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    with pytest.raises(ToolError) as refused:
        _run(
            effects=(_column("effect", EFFECTS),),
            standard_errors=(_column("se", ERRORS),),
            level=level,
        )

    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_meta(object())  # type: ignore[arg-type]

    assert refused.value.code == "INVALID_OPTION"


# --------------------------------------------------------------------------- the contract


def test_the_result_is_the_contract() -> None:
    result = _plain()

    assert type(result) is ToolResult
    assert result.tool == "meta"
    assert result.title == "Meta-analysis"
    assert result.level == 0.95
    assert [estimate.label for estimate in result.estimates] == [
        "Pooled effect (fixed)",
        "Pooled effect (random)",
        "Heterogeneity I squared %",
        "Tau squared",
    ]
    assert [table.title for table in result.tables] == ["Studies", "Pooled"]
    assert [plot.kind for plot in result.plots] == ["lines", "scatter"]
    assert result.counts == Counts(n_offered=7, n_used=7, n_excluded=0)


def test_the_result_survives_a_json_round_trip_without_a_nan() -> None:
    result = _plain()
    payload = result.model_dump_json()

    assert ToolResult.model_validate_json(payload) == result
    assert "NaN" not in payload


def test_a_ratio_result_survives_a_json_round_trip() -> None:
    result = _run(
        measure="risk_ratio",
        events_a=_column("events_a", [10.0, 20.0, 15.0]),
        n_a=_column("n_a", [100.0, 200.0, 120.0]),
        events_b=_column("events_b", [20.0, 30.0, 18.0]),
        n_b=_column("n_b", [100.0, 200.0, 140.0]),
    )
    payload = result.model_dump_json()

    assert ToolResult.model_validate_json(payload) == result
    assert "NaN" not in payload


# --------------------------------------------------------------------------- registration


def test_the_registry_lists_and_runs_the_tool() -> None:
    assert "curvestudio.tools.stats.meta" in TOOL_MODULES
    assert ("evidence", "Evidence synthesis") in FAMILIES
    assert "meta" in TOOLS
    spec = TOOLS["meta"]
    assert spec.family == "evidence"
    assert spec.title == "Meta-analysis"

    info = TOOLS.info("meta")
    assert info.needs_data
    assert [field.name for field in info.fields] == [
        "effects",
        "standard_errors",
        "labels",
        "measure",
        "level",
        "events_a",
        "n_a",
        "events_b",
        "n_b",
    ]
    assert [field.kind for field in info.fields] == [
        "columns",
        "columns",
        "group",
        "option",
        "option",
        "column",
        "column",
        "column",
        "column",
    ]
    assert [field.label for field in info.fields] == [
        "Effects",
        "Standard errors",
        "Study labels",
        "Measure",
        "Level",
        "Events (group A)",
        "Total (group A)",
        "Events (group B)",
        "Total (group B)",
    ]
    assert all(field.description for field in info.fields)
    assert info.fields[3].option_schema is not None
    assert "meta" in {entry.name for entry in catalogue() if entry.family == "evidence"}

    columns = {
        "effect": _column("effect", [0.1, 0.2, 0.3]),
        "se": _column("se", [0.1, 0.2, 0.3]),
    }
    groups = {"trial": _group("trial", ["A", "B", "C"])}

    request = build_request(
        "meta",
        {
            "effects": ["effect"],
            "standard_errors": ["se"],
            "labels": "trial",
            "measure": "generic",
            "level": 0.9,
        },
        column=columns.__getitem__,
        group=groups.__getitem__,
    )
    result = run_tool("meta", request)

    assert result.tool == "meta"
    assert result.level == 0.9
    assert math.isfinite(result.estimates[0].value)
    assert [row[0] for row in _table(result, "Studies").rows] == ["A", "B", "C"]
