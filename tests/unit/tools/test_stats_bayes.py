"""Bayesian versions of the Stage 1 tests: the JZS Bayes factor, the posterior and the ROPE.

The one-sample and two-sample JZS numbers are held to Rouder, Speckman, Sun, Morey and
Iverson (2009), *Psychonomic Bulletin & Review* 16(2), 225-237, fetched and read on
2026-11-25 from the publisher's PDF (link.springer.com/content/pdf/10.3758/PBR.16.2.225.pdf)
through the Jina reader, which renders the page images' text. The paper tabulates one table,
its Table 1 on page 233, of the critical t values at which the JZS Bayes factor reaches the
odds 10, 3, 1/3 and 1/10 for each sample size. Its row for N = 100 prints 0.69 for odds 10
on the null's side, 1.72 for odds 3 on the null's side, 2.76 for odds 3 on the
alternative's side, and 3.20 for odds 10 on the alternative's side; those four values are
checked below. The paper's worked example on page 232, Grider and Malmberg's Experiment 3
with t(79) = 2.03 and N = 80, prints B01 = 1.56 (and t(79) = 2.24 prints B01 = 1.02); both
are checked exactly. The paper has no two-sample table: page 234 gives the two-sample case
in prose, as Equation 1 with the effective sample size n1 n2 / (n1 + n2) and n1 + n2 - 2
degrees of freedom, and the test below holds that form to an independent quadrature over the
standardised effect, which is the same model written in the other parameterisation.

The proportion design is held to the Savage-Dickey ratio's closed form written out in the
test with gamma functions, independent of SciPy's beta distribution.
"""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import Counts, ToolResult
from curvestudio.tools.stats.bayes import (
    BayesRequest,
    _effect_posterior,
    jzs_bayes_factor,
    run_bayes,
)
from curvestudio.tools.stats.means import TTestRequest
from curvestudio.tools.stats.registry import TOOL_MODULES, TOOLS, catalogue

REL = 0.01
"""The tolerance on the paper's rounded values and the closed-form references."""


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _group(name: str, labels: object, rows: object | None = None) -> GroupColumn:
    """A group column whose labels are the data layer's object array of strings."""
    array = np.asarray(labels, dtype=object)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return GroupColumn(name=name, labels=array, rows=np.asarray(positions, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_bayes(BayesRequest(**options))  # type: ignore[arg-type]


def _one(values: object, **options: object) -> ToolResult:
    return _run(design="one_sample", a=_column("y", values), **options)


def _proportion(successes: int, trials: int, **options: object) -> ToolResult:
    return _run(design="proportion", successes=successes, trials=trials, null=0.5, **options)


def _bayes_factor(result: ToolResult) -> float:
    test = result.tests[0]
    assert test.effect is not None
    return test.effect.value


def _estimate(result: ToolResult, label: str) -> float:
    return next(estimate.value for estimate in result.estimates if estimate.label == label)


def _rows(result: ToolResult, title: str) -> dict[str, object]:
    """The rows of a table by its first cell, for the two-column summary."""
    found = next(table for table in result.tables if table.title == title)
    return {str(row[0]): row[1] for row in found.rows}


def _paired_data() -> tuple[Column, Column]:
    return (
        _column("before", [1.0, 2.0, 4.0, 8.0, 9.0]),
        _column("after", [0.0, 1.0, 2.0, 3.0, 5.0]),
    )


def _two_sample_data() -> tuple[Column, Column]:
    return (
        _column("first", np.linspace(0.0, 1.0, 25)),
        _column("second", np.linspace(1.2, 2.8, 35)),
    )


def _effect_at(t: float, n: int) -> Column:
    """A column whose one-sample t is close to ``t``: a straight line with a shifted mean."""
    values = np.linspace(-math.sqrt(3.0), math.sqrt(3.0), n) + t / math.sqrt(n)
    return _column("y", values)


def _posterior_oracle(t: float, n_eff: float, df: float, scale: float) -> float:
    """The JZS Bayes factor from the posterior's normalisation over the standardised effect.

    The alternative's marginal likelihood is the non-central t likelihood of ``t`` at
    non-centrality ``delta * sqrt(n_eff)``, averaged over the Cauchy prior; the null's is the
    central t density at ``t``. This is Equation 1 of Rouder et al. written in delta instead
    of g, so it is an independent check of the g integral, not a copy of it.
    """
    null = float(stats.t.pdf(t, df))
    alternative = integrate.quad(
        lambda delta: (
            float(stats.nct.pdf(t, df, delta * math.sqrt(n_eff)))
            * float(stats.cauchy.pdf(delta, loc=0.0, scale=scale))
        ),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=500,
    )[0]
    return alternative / null


def _savage_dickey(successes: int, trials: int, null: float, alpha: float, beta: float) -> float:
    """The Savage-Dickey ratio in closed form, from gamma functions, for a beta prior."""
    failures = trials - successes
    post_alpha = alpha + successes
    post_beta = beta + failures

    def log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    log_prior = (
        (alpha - 1.0) * math.log(null) + (beta - 1.0) * math.log(1.0 - null) - log_beta(alpha, beta)
    )
    log_posterior = (
        (post_alpha - 1.0) * math.log(null)
        + (post_beta - 1.0) * math.log(1.0 - null)
        - log_beta(post_alpha, post_beta)
    )
    return math.exp(log_prior - log_posterior)


# --- The published values -------------------------------------------------------------


def test_the_jzs_bayes_factor_reproduces_the_papers_worked_examples() -> None:
    """The paper's page 232 example: t(79) = 2.03 and N = 80 give B01 = 1.56."""
    assert 1.0 / jzs_bayes_factor(2.03, 80.0, 79.0, 1.0) == pytest.approx(1.56, rel=REL)
    assert 1.0 / jzs_bayes_factor(2.24, 80.0, 79.0, 1.0) == pytest.approx(1.02, rel=REL)


def test_the_jzs_bayes_factor_reproduces_the_papers_table_1_row_for_100() -> None:
    """Table 1, page 233, the N = 100 row, all four printed critical t values."""
    assert 1.0 / jzs_bayes_factor(0.69, 100.0, 99.0, 1.0) == pytest.approx(10.0, rel=0.02)
    assert 1.0 / jzs_bayes_factor(1.72, 100.0, 99.0, 1.0) == pytest.approx(3.0, rel=0.02)
    assert jzs_bayes_factor(2.76, 100.0, 99.0, 1.0) == pytest.approx(3.0, rel=0.02)
    assert jzs_bayes_factor(3.20, 100.0, 99.0, 1.0) == pytest.approx(10.0, rel=0.02)


def test_the_two_sample_factor_uses_the_papers_effective_sample_size() -> None:
    """Page 234: n1 n2 / (n1 + n2) and n1 + n2 - 2 df, checked in the delta parameterisation."""
    n_effect, df = 25.0 * 35.0 / 60.0, 58.0
    for t in (0.0, 1.5, 2.5, 4.0):
        expected = _posterior_oracle(t, n_effect, df, 0.707)
        assert jzs_bayes_factor(t, n_effect, df, 0.707) == pytest.approx(expected, rel=1e-6)


def test_the_two_sample_tool_agrees_with_the_delta_form_of_the_same_model() -> None:
    first, second = _two_sample_data()
    result = _run(design="independent", a=first, b=second)
    test = result.tests[0]
    assert test.statistic is not None and test.df is not None
    expected = _posterior_oracle(test.statistic, 25.0 * 35.0 / 60.0, test.df, 0.707)
    assert _bayes_factor(result) == pytest.approx(expected, rel=1e-6)


def test_the_savage_dickey_factor_matches_its_closed_form() -> None:
    options = {"prior_alpha": 2.0, "prior_beta": 3.0}
    result = _run(design="proportion", successes=3, trials=10, null=0.4, **options)
    expected = _savage_dickey(3, 10, 0.4, 2.0, 3.0)
    assert _bayes_factor(result) == pytest.approx(expected, rel=1e-12)


# --- Properties -----------------------------------------------------------------------


def test_a_very_large_t_and_a_t_of_zero_fall_on_opposite_sides_of_one() -> None:
    zero = _one([-2.0, -1.0, 0.0, 1.0, 2.0])
    large = _one(np.linspace(9.0, 11.0, 20))
    assert _bayes_factor(zero) < 1.0
    assert _bayes_factor(large) > 1.0


def test_adding_a_constant_to_the_data_leaves_the_one_sample_test_unchanged() -> None:
    values = [0.2, 0.8, 0.4, 1.1, 0.6, 0.9]
    plain = _one(values, null=0.0)
    shifted = _one(np.asarray(values) + 5.0, null=5.0)
    assert _bayes_factor(shifted) == pytest.approx(_bayes_factor(plain), rel=1e-12)
    assert _estimate(shifted, "Standardised effect delta (posterior mean; credible interval)") == (
        pytest.approx(
            _estimate(plain, "Standardised effect delta (posterior mean; credible interval)"),
            rel=1e-12,
        )
    )


def test_adding_the_same_constant_to_both_columns_leaves_the_paired_test_unchanged() -> None:
    left, right = _paired_data()
    plain = _run(design="paired", a=left, b=right)
    shifted = _run(
        design="paired",
        a=_column("before", left.values + 3.0),
        b=_column("after", right.values + 3.0),
    )
    assert _bayes_factor(shifted) == pytest.approx(_bayes_factor(plain), rel=1e-12)
    assert shifted.estimates[0].lower == pytest.approx(plain.estimates[0].lower, rel=1e-12)


def _width(result: ToolResult) -> float:
    """The width of the first estimate's interval."""
    estimate = result.estimates[0]
    assert estimate.lower is not None and estimate.upper is not None
    return estimate.upper - estimate.lower


def test_the_credible_interval_narrows_as_the_sample_grows() -> None:
    small = _one(_effect_at(2.0, 20).values)
    large = _one(_effect_at(2.0, 200).values)
    assert _width(large) < _width(small)


def test_the_posterior_grid_is_positive_and_integrates_to_one() -> None:
    posterior = _effect_posterior(2.0, 50.0, 49.0, 0.707, 0.95, (-0.1, 0.1))
    assert np.all(posterior.density > 0.0)
    assert float(np.trapezoid(posterior.density, posterior.grid)) == pytest.approx(1.0, rel=1e-9)
    assert posterior.lower < posterior.mean < posterior.upper


def test_the_plotted_density_stays_normalised_under_the_stride() -> None:
    result = _one(_effect_at(2.0, 50).values)
    plot = result.plots[0]
    series = plot.series[0]
    assert plot.title == "Posterior density"
    assert len(series.x) <= 200 and len(series.x) == len(series.y)
    assert float(np.trapezoid(series.y, series.x)) == pytest.approx(1.0, rel=1e-3)
    assert plot.note is not None and "200 of 4,001" in plot.note
    assert plot.x_labels is not None and len(plot.x_labels) == len(series.x)


def test_the_posterior_interval_contains_the_posterior_mean_and_the_rope_share_is_a_share() -> None:
    result = _one([0.1, 0.5, 0.2, 0.4, 0.8, 0.6, 0.3, 0.7])
    effect = result.estimates[0]
    assert effect.lower is not None and effect.upper is not None
    assert effect.lower < effect.value < effect.upper
    rope = _estimate(result, "Posterior probability of the null region")
    assert 0.0 <= rope <= 1.0


# --- The result contract ---------------------------------------------------------------


def test_the_one_sample_result_states_the_bayes_factor_and_never_a_p_value() -> None:
    result = _one([0.1, 0.5, 0.2, 0.4, 0.8, 0.6, 0.3, 0.7])
    assert type(result) is ToolResult
    test = result.tests[0]
    assert test.name == "Bayes factor (JZS)"
    assert test.p_value is None and test.p_value_text == "not computed"
    assert test.reason is not None and "not a p-value" in test.reason
    assert "JZS" in test.method
    assert result.assumptions[0].verdict.value == "pass"
    assert "0.707" in result.assumptions[0].evidence
    assert result.counts == Counts(n_offered=8, n_used=8, n_excluded=0)


def test_the_result_carries_the_summary_table_and_the_credible_labels() -> None:
    result = _one([0.1, 0.5, 0.2, 0.4, 0.8, 0.6, 0.3, 0.7])
    rows = _rows(result, "Posterior summary")
    assert list(rows) == [
        "Posterior mean",
        "Posterior standard deviation",
        "Credible lower",
        "Credible upper",
        "Bayes factor (BF10)",
        "ROPE probability",
    ]
    assert rows["Bayes factor (BF10)"] == pytest.approx(_bayes_factor(result), rel=1e-12)
    assert result.estimates[0].label.endswith("(posterior mean; credible interval)")
    assert "credible interval" in result.interpretation
    assert "not a demonstration" in result.interpretation


def test_the_words_name_the_prior_the_bands_and_the_integration_tolerance() -> None:
    result = _one([0.1, 0.5, 0.2, 0.4, 0.8, 0.6, 0.3, 0.7])
    assert "3, 10 and 30" in result.interpretation
    assert "Cauchy" in result.methods_fragment
    assert "0.707" in result.methods_fragment
    assert "1e-10" in result.methods_fragment
    assert "null region" in result.methods_fragment
    for text in (result.interpretation, result.methods_fragment, result.title):
        assert not claims_proof(text), text


def test_the_result_round_trips_through_json_without_a_nan() -> None:
    result = _run(design="proportion", successes=3, trials=10, null=0.5)
    text = result.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    assert ToolResult.model_validate_json(text) == result


# --- The designs -----------------------------------------------------------------------


def test_the_paired_design_uses_complete_pairs_and_the_differences() -> None:
    left, right = _paired_data()
    result = _run(design="paired", a=left, b=right)
    assert result.title == "Bayesian paired test of before and after"
    assert result.counts == Counts(n_offered=5, n_used=5, n_excluded=0)


def test_the_independent_design_by_groups_reports_the_pooled_two_sample_factor() -> None:
    first, second = _two_sample_data()
    values = np.concatenate([first.values, second.values])
    labels = ["first"] * first.values.size + ["second"] * second.values.size
    by_groups = _run(
        design="independent",
        a=_column("y", values),
        groups=_group("g", labels),
    )
    by_columns = _run(design="independent", a=first, b=second)
    assert by_groups.title == "Bayesian two-sample test of y by g"
    assert _bayes_factor(by_groups) == pytest.approx(_bayes_factor(by_columns), rel=1e-12)
    assert by_groups.counts == Counts(n_offered=60, n_used=60, n_excluded=0)


def test_a_missing_value_is_excluded_and_listed() -> None:
    result = _one([1.0, 2.0, np.nan, 4.0, 5.0])
    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded[0].index == 2
    assert result.excluded[0].reason == "missing in y"
    assert result.excluded_unlisted == 0


def test_the_proportion_result_states_the_beta_prior_and_the_trial_counts() -> None:
    result = _proportion(3, 10)
    assert result.title == "Bayesian test of a proportion: 3 of 10"
    assert result.counts == Counts(n_offered=0, n_used=0, n_excluded=0)
    assert result.assumptions[0].evidence == "Beta(1, 1) prior on the proportion."
    rows = _rows(result, "Posterior summary")
    assert rows["Successes"] == 3 and rows["Trials"] == 10
    assert result.tests[0].name.startswith("Bayes factor (Savage")
    assert result.tests[0].p_value_text == "not computed"
    assert _estimate(result, "Posterior probability that p exceeds the null") == pytest.approx(
        float(stats.beta(4, 8).sf(0.5)), rel=1e-12
    )


def test_the_proportion_interval_is_the_beta_quantile_interval() -> None:
    result = _proportion(3, 10, level=0.9)
    estimate = result.estimates[0]
    assert estimate.lower == pytest.approx(float(stats.beta(4, 8).ppf(0.05)), rel=1e-12)
    assert estimate.upper == pytest.approx(float(stats.beta(4, 8).ppf(0.95)), rel=1e-12)


# --- Refusals --------------------------------------------------------------------------


def test_one_value_is_too_few_rows() -> None:
    with pytest.raises(ToolError) as refused:
        _one([1.0])
    assert refused.value.code == "TOO_FEW_ROWS"


def test_a_constant_column_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _one([2.0, 2.0, 2.0, 2.0])
    assert refused.value.code == "CONSTANT_COLUMN"


def test_two_independent_values_are_too_few_rows_together() -> None:
    with pytest.raises(ToolError) as refused:
        _run(design="independent", a=_column("first", [1.0]), b=_column("second", [2.0]))
    assert refused.value.code == "TOO_FEW_ROWS"


def test_one_group_level_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(
            design="independent",
            a=_column("y", [1.0, 2.0, 3.0]),
            groups=_group("g", ["only", "only", "only"]),
        )
    assert refused.value.code == "LEVELS_REQUIRED"


def test_a_prior_scale_of_zero_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _one([1.0, 2.0, 3.0], prior_scale=0.0)
    assert refused.value.code == "INVALID_OPTION"


def test_an_unordered_null_region_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _one([1.0, 2.0, 3.0], rope=(0.2, -0.2))
    assert refused.value.code == "INVALID_OPTION"


def test_a_confidence_level_outside_zero_and_one_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _one([1.0, 2.0, 3.0], level=1.0)
    assert refused.value.code == "INVALID_OPTION"


def test_more_successes_than_trials_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(design="proportion", successes=11, trials=10, null=0.5)
    assert refused.value.code == "INVALID_OPTION"


def test_a_proportion_with_no_trials_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(design="proportion", successes=0, trials=0, null=0.5)
    assert refused.value.code == "INVALID_OPTION"


def test_a_hypothesised_probability_outside_zero_and_one_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(design="proportion", successes=3, trials=10, null=1.0)
    assert refused.value.code == "INVALID_OPTION"


def test_a_proportion_takes_no_column() -> None:
    with pytest.raises(ToolError) as refused:
        _run(
            design="proportion",
            successes=3,
            trials=10,
            null=0.5,
            a=_column("y", [1.0, 2.0]),
        )
    assert refused.value.code == "INVALID_OPTION"


def test_an_effect_design_takes_no_counts() -> None:
    with pytest.raises(ToolError) as refused:
        _run(design="one_sample", a=_column("y", [1.0, 2.0]), successes=1, trials=2)
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_that_is_not_the_tools_own_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_bayes(TTestRequest(kind="one_sample", a=_column("y", [1.0, 2.0])))
    assert refused.value.code == "INVALID_OPTION"


# --- The registry ----------------------------------------------------------------------


def test_the_tool_is_registered_with_its_family_and_title() -> None:
    info = TOOLS.info("bayes")
    assert (info.name, info.family, info.title, info.needs_data) == (
        "bayes",
        "bayes",
        "Bayesian test",
        True,
    )
    assert "curvestudio.tools.stats.bayes" in TOOL_MODULES
    assert any(found.name == "bayes" for found in catalogue())
    assert [field.name for field in info.fields] == [
        "design",
        "a",
        "b",
        "groups",
        "null",
        "prior_scale",
        "rope",
        "successes",
        "trials",
        "prior_alpha",
        "prior_beta",
        "level",
    ]
