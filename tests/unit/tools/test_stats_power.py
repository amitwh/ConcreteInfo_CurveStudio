"""Sample size and power: the size a target power needs, and the power a size reaches.

The numbers are checked against the reference solvers themselves: statsmodels' ``TTestPower``
(the one-sample and paired tests), ``TTestIndPower``, ``FTestAnovaPower`` and, through
``proportion_effectsize``, ``NormalIndPower``; the analysis of variance is held to the
non-central F with ``k`` groups. The Pearson correlation has no statsmodels power function,
so it is held to the formula of Fisher's z transformation with the t correction, written out
from its definition in :func:`_correlation_reference`, which is the formula of R's
``pwr.r.test`` and pingouin's ``power_corr`` (see the sources below).

Sources of the worked examples, all fetched on 2026-11-24:

- NIST/SEMATECH e-Handbook of Statistical Methods, 7.2.2.2 "Sample sizes required",
  https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm. Its "Sample Size Table for
  Two-Sided Tests" tabulates sizes only for the one-sample normal test of a mean with the
  standard deviation known, which this tool does not compute; the handbook has no tabulated
  sample-size example for the two-sample t test or the one-way analysis of variance, and its
  detailed contents (https://www.itl.nist.gov/div898/handbook/dtoc.htm) list no such page.
  The one NIST page with numerical analysis-of-variance data, 7.4.3.3
  (https://www.itl.nist.gov/div898/handbook/prc/section4/prc433.htm), tabulates the resistor
  example with n = 5 per group in three groups, F = 9.59 and F(0.05; 2, 12) = 3.89; its
  tabulated n and critical F are checked in
  :func:`test_the_nist_analysis_of_variance_example_supplies_its_tabulated_n_and_critical_f`.
- The R ``pwr`` reference manual (CRAN, https://cran.r-project.org/web/packages/pwr/pwr.pdf)
  states in ``pwr.r.test`` that it uses the Z' transformation with the bias correction
  ``zp' = arctanh(rp) + rp / (2 (n - 1))``; the pingouin ``power_corr`` source (fetched the
  same day) prints that ``power_corr(n=20, power=0.80, alpha=0.05)`` gives ``r = 0.5822``,
  which :func:`test_the_correlation_worked_example_is_reproduced` checks.
- The two-sample t, analysis-of-variance and proportions worked examples come from the
  reference solvers (statsmodels 0.15.0 ``solve_power``), which are computed in the test, not
  typed from memory.
"""

import math
from collections.abc import Callable

import pytest
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize

from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import Counts, ToolResult, build_request, run_tool
from curvestudio.tools.stats.power import PowerRequest, run_power
from curvestudio.tools.stats.registry import TOOLS

ALPHA = 0.05


def _run(**options: object) -> ToolResult:
    return run_power(PowerRequest(**options))  # type: ignore[arg-type]


def _cells(result: ToolResult) -> dict[str, object]:
    """The one row of the Result table, by column key."""
    result_table = next(found for found in result.tables if found.title == "Result")
    keys = [column.key for column in result_table.columns]
    return dict(zip(keys, result_table.rows[0], strict=True))


def _scan(result: ToolResult) -> list[dict[str, object]]:
    """The rows of the Power by sample size table, by column key."""
    scan = next(found for found in result.tables if found.title == "Power by sample size")
    keys = [column.key for column in scan.columns]
    return [dict(zip(keys, row, strict=True)) for row in scan.rows]


def _estimate(result: ToolResult, label: str) -> float:
    return next(estimate.value for estimate in result.estimates if estimate.label == label)


def _correlation_reference(r: float, n: int, alpha: float) -> float:
    """The power of a correlation test, from Fisher's z with the t correction.

    The critical correlation comes from the t distribution on n - 2 degrees of freedom, the
    coefficient goes through the Z' transformation with its bias correction, and the normal
    approximation uses n - 3. This is R's ``pwr.r.test`` and pingouin's ``power_corr``.
    """
    dof = n - 2
    critical_t = stats.t.ppf(1.0 - alpha / 2.0, dof)
    critical_r = math.sqrt(critical_t**2 / (critical_t**2 + dof))
    magnitude = abs(r)
    z_r = math.atanh(magnitude) + magnitude / (2.0 * (n - 1))
    z_critical = math.atanh(critical_r)
    root = math.sqrt(n - 3)
    return float(
        stats.norm.cdf((z_r - z_critical) * root) + stats.norm.cdf((-z_r - z_critical) * root)
    )


def _reference_power(
    design: str,
    effect: float,
    n: int,
    *,
    alpha: float = ALPHA,
    ratio: float = 1.0,
    groups: int = 2,
) -> float:
    """The power of one design, as the reference solver gives it, at ``n`` per group."""
    if design in ("one_sample", "paired"):
        return float(TTestPower().power(effect, n, alpha))
    if design == "independent":
        return float(TTestIndPower().power(effect, n, alpha, ratio=ratio))
    if design == "anova":
        return float(FTestAnovaPower().power(effect, n * groups, alpha, k_groups=groups))
    if design == "proportions":
        return float(NormalIndPower().power(effect, n, alpha, ratio=ratio))
    return _correlation_reference(effect, n, alpha)


def _smallest_reference(power: Callable[[int], float], target: float, start: int) -> int:
    """The least ``n`` from ``start`` at which the reference power reaches ``target``."""
    n = start
    while power(n) < target:
        n += 1
        assert n < 1_000_000, "the reference power never reached the target"
    return n


# --------------------------------------------------------------------------- the contract


def test_the_result_is_the_contract_with_the_two_tables_and_the_power_plot() -> None:
    result = _run(design="one_sample", effect_size=0.5, power=0.8)

    assert type(result) is ToolResult
    assert result.tool == "power"
    assert result.title == "Sample size and power for a one-sample t test"
    assert result.level == pytest.approx(1.0 - ALPHA)
    assert result.counts == Counts(n_offered=0, n_used=0, n_excluded=0)
    assert result.tests == () and result.assumptions == () and result.excluded == ()
    assert [found.title for found in result.tables] == ["Result", "Power by sample size"]
    assert [column.label for column in result.tables[0].columns] == [
        "Design",
        "Effect size",
        "Alpha",
        "Power",
        "n per group",
        "Total N",
        "Ratio",
    ]
    assert [column.kind for column in result.tables[0].columns] == [
        "text",
        "number",
        "number",
        "number",
        "int",
        "number",
        "number",
    ]
    assert [column.label for column in result.tables[1].columns] == ["n per group", "Power"]
    assert [estimate.label for estimate in result.estimates] == [
        "n per group",
        "Total number of observations",
        "Power",
    ]
    assert len(result.plots) == 1
    plot = result.plots[0]
    assert plot.kind == "lines" and plot.title == "Power against sample size"
    assert plot.x_label == "n per group" and plot.y_label == "Power"
    assert [series.label for series in plot.series] == ["Power"]


def test_the_whole_number_and_the_total_follow_the_design() -> None:
    one = _run(design="one_sample", effect_size=0.5, power=0.8)
    n_one = int(_cells(one)["n"])  # type: ignore[arg-type]
    assert n_one == _smallest_reference(lambda n: float(TTestPower().power(0.5, n, ALPHA)), 0.8, 2)
    assert _cells(one)["total"] == n_one
    assert _cells(one)["ratio"] is None

    paired = _run(design="paired", effect_size=0.5, power=0.8)
    assert _cells(paired)["n"] == n_one and _cells(paired)["total"] == 2 * n_one

    independent = _run(design="independent", effect_size=0.5, power=0.8, ratio=2.0)
    n_independent = int(_cells(independent)["n"])  # type: ignore[arg-type]
    assert n_independent == _smallest_reference(
        lambda n: float(TTestIndPower().power(0.5, n, ALPHA, ratio=2.0)), 0.8, 2
    )
    assert _cells(independent)["total"] == 3 * n_independent
    assert _cells(independent)["ratio"] == 2.0

    anova = _run(design="anova", effect_size=0.25, power=0.8, n_groups=4)
    n_anova = int(_cells(anova)["n"])  # type: ignore[arg-type]
    assert n_anova == _smallest_reference(
        lambda n: float(FTestAnovaPower().power(0.25, n * 4, ALPHA, k_groups=4)), 0.8, 2
    )
    assert _cells(anova)["total"] == 4 * n_anova

    correlation = _run(design="correlation", effect_size=0.3, power=0.8)
    n_correlation = int(_cells(correlation)["n"])  # type: ignore[arg-type]
    assert n_correlation == _smallest_reference(
        lambda n: _correlation_reference(0.3, n, ALPHA), 0.8, 5
    )
    assert _cells(correlation)["total"] == n_correlation

    proportions = _run(design="proportions", p1=0.6, p2=0.4, power=0.8)
    n_proportions = int(_cells(proportions)["n"])  # type: ignore[arg-type]
    effect = abs(proportion_effectsize(0.6, 0.4))
    assert n_proportions == _smallest_reference(
        lambda n: float(NormalIndPower().power(effect, n, ALPHA, ratio=1.0)), 0.8, 2
    )
    assert _cells(proportions)["total"] == 2 * n_proportions


def test_the_result_is_a_pure_function_of_the_request() -> None:
    first = _run(design="independent", effect_size=0.4, power=0.85, ratio=1.5)
    second = _run(design="independent", effect_size=0.4, power=0.85, ratio=1.5)
    assert first == second


# --------------------------------------------------------------------------- references


@pytest.mark.parametrize(
    ("effect", "n", "alpha"),
    [(0.5, 34, 0.05), (0.8, 12, 0.01), (0.3, 100, 0.05), (0.2, 7, 0.05)],
)
def test_the_one_sample_power_matches_statsmodels(effect: float, n: int, alpha: float) -> None:
    result = _run(design="one_sample", effect_size=effect, n=n, alpha=alpha)
    expected = float(TTestPower().power(effect, n, alpha))

    assert _estimate(result, "Power") == pytest.approx(expected, rel=1e-8)
    assert _cells(result)["power"] == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize(
    ("effect", "n", "alpha"),
    [(0.5, 34, 0.05), (0.8, 12, 0.01), (0.3, 100, 0.05)],
)
def test_the_paired_power_matches_the_one_sample_t(effect: float, n: int, alpha: float) -> None:
    result = _run(design="paired", effect_size=effect, n=n, alpha=alpha)
    expected = float(TTestPower().power(effect, n, alpha))

    assert _estimate(result, "Power") == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize(
    ("effect", "n", "ratio", "alpha"),
    [(0.5, 30, 1.0, 0.05), (0.8, 12, 2.0, 0.01), (0.3, 100, 0.5, 0.05)],
)
def test_the_independent_power_matches_statsmodels(
    effect: float, n: int, ratio: float, alpha: float
) -> None:
    result = _run(design="independent", effect_size=effect, n=n, alpha=alpha, ratio=ratio)
    expected = float(TTestIndPower().power(effect, n, alpha, ratio=ratio))

    assert _estimate(result, "Power") == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize(
    ("effect", "n", "groups", "alpha"),
    [(0.25, 45, 4, 0.05), (0.4, 20, 3, 0.05), (0.1, 250, 5, 0.01)],
)
def test_the_analysis_of_variance_power_matches_statsmodels(
    effect: float, n: int, groups: int, alpha: float
) -> None:
    result = _run(design="anova", effect_size=effect, n=n, n_groups=groups, alpha=alpha)
    expected = float(FTestAnovaPower().power(effect, n * groups, alpha, k_groups=groups))

    assert _estimate(result, "Power") == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize(
    ("p1", "p2", "n", "ratio", "alpha"),
    [(0.6, 0.4, 50, 1.0, 0.05), (0.55, 0.5, 200, 1.0, 0.01), (0.7, 0.3, 20, 2.0, 0.05)],
)
def test_the_proportions_power_matches_statsmodels(
    p1: float, p2: float, n: int, ratio: float, alpha: float
) -> None:
    effect = abs(proportion_effectsize(p1, p2))
    result = _run(design="proportions", p1=p1, p2=p2, n=n, ratio=ratio, alpha=alpha)
    expected = float(NormalIndPower().power(effect, n, alpha, ratio=ratio))

    assert _estimate(result, "Power") == pytest.approx(expected, rel=1e-8)
    assert _cells(result)["effect_size"] == pytest.approx(effect, rel=1e-12)


@pytest.mark.parametrize(
    ("r", "n", "alpha"),
    [(0.3, 85, 0.05), (0.5, 29, 0.05), (0.1, 200, 0.01), (-0.4, 60, 0.05)],
)
def test_the_correlation_power_matches_the_fisher_z_with_the_t_correction(
    r: float, n: int, alpha: float
) -> None:
    result = _run(design="correlation", effect_size=r, n=n, alpha=alpha)
    expected = _correlation_reference(r, n, alpha)

    assert _estimate(result, "Power") == pytest.approx(expected, rel=1e-8)
    # The effect size a person sees is the magnitude the power depends on.
    assert _cells(result)["effect_size"] == pytest.approx(abs(r), rel=1e-12)


def test_the_correlation_converges_on_the_normal_approximation_for_a_large_sample() -> None:
    r, n = 0.25, 4_000
    result = _run(design="correlation", effect_size=r, n=n)
    approximation = float(NormalIndPower(ddof=3).power(math.atanh(r), n, ALPHA, ratio=0.0))

    # The t and bias corrections of the exact formula vanish as n grows.
    assert _estimate(result, "Power") == pytest.approx(approximation, abs=5e-3)


# --------------------------------------------------------------------------- solving for n


@pytest.mark.parametrize(
    ("design", "effect", "options", "start"),
    [
        ("one_sample", 0.5, {}, 2),
        ("independent", 0.5, {"ratio": 1.0}, 2),
        ("paired", 0.5, {}, 2),
        ("anova", 0.25, {"n_groups": 4}, 2),
        ("correlation", 0.3, {}, 5),
        ("proportions", None, {"p1": 0.6, "p2": 0.4}, 2),
    ],
)
def test_solving_for_n_gives_the_least_size_the_reference_reaches(
    design: str, effect: float | None, options: dict[str, float], start: int
) -> None:
    target = 0.8
    ratio = options.get("ratio", 1.0)
    groups = int(options.get("n_groups", 2))
    if design == "proportions":
        chosen = abs(proportion_effectsize(options["p1"], options["p2"]))
        request: dict[str, object] = {**options}
    else:
        assert effect is not None
        chosen = effect
        request = {"effect_size": effect, **options}

    def reference(n: int) -> float:
        if design == "proportions":
            return float(NormalIndPower().power(chosen, n, ALPHA, ratio=ratio))
        return _reference_power(design, chosen, n, ratio=ratio, groups=groups)

    expected = _smallest_reference(reference, target, start)

    result = _run(design=design, power=target, **request)

    assert _cells(result)["n"] == expected
    assert _estimate(result, "n per group") == float(expected)
    assert _estimate(result, "Power") == pytest.approx(reference(expected), rel=1e-8)


@pytest.mark.parametrize(
    ("design", "effect", "options"),
    [
        ("one_sample", 0.2, {}),
        ("independent", 0.3, {"ratio": 1.0}),
        ("paired", 0.2, {}),
        ("anova", 0.15, {"n_groups": 4}),
        ("correlation", 0.2, {}),
        ("proportions", None, {"p1": 0.55, "p2": 0.5}),
    ],
)
def test_the_power_at_the_solved_size_is_the_power_asked_for(
    design: str, effect: float | None, options: dict[str, object]
) -> None:
    target = 0.8
    request = {**options} if design == "proportions" else {"effect_size": effect, **options}

    solved = _run(design=design, power=target, **request)
    n = int(_cells(solved)["n"])  # type: ignore[arg-type]
    again = _run(design=design, n=n, **request)

    assert target <= _estimate(again, "Power") <= target + 0.005


def test_a_larger_effect_needs_a_smaller_sample() -> None:
    strong = _run(design="independent", effect_size=0.8, power=0.8)
    weak = _run(design="independent", effect_size=0.4, power=0.8)
    assert int(_cells(strong)["n"]) < int(_cells(weak)["n"])  # type: ignore[arg-type]

    broad = _run(design="anova", effect_size=0.4, power=0.8, n_groups=4)
    narrow = _run(design="anova", effect_size=0.2, power=0.8, n_groups=4)
    assert int(_cells(broad)["n"]) < int(_cells(narrow)["n"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the scan


def test_the_power_by_sample_size_table_spans_five_to_ninety_nine_percent() -> None:
    result = _run(design="independent", effect_size=0.5, power=0.8)
    rows = _scan(result)
    sizes = [row["n"] for row in rows]
    powers = [row["power"] for row in rows]

    assert 20 <= len(rows) <= 40
    assert sizes == sorted(set(sizes))
    assert powers == sorted(powers)

    def reference(n: int) -> float:
        return float(TTestIndPower().power(0.5, n, ALPHA, ratio=1.0))

    assert sizes[0] == _smallest_reference(reference, 0.05, 2)
    assert sizes[-1] == _smallest_reference(reference, 0.99, 2)
    assert powers[0] >= 0.05 and powers[-1] >= 0.99

    plot = result.plots[0]
    assert plot.series[0].x == tuple(float(size) for size in sizes)  # type: ignore[arg-type]
    assert plot.series[0].y == tuple(powers)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- worked examples


def test_the_nist_analysis_of_variance_example_supplies_its_tabulated_n_and_critical_f() -> None:
    """The resistor example of the NIST/SEMATECH e-Handbook, 7.4.3.3 (fetched 2026-11-24).

    The page tabulates three temperature groups of n = 5 resistors each, the sums of squares
    below, and F = 9.59 with F(0.05; 2, 12) = 3.89. The effect size is read from the table as
    Cohen's f = sqrt(eta^2 / (1 - eta^2)), with eta^2 the treatment share of the total sum of
    squares, and the tabulated n = 5 per group is the size the tool is run at.
    """
    assert round(float(stats.f.isf(ALPHA, 2, 12)), 2) == 3.89

    treatment, error = 27.897, 17.452
    total = treatment + error
    eta_squared = treatment / total
    f = math.sqrt(eta_squared / (1.0 - eta_squared))

    result = _run(design="anova", effect_size=f, n=5, n_groups=3, alpha=ALPHA)
    expected = float(FTestAnovaPower().power(f, 5 * 3, ALPHA, k_groups=3))

    assert _cells(result)["n"] == 5 and _cells(result)["total"] == 15
    assert _estimate(result, "Power") == pytest.approx(expected, rel=1e-8)


def test_the_reference_worked_examples_are_reproduced() -> None:
    """The published example calls of the reference solvers, solved by this tool in turn.

    statsmodels 0.15.0: ``TTestIndPower.solve_power(effect_size=0.5, alpha=0.05,
    power=0.8)`` gives 63.7656 per group, ``FTestAnovaPower.solve_power(effect_size=0.25,
    alpha=0.05, power=0.8, k_groups=4)`` gives 178.397 total, and ``NormalIndPower.solve_power``
    at the effect size of two proportions 0.6 and 0.4 gives 96.79 per group. The smallest
    whole number reaching each target is what the tool reports.
    """
    two_sample = _run(design="independent", effect_size=0.5, power=0.8)
    assert _cells(two_sample)["n"] == math.ceil(
        float(TTestIndPower().solve_power(effect_size=0.5, alpha=ALPHA, power=0.8))
    )

    analysis = _run(design="anova", effect_size=0.25, power=0.8, n_groups=4)
    total = float(
        FTestAnovaPower().solve_power(effect_size=0.25, alpha=ALPHA, power=0.8, k_groups=4)
    )
    assert _cells(analysis)["n"] == math.ceil(total / 4)

    effect = abs(proportion_effectsize(0.6, 0.4))
    proportions = _run(design="proportions", p1=0.6, p2=0.4, power=0.8)
    assert _cells(proportions)["n"] == math.ceil(
        float(NormalIndPower().solve_power(effect_size=effect, alpha=ALPHA, power=0.8))
    )


def test_the_correlation_worked_example_is_reproduced() -> None:
    """pingouin's ``power_corr`` prints r = 0.5822 for n = 20 and a power of 0.80.

    The source (fetched 2026-11-24) shows the same formula this tool uses, so the tool's
    power at that r and n is the power that was asked for, to the four places printed.
    """
    result = _run(design="correlation", effect_size=0.5822, n=20, alpha=ALPHA)
    assert _estimate(result, "Power") == pytest.approx(0.80, abs=1e-3)
    assert int(_cells(result)["n"]) == 20  # type: ignore[arg-type]


# --------------------------------------------------------------------------- edge cases


@pytest.mark.parametrize(
    "options",
    [
        {"design": "one_sample", "effect_size": 0.0, "power": 0.8},
        {"design": "one_sample", "effect_size": 0.5, "alpha": 0.0, "power": 0.8},
        {"design": "one_sample", "effect_size": 0.5, "alpha": 1.0, "power": 0.8},
        {"design": "one_sample", "effect_size": 0.5, "alpha": -0.1, "power": 0.8},
        {"design": "one_sample", "effect_size": 0.5, "alpha": 1.5, "power": 0.8},
        {"design": "one_sample", "effect_size": 0.5},
        {"design": "one_sample", "effect_size": 0.5, "power": 0.8, "n": 10},
        {"design": "one_sample", "effect_size": 0.5, "power": 0.0},
        {"design": "one_sample", "effect_size": 0.5, "power": 1.0},
        {"design": "one_sample", "effect_size": 0.5, "n": 1},
        {"design": "correlation", "effect_size": 1.0, "power": 0.8},
        {"design": "correlation", "effect_size": -1.0, "power": 0.8},
        {"design": "independent", "effect_size": 0.5, "power": 0.8, "ratio": 0.0},
        {"design": "independent", "effect_size": 0.5, "power": 0.8, "n_groups": 1},
        {"design": "one_sample", "effect_size": 0.5, "power": 0.8, "p1": 0.4},
        {"design": "proportions", "effect_size": 0.5, "p1": 0.6, "p2": 0.4, "power": 0.8},
        {"design": "proportions", "power": 0.8},
        {"design": "proportions", "p1": 0.6, "p2": 0.6, "power": 0.8},
        {"design": "proportions", "p1": 0.0, "p2": 0.4, "power": 0.8},
        {"design": "proportions", "p1": 1.0, "p2": 0.4, "power": 0.8},
    ],
)
def test_an_out_of_range_or_contradictory_request_is_refused(options: dict[str, object]) -> None:
    with pytest.raises(ToolError) as refused:
        _run(**options)
    assert refused.value.code == "INVALID_OPTION"


def test_run_power_refuses_a_request_that_is_not_its_own() -> None:
    class Other(BaseModel):
        x: int = 1

    with pytest.raises(ToolError) as refused:
        run_power(Other())
    assert refused.value.code == "INVALID_OPTION"
    assert "PowerRequest" in refused.value.message


# --------------------------------------------------------------------------- words


@pytest.mark.parametrize(
    ("options", "phrases"),
    [
        (
            {"design": "one_sample", "effect_size": 0.5, "power": 0.8},
            ["two-sided t test; non-central t distribution", "Cohen's d"],
        ),
        (
            {"design": "independent", "effect_size": 0.5, "power": 0.8},
            [
                "two-sided t test; non-central t distribution",
                "equal-variance test",
                "TTestIndPower",
                "Cohen's d",
            ],
        ),
        (
            {"design": "paired", "effect_size": 0.5, "power": 0.8},
            ["non-central t distribution", "Cohen's dz"],
        ),
        (
            {"design": "anova", "effect_size": 0.25, "power": 0.8, "n_groups": 4},
            ["one-way analysis of variance with 4 groups; non-central F", "Cohen's f"],
        ),
        (
            {"design": "correlation", "effect_size": 0.3, "power": 0.8},
            ["Pearson correlation; Fisher z with the t correction"],
        ),
        (
            {"design": "proportions", "p1": 0.6, "p2": 0.4, "power": 0.8},
            [
                "difference of two proportions; the normal approximation on the arcsine "
                "scale (Cohen's h)",
                "Cohen's h",
            ],
        ),
    ],
)
def test_the_methods_fragment_names_the_method_and_the_effect_size(
    options: dict[str, object], phrases: list[str]
) -> None:
    text = _run(**options).methods_fragment
    for phrase in phrases:
        assert phrase in text, phrase


def test_the_sentences_use_the_three_significant_digit_numbers_and_claim_no_proof() -> None:
    result = _run(design="independent", effect_size=0.5, power=0.8, alpha=0.05)

    assert "alpha = 0.05" in result.interpretation
    assert "power of 0.8" in result.interpretation
    for words in (result.title, result.interpretation, result.methods_fragment):
        assert not claims_proof(words)


# --------------------------------------------------------------------------- registration


def test_the_tool_is_registered_under_power_with_described_option_fields() -> None:
    assert "power" in TOOLS
    spec = TOOLS["power"]
    assert (spec.family, spec.title, spec.request) == (
        "power",
        "Sample size and power",
        PowerRequest,
    )

    info = TOOLS.info("power")
    assert info.needs_data is False
    assert [field.name for field in info.fields] == [
        "design",
        "effect_size",
        "alpha",
        "power",
        "n",
        "ratio",
        "n_groups",
        "p1",
        "p2",
    ]
    assert [field.kind for field in info.fields] == ["option"] * 9
    assert [field.required for field in info.fields] == [True] + [False] * 8
    assert all(field.description for field in info.fields)
    assert [field.label for field in info.fields] == [
        "Design",
        "Effect size",
        "Alpha",
        "Power",
        "N per group",
        "Group size ratio",
        "Number of groups",
        "First proportion",
        "Second proportion",
    ]
    assert all(field.option_schema is not None for field in info.fields)
    design = info.fields[0].option_schema
    assert design is not None
    assert design["enum"] == [
        "one_sample",
        "independent",
        "paired",
        "anova",
        "correlation",
        "proportions",
    ]


def test_the_registry_builds_the_request_from_options_and_runs_without_a_dataset() -> None:
    def refuse(_name: str) -> object:
        raise AssertionError("a power request names no column, so no lookup may happen")

    request = build_request(
        "power",
        {"design": "independent", "effect_size": 0.5, "power": 0.8, "ratio": 1.0},
        column=refuse,  # type: ignore[arg-type]
        group=refuse,  # type: ignore[arg-type]
    )
    assert isinstance(request, PowerRequest)
    assert (request.design, request.effect_size, request.power, request.ratio) == (
        "independent",
        0.5,
        0.8,
        1.0,
    )

    result = run_tool("power", request)
    assert result.tool == "power" and _cells(result)["n"] == 64
