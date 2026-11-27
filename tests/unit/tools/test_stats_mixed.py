"""The linear mixed model: REML fixed effects, variance components, BLUPs and intervals.

The fixed effects, the variance components and the BLUPs are checked against
``statsmodels.formula.api.mixedlm(...).fit(reml=True, method="bfgs", gtol=1e-12)`` at 1e-6 on
a balanced and an unbalanced fixture. The standard errors are checked at 1e-6 against the
generalized least squares form computed from the reference's own variance components: the
tool reports the conditional covariance at the optimum, whereas statsmodels' ``bse`` inverts
the Hessian over the fixed effects and the variance parameters together, so the two are
different estimators and only the model's own form is comparable. The denominator degrees of
freedom follow the stated Satterthwaite-type rule for a random-intercept model.

The published example is the Sitka growth data's random-intercept REML fit printed in the
statsmodels documentation page "Linear Mixed Effects Models"
(``statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html``), read on
2026-11-25. That section prints both the statsmodels ``MixedLM`` REML summary (Intercept
2.273, Time 0.013, Intercept Var 0.374, Scale 0.0392) and the lme4 fit it compares with
(2.2732443, 0.0126855, tree variance 0.37451, residual 0.03921). The page uses the Sitka data
from ``MASS``, fetched from the RDatasets mirror on 2026-11-25 and embedded below, because
statsmodels ships no copy and the tests run without a network. The page's other mixed-model
example uses the dietox data; the ``sleepstudy`` fit named in the task's plan is not shipped
or tabulated by statsmodels 0.15, so its documented random-intercept example is used instead.
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from pydantic import ValidationError
from scipy import stats as scipy_stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ExcludedRow, ToolError
from curvestudio.tools.stats import FAMILIES, ToolResult, catalogue
from curvestudio.tools.stats.mixed import MixedRequest, _fit, run_mixed

STAT_RTOL = 1e-6
"""The tolerance on an estimate the reference implementation also computes."""

P_ATOL = 1e-12
"""The tolerance on a p-value the model's own distribution gives."""

SITKA_ROWS = """1 152 4.51
1 174 4.98
1 201 5.41
1 227 5.9
1 258 6.15
2 152 4.24
2 174 4.2
2 201 4.68
2 227 4.92
2 258 4.96
3 152 3.98
3 174 4.36
3 201 4.79
3 227 4.99
3 258 5.03
4 152 4.36
4 174 4.77
4 201 5.1
4 227 5.3
4 258 5.36
5 152 4.34
5 174 4.95
5 201 5.42
5 227 5.97
5 258 6.28
6 152 4.59
6 174 5.08
6 201 5.36
6 227 5.76
6 258 6
7 152 4.41
7 174 4.56
7 201 4.95
7 227 5.23
7 258 5.33
8 152 4.24
8 174 4.64
8 201 4.95
8 227 5.38
8 258 5.48
9 152 4.82
9 174 5.17
9 201 5.76
9 227 6.12
9 258 6.24
10 152 3.84
10 174 4.17
10 201 4.67
10 227 4.67
10 258 4.8
11 152 4.07
11 174 4.31
11 201 4.9
11 227 5.1
11 258 5.1
12 152 4.28
12 174 4.8
12 201 5.27
12 227 5.55
12 258 5.65
13 152 4.47
13 174 4.89
13 201 5.23
13 227 5.55
13 258 5.74
14 152 4.46
14 174 4.84
14 201 5.11
14 227 5.34
14 258 5.46
15 152 4.6
15 174 4.08
15 201 4.17
15 227 4.35
15 258 4.59
16 152 3.73
16 174 4.15
16 201 4.61
16 227 4.87
16 258 4.93
17 152 4.67
17 174 4.88
17 201 5.18
17 227 5.34
17 258 5.49
18 152 2.96
18 174 3.47
18 201 3.76
18 227 3.89
18 258 4.3
19 152 3.24
19 174 3.93
19 201 4.76
19 227 4.62
19 258 4.64
20 152 4.36
20 174 4.77
20 201 5.02
20 227 5.26
20 258 5.45
21 152 4.04
21 174 4.64
21 201 4.86
21 227 5.09
21 258 5.25
22 152 3.53
22 174 4.25
22 201 4.68
22 227 4.97
22 258 5.18
23 152 4.22
23 174 4.69
23 201 5.07
23 227 5.37
23 258 5.58
24 152 2.79
24 174 3.1
24 201 3.3
24 227 3.38
24 258 3.55
25 152 3.3
25 174 3.9
25 201 4.34
25 227 4.96
25 258 5.4
26 152 3.34
26 174 3.81
26 201 4.21
26 227 4.54
26 258 4.86
27 152 3.76
27 174 4.36
27 201 4.7
27 227 5.44
27 258 5.32
28 152 4.49
28 174 4.76
28 201 5.15
28 227 5.37
28 258 5.56
29 152 4.88
29 174 5.14
29 201 5.52
29 227 6.08
29 258 6.17
30 152 4.88
30 174 5.32
30 201 5.63
30 227 5.75
30 258 5.94
31 152 3.8
31 174 4.16
31 201 4.45
31 227 4.89
31 258 5.05
32 152 4.46
32 174 4.62
32 201 5
32 227 5.4
32 258 5.49
33 152 4.29
33 174 4.82
33 201 5.32
33 227 5.46
33 258 5.5
34 152 4.06
34 174 4.58
34 201 4.81
34 227 5.12
34 258 5.27
35 152 5.16
35 174 5.43
35 201 5.71
35 227 6.08
35 258 6.21
36 152 3.81
36 174 4.12
36 201 4.42
36 227 4.62
36 258 4.6
37 152 5.09
37 174 5.62
37 201 5.9
37 227 6.36
37 258 6.49
38 152 4.13
38 174 4.71
38 201 5.27
38 227 5.56
38 258 5.72
39 152 4.85
39 174 5.36
39 201 5.52
39 227 5.96
39 258 6.13
40 152 4.11
40 174 4.62
40 201 4.95
40 227 5.28
40 258 5.43
41 152 4.95
41 174 5.39
41 201 5.82
41 227 6.42
41 258 6.48
42 152 4.36
42 174 4.65
42 201 5.04
42 227 5.38
42 258 5.47
43 152 4.05
43 174 4.65
43 201 5.09
43 227 5.44
43 258 5.6
44 152 3.76
44 174 4.27
44 201 4.59
44 227 5.1
44 258 5.25
45 152 2.84
45 174 3.25
45 201 3.69
45 227 4.16
45 258 4.21
46 152 4.33
46 174 4.8
46 201 5.09
46 227 5.42
46 258 5.61
47 152 3.99
47 174 4.55
47 201 4.91
47 227 5.26
47 258 5.3
48 152 3.5
48 174 3.75
48 201 3.97
48 227 4.71
48 258 4.85
49 152 3.31
49 174 3.45
49 201 4.16
49 227 4.48
49 258 4.54
50 152 3.03
50 174 3.55
50 201 3.97
50 227 4.4
50 258 4.58
51 152 3.27
51 174 3.83
51 201 4.44
51 227 4.8
51 258 4.89
52 152 3.56
52 174 4.18
52 201 4.7
52 227 5.27
52 258 5.28
53 152 3.39
53 174 3.73
53 201 3.92
53 227 4.11
53 258 4.15
54 152 3.72
54 174 4.16
54 201 4.55
54 227 5.03
54 258 5.02
55 152 4.53
55 174 5.05
55 201 5.18
55 227 5.41
55 258 5.42
56 152 4.97
56 174 5.32
56 201 5.83
56 227 6.29
56 258 6.45
57 152 4.37
57 174 4.81
57 201 5.03
57 227 5.19
57 258 5.4
58 152 4.58
58 174 4.99
58 201 5.37
58 227 5.68
58 258 5.93
59 152 4
59 174 4.5
59 201 4.92
59 227 5.44
59 258 5.87
60 152 4.73
60 174 5.05
60 201 5.33
60 227 5.92
60 258 6.01
61 152 5.15
61 174 5.63
61 201 6.11
61 227 6.39
61 258 6.61
62 152 4.1
62 174 4.46
62 201 4.84
62 227 5.29
62 258 5.48
63 152 3.22
63 174 3.85
63 201 4.47
63 227 4.85
63 258 5.11
64 152 2.23
64 174 2.89
64 201 3.16
64 227 3.4
64 258 3.52
65 152 3.65
65 174 4.36
65 201 4.76
65 227 5.18
65 258 5.44
66 152 3.4
66 174 3.92
66 201 4.5
66 227 4.97
66 258 5.14
67 152 5.16
67 174 5.49
67 201 5.74
67 227 6.05
67 258 6.21
68 152 4.04
68 174 4.52
68 201 5.15
68 227 5.59
68 258 5.87
69 152 4.52
69 174 4.91
69 201 5.04
69 227 5.71
69 258 5.97
70 152 4.56
70 174 5.12
70 201 5.4
70 227 5.69
70 258 5.89
71 152 4.9
71 174 5.35
71 201 5.71
71 227 6.12
71 258 6.25
72 152 4.83
72 174 5.1
72 201 5.43
72 227 5.59
72 258 6.04
73 152 5.46
73 174 5.79
73 201 6.12
73 227 6.41
73 258 6.63
74 152 4.17
74 174 4.67
74 201 5.16
74 227 5.56
74 258 5.75
75 152 3.35
75 174 4.05
75 201 4.51
75 227 5.22
75 258 5.44
76 152 3.33
76 174 3.82
76 201 4.38
76 227 4.99
76 258 5.17
77 152 3.41
77 174 3.68
77 201 4.03
77 227 4.28
77 258 4.54
78 152 4.5
78 174 4.8
78 201 5.28
78 227 5.83
78 258 6.16
79 152 2.99
79 174 3.61
79 201 4.48
79 227 4.91
79 258 5.06"""
"""The Sitka rows, ``tree time size``, from MASS's Sitka data (fetched 2026-11-25)."""


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _group(name: str, labels: Sequence[str], rows: object | None = None) -> GroupColumn:
    """A group column over positions 0..n-1, unless ``rows`` names them."""
    text = np.asarray(labels, dtype=object)
    positions = np.arange(text.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return GroupColumn(name=name, labels=text, rows=np.asarray(positions, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_mixed(MixedRequest(**options))  # type: ignore[arg-type]


def _table(result: ToolResult, title: str) -> list[tuple[object, ...]]:
    found = next(table for table in result.tables if table.title == title)
    return list(found.rows)


def _keyed(result: ToolResult, title: str) -> dict[object, dict[str, object]]:
    found = next(table for table in result.tables if table.title == title)
    keys = [column.key for column in found.columns]
    return {row[0]: dict(zip(keys, row, strict=True)) for row in found.rows}


def _estimate(result: ToolResult, label: str) -> float:
    return next(item.value for item in result.estimates if item.label == label)


def _fixture(sizes: Sequence[int], seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A random-intercept sample with the group sizes given, and a fixed effect."""
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(len(sizes)), sizes)
    predictor = rng.normal(size=codes.size)
    shifts = rng.normal(scale=0.9, size=len(sizes))[codes]
    outcome = 1.5 + 0.8 * predictor + shifts + rng.normal(scale=0.7, size=codes.size)
    labels = [f"g{code}" for code in codes]
    return outcome, predictor, labels


def _reference(frame: pd.DataFrame) -> object:
    """The statsmodels REML fit, at a tight tolerance so its optimum is the same."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return smf.mixedlm("y ~ x", frame, groups=frame["g"]).fit(
            reml=True, method="bfgs", gtol=1e-12, maxiter=10_000
        )


def _fitted(sizes: Sequence[int], seed: int) -> tuple[ToolResult, pd.DataFrame, object]:
    outcome, predictor, labels = _fixture(sizes, seed)
    frame = pd.DataFrame({"y": outcome, "x": predictor, "g": labels})
    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
    )
    return result, frame, _reference(frame)


def _gls_standard_errors(frame: pd.DataFrame, reference: object) -> np.ndarray:
    """The GLS standard errors at the reference's own variance components."""
    group_variance = float(reference.cov_re.to_numpy()[0, 0])  # type: ignore[attr-defined]
    residual_variance = float(reference.scale)  # type: ignore[attr-defined]
    lam = group_variance / residual_variance
    labels = list(dict.fromkeys(frame["g"]))
    position = {label: index for index, label in enumerate(labels)}
    codes = np.array([position[label] for label in frame["g"]], dtype=np.intp)
    counts = np.bincount(codes, minlength=len(labels)).astype(np.float64)
    design = np.column_stack((np.ones(len(frame)), frame["x"].to_numpy(float)))
    sums = np.zeros((len(labels), design.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, design)
    shrink = lam / (1.0 + lam * counts)
    information = design.T @ (design - shrink[codes, None] * sums[codes])
    covariance = residual_variance * np.linalg.inv(information)
    return np.sqrt(np.diag(covariance))


# --------------------------------------------------------------------------- the references


@pytest.mark.parametrize(
    ("sizes", "seed"),
    [([6] * 8, 11), ([3, 11, 4, 9, 2, 7, 5, 8], 12)],
    ids=["balanced", "unbalanced"],
)
def test_the_fixed_effects_and_variance_components_are_statsmodels(
    sizes: Sequence[int], seed: int
) -> None:
    result, _frame, reference = _fitted(sizes, seed)
    fixed = _keyed(result, "Fixed effects")
    params = np.asarray(reference.params)  # type: ignore[attr-defined]

    assert fixed["(Intercept)"]["estimate"] == pytest.approx(params[0], rel=STAT_RTOL)
    assert fixed["x"]["estimate"] == pytest.approx(params[1], rel=STAT_RTOL)
    assert _estimate(result, "Residual variance (sigma squared)") == pytest.approx(
        float(reference.scale),
        rel=STAT_RTOL,  # type: ignore[attr-defined]
    )
    assert _estimate(result, "Between-group variance (tau squared)") == pytest.approx(
        float(reference.cov_re.to_numpy()[0, 0]),
        rel=STAT_RTOL,  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize(
    ("sizes", "seed"),
    [([6] * 8, 11), ([3, 11, 4, 9, 2, 7, 5, 8], 12)],
    ids=["balanced", "unbalanced"],
)
def test_the_blups_are_statsmodels_random_effects(sizes: Sequence[int], seed: int) -> None:
    result, _frame, reference = _fitted(sizes, seed)
    rows = _table(result, "Group effects")
    effects = reference.random_effects  # type: ignore[attr-defined]

    assert len(rows) == len(sizes)
    for row in rows:
        label, _count, blup = row[0], row[1], row[2]
        expected = float(effects[str(label)].to_numpy()[0])
        assert blup == pytest.approx(expected, rel=STAT_RTOL)


@pytest.mark.parametrize(
    ("sizes", "seed"),
    [([6] * 8, 11), ([3, 11, 4, 9, 2, 7, 5, 8], 12)],
    ids=["balanced", "unbalanced"],
)
def test_the_standard_errors_are_the_models_own_gls_ones(sizes: Sequence[int], seed: int) -> None:
    result, frame, reference = _fitted(sizes, seed)
    fixed = _keyed(result, "Fixed effects")
    expected = _gls_standard_errors(frame, reference)

    assert fixed["(Intercept)"]["se"] == pytest.approx(expected[0], rel=STAT_RTOL)
    assert fixed["x"]["se"] == pytest.approx(expected[1], rel=STAT_RTOL)


def test_the_denominator_degrees_of_freedom_follow_the_stated_rule() -> None:
    result, _frame, _reference_fit = _fitted([6] * 8, 11)
    fixed = _keyed(result, "Fixed effects")
    n = sum([6] * 8)
    groups = 8

    assert fixed["(Intercept)"]["df"] == float(groups - 1)
    assert fixed["x"]["df"] == float(n - 3)
    tests = {test.name: test for test in result.tests}
    assert "number of groups minus one" in tests["t test of (Intercept)"].method
    assert "rows minus fixed parameters minus one" in tests["t test of x"].method
    for term in ("(Intercept)", "x"):
        test = tests[f"t test of {term}"]
        assert test.statistic == pytest.approx(
            fixed[term]["estimate"] / fixed[term]["se"],
            rel=1e-12,  # type: ignore[operator]
        )
        assert test.p_value == pytest.approx(
            2.0 * scipy_stats.t.sf(abs(test.statistic), test.df),
            abs=P_ATOL,  # type: ignore[arg-type]
        )
        assert test.p_value_text.startswith("p ")


def test_the_likelihood_ratio_test_uses_the_half_half_mixture() -> None:
    result, _frame, _reference_fit = _fitted([6] * 8, 11)
    test = next(test for test in result.tests if test.name.startswith("Likelihood-ratio"))

    assert test.statistic is not None and test.statistic > 0.0
    assert test.p_value == pytest.approx(0.5 * scipy_stats.chi2.sf(test.statistic, 1), abs=P_ATOL)
    assert "mixture" in test.p_value_text
    assert "0.5 chi-square(0) + 0.5 chi-square(1)" in test.p_value_text


def test_nakagawas_r_squared_follows_its_definition() -> None:
    result, frame, _reference_fit = _fitted([6] * 8, 11)
    fixed = _keyed(result, "Fixed effects")
    fitted = (
        float(fixed["(Intercept)"]["estimate"])  # type: ignore[arg-type]
        + float(fixed["x"]["estimate"]) * frame["x"].to_numpy(float)  # type: ignore[arg-type]
    )
    fixed_variance = float(np.var(fitted, ddof=1))
    tau2 = _estimate(result, "Between-group variance (tau squared)")
    sigma2 = _estimate(result, "Residual variance (sigma squared)")
    total = fixed_variance + tau2 + sigma2

    assert _estimate(result, "Marginal R squared") == pytest.approx(
        fixed_variance / total, rel=1e-12
    )
    assert _estimate(result, "Conditional R squared") == pytest.approx(
        (fixed_variance + tau2) / total, rel=1e-12
    )


def test_the_returned_fit_satisfies_the_models_own_likelihood_equations() -> None:
    outcome, predictor, labels = _fixture([6] * 8, 11)
    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
    )
    fixed = _keyed(result, "Fixed effects")
    beta = np.array([fixed["(Intercept)"]["estimate"], fixed["x"]["estimate"]], dtype=np.float64)
    tau2 = _estimate(result, "Between-group variance (tau squared)")
    sigma2 = _estimate(result, "Residual variance (sigma squared)")
    design = np.column_stack((np.ones(outcome.size), predictor))
    order = list(dict.fromkeys(labels))
    position = {label: index for index, label in enumerate(order)}
    codes = np.array([position[label] for label in labels], dtype=np.intp)
    counts = np.bincount(codes, minlength=len(order)).astype(np.float64)
    lam = tau2 / sigma2
    scale = 1.0 + lam * counts
    shrink = lam / scale
    residuals = outcome - design @ beta
    sums = np.bincount(codes, weights=residuals, minlength=len(order))
    inverse_residuals = residuals - shrink[codes] * sums[codes]
    gradient = design.T @ inverse_residuals

    assert float(np.max(np.abs(gradient))) <= 1e-8 * float(np.max(np.abs(inverse_residuals)))

    def criterion(value: float) -> float:
        block = 1.0 + value * counts
        squeeze = value / block
        sums_design = np.zeros((len(order), 2), dtype=np.float64)
        np.add.at(sums_design, codes, design)
        sums_outcome = np.bincount(codes, weights=outcome, minlength=len(order))
        information = design.T @ (design - squeeze[codes, None] * sums_design[codes])
        estimate = np.linalg.solve(
            information,
            design.T @ (outcome - squeeze[codes] * sums_outcome[codes]),
        )
        difference = outcome - design @ estimate
        squares = float(
            difference
            @ (
                difference
                - squeeze[codes]
                * np.bincount(codes, weights=difference, minlength=len(order))[codes]
            )
        )
        return float(
            np.sum(np.log(block))
            + np.linalg.slogdet(information)[1]
            + (outcome.size - 2) * np.log(squares / (outcome.size - 2))
        )

    at_optimum = criterion(lam)
    assert at_optimum <= criterion(lam * 1.01) + 1e-9
    assert at_optimum <= criterion(lam / 1.01) + 1e-9


def test_the_icc_is_the_between_group_share_and_lies_in_zero_to_one() -> None:
    result, _frame, _reference_fit = _fitted([6] * 8, 11)
    tau2 = _estimate(result, "Between-group variance (tau squared)")
    sigma2 = _estimate(result, "Residual variance (sigma squared)")
    icc = _estimate(result, "Intraclass correlation")

    assert icc == pytest.approx(tau2 / (tau2 + sigma2), rel=1e-12)
    assert 0.0 <= icc <= 1.0


# --------------------------------------------------------------------------- the published fit


def _sitka() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [line.split() for line in SITKA_ROWS.strip().splitlines()]
    trees = np.array([row[0] for row in rows], dtype=object)
    times = np.array([float(row[1]) for row in rows])
    sizes = np.array([float(row[2]) for row in rows])
    return trees, times, sizes


def test_the_published_sitka_fit_is_reproduced() -> None:
    trees, times, sizes = _sitka()
    result = _run(
        outcome=_column("size", sizes),
        predictors=(_column("Time", times),),
        group=_group("tree", [str(tree) for tree in trees]),
    )
    fixed = _keyed(result, "Fixed effects")

    # The lme4 output printed in the statsmodels documentation, to its printed precision.
    assert fixed["(Intercept)"]["estimate"] == pytest.approx(2.2732443, abs=5e-7)
    assert fixed["Time"]["estimate"] == pytest.approx(0.0126855, abs=5e-8)
    assert fixed["(Intercept)"]["se"] == pytest.approx(0.0878955, abs=5e-7)
    assert fixed["Time"]["se"] == pytest.approx(0.0002654, abs=5e-8)
    assert _estimate(result, "Between-group variance (tau squared)") == pytest.approx(
        0.37451, abs=5e-6
    )
    assert _estimate(result, "Residual variance (sigma squared)") == pytest.approx(
        0.03921, abs=5e-6
    )
    # The statsmodels MixedLM summary on the same page.
    assert fixed["(Intercept)"]["estimate"] == pytest.approx(2.273, abs=5e-4)
    assert fixed["Time"]["estimate"] == pytest.approx(0.013, abs=5e-4)
    assert _estimate(result, "Between-group variance (tau squared)") == pytest.approx(
        0.374, abs=1e-3
    )
    assert _estimate(result, "Residual variance (sigma squared)") == pytest.approx(0.0392, abs=5e-5)
    assert result.counts.n_used == 395


# --------------------------------------------------------------------------- the properties


def test_a_single_group_has_no_between_group_variance_and_ols_coefficients() -> None:
    outcome, predictor, _labels = _fixture([20], 21)
    design = np.column_stack((np.ones(outcome.size), predictor))
    codes = np.zeros(outcome.size, dtype=np.intp)

    model = _fit(design, outcome, codes, ("only",), 0.95)
    expected = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ expected

    assert model.tau2 == 0.0
    assert np.allclose(model.beta, expected, rtol=1e-12, atol=1e-12)
    assert model.sigma2 == pytest.approx(
        float(residuals @ residuals) / (outcome.size - 2), rel=1e-12
    )
    assert model.lr == 0.0 and model.lr_p == 1.0
    assert np.all(np.isfinite(model.p_value))


def test_adding_a_constant_to_the_outcome_moves_only_the_intercept() -> None:
    outcome, predictor, labels = _fixture([6] * 8, 11)
    base = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
    )
    shifted = _run(
        outcome=_column("y", outcome + 10.0),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
    )
    base_fixed = _keyed(base, "Fixed effects")
    shifted_fixed = _keyed(shifted, "Fixed effects")

    assert shifted_fixed["(Intercept)"]["estimate"] == pytest.approx(
        float(base_fixed["(Intercept)"]["estimate"]) + 10.0,
        rel=1e-9,  # type: ignore[arg-type]
        # shift invariance is exact in exact arithmetic; 1e-12 was below the
        # BLAS rounding the CI runner's libraries produce (rel ~5e-12 observed)
    )
    assert shifted_fixed["x"]["estimate"] == pytest.approx(
        base_fixed["x"]["estimate"],
        rel=1e-10,  # type: ignore[arg-type]
    )
    for label in (
        "Residual variance (sigma squared)",
        "Between-group variance (tau squared)",
        "Intraclass correlation",
    ):
        assert _estimate(shifted, label) == pytest.approx(_estimate(base, label), rel=1e-10)


# --------------------------------------------------------------------------- the edge cases


def test_five_rows_are_refused() -> None:
    outcome, predictor, labels = _fixture([2, 2, 1], 31)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("x", predictor),),
            group=_group("g", labels),
        )
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "6" in refused.value.message


def test_two_groups_are_refused() -> None:
    outcome, predictor, labels = _fixture([6, 6], 32)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("x", predictor),),
            group=_group("g", labels),
        )
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "3" in refused.value.message


def test_a_constant_predictor_is_refused() -> None:
    outcome, _predictor, labels = _fixture([4, 4, 4], 33)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("flat", np.ones(outcome.size)),),
            group=_group("g", labels),
        )
    assert refused.value.code == "CONSTANT_COLUMN"
    assert "flat" in refused.value.message


def test_collinear_predictors_are_refused() -> None:
    outcome, predictor, labels = _fixture([4, 4, 4], 34)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", outcome),
            predictors=(_column("x", predictor), _column("twice", 2.0 * predictor)),
            group=_group("g", labels),
        )
    assert refused.value.code == "COLLINEAR"
    assert "x and twice" in refused.value.message


def test_a_constant_outcome_is_refused() -> None:
    _outcome, predictor, labels = _fixture([4, 4, 4], 35)
    with pytest.raises(ToolError) as refused:
        _run(
            outcome=_column("y", np.ones(predictor.size)),
            predictors=(_column("x", predictor),),
            group=_group("g", labels),
        )
    assert refused.value.code == "CONSTANT_COLUMN"


def test_a_group_of_one_is_a_warning_not_a_refusal() -> None:
    outcome, predictor, labels = _fixture([1, 3, 3, 3], 36)
    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
    )
    joined = " ".join(result.warnings)

    assert "one row" in joined
    assert "no information about the between-group variance" in joined
    assert "fewer than 5 rows" in joined


def test_a_between_group_variance_at_zero_is_reported_and_warned() -> None:
    rng = np.random.default_rng(37)
    sizes = [6] * 10
    labels = [f"g{index}" for index, size in enumerate(sizes) for _ in range(size)]
    # The same six rows in every group: the group means are identical, so the between-group
    # sample variance is exactly zero and the fit puts tau squared at its boundary.
    predictor = np.tile(rng.normal(size=6), len(sizes))
    outcome = 2.0 + 1.5 * predictor + np.tile(rng.normal(scale=1.0, size=6), len(sizes))

    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
    )
    test = next(test for test in result.tests if test.name.startswith("Likelihood-ratio"))

    assert _estimate(result, "Between-group variance (tau squared)") == 0.0
    assert test.statistic == 0.0 and test.p_value == 1.0
    assert any("estimated at zero" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- the contract


def test_rows_are_counted_and_listed_exactly() -> None:
    size = 12
    outcome = np.random.default_rng(51).normal(size=size)
    predictor = np.arange(size, dtype=np.float64)
    outcome[0] = np.nan
    predictor[1] = np.inf
    group_rows = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    group_labels = ["a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
    kept = 9

    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", group_labels, rows=group_rows),
    )

    assert result.counts.n_used == kept
    assert result.counts.n_excluded == 3
    assert result.counts.n_offered == kept + 3 == size
    assert result.excluded == (
        ExcludedRow(index=0, reason="missing in y"),
        ExcludedRow(index=1, reason="not a finite number in x"),
        ExcludedRow(index=2, reason="no group label in g"),
    )
    assert result.excluded_unlisted == 0


def test_predictions_are_bounded_shaped_and_wider_for_a_new_group() -> None:
    outcome, predictor, labels = _fixture([6] * 8, 11)
    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
        predictions=((0.0,), (1.0,)),
    )
    fixed = _keyed(result, "Fixed effects")
    rows = _table(result, "Predictions")

    assert len(rows) == 2
    first = dict(zip([column.key for column in result.tables[-1].columns], rows[0], strict=True))
    assert first["predicted"] == pytest.approx(
        fixed["(Intercept)"]["estimate"],
        rel=1e-12,  # type: ignore[arg-type]
    )
    assert first["ci_lower"] < first["predicted"] < first["ci_upper"]  # type: ignore[operator]
    assert first["pi_lower"] < first["ci_lower"]  # type: ignore[operator]
    assert first["pi_upper"] > first["ci_upper"]  # type: ignore[operator]


def test_predictions_are_refused_over_a_thousand_rows_or_malformed() -> None:
    outcome, predictor, labels = _fixture([6] * 8, 11)
    base: dict[str, object] = {
        "outcome": _column("y", outcome),
        "predictors": (_column("x", predictor),),
        "group": _group("g", labels),
    }
    with pytest.raises(ToolError) as too_many:
        _run(**base, predictions=((0.0,),) * 1_001)
    assert too_many.value.code == "INVALID_OPTION"
    with pytest.raises(ToolError) as wrong_width:
        _run(**base, predictions=((0.0, 1.0),))
    assert wrong_width.value.code == "INVALID_OPTION"
    with pytest.raises(ToolError) as not_finite:
        _run(**base, predictions=((np.inf,),))
    assert not_finite.value.code == "INVALID_OPTION"


def test_the_group_table_is_capped_at_two_hundred_groups() -> None:
    groups = 201
    labels = [f"g{index:03d}" for index in range(groups) for _ in range(3)]
    outcome = np.random.default_rng(41).normal(size=len(labels))
    predictor = np.random.default_rng(42).normal(size=len(labels))

    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
    )
    table = next(table for table in result.tables if table.title == "Group effects")

    assert len(table.rows) == 200
    assert table.note is not None
    assert "first 200 of 201 groups" in table.note


def test_the_result_is_a_tool_result_round_trips_and_holds_no_nan() -> None:
    outcome, predictor, labels = _fixture([6] * 8, 11)
    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
        predictions=((0.5,),),
    )

    assert type(result) is ToolResult
    assert "NaN" not in result.model_dump_json()
    assert "Infinity" not in result.model_dump_json()
    assert ToolResult.model_validate_json(result.model_dump_json()) == result
    assert result.tool == "mixed" and result.level == 0.95


def test_the_tool_is_registered_in_the_advanced_family() -> None:
    info = next(item for item in catalogue() if item.name == "mixed")

    assert info.family == "advanced"
    assert info.title == "Linear mixed model"
    assert info.needs_data
    assert ("advanced", "Advanced models") in FAMILIES


def test_names_override_the_column_names() -> None:
    outcome, predictor, labels = _fixture([6] * 8, 11)
    result = _run(
        outcome=_column("y", outcome),
        predictors=(_column("x", predictor),),
        group=_group("g", labels),
        names=("Dose",),
    )
    fixed = _keyed(result, "Fixed effects")

    assert set(fixed) == {"(Intercept)", "Dose"}
    assert "Dose" in result.title


def test_the_request_refuses_bad_options() -> None:
    outcome, predictor, labels = _fixture([6] * 8, 11)
    base: dict[str, object] = {
        "outcome": _column("y", outcome),
        "predictors": (_column("x", predictor),),
        "group": _group("g", labels),
    }
    for level in (0.0, 1.0, -0.1):
        with pytest.raises(ToolError) as refused:
            _run(**base, level=level)
        assert refused.value.code == "INVALID_OPTION"
    with pytest.raises(ToolError) as no_predictors:
        _run(**{**base, "predictors": ()})
    assert no_predictors.value.code == "INVALID_OPTION"
    with pytest.raises(ToolError) as too_many_predictors:
        _run(**{**base, "predictors": tuple(_column(f"x{i}", predictor) for i in range(21))})
    assert too_many_predictors.value.code == "INVALID_OPTION"
    with pytest.raises(ToolError) as wrong_names:
        _run(**base, names=("a", "b"))
    assert wrong_names.value.code == "INVALID_OPTION"
    with pytest.raises(ValidationError):
        MixedRequest(**base, extra="field")  # type: ignore[arg-type]
