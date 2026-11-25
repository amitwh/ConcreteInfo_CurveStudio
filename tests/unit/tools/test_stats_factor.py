"""The factor analysis tool: principal-axis loadings, varimax rotation and the scores.

The unrotated loadings and communalities are checked against
``statsmodels.multivariate.factor.Factor`` with the same extraction (``method="pa"``, squared
multiple correlations as the start). The rotation is checked against the same library's
varimax applied to Kaiser-normalised loadings, and against an independent implementation of
the published SVD iteration in this file.

The published worked examples are from William Revelle's *An introduction to psychometric
theory, with applications in R*, chapter 6, fetched from
``personality-project.org/r/book/Chapter6.pdf`` and read on 2026-11-25:

- Table 6.1 and Table 6.4: the six-variable single-factor model ``f = (.9, .8, .7, .6, .5,
  .4)``, whose correlation matrix is ``f f'`` with ones on the diagonal; ``factor.pa`` gives
  loadings ``.9 ... .4`` and communalities ``.81 ... .16``.
- Table 6.10 and Table 6.11: the eight physical variables of Harman (1976), with the
  unrotated maximum-likelihood loadings of Table 6.10 and their varimax rotation in Table
  6.11. The rotation is checked from the published unrotated loadings; the 3-decimal input
  limits the agreement to about 0.01.

The properties are checked too: rotation leaves communalities unchanged, varimax never
lowers its criterion, running twice gives the same JSON, scaling every column by a positive
constant leaves the correlation-based solution unchanged, and the scores follow Thurstone's
regression method.
"""

import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.factor_rotation import rotate_factors

from curvestudio.data import Column
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    ScatterPlot,
    ToolResult,
    Verdict,
    build_request,
    catalogue,
    run_tool,
)
from curvestudio.tools.stats.factor import (
    FactorRequest,
    _ordered_and_signed,
    _principal_axis,
    _varimax,
    _varimax_criterion,
    run_factor,
)
from curvestudio.tools.stats.pca import _bartlett, _kaiser_meyer_olkin
from curvestudio.tools.stats.registry import TOOL_MODULES

STAT_RTOL = 1e-6
"""The tolerance on a loading or communality from the same extraction."""

PUBLISHED_RTOL = 1e-5
"""The tolerance on the published single-factor table, whose iteration stops at 1e-6."""

PUBLISHED_ROTATION_RTOL = 0.01
"""The tolerance on the published rotation, limited by its 3-decimal unrotated input."""

_PUBLISHED_FACTOR = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
"""The single-factor model of chapter 6, Tables 6.1 and 6.4."""

_PUBLISHED_LOADINGS = (
    (0.880, -0.237),
    (0.874, -0.360),
    (0.846, -0.344),
    (0.855, -0.263),
    (0.705, 0.644),
    (0.589, 0.538),
    (0.526, 0.554),
    (0.574, 0.365),
)
"""Table 6.10: the unrotated maximum-likelihood loadings of Harman's 8 variables."""

_PUBLISHED_ROTATED = (
    (0.865, 0.287),
    (0.927, 0.181),
    (0.895, 0.179),
    (0.859, 0.252),
    (0.233, 0.925),
    (0.194, 0.774),
    (0.134, 0.752),
    (0.278, 0.621),
)
"""Table 6.11: the varimax rotation of the loadings above."""

_HEYWOOD_CORRELATION = np.array(
    [
        [1.0000, 0.3897, -0.4386, -0.5792, -0.6104],
        [0.3897, 1.0000, 0.0063, -0.8714, -0.0731],
        [-0.4386, 0.0063, 1.0000, -0.1640, 0.0669],
        [-0.5792, -0.8714, -0.1640, 1.0000, 0.2321],
        [-0.6104, -0.0731, 0.0669, 0.2321, 1.0000],
    ]
)
"""A correlation matrix whose two-factor principal-axis solution has Heywood communalities."""


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _columns(data: np.ndarray) -> tuple[Column, ...]:
    return tuple(_column(f"x{index + 1}", data[:, index]) for index in range(data.shape[1]))


def _run(**options: object) -> ToolResult:
    return run_factor(FactorRequest(**options))  # type: ignore[arg-type]


def _table(result: ToolResult, title: str) -> list[tuple[object, ...]]:
    return list(next(table for table in result.tables if table.title == title).rows)


def _sample(n: int = 200, seed: int = 0, columns: int = 6) -> np.ndarray:
    """Correlated columns drawn from two shared factors, plus their own noise."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    true = np.array(
        [
            [0.8, 0.1],
            [0.7, 0.2],
            [0.6, 0.0],
            [0.1, 0.8],
            [0.2, 0.7],
            [0.0, 0.6],
        ]
    )
    data = factors @ true.T + rng.normal(scale=0.5, size=(n, 6))
    return data[:, :columns]


def _data_with_correlation(correlation: np.ndarray, n: int = 40, seed: int = 0) -> np.ndarray:
    """Rows whose sample correlation matrix is ``correlation``, exactly.

    An orthonormal set of columns with zero mean (a QR decomposition of centred noise) has
    the identity correlation matrix; multiplying it by a square root of ``correlation``
    gives rows with that correlation.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, correlation.shape[0]))
    raw -= np.mean(raw, axis=0)
    orthonormal, _ = np.linalg.qr(raw)
    standardised = orthonormal * math.sqrt(n - 1)
    root = np.linalg.cholesky(correlation).T
    return standardised @ root


def _loadings(result: ToolResult) -> np.ndarray:
    """The factor loadings of the result's table, one column per factor."""
    rows = _table(result, "Factor loadings")
    factors = len(result.tables[0].columns) - 3
    return np.array(
        [[float(row[1 + index]) for index in range(factors)] for row in rows],  # type: ignore[arg-type]
        dtype=np.float64,
    )


def _varimax_svd(loadings: np.ndarray, maxit: int = 5_000, tol: float = 1e-15) -> np.ndarray:
    """The published SVD varimax iteration (R's ``stats::varimax``), Kaiser normalised."""
    rows, factors = loadings.shape
    norms = np.sqrt((loadings**2).sum(axis=1))
    scaled = loadings / norms[:, None]
    turn = np.eye(factors)
    previous = 0.0
    for _ in range(maxit):
        turned = scaled @ turn
        matrix = scaled.T @ (turned**3 - turned @ np.diag((turned**2).sum(axis=0)) / rows)
        left, singular, right = np.linalg.svd(matrix)
        turn = left @ right
        current = float(singular.sum())
        if current < previous - tol or abs(current - previous) < tol:
            break
        previous = current
    return (scaled @ turn) * norms[:, None]


def _aligned(reference: np.ndarray, found: np.ndarray) -> np.ndarray:
    """``found`` with each column sign-matched to the corresponding reference column."""
    aligned = found.copy()
    for index in range(aligned.shape[1]):
        if float(np.dot(aligned[:, index], reference[:, index])) < 0.0:
            aligned[:, index] = -aligned[:, index]
    return aligned


def _statsmodels(data: np.ndarray, factors: int) -> Any:
    """The principal-axis fit of statsmodels, with the same start and tolerance."""
    model = Factor(data, factors, method="pa", smc=True)
    return model.fit(maxiter=100, tol=1e-6)


# --------------------------------------------------------------------------- the references


def test_the_unrotated_loadings_and_communalities_match_statsmodels() -> None:
    data = _sample(seed=1)
    reference = _statsmodels(data, 2)

    result = _run(columns=_columns(data), factors=2, rotation="none")

    loadings = _loadings(result)
    expected = _aligned(loadings, np.asarray(reference.loadings_no_rot))  # type: ignore[attr-defined]
    assert loadings == pytest.approx(expected, abs=STAT_RTOL)
    rows = _table(result, "Factor loadings")
    for index, row in enumerate(rows):
        assert row[3] == pytest.approx(
            float(reference.communality[index]),
            abs=STAT_RTOL,  # type: ignore[attr-defined]
        )
        assert row[4] == pytest.approx(
            1.0 - float(reference.communality[index]),
            abs=STAT_RTOL,  # type: ignore[attr-defined]
        )


def test_the_varimax_rotation_matches_statsmodels_with_kaiser_normalisation() -> None:
    data = _sample(seed=2)
    reference = _statsmodels(data, 2)
    unrotated = np.asarray(reference.loadings_no_rot)  # type: ignore[attr-defined]
    norms = np.sqrt((unrotated**2).sum(axis=1))
    normalised = unrotated / norms[:, None]
    turned = rotate_factors(normalised, "varimax", tol=1e-12, max_tries=5_000)[0]
    expected = _ordered_and_signed(turned * norms[:, None])

    result = _run(columns=_columns(data), factors=2, rotation="varimax")

    assert _loadings(result) == pytest.approx(expected, abs=STAT_RTOL)


def test_the_rotation_matches_an_independent_varimax_oracle() -> None:
    data = _sample(seed=3)
    correlation = np.corrcoef(data, rowvar=False)
    solution = _principal_axis(correlation, 2)

    ours = _ordered_and_signed(_varimax(solution.loadings))
    expected = _ordered_and_signed(_varimax_svd(solution.loadings))

    assert ours == pytest.approx(expected, abs=STAT_RTOL)


def test_the_published_single_factor_example() -> None:
    model = np.array(_PUBLISHED_FACTOR)
    correlation = np.outer(model, model)
    np.fill_diagonal(correlation, 1.0)
    data = _data_with_correlation(correlation, n=40, seed=4)

    result = _run(columns=_columns(data), factors=1, rotation="none")

    rows = _table(result, "Factor loadings")
    for index, loading in enumerate(_PUBLISHED_FACTOR):
        assert rows[index][1] == pytest.approx(loading, abs=PUBLISHED_RTOL)
        assert rows[index][2] == pytest.approx(loading**2, abs=PUBLISHED_RTOL)
        assert rows[index][3] == pytest.approx(1.0 - loading**2, abs=PUBLISHED_RTOL)


def test_the_published_varimax_rotation() -> None:
    unrotated = np.array(_PUBLISHED_LOADINGS)

    found = _ordered_and_signed(_varimax(unrotated))

    assert found == pytest.approx(np.array(_PUBLISHED_ROTATED), abs=PUBLISHED_ROTATION_RTOL)


# --------------------------------------------------------------------------- the properties


def test_rotation_leaves_the_communalities_unchanged() -> None:
    data = _sample(seed=5)
    correlation = np.corrcoef(data, rowvar=False)
    solution = _principal_axis(correlation, 2)

    before = (solution.loadings**2).sum(axis=1)
    after = (_varimax(solution.loadings) ** 2).sum(axis=1)

    assert after == pytest.approx(before, abs=1e-12)


def test_varimax_never_lowers_its_criterion() -> None:
    data = _sample(seed=6)
    correlation = np.corrcoef(data, rowvar=False)
    solution = _principal_axis(correlation, 2)
    norms = np.sqrt((solution.loadings**2).sum(axis=1))
    normalised = solution.loadings / norms[:, None]

    turned = _varimax(solution.loadings)
    turned_norms = np.sqrt((turned**2).sum(axis=1))
    turned_normalised = turned / turned_norms[:, None]

    assert _varimax_criterion(turned_normalised) >= (_varimax_criterion(normalised) - 1e-12)


def test_running_twice_gives_the_same_json() -> None:
    data = _sample(seed=7)

    first = _run(columns=_columns(data))
    second = _run(columns=_columns(data))

    assert first.model_dump_json() == second.model_dump_json()


def test_scaling_every_column_leaves_the_solution_unchanged() -> None:
    data = _sample(n=120, seed=8)
    scales = np.array([3.0, 0.5, 7.0, 2.0, 1.5, 0.25])

    plain = _run(columns=_columns(data))
    rescaled = _run(columns=_columns(data * scales))

    plain_rows = _table(plain, "Factor loadings")
    rescaled_rows = _table(rescaled, "Factor loadings")
    for plain_row, rescaled_row in zip(plain_rows, rescaled_rows, strict=True):
        for index in range(1, len(plain_row)):
            assert float(rescaled_row[index]) == pytest.approx(  # type: ignore[arg-type]
                float(plain_row[index]),
                rel=1e-9,
                abs=1e-12,  # type: ignore[arg-type]
            )
    plain_variance = _table(plain, "Variance explained")
    rescaled_variance = _table(rescaled, "Variance explained")
    for plain_row, rescaled_row in zip(plain_variance, rescaled_variance, strict=True):
        for index in range(1, len(plain_row)):
            assert float(rescaled_row[index]) == pytest.approx(  # type: ignore[arg-type]
                float(plain_row[index]),
                rel=1e-9,
                abs=1e-12,  # type: ignore[arg-type]
            )
    plain_scores = plain.plots[0]
    rescaled_scores = rescaled.plots[0]
    assert isinstance(plain_scores, ScatterPlot)
    assert isinstance(rescaled_scores, ScatterPlot)
    assert rescaled_scores.x == pytest.approx(plain_scores.x, rel=1e-9, abs=1e-12)
    assert rescaled_scores.y == pytest.approx(plain_scores.y, rel=1e-9, abs=1e-12)


def test_the_scores_follow_the_regression_method() -> None:
    data = _sample(n=60, seed=9)
    result = _run(columns=_columns(data), factors=2)
    loadings = _loadings(result)
    standardised = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    raw = standardised @ np.linalg.inv(np.corrcoef(data, rowvar=False)) @ loadings
    expected = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)

    plot = result.plots[0]
    assert isinstance(plot, ScatterPlot)
    assert plot.x == pytest.approx(expected[:, 0], rel=1e-9, abs=1e-12)
    assert plot.y == pytest.approx(expected[:, 1], rel=1e-9, abs=1e-12)


def test_the_variance_explained_adds_up_to_the_communalities() -> None:
    data = _sample(seed=10)

    result = _run(columns=_columns(data), factors=2)

    rows = _table(result, "Variance explained")
    loadings = _table(result, "Factor loadings")
    total_squares = sum(float(row[1]) for row in rows)  # type: ignore[arg-type]
    total_communalities = sum(float(row[3]) for row in loadings)  # type: ignore[arg-type]
    assert total_squares == pytest.approx(total_communalities, abs=1e-10)
    assert float(rows[-1][3]) == pytest.approx(  # type: ignore[arg-type]
        sum(float(row[2]) for row in rows),
        abs=1e-12,  # type: ignore[arg-type]
    )
    assert result.estimates[1].value == pytest.approx(
        total_squares / len(loadings) * 100.0, abs=1e-10
    )


def test_the_kmo_and_bartlett_match_their_definitions() -> None:
    data = _sample(seed=11)
    correlation = np.corrcoef(data, rowvar=False)

    result = _run(columns=_columns(data), factors=2)

    sphericity = _bartlett(correlation, data.shape[0])
    assert result.tests[0].statistic == pytest.approx(sphericity.statistic, rel=1e-12)  # type: ignore[arg-type]
    assert result.tests[0].df == sphericity.df
    assert result.tests[0].p_value == pytest.approx(sphericity.p_value, rel=1e-12)  # type: ignore[arg-type]
    assert result.estimates[0].value == pytest.approx(
        float(_kaiser_meyer_olkin(correlation)),
        rel=1e-12,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- the edge cases


def test_nine_rows_are_refused() -> None:
    data = _sample(n=9, seed=12, columns=3)

    with pytest.raises(ToolError) as refused:
        _run(columns=_columns(data))

    assert refused.value.code == "TOO_FEW_ROWS"


def test_a_constant_column_is_refused() -> None:
    data = _sample(n=20, seed=13, columns=3)
    columns = (*_columns(data), _column("constant", [5.0] * 20))

    with pytest.raises(ToolError) as refused:
        _run(columns=columns)

    assert refused.value.code == "CONSTANT_COLUMN"
    assert "constant" in refused.value.message


def test_a_factor_count_the_columns_cannot_carry_is_refused() -> None:
    data = _sample(n=40, seed=14, columns=4)

    for count in (0, 4, 11):
        with pytest.raises(ToolError) as refused:
            _run(columns=_columns(data), factors=count)
        assert refused.value.code == "INVALID_OPTION"


def test_one_fewer_factor_than_columns_is_accepted() -> None:
    data = _sample(n=80, seed=15, columns=4)

    result = _run(columns=_columns(data), factors=3)

    assert len(_table(result, "Factor loadings")[0]) == 3 + 3
    assert len(_table(result, "Variance explained")) == 3


def test_two_columns_or_more_than_thirty_are_refused() -> None:
    rng = np.random.default_rng(16)
    with pytest.raises(ToolError) as refused:
        _run(columns=tuple(_column(f"x{index}", rng.normal(size=20)) for index in range(2)))
    assert refused.value.code == "INVALID_OPTION"

    many = tuple(_column(f"x{index}", rng.normal(size=40)) for index in range(31))
    with pytest.raises(ToolError) as refused:
        _run(columns=many)
    assert refused.value.code == "INVALID_OPTION"
    assert "between 3 and 30" in refused.value.message


def test_a_heywood_case_is_warned_about() -> None:
    data = _data_with_correlation(_HEYWOOD_CORRELATION, n=40, seed=17)

    result = _run(columns=_columns(data), factors=2)

    assert any("Heywood" in warning for warning in result.warnings)
    assert any(float(row[3]) >= 1.0 for row in _table(result, "Factor loadings"))  # type: ignore[arg-type]


def test_rotation_none_is_named_and_shows_the_unrotated_loadings() -> None:
    data = _sample(seed=18)

    result = _run(columns=_columns(data), rotation="none")

    assert any("rotation='none'" in warning for warning in result.warnings)
    assert "no rotation" in result.methods_fragment


def test_one_factor_shows_no_scatter_plot() -> None:
    data = _sample(n=80, seed=19, columns=4)

    result = _run(columns=_columns(data), factors=1)

    assert result.plots == ()
    assert len(_table(result, "Factor loadings")[0]) == 1 + 3


def test_rows_missing_in_a_column_are_excluded_and_listed() -> None:
    data = _sample(n=20, seed=20, columns=3)
    values = data[:, 1].copy()
    values[4] = np.nan
    columns = (_column("a", data[:, 0]), _column("b", values), _column("c", data[:, 2]))

    result = _run(columns=columns, factors=2)

    assert result.counts == Counts(n_offered=20, n_used=19, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=4, reason="missing in b"),)


def test_fewer_than_five_rows_per_column_is_warned_about() -> None:
    data = _sample(n=12, seed=21, columns=3)

    result = _run(columns=_columns(data), factors=2)

    assert any("fewer than 5 rows per column" in warning for warning in result.warnings)


def test_an_outlier_heavy_column_is_a_caution() -> None:
    data = _sample(n=40, seed=22, columns=4)
    data[0:5, 0] = 100.0

    result = _run(columns=_columns(data), factors=2)

    check = result.assumptions[0]
    assert check.id == "outliers_1"
    assert check.verdict is Verdict.CAUTION
    assert "5 of 40" in check.evidence


def test_the_linearity_check_states_the_assumption() -> None:
    data = _sample(seed=23)

    result = _run(columns=_columns(data))

    check = result.assumptions[-1]
    assert check.id == "linearity"
    assert check.verdict is Verdict.NOT_APPLICABLE
    assert (
        "factor analysis assumes linear relations between the columns; look at the scatter "
        "plots" in check.evidence
    )


def test_collinear_columns_are_refused() -> None:
    data = _sample(n=30, seed=24, columns=2)
    columns = (
        _column("a", data[:, 0]),
        _column("b", data[:, 1]),
        _column("copy", data[:, 0]),
    )

    with pytest.raises(ToolError) as refused:
        _run(columns=columns, factors=2)

    assert refused.value.code == "COLLINEAR"


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    data = _sample(n=20, seed=25, columns=3)

    with pytest.raises(ToolError) as refused:
        _run(columns=_columns(data), level=level)

    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_factor(object())  # type: ignore[arg-type]

    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_option_is_refused_by_the_model() -> None:
    data = _sample(n=20, seed=26, columns=3)

    with pytest.raises(ValidationError):
        FactorRequest(columns=_columns(data), method="varimax")  # type: ignore[call-arg]


# ---------------------------------------------------------------------- the words and the registry


def test_the_result_is_the_contract() -> None:
    data = _sample(seed=27)

    result = _run(columns=_columns(data))

    assert type(result) is ToolResult
    assert result.tool == "factor"
    assert result.title == "Factor analysis"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=200, n_used=200, n_excluded=0)
    assert result.excluded == ()
    assert result.excluded_unlisted == 0
    assert result.estimates[0].label.startswith("Kaiser-Meyer-Olkin measure (")
    assert result.estimates[1].label == "Variance explained by the factors %"
    assert result.estimates[2].label == "Kaiser's criterion: factors with an eigenvalue above 1"
    assert [test.name for test in result.tests] == ["Bartlett's test of sphericity"]
    assert result.tests[0].df == 15.0
    assert result.tests[0].effect is not None
    assert result.tests[0].effect.label == "Chi-square"
    assert result.tests[0].effect.value == result.tests[0].statistic
    assert result.tests[0].p_value_text == "p < 0.0001"
    assert [check.id for check in result.assumptions] == [
        "outliers_1",
        "outliers_2",
        "outliers_3",
        "outliers_4",
        "outliers_5",
        "outliers_6",
        "linearity",
    ]
    assert [table.title for table in result.tables] == [
        "Factor loadings",
        "Variance explained",
        "Factor scores summary",
    ]
    assert [column.label for column in result.tables[0].columns] == [
        "Column",
        "Factor 1",
        "Factor 2",
        "Communality",
        "Uniqueness",
    ]
    assert [column.label for column in result.tables[1].columns] == [
        "Factor",
        "Sum of squared loadings",
        "Proportion",
        "Cumulative",
    ]
    assert [column.label for column in result.tables[2].columns] == [
        "Factor",
        "Mean",
        "SD",
        "Minimum",
        "Maximum",
    ]
    assert [type(plot) for plot in result.plots] == [ScatterPlot]
    assert result.plots[0].title == "Scores on the first two factors"
    assert result.plots[0].note is not None
    assert "loadings are in the Factor loadings table" in result.plots[0].note


def test_the_result_round_trips_and_holds_no_non_finite_number() -> None:
    data = _sample(seed=28)

    result = _run(columns=_columns(data))

    document = result.model_dump_json()
    assert "NaN" not in document
    assert "Infinity" not in document
    assert ToolResult.model_validate_json(document) == result


def test_the_words_state_evidence_and_claim_no_proof() -> None:
    data = _sample(seed=29)

    result = _run(columns=_columns(data))

    texts = [result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])
    for text in texts:
        assert not claims_proof(text), text


def test_the_interpretation_names_the_largest_loading_of_each_factor() -> None:
    data = _sample(seed=30)

    result = _run(columns=_columns(data), factors=2)

    assert "factor 1" in result.interpretation
    assert "factor 2" in result.interpretation
    assert "not a latent cause" in result.interpretation


def test_every_request_field_is_described_and_the_options_are_labelled() -> None:
    found = next(info for info in catalogue() if info.name == "factor")

    assert found.family == "relationships"
    assert found.title == "Factor analysis"
    assert [field.name for field in found.fields] == [
        "columns",
        "factors",
        "rotation",
        "level",
    ]
    assert all(field.description for field in found.fields)
    assert found.fields[0].kind == "columns"
    assert found.fields[1].label == "Number of factors"
    assert found.fields[2].label == "Rotation"


def test_the_registry_builds_and_runs_the_tool_from_column_names() -> None:
    data = _sample(n=60, seed=31, columns=3)
    columns = {f"x{index + 1}": _column(f"x{index + 1}", data[:, index]) for index in range(3)}

    def group(_name: str) -> object:
        raise AssertionError("factor takes no group column")

    request = build_request(
        "factor",
        {"columns": ["x1", "x2", "x3"], "factors": 1, "rotation": "none", "level": 0.9},
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    result = run_tool("factor", request)

    assert isinstance(request, FactorRequest)
    assert result.tool == "factor"
    assert result.level == 0.9
    assert result.plots == ()
    assert "curvestudio.tools.stats.factor" in TOOL_MODULES
