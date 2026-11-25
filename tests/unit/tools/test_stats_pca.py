"""The principal component analysis tool: eigenvalues, loadings, scores and the two checks.

Every number is checked against ``numpy.linalg.eigh`` on the same matrix the tool decomposes:
the correlation matrix of the standardised columns, and the covariance matrix of the centred
ones. The eigenvectors are sorted by descending eigenvalue and their signs are fixed the way
the tool fixes them (the largest absolute loading positive) before they are compared.

The worked example is the principal component analysis of the ``USArrests`` data that
Statology publishes at ``statology.org/principal-components-analysis-in-r/``, read on
2026-11-25: the ``results$rotation`` table after its sign flip (the loadings of PC1 to PC4),
``results$sdev^2 / sum(results$sdev^2)`` (the proportions of variance), and
``head(results$x)`` after its sign flip (the scores of the first six states). The raw 50 by 4
dataset is the CSV that Rdatasets serves for R's ``datasets::USArrests`` at
``vincentarelbundock.github.io/Rdatasets/csv/datasets/USArrests.csv``, read on 2026-11-25;
the values are reproduced here only so the test runs without the network. The NIST/SEMATECH
e-Handbook's principal components pages (``pmc55.htm`` and ``pmc551.htm``, read on
2026-11-25) were checked first and hold no tabulated worked example, so the Statology
walkthrough is used instead.

The tool's properties are checked too: the standardised eigenvalues sum to the number of
columns, the covariance eigenvalues to the total variance, the loadings are orthonormal,
running twice gives the same JSON, and a strictly increasing linear rescaling of every column
leaves the standardised solution unchanged.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    MultiLinePlot,
    ScatterPlot,
    ToolResult,
    Verdict,
    build_request,
    catalogue,
    run_tool,
)
from curvestudio.tools.stats.pca import (
    PcaRequest,
    _bartlett,
    _kaiser_meyer_olkin,
    run_pca,
)
from curvestudio.tools.stats.registry import TOOL_MODULES

STAT_RTOL = 1e-10
"""The tolerance on an eigenvalue, a percentage or a loading from the same decomposition."""

PUBLISHED_RTOL = 1e-6
"""The tolerance on a number the published table prints to seven decimals."""

_USARRESTS = """\
Alabama,13.2,236,58,21.2
Alaska,10,263,48,44.5
Arizona,8.1,294,80,31
Arkansas,8.8,190,50,19.5
California,9,276,91,40.6
Colorado,7.9,204,78,38.7
Connecticut,3.3,110,77,11.1
Delaware,5.9,238,72,15.8
Florida,15.4,335,80,31.9
Georgia,17.4,211,60,25.8
Hawaii,5.3,46,83,20.2
Idaho,2.6,120,54,14.2
Illinois,10.4,249,83,24
Indiana,7.2,113,65,21
Iowa,2.2,56,57,11.3
Kansas,6,115,66,18
Kentucky,9.7,109,52,16.3
Louisiana,15.4,249,66,22.2
Maine,2.1,83,51,7.8
Maryland,11.3,300,67,27.8
Massachusetts,4.4,149,85,16.3
Michigan,12.1,255,74,35.1
Minnesota,2.7,72,66,14.9
Mississippi,16.1,259,44,17.1
Missouri,9,178,70,28.2
Montana,6,109,53,16.4
Nebraska,4.3,102,62,16.5
Nevada,12.2,252,81,46
New Hampshire,2.1,57,56,9.5
New Jersey,7.4,159,89,18.8
New Mexico,11.4,285,70,32.1
New York,11.1,254,86,26.1
North Carolina,13,337,45,16.1
North Dakota,0.8,45,44,7.3
Ohio,7.3,120,75,21.4
Oklahoma,6.6,151,68,20
Oregon,4.9,159,67,29.3
Pennsylvania,6.3,106,72,14.9
Rhode Island,3.4,174,87,8.3
South Carolina,14.4,279,48,22.5
South Dakota,3.8,86,45,12.8
Tennessee,13.2,188,59,26.9
Texas,12.7,201,80,25.5
Utah,3.2,120,80,22.9
Vermont,2.2,48,32,11.2
Virginia,8.5,156,63,20.7
Washington,4,145,73,26.2
West Virginia,5.7,81,39,9.3
Wisconsin,2.6,53,66,10.8
Wyoming,6.8,161,60,15.6
"""

_NAMES = ("Murder", "Assault", "UrbanPop", "Rape")

_PUBLISHED_PROPORTIONS = (0.62006039, 0.24744129, 0.08914080, 0.04335752)

_PUBLISHED_LOADINGS = (
    (0.5358995, 0.5831836, 0.2781909, 0.5434321),
    (-0.4181809, -0.1879856, 0.8728062, 0.1673186),
)
"""The loadings of PC1 and PC2, in the order Murder, Assault, UrbanPop, Rape."""

_PUBLISHED_OTHER_LOADINGS = (
    (0.3412327, 0.2681484, 0.3780158, 0.8177779),
    (0.64922780, 0.74340748, 0.13387773, 0.08902432),
)
"""The magnitudes of PC3 and PC4; the tool's sign convention negates the page's PC3."""

_PUBLISHED_SCORES = (
    (0.9756604, -1.1220012),
    (1.9305379, -1.0624269),
    (1.7454429, 0.7384595),
    (-0.1399989, -1.1085423),
    (2.4986128, 1.5274267),
    (1.4993407, 0.9776297),
)
"""The PC1 and PC2 scores of the first six states, in the CSV's order."""


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_pca(PcaRequest(**options))  # type: ignore[arg-type]


def _table(result: ToolResult, title: str) -> list[tuple[object, ...]]:
    return list(next(table for table in result.tables if table.title == title).rows)


def _sample(n: int = 40, seed: int = 0, columns: int = 4) -> np.ndarray:
    """Correlated columns drawn from two shared factors, plus their own noise."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    data = np.column_stack(
        (
            factors[:, 0] + 0.2 * rng.normal(size=n),
            factors[:, 0] - factors[:, 1] + 0.2 * rng.normal(size=n),
            factors[:, 1] + 0.2 * rng.normal(size=n),
            factors.sum(axis=1) + rng.normal(size=n),
            rng.normal(size=n),
        )
    )
    return data[:, :columns]


def _columns(data: np.ndarray) -> tuple[Column, ...]:
    return tuple(_column(f"x{index + 1}", data[:, index]) for index in range(data.shape[1]))


def _usa_arrests() -> tuple[Column, ...]:
    """The R ``USArrests`` data as the Rdatasets CSV serves it, one column per variable."""
    rows = [line.split(",") for line in _USARRESTS.strip().splitlines()]
    return tuple(
        _column(name, [float(row[index + 1]) for row in rows]) for index, name in enumerate(_NAMES)
    )


def _eigh_reference(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eigenvalues and sign-fixed eigenvectors of a symmetric matrix, descending."""
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    for index in range(vectors.shape[1]):
        largest = int(np.argmax(np.abs(vectors[:, index])))
        if vectors[largest, index] < 0.0:
            vectors[:, index] = -vectors[:, index]
    return values, vectors


# --------------------------------------------------------------------------- the references


@pytest.mark.parametrize("standardise", [True, False])
def test_the_components_are_the_eigh_decomposition_of_the_matrix(standardise: bool) -> None:
    data = _sample(seed=1)
    matrix = np.corrcoef(data, rowvar=False) if standardise else np.cov(data, rowvar=False, ddof=1)
    reference_values, reference_vectors = _eigh_reference(matrix)
    reference_percentages = reference_values / reference_values.sum() * 100.0
    reference_cumulative = np.cumsum(reference_percentages)

    result = _run(columns=_columns(data), standardise=standardise)
    rows = _table(result, "Components")

    for index, row in enumerate(rows):
        assert row[1] == pytest.approx(float(reference_values[index]), abs=STAT_RTOL)
        assert row[2] == pytest.approx(float(reference_percentages[index]), abs=STAT_RTOL)
        assert row[3] == pytest.approx(float(reference_cumulative[index]), abs=STAT_RTOL)
        for column in range(data.shape[1]):
            assert row[4 + column] == pytest.approx(
                float(reference_vectors[column, index]), abs=STAT_RTOL
            )


def test_the_usa_arrests_analysis_reproduces_the_published_walkthrough() -> None:
    result = _run(columns=_usa_arrests())
    rows = _table(result, "Components")

    assert [row[0] for row in rows] == [1, 2, 3, 4]
    for index, proportion in enumerate(_PUBLISHED_PROPORTIONS):
        assert rows[index][2] == pytest.approx(proportion * 100.0, abs=PUBLISHED_RTOL)
    for component, loadings in enumerate(_PUBLISHED_LOADINGS):
        for column, expected in enumerate(loadings):
            assert rows[component][4 + column] == pytest.approx(expected, abs=PUBLISHED_RTOL)
    for component, loadings in enumerate(_PUBLISHED_OTHER_LOADINGS, start=2):
        for column, expected in enumerate(loadings):
            found = abs(float(rows[component][4 + column]))  # type: ignore[arg-type]
            assert found == pytest.approx(expected, abs=PUBLISHED_RTOL)

    scores = result.plots[1]
    assert isinstance(scores, ScatterPlot)
    for index, (first, second) in enumerate(_PUBLISHED_SCORES):
        assert scores.x[index] == pytest.approx(first, abs=PUBLISHED_RTOL)
        assert scores.y[index] == pytest.approx(second, abs=PUBLISHED_RTOL)

    assert result.estimates[0].value == pytest.approx(
        _PUBLISHED_PROPORTIONS[0] * 100.0, abs=PUBLISHED_RTOL
    )
    assert result.estimates[1].value == pytest.approx(
        sum(_PUBLISHED_PROPORTIONS[:2]) * 100.0, abs=PUBLISHED_RTOL
    )
    assert result.estimates[2].value == 2.0


def test_bartlett_and_kmo_match_their_definitions() -> None:
    data = _sample(seed=2)
    correlation = np.corrcoef(data, rowvar=False)
    size = data.shape[1]
    n = data.shape[0]

    found = _bartlett(correlation, n)
    _sign, log_determinant = np.linalg.slogdet(correlation)
    expected = -(n - 1.0 - (2.0 * size + 5.0) / 6.0) * float(log_determinant)
    assert found.statistic == pytest.approx(float(expected), rel=1e-12)
    assert found.df == size * (size - 1) / 2
    assert found.p_value == pytest.approx(float(scipy_stats.chi2.sf(expected, found.df)), rel=1e-12)

    inverse = np.linalg.inv(correlation)
    diagonal = np.sqrt(np.diag(inverse))
    partial = -inverse / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 0.0)
    off_diagonal = ~np.eye(size, dtype=bool)
    shared = float(np.sum(correlation[off_diagonal] ** 2))
    unique = float(np.sum(partial[off_diagonal] ** 2))

    assert _kaiser_meyer_olkin(correlation) == pytest.approx(shared / (shared + unique), rel=1e-12)


# --------------------------------------------------------------------------- the properties


def test_the_standardised_eigenvalues_sum_to_the_number_of_columns() -> None:
    data = _sample(seed=3)

    result = _run(columns=_columns(data))

    total = sum(float(row[1]) for row in _table(result, "Components"))  # type: ignore[arg-type]
    assert total == pytest.approx(data.shape[1], abs=STAT_RTOL)


def test_the_covariance_eigenvalues_sum_to_the_total_variance() -> None:
    data = _sample(seed=4)

    result = _run(columns=_columns(data), standardise=False)

    total = sum(float(row[1]) for row in _table(result, "Components"))  # type: ignore[arg-type]
    assert total == pytest.approx(float(np.var(data, axis=0, ddof=1).sum()), rel=STAT_RTOL)


def test_the_loadings_are_orthonormal() -> None:
    data = _sample(seed=5)

    result = _run(columns=_columns(data))

    rows = _table(result, "Components")
    loadings = np.array(
        [[float(row[4 + index]) for index in range(data.shape[1])] for row in rows],  # type: ignore[arg-type]
        dtype=np.float64,
    )
    assert np.allclose(loadings @ loadings.T, np.eye(data.shape[1]), atol=STAT_RTOL)


def test_running_twice_gives_the_same_json() -> None:
    data = _sample(seed=6)

    first = _run(columns=_columns(data))
    second = _run(columns=_columns(data))

    assert first.model_dump_json() == second.model_dump_json()


def test_a_strictly_increasing_rescaling_leaves_the_standardised_solution() -> None:
    data = _sample(seed=7)
    scales = np.array([3.0, 0.5, 7.0, 2.0])
    shifts = np.array([10.0, -4.0, 0.0, 100.0])

    plain = _run(columns=_columns(data))
    rescaled = _run(columns=_columns(data * scales + shifts))

    plain_rows = _table(plain, "Components")
    rescaled_rows = _table(rescaled, "Components")
    for plain_row, rescaled_row in zip(plain_rows, rescaled_rows, strict=True):
        for index in range(1, len(plain_row)):
            assert rescaled_row[index] == pytest.approx(
                float(plain_row[index]),
                rel=1e-9,
                abs=1e-12,  # type: ignore[arg-type]
            )
    plain_scores = plain.plots[1]
    rescaled_scores = rescaled.plots[1]
    assert isinstance(plain_scores, ScatterPlot)
    assert isinstance(rescaled_scores, ScatterPlot)
    assert rescaled_scores.x == pytest.approx(plain_scores.x, rel=1e-9, abs=1e-12)
    assert rescaled_scores.y == pytest.approx(plain_scores.y, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- the edge cases


def test_three_rows_are_refused() -> None:
    columns = (_column("a", [1.0, 2.0, 3.0]), _column("b", [2.0, 4.0, 7.0]))

    with pytest.raises(ToolError) as refused:
        _run(columns=columns)

    assert refused.value.code == "TOO_FEW_ROWS"


def test_a_constant_column_is_refused_when_standardised() -> None:
    data = _sample(n=12, seed=8)
    columns = (*_columns(data), _column("constant", [5.0] * 12))

    with pytest.raises(ToolError) as refused:
        _run(columns=columns)

    assert refused.value.code == "CONSTANT_COLUMN"
    assert "constant" in refused.value.message


def test_a_constant_column_is_dropped_with_a_warning_when_not_standardised() -> None:
    data = _sample(n=12, seed=9, columns=3)
    columns = (*_columns(data), _column("constant", [5.0] * 12))

    result = _run(columns=columns, standardise=False)

    labels = [column.label for column in result.tables[0].columns]
    assert labels[4:] == ["x1", "x2", "x3"]
    assert len(result.tables[0].rows) == 3
    assert any("'constant'" in warning for warning in result.warnings)
    assert result.counts == Counts(n_offered=12, n_used=12, n_excluded=0)


def test_one_column_or_more_than_thirty_is_refused() -> None:
    rng = np.random.default_rng(10)
    with pytest.raises(ToolError) as refused:
        _run(columns=(_column("only", rng.normal(size=12)),))
    assert refused.value.code == "INVALID_OPTION"

    many = tuple(_column(f"x{index}", rng.normal(size=12)) for index in range(31))
    with pytest.raises(ToolError) as refused:
        _run(columns=many)
    assert refused.value.code == "INVALID_OPTION"
    assert "between 2 and 30" in refused.value.message


def test_one_component_limits_the_tables_and_the_plots() -> None:
    data = _sample(n=20, seed=11)

    result = _run(columns=_columns(data), max_components=1)

    assert len(_table(result, "Components")) == 1
    assert len(_table(result, "Scores summary")) == 1
    assert [type(plot) for plot in result.plots] == [MultiLinePlot]
    scree = result.plots[0]
    assert isinstance(scree, MultiLinePlot)
    assert len(scree.series[0].x) == 1
    assert len(scree.series[0].y) == 1
    assert any("Only the first 1" in warning for warning in result.warnings)


def test_an_out_of_range_component_count_is_refused() -> None:
    data = _sample(n=12, seed=12)

    for count in (0, 5):
        with pytest.raises(ToolError) as refused:
            _run(columns=_columns(data), max_components=count)
        assert refused.value.code == "INVALID_OPTION"


def test_a_column_entirely_missing_leaves_no_rows() -> None:
    values = np.arange(8.0)
    columns = (_column("a", values), _column("b", [float("nan")] * 8))

    with pytest.raises(ToolError) as refused:
        _run(columns=columns)

    assert refused.value.code == "TOO_FEW_ROWS"


def test_rows_missing_in_a_column_are_excluded_and_listed() -> None:
    first = _column("a", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    second = _column("b", [2.0, 3.0, float("nan"), 5.0, 7.0, 8.0, 9.0, 10.0])

    result = _run(columns=(first, second))

    assert result.counts == Counts(n_offered=8, n_used=7, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="missing in b"),)


def test_fewer_than_ten_rows_are_warned_about() -> None:
    data = _sample(n=6, seed=13, columns=2)

    result = _run(columns=_columns(data))

    assert any("Only 6 rows" in warning for warning in result.warnings)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    data = _sample(n=12, seed=14, columns=2)

    with pytest.raises(ToolError) as refused:
        _run(columns=_columns(data), level=level)

    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_pca(object())  # type: ignore[arg-type]

    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_option_is_refused_by_the_model() -> None:
    data = _sample(n=12, seed=15, columns=2)

    with pytest.raises(ValidationError):
        PcaRequest(  # type: ignore[call-arg]
            columns=_columns(data), method="varimax"
        )


# ---------------------------------------------------------------------- the words and the registry


def test_the_result_is_the_contract() -> None:
    data = _sample(seed=16)

    result = _run(columns=_columns(data))

    assert type(result) is ToolResult
    assert result.tool == "pca"
    assert result.title == "Principal component analysis"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=40, n_used=40, n_excluded=0)
    assert result.excluded == ()
    assert result.excluded_unlisted == 0
    assert result.tests == ()
    assert [estimate.label for estimate in result.estimates] == [
        "Variance explained by the first component",
        "Variance explained by the first two components",
        "Number of components for 80 % of variance",
    ]
    assert [check.id for check in result.assumptions] == ["sphericity", "factorability"]
    assert [table.title for table in result.tables] == ["Components", "Scores summary"]
    assert [column.label for column in result.tables[0].columns] == [
        "Component",
        "Eigenvalue",
        "Variance explained %",
        "Cumulative %",
        "x1",
        "x2",
        "x3",
        "x4",
    ]
    assert [type(plot) for plot in result.plots] == [MultiLinePlot, ScatterPlot]
    assert result.plots[0].title == "Scree plot"
    assert result.plots[1].title == "Scores on the first two components"


def test_the_result_round_trips_and_holds_no_non_finite_number() -> None:
    result = _run(columns=_usa_arrests())

    document = result.model_dump_json()

    assert "NaN" not in document
    assert "Infinity" not in document
    assert ToolResult.model_validate_json(document) == result


def test_the_fetched_checks_read_the_correlation_of_the_usa_arrests() -> None:
    result = _run(columns=_usa_arrests())

    sphericity = result.assumptions[0]
    assert sphericity.verdict is Verdict.FAIL
    assert "strongly correlated" in sphericity.title
    assert sphericity.statistic is not None and sphericity.statistic.startswith("chi-square")

    factorability = result.assumptions[1]
    assert factorability.verdict is Verdict.CAUTION
    assert factorability.statistic is not None and factorability.statistic.startswith("KMO")


def test_the_interpretation_names_the_largest_loading_of_the_first_two_components() -> None:
    result = _run(columns=_usa_arrests())

    assert "Assault" in result.interpretation
    assert "UrbanPop" in result.interpretation
    assert "not a cause" in result.interpretation


def test_the_words_state_evidence_and_claim_no_proof() -> None:
    result = _run(columns=_usa_arrests())

    texts = [result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])

    for text in texts:
        assert not claims_proof(text), text


def test_every_request_field_is_described_and_the_options_are_labelled() -> None:
    found = next(info for info in catalogue() if info.name == "pca")

    assert found.family == "relationships"
    assert found.title == "Principal component analysis"
    assert [field.name for field in found.fields] == [
        "columns",
        "standardise",
        "max_components",
        "level",
    ]
    assert all(field.description for field in found.fields)
    assert found.fields[0].kind == "columns"
    assert found.fields[1].label == "Standardise the columns"
    assert found.fields[2].label == "Most components to show"


def test_the_registry_builds_and_runs_the_tool_from_column_names() -> None:
    data = _sample(n=15, seed=17, columns=2)
    columns = {"x1": _column("x1", data[:, 0]), "x2": _column("x2", data[:, 1])}

    def group(_name: str) -> object:
        raise AssertionError("pca takes no group column")

    request = build_request(
        "pca",
        {"columns": ["x1", "x2"], "standardise": True, "max_components": 1, "level": 0.9},
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    result = run_tool("pca", request)

    assert isinstance(request, PcaRequest)
    assert result.tool == "pca"
    assert result.level == 0.9
    assert len(result.tables[0].rows) == 1
    assert "curvestudio.tools.stats.pca" in TOOL_MODULES
