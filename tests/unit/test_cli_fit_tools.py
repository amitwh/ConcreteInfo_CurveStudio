"""``curvestudio discern``, ``simulate`` and ``advise``: the fit tools from the command line.

Each command prints the one :class:`~curvestudio.tools.stats.ToolResult` the core function
returns, so the same-numbers rule is pinned for all three: ``--json`` is compared, as a
parsed dict, with ``model_dump(mode="json")`` of the same call built in the test from the
same data, specification and request. The readable reports are checked to carry the title,
every table and the interpretation, ``discern`` is checked to run Auto-fit with the
``autofit`` command's defaults, and the usage and input errors are checked to exit ``3`` with
the tool's code on standard error.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from curvestudio.cli import main
from curvestudio.data import XY, Dataset, read_dataset, validate
from curvestudio.discovery import AutoFitConfig, AutoFitResult, autofit
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.tools.advisor import AdvisorRequest, advise
from curvestudio.tools.discernibility import DiscernibilityRequest, discernibility
from curvestudio.tools.feasibility import DesignSpec, NoiseSpec, SimulationRequest, simulate
from curvestudio.tools.stats import ToolResult
from curvestudio.workflow import FitSpec, PolynomialRef, RegisteredRef, fit_dataset


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    """The starter models, as the CLI fit tests expect them."""
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()


@pytest.fixture(scope="module")
def data_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A quadratic with seeded noise: Auto-fit ranks the polynomial, and it fits in closed form."""
    x = np.linspace(0.0, 10.0, 36)
    y = 1.0 + 0.5 * x - 0.03 * x**2 + np.random.default_rng(0).normal(0.0, 0.1, x.size)
    path = tmp_path_factory.mktemp("fit-tools") / "quadratic.csv"
    rows = [",".join(repr(float(v)) for v in row) for row in zip(x, y, strict=True)]
    path.write_text("x,y\n" + "\n".join(rows) + "\n")
    return path


def read_xy(path: Path) -> XY:
    """The file's data as the command reads it, with the same validation it applies."""
    dataset: Dataset = read_dataset(str(path))
    issues = validate(dataset, "x", "y", None)
    assert not [issue for issue in issues if issue.severity == "error"], issues
    return dataset.xy("x", "y", None)


_RANKINGS: dict[str, tuple[XY, AutoFitResult]] = {}
"""Auto-fit is the slow part: one run per file, after the registry fixture is in force."""


def cached_ranking(path: Path) -> tuple[XY, AutoFitResult]:
    """The data and what Auto-fit with the ``autofit`` command's defaults returns for it."""
    key = str(path)
    if key not in _RANKINGS:
        xy = read_xy(path)
        _RANKINGS[key] = (xy, autofit(xy, AutoFitConfig()))
    return _RANKINGS[key]


def json_of(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


DISCERN_REQUEST = DiscernibilityRequest(level=0.9, grid_points=40, max_models=4)
DISCERN_FLAGS = ["--level", "0.9", "--grid-points", "40", "--max-models", "4", "--json"]


# --- the same numbers, all three ------------------------------------------------------------


def test_discern_json_is_exactly_what_the_report_returns_and_uses_the_autofit_defaults(
    data_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["discern", str(data_csv), "--x", "x", "--y", "y", *DISCERN_FLAGS])
    printed = json_of(capsys)
    xy, ranking = cached_ranking(data_csv)
    expected = discernibility(xy, ranking, request=DISCERN_REQUEST)

    assert code == 0
    assert type(expected) is ToolResult
    assert printed == expected.model_dump(mode="json")
    # The ranking is the one autofit gives on the same file, model for model.
    models = next(block for block in expected.tables if block.title == "Models")
    assert ranking.ranked[0].model_id in str(models.rows[0][1])
    assert str(models.rows[0][0]) == "1"


def test_simulate_json_is_exactly_what_the_tool_returns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "simulate",
            "--model",
            "exp_growth",
            "--params",
            "a=2,b=0.5",
            "--design",
            "0:4:5",
            "--noise",
            "0.1",
            "--runs",
            "5",
            "--seed",
            "7",
            "--json",
        ]
    )
    printed = json_of(capsys)
    expected = simulate(
        FitSpec(model=RegisteredRef(id="exp_growth")),
        SimulationRequest(
            truth={"a": 2.0, "b": 0.5},
            design=DesignSpec(low=0.0, high=4.0, count=5, replicates=1),
            noise=NoiseSpec(kind="sd", value=0.1),
            runs=5,
            seed=7,
            level=0.95,
        ),
    )

    assert code == 0
    assert printed == expected.model_dump(mode="json")


def test_advise_json_is_exactly_what_the_tool_returns(
    data_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = FitSpec(model=PolynomialRef(degree=2))
    code = main(
        [
            "advise",
            str(data_csv),
            "--x",
            "x",
            "--y",
            "y",
            "--model",
            "poly",
            "--degree",
            "2",
            "--points",
            "2",
            "--candidates",
            "20",
            "--level",
            "0.9",
            "--json",
        ]
    )
    printed = json_of(capsys)
    xy = read_xy(data_csv)
    fit = fit_dataset(xy, spec).result
    expected = advise(
        xy,
        spec,
        fit,
        request=AdvisorRequest(points=2, candidates=20, level=0.9),
    )

    assert code == 0
    assert printed == expected.model_dump(mode="json")


# --- the readable reports -------------------------------------------------------------------


def test_discern_report_prints_the_title_tables_and_interpretation(
    data_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = [flag for flag in DISCERN_FLAGS if flag != "--json"]
    code = main(["discern", str(data_csv), "--x", "x", "--y", "y", *flags])
    out = capsys.readouterr().out
    xy, ranking = cached_ranking(data_csv)
    expected = discernibility(xy, ranking, request=DISCERN_REQUEST)

    assert code == 0
    assert expected.title in out
    for block in expected.tables:
        assert block.title in out
    assert f"Interpretation:\n{expected.interpretation}" in out


def test_simulate_report_prints_the_title_tables_and_interpretation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    flags = [
        "--model",
        "exp_growth",
        "--params",
        "a=2,b=0.5",
        "--design",
        "0:4:5",
        "--noise",
        "0.1",
        "--runs",
        "5",
        "--seed",
        "7",
    ]
    code = main(["simulate", *flags])
    out = capsys.readouterr().out
    expected = simulate(
        FitSpec(model=RegisteredRef(id="exp_growth")),
        SimulationRequest(
            truth={"a": 2.0, "b": 0.5},
            design=DesignSpec(low=0.0, high=4.0, count=5, replicates=1),
            noise=NoiseSpec(kind="sd", value=0.1),
            runs=5,
            seed=7,
        ),
    )

    assert code == 0
    assert expected.title in out
    for block in expected.tables:
        assert block.title in out
    assert f"Interpretation:\n{expected.interpretation}" in out


def test_advise_report_prints_the_title_tables_and_interpretation(
    data_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "advise",
            str(data_csv),
            "--x",
            "x",
            "--y",
            "y",
            "--model",
            "poly",
            "--degree",
            "2",
            "--points",
            "2",
            "--candidates",
            "20",
        ]
    )
    out = capsys.readouterr().out
    spec = FitSpec(model=PolynomialRef(degree=2))
    xy = read_xy(data_csv)
    expected = advise(
        xy,
        spec,
        fit_dataset(xy, spec).result,
        request=AdvisorRequest(points=2, candidates=20),
    )

    assert code == 0
    assert expected.title in out
    for block in expected.tables:
        assert block.title in out
    assert f"Interpretation:\n{expected.interpretation}" in out


# --- simulate without a file, and with one --------------------------------------------------


def test_simulate_takes_its_truth_from_the_model_example_with_no_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["simulate", "--model", "exp_growth", "--runs", "3", "--json"])
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    recovery = next(b for b in printed["tables"] if b["title"] == "Parameter recovery")

    assert code == 0
    # exp_growth's example is (a=2, b=0.5), and the note says so.
    assert [row[1] for row in recovery["rows"]] == [2.0, 0.5]
    assert "the model's example supplies the true values" in captured.err


def test_from_file_changes_the_truth_to_the_fitted_values(
    data_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "simulate",
            "--model",
            "poly",
            "--degree",
            "2",
            "--from-file",
            str(data_csv),
            "--x",
            "x",
            "--y",
            "y",
            "--design",
            "0:10:5",
            "--noise",
            "0.1",
            "--runs",
            "3",
            "--json",
        ]
    )
    printed = json_of(capsys)
    recovery = next(b for b in printed["tables"] if b["title"] == "Parameter recovery")
    spec = FitSpec(model=PolynomialRef(degree=2))
    fitted = {p.name: p.value for p in fit_dataset(read_xy(data_csv), spec).result.parameters}

    assert code == 0
    assert [row[0] for row in recovery["rows"]] == ["b0", "b1", "b2"]
    for row in recovery["rows"]:
        assert row[1] == pytest.approx(fitted[row[0]])
    # The example's quadratic would have been (1, 0.5, -0.03); the fit moved the truth.
    assert recovery["rows"][2][1] != pytest.approx(-0.03, abs=1e-6)


# --- the exit codes -------------------------------------------------------------------------


def test_a_missing_column_exits_three_with_the_code(
    data_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["discern", str(data_csv), "--x", "nosuch", "--y", "y"])
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    assert "UNKNOWN_COLUMN" in captured.err


def test_an_unknown_model_exits_three_with_the_code(
    data_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["advise", str(data_csv), "--x", "x", "--y", "y", "--model", "nosuch"])
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    assert "UNKNOWN_MODEL" in captured.err


def test_a_bad_design_exits_three_with_the_code(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "simulate",
            "--model",
            "exp_growth",
            "--params",
            "a=2,b=0.5",
            "--design",
            "1:0:5",
            "--noise",
            "0.1",
            "--runs",
            "2",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    assert "INVALID_OPTION" in captured.err


def test_a_difference_for_an_unknown_parameter_exits_three_with_the_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "simulate",
            "--model",
            "exp_growth",
            "--params",
            "a=2,b=0.5",
            "--design",
            "0:4:5",
            "--noise",
            "0.1",
            "--difference",
            "z=0.5",
            "--runs",
            "2",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    assert "INVALID_OPTION" in captured.err


def test_more_than_five_thousand_runs_exits_three_with_the_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "simulate",
            "--model",
            "exp_growth",
            "--params",
            "a=2,b=0.5",
            "--design",
            "0:4:5",
            "--noise",
            "0.1",
            "--runs",
            "5001",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    assert "TOO_MANY_ROWS" in captured.err
