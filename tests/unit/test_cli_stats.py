"""``curvestudio stats``: every statistical tool, from a file, with the tools' own numbers.

The command is generated from the catalogue, so this file runs all nineteen tools the way a
person would and compares what ``--json`` prints with what ``run_tool`` returns for the same
choice of columns and options on the same dataset. That is the same-numbers rule: the command
line, the desktop app and the web app all show one result. The readable report is checked to
carry every block -- the tables, the interpretation, the methods fragment and the rows left
out -- and the usage and input errors are checked to exit ``3`` with their code.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from curvestudio.cli import main
from curvestudio.data import Dataset, read_dataset
from curvestudio.tools.stats import (
    FAMILIES,
    ToolResult,
    build_request,
    catalogue,
    run_tool,
)

CASES: tuple[tuple[str, tuple[str, ...], dict[str, object]], ...] = (
    ("anova", ("--values", "y", "--groups", "g1"), {"values": "y", "groups": "g1"}),
    ("kruskal", ("--values", "y", "--groups", "g1"), {"values": "y", "groups": "g1"}),
    (
        "twoway",
        ("--values", "y", "--factor-a", "g1", "--factor-b", "g2"),
        {"values": "y", "factor_a": "g1", "factor_b": "g2"},
    ),
    ("repeated", ("--measures", "m1,m2,m3"), {"measures": ["m1", "m2", "m3"]}),
    ("friedman", ("--measures", "m1,m2,m3"), {"measures": ["m1", "m2", "m3"]}),
    (
        "blandaltman",
        ("--method-a", "m1", "--method-b", "m2"),
        {"method_a": "m1", "method_b": "m2"},
    ),
    ("deming", ("--method-a", "m1", "--method-b", "m2"), {"method_a": "m1", "method_b": "m2"}),
    (
        "passingbablok",
        ("--method-a", "m1", "--method-b", "m2"),
        {"method_a": "m1", "method_b": "m2"},
    ),
    ("contingency", ("--rows", "g1", "--columns", "g2"), {"rows": "g1", "columns": "g2"}),
    (
        "mcnemar",
        ("--before", "before", "--after", "after"),
        {"before": "before", "after": "after"},
    ),
    ("correlation", ("--columns", "x,y,z"), {"columns": ["x", "y", "z"]}),
    ("descriptives", ("--values", "y", "--groups", "g1"), {"values": "y", "groups": "g1"}),
    ("distribution", ("--values", "y", "--bins", "6"), {"values": "y", "bins": 6}),
    (
        "ttest",
        (
            "--kind",
            "independent",
            "--a",
            "y",
            "--groups",
            "g1",
            "--null",
            "0.1",
            "--equal-variances",
            "--level",
            "0.9",
        ),
        {
            "kind": "independent",
            "a": "y",
            "groups": "g1",
            "null": 0.1,
            "equal_variances": True,
            "level": 0.9,
        },
    ),
    (
        "power",
        ("--design", "independent", "--effect-size", "0.5", "--power", "0.8"),
        {"design": "independent", "effect_size": 0.5, "power": 0.8},
    ),
    ("mannwhitney", ("--a", "y", "--groups", "g1"), {"a": "y", "groups": "g1"}),
    ("wilcoxon", ("--a", "m1"), {"a": "m1"}),
    ("signtest", ("--a", "m1"), {"a": "m1"}),
    (
        "regression",
        ("--outcome", "y", "--predictors", "x,z"),
        {"outcome": "y", "predictors": ["x", "z"]},
    ),
)
"""One invocation per tool: its flags, and the same choices as the tools' request takes them."""

TOOL_IDS = [case[0] for case in CASES]
FAMILY_TITLES = [title for _, title in FAMILIES]


def _fixture_arrays() -> dict[str, np.ndarray]:
    """The columns of the starter file: numbers, two group columns and two binary columns."""
    rng = np.random.default_rng(7)
    size = 40
    x = np.linspace(-2.0, 2.0, size) + rng.normal(0.0, 0.3, size)
    z = rng.normal(0.0, 1.0, size)
    g1 = np.array(["A" if index % 2 == 0 else "B" for index in range(size)], dtype=object)
    g2 = np.array(["X" if (index // 2) % 2 == 0 else "Y" for index in range(size)], dtype=object)
    y = 1.0 + 0.5 * x - 0.2 * z + 0.3 * (g1 == "A") + 0.2 * (g2 == "X") + rng.normal(0.0, 0.4, size)
    m1 = rng.normal(0.0, 1.0, size)
    m2 = m1 + 0.5 + rng.normal(0.0, 0.5, size)
    m3 = m1 + 1.0 + rng.normal(0.0, 0.5, size)
    before = np.array(["no"] * 20 + ["yes"] * 20, dtype=object)
    after = np.array(["yes"] * 8 + ["no"] * 12 + ["yes"] * 14 + ["no"] * 6, dtype=object)
    return {
        "x": x,
        "y": y,
        "z": z,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "g1": g1,
        "g2": g2,
        "before": before,
        "after": after,
    }


def _write_fixture(path: Path) -> Path:
    """Write the starter file: one header and one line per row, numbers to twelve digits."""
    arrays = _fixture_arrays()
    names = ["x", "y", "z", "m1", "m2", "m3", "g1", "g2", "before", "after"]
    lines = [",".join(names)]
    for index in range(len(arrays["y"])):
        cells = [
            f"{arrays[name][index]:.12g}"
            if isinstance(arrays[name][index], (float, np.floating))
            else str(arrays[name][index])
            for name in names
        ]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def stats_csv(tmp_path: Path) -> Path:
    """The starter file, written once for each test that reads it."""
    return _write_fixture(tmp_path / "stats.csv")


def _unused(name: str) -> object:
    raise AssertionError(f"a tool with no data read a column: {name!r}")


def _expected(tool: str, spec: dict[str, object], dataset: Dataset | None) -> ToolResult:
    """What the tool returns for the same choices, read through the same data layer."""
    column = _unused if dataset is None else dataset.column
    group = _unused if dataset is None else dataset.group
    return run_tool(tool, build_request(tool, spec, column=column, group=group))


def _words(tool: str, path: Path | None, flags: tuple[str, ...], *extra: str) -> list[str]:
    """The command line a person would type for one tool."""
    words = ["stats", tool]
    if path is not None:
        words.append(str(path))
    return [*words, *flags, *extra]


# --- the same numbers, tool by tool --------------------------------------------------------


@pytest.mark.parametrize(("tool", "flags", "spec"), CASES, ids=TOOL_IDS)
def test_the_json_is_exactly_what_the_tool_returns(
    tool: str,
    flags: tuple[str, ...],
    spec: dict[str, object],
    stats_csv: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = None if tool == "power" else stats_csv
    code = main(_words(tool, path, flags, "--json"))
    out = capsys.readouterr().out
    dataset = None if path is None else read_dataset(str(path))
    expected = _expected(tool, spec, dataset)

    assert code == 0
    assert type(expected) is ToolResult  # no tool subclasses the result
    assert set(json.loads(out)) == set(ToolResult.model_fields)  # and no field is added
    assert json.loads(out) == expected.model_dump(mode="json")
    # --json is the dump itself, not a reformatted one: a surface can trust the numbers
    assert out.strip() == json.dumps(expected.model_dump(mode="json"))


# --- the readable report -------------------------------------------------------------------


def test_the_report_prints_every_table_and_the_words(
    stats_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = ("--outcome", "y", "--predictors", "x,z", "--predictions", "0,0;1,1")
    spec = {
        "outcome": "y",
        "predictors": ["x", "z"],
        "predictions": ((0.0, 0.0), (1.0, 1.0)),
    }
    expected = _expected("regression", spec, read_dataset(str(stats_csv)))
    code = main(_words("regression", stats_csv, flags))
    out = capsys.readouterr().out

    assert code == 0
    assert expected.tables  # there is a table to print
    assert expected.title in out
    for block in expected.tables:
        assert block.title in out
    for test in expected.tests:
        assert test.name in out
    for check in expected.assumptions:
        assert check.title in out
    assert expected.interpretation in out
    assert expected.methods_fragment in out
    assert f"Rows: {expected.counts.n_offered} offered" in out


def test_the_report_names_the_excluded_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "gaps.csv"
    path.write_text("x,y\n1,2\n2,\n3,4\n4,oops\n5,6\n")
    expected = _expected("descriptives", {"values": "y"}, read_dataset(str(path)))
    code = main(_words("descriptives", path, ("--values", "y")))
    out = capsys.readouterr().out

    assert code == 0
    assert expected.counts.n_excluded == 2
    assert f"{expected.counts.n_excluded} excluded" in out
    assert "Excluded rows" in out
    for row in expected.excluded:
        assert f"row {row.index + 1}: {row.reason}" in out


def test_predictions_and_names_reach_the_request(
    stats_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = (
        "--outcome",
        "y",
        "--predictors",
        "x,z",
        "--names",
        "first,second",
        "--predictions",
        "0,0;1,1",
        "--json",
    )
    spec = {
        "outcome": "y",
        "predictors": ["x", "z"],
        "names": ["first", "second"],
        "predictions": ((0.0, 0.0), (1.0, 1.0)),
    }
    expected = _expected("regression", spec, read_dataset(str(stats_csv)))
    code = main(_words("regression", stats_csv, flags))
    printed = json.loads(capsys.readouterr().out)

    assert code == 0
    assert any(block.title == "Predictions" for block in expected.tables)
    assert printed == expected.model_dump(mode="json")


def test_power_runs_with_no_file(capsys: pytest.CaptureFixture[str]) -> None:
    flags = ("--design", "independent", "--effect-size", "0.5", "--power", "0.8")
    code = main(_words("power", None, flags))
    out = capsys.readouterr().out

    assert code == 0
    assert "Sample size and power" in out


# --- the catalogue as a list ---------------------------------------------------------------


def test_list_groups_the_tools_by_family(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["stats", "--list"])
    out = capsys.readouterr().out
    lines = out.splitlines()

    assert code == 0
    assert [line for line in lines if line in FAMILY_TITLES] == FAMILY_TITLES
    tools = catalogue()
    for info in tools:
        # each tool is one indented line, under its family's title
        assert any(line.startswith(f"  {info.name} ") for line in lines)
        found = next(line for line in lines if line.startswith(f"  {info.name} "))
        family_title = dict(FAMILIES)[info.family]
        assert lines.index(family_title) < lines.index(found)


# --- the exit codes ------------------------------------------------------------------------


def test_a_missing_column_exits_three_with_the_code(
    stats_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(_words("anova", stats_csv, ("--values", "nosuch", "--groups", "g1")))
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    assert "UNKNOWN_COLUMN" in captured.err


def test_an_unknown_tool_exits_three(stats_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["stats", "nosuchtool", str(stats_csv)])
    captured = capsys.readouterr()

    assert code == 3
    assert "invalid choice" in captured.err


def test_a_wrong_option_exits_three_with_the_code(
    stats_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(_words("anova", stats_csv, ("--values", "y", "--groups", "g1", "--level", "5")))
    captured = capsys.readouterr()

    assert code == 3
    assert captured.out == ""
    assert "INVALID_OPTION" in captured.err
