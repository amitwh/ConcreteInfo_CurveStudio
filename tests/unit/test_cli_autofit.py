"""``curvestudio autofit``: the ranked table, the ordering header, the lists, JSON, errors.

Every test drives ``main`` through the real data, discovery and model layers on the starter
models (the installed catalogues are hidden by the test fixture), in-process with
``--workers 1`` so no worker interpreters are started.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from curvestudio.cli import main
from curvestudio.discovery import AutoFitResult, RankBy
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models

REPO = Path(__file__).resolve().parents[2]
DECAY = REPO / "examples" / "decay.csv"
FAST = ["--workers", "1", "--n-starts", "3"]


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()  # leave the registry as other tests (the CLI fit tests) expect it


def write_csv(path: Path, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> Path:
    columns = [x, y] if weights is None else [x, y, weights]
    header = "x,y" if weights is None else "x,y,w"
    rows = [",".join(repr(float(v)) for v in row) for row in zip(*columns, strict=True)]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def noisy_line(tmp_path: Path) -> Path:
    """Seeded straight line plus noise: a polynomial of high degree beats it on R2 only."""
    x = np.linspace(1.0, 10.0, 30)
    y = 2.0 + 0.5 * x + np.random.default_rng(3).normal(0.0, 0.15, x.size)
    return write_csv(tmp_path / "line.csv", x, y)


def test_table_lists_ranks_models_and_fitted_equations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["autofit", str(DECAY), "--x", "x", "--y", "y", *FAST])
    out = capsys.readouterr().out
    assert code == 0
    assert "Ranked by: best overall" in out
    for column in ("Rank", "Model", "R²", "Adj. R²", "RMSE", "AICc", "BIC", "CV-RMSE", "k"):
        assert column in out
    assert "exp_decay_offset" in out
    assert "y = 5.00" in out  # the fitted equation of the true model, numbers filled in
    first_row = next(line for line in out.splitlines() if "exp_decay_offset" in line)
    assert first_row.split()[0] == "1"


def test_skipped_candidates_are_listed_with_their_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # decay.csv starts at x = 0, which rules out every power model
    code = main(["autofit", str(DECAY), "--x", "x", "--y", "y", *FAST])
    out = capsys.readouterr().out
    assert code == 0
    assert "Skipped" in out
    assert "x must be strictly positive" in out
    assert "power" in out.split("Skipped", 1)[1]


def test_top_limits_the_rows_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["autofit", str(DECAY), "--x", "x", "--y", "y", "--top", "2", *FAST])
    out = capsys.readouterr().out
    assert code == 0
    assert "showing the top 2 of" in out
    ranks = [
        line.split()[0]
        for line in out.splitlines()
        if line.split()[:1] and line.split()[0].isdigit()
    ]
    assert len(ranks) == 2  # tied rows share a rank, so only the row count is fixed


def test_rank_by_r2_relabels_the_header_and_shows_the_overfitting_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = noisy_line(tmp_path)
    assert main(["autofit", str(path), "--x", "x", "--y", "y", "--rank-by", "r2", *FAST]) == 0
    by_r2 = capsys.readouterr().out
    assert "Ranked by: R²" in by_r2
    assert "Overfitting warning" in by_r2
    assert main(["autofit", str(path), "--x", "x", "--y", "y", *FAST]) == 0
    best = capsys.readouterr().out
    assert "Ranked by: best overall" in best
    assert "Overfitting warning" not in best


def test_json_prints_a_result_that_round_trips(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["autofit", str(DECAY), "--x", "x", "--y", "y", "--json", "--seed", "5"]
    code = main([*args, "--rank-by", "bic", *FAST])
    out = capsys.readouterr().out
    assert code == 0
    result = AutoFitResult.from_json(out)  # the whole stdout is one JSON document
    assert result.config.rank_by is RankBy.BIC
    assert (result.config.seed, result.config.n_starts, result.config.workers) == (5, 3, 1)
    assert result.ranked[0].model_id == "exp_decay_offset"
    assert AutoFitResult.from_json(result.to_json()) == result


def test_exhaustive_and_time_limit_reach_the_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["autofit", str(DECAY), "--x", "x", "--y", "y", "--json", "--exhaustive"]
    code = main([*args, "--time-limit", "7.5", *FAST])
    result = AutoFitResult.from_json(capsys.readouterr().out)
    assert code == 0
    assert result.config.exhaustive is True
    assert result.config.time_limit == 7.5


def test_weights_column_is_used(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    x = np.linspace(0.5, 6.0, 40)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + np.random.default_rng(1).normal(0.0, 0.03, x.size)
    path = write_csv(tmp_path / "weighted.csv", x, y, weights=np.ones(x.size))
    code = main(["autofit", str(path), "--x", "x", "--y", "y", "--weights", "w", *FAST])
    out = capsys.readouterr().out
    assert code == 0
    assert "weights='w'" in out


def test_too_few_points_exits_1_naming_the_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv(tmp_path / "three.csv", np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.1, 5.9]))
    code = main(["autofit", str(path), "--x", "x", "--y", "y", *FAST])
    captured = capsys.readouterr()
    assert code == 1
    assert "TOO_FEW_POINTS" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_constant_response_exits_1_naming_the_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    x = np.linspace(0.0, 5.0, 12)
    path = write_csv(tmp_path / "flat.csv", x, np.full(x.size, 4.0))
    code = main(["autofit", str(path), "--x", "x", "--y", "y", *FAST])
    assert code == 1
    assert "CONSTANT_RESPONSE" in capsys.readouterr().err


def test_missing_column_and_missing_file_exit_1_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["autofit", str(DECAY), "--x", "x", "--y", "nope", *FAST]) == 1
    assert "nope" in capsys.readouterr().err
    assert main(["autofit", str(tmp_path / "absent.csv"), "--x", "x", "--y", "y", *FAST]) == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [
        ["--rank-by", "nonsense"],
        ["--time-limit", "0"],
        ["--time-limit", "nan"],
        ["--workers", "0"],
        ["--n-starts", "0"],
        ["--top", "0"],
        ["--seed", "-1"],
    ],
)
def test_bad_option_values_are_usage_errors(
    flags: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["autofit", str(DECAY), "--x", "x", "--y", "y", *flags])
    assert code == 2
    assert capsys.readouterr().err.strip() != ""
