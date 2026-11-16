"""End-to-end CLI tests: import -> validate -> fit -> report through main() and a subprocess.

Every test drives the real layers (data, fitting, models); nothing is mocked. Exit-code
contract under test: 0 converged, 2 invalid input, 3 the fit ran but did not converge.
"""

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import pytest

from curvestudio.cli import main
from curvestudio.fitting import FitResult, FitStatus

REPO = Path(__file__).resolve().parents[2]
DECAY = REPO / "examples" / "decay.csv"


def _param_value(captured: str, name: str) -> float:
    """The printed value of parameter ``name`` from the report's parameter table."""
    match = re.search(rf"^{name}\s+([0-9.eE+-]+)\s", captured, re.MULTILINE)
    assert match is not None, f"parameter {name!r} not found in output:\n{captured}"
    return float(match.group(1))


def test_fit_decay_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y", "--model", "exp_decay_offset"])
    captured = capsys.readouterr()
    assert code == 0
    assert "R²" in captured.out
    assert "converged" in captured.out
    assert "exp_decay_offset" in captured.out
    assert abs(_param_value(captured.out, "a") - 5.0) <= 0.05 * 5.0


def test_blank_y_cell_exits_2_and_names_the_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Blank y in the 3rd data row (0-based table position 2 -> displayed data row 3).
    path = tmp_path / "blank.csv"
    path.write_text("x,y\n0,7.0\n1,5.6\n2,\n3,3.6\n4,2.9\n5,2.4\n")
    code = main(["fit", str(path), "--x", "x", "--y", "y", "--model", "exp_decay_offset"])
    captured = capsys.readouterr()
    assert code == 2
    assert "MISSING_VALUE" in captured.err
    assert "data row 3" in captured.err


def test_json_round_trips_through_from_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "fit.json"
    code = main(
        [
            "fit",
            str(DECAY),
            "--x",
            "x",
            "--y",
            "y",
            "--model",
            "exp_decay_offset",
            "--json",
            str(target),
        ]
    )
    assert code == 0
    capsys.readouterr()
    result = FitResult.from_json(target.read_text(encoding="utf-8"))
    assert result.status is FitStatus.CONVERGED
    assert result.model_id == "exp_decay_offset"
    value = {p.name: p.value for p in result.parameters}["a"]
    assert abs(value - 5.0) <= 0.05 * 5.0


def test_unknown_model_with_near_match_hint(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y", "--model", "exp_decay_offse"])
    captured = capsys.readouterr()
    assert code == 2
    assert "exp_decay_offset" in captured.err


def test_unknown_model_without_near_match_points_at_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y", "--model", "nope"])
    captured = capsys.readouterr()
    assert code == 2
    assert "curvestudio models" in captured.err


def test_unconverged_fit_exits_3(capsys: pytest.CaptureFixture[str]) -> None:
    # A single symmetric peak genuinely cannot fit a monotone decay with asymptote:
    # the optimizer wanders and exhausts its function evaluations (verified up to
    # max_nfev=5000), so this is a real not-converged outcome, not a mocked one.
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y", "--model", "gaussian"])
    captured = capsys.readouterr()
    assert code == 3
    assert "not_converged" in captured.out


def test_poly_path(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y", "--poly", "2"])
    captured = capsys.readouterr()
    assert code == 0
    assert "R²" in captured.out
    assert "b2" in captured.out
    assert "converged" in captured.out


def test_weights_column_reins_in_outlier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = ["x,y,w"]
    for i in range(21):
        x = 0.5 * i
        lines.append(f"{x},{5 * 2.718281828459045 ** (-0.4 * x) + 2:.12g},1")
    lines.append("2.0,1000.0,1e-9")  # a wild outlier, effectively ignored by its tiny weight
    path = tmp_path / "weighted.csv"
    path.write_text("\n".join(lines) + "\n")
    code = main(
        ["fit", str(path), "--x", "x", "--y", "y", "--weights", "w", "--model", "exp_decay_offset"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert abs(_param_value(captured.out, "a") - 5.0) <= 0.05 * 5.0


def test_starts_and_seed_are_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "fit",
        str(DECAY),
        "--x",
        "x",
        "--y",
        "y",
        "--model",
        "exp_decay_offset",
        "--starts",
        "5",
        "--seed",
        "7",
    ]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert "starts=5" in first
    assert main(argv) == 0
    second = capsys.readouterr().out
    assert first == second


def test_models_lists_registry(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["models"])
    captured = capsys.readouterr()
    assert code == 0
    assert "exp_decay_offset" in captured.out
    assert "Exponential decay with offset" in captured.out
    assert "exponential" in captured.out


def test_missing_file_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", "does/not/exist.csv", "--x", "x", "--y", "y", "--model", "exp_growth"])
    captured = capsys.readouterr()
    assert code == 2
    assert "FILE_NOT_FOUND" in captured.err


def test_missing_column_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", str(DECAY), "--x", "nope", "--y", "y", "--model", "exp_growth"])
    captured = capsys.readouterr()
    assert code == 2
    assert "UNKNOWN_COLUMN" in captured.err
    assert "nope" in captured.err


def test_model_and_poly_are_mutually_exclusive(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y", "--model", "exp_growth", "--poly", "2"])
    assert code == 2
    capsys.readouterr()


def test_model_or_poly_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y"])
    assert code == 2
    capsys.readouterr()


def test_poly_rejects_model_only_options(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fit", str(DECAY), "--x", "x", "--y", "y", "--poly", "2", "--method", "lm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--method" in captured.err


def test_robust_loss_flag_reaches_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "fit",
            str(DECAY),
            "--x",
            "x",
            "--y",
            "y",
            "--model",
            "exp_decay_offset",
            "--loss",
            "huber",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "loss=huber" in captured.out


def test_console_script_entry_point_is_registered() -> None:
    entries = importlib.metadata.entry_points(group="console_scripts")
    values = {e.name: e.value for e in entries if e.name == "curvestudio"}
    assert values.get("curvestudio") == "curvestudio.cli:main"


def test_python_dash_m_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "curvestudio.cli", "models"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    assert proc.returncode == 0
    assert "exp_decay_offset" in proc.stdout
