"""End-to-end CLI tests: import -> validate -> fit -> report through main() and a subprocess.

Every test drives the real layers (data, fitting, models); nothing is mocked. Exit-code
contract under test: 0 converged, 2 invalid input, 3 the fit ran but did not converge.
"""

import importlib.metadata
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from openpyxl import Workbook

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


def test_too_few_points_for_the_model_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Three points, four parameters (degree-3 polynomial): the fit result is
    # INVALID_INPUT (n <= k), which is invalid input (exit 2), not a failed fit (3).
    path = tmp_path / "sparse.csv"
    path.write_text("x,y\n1,2.0\n2,4.1\n3,6.2\n")
    code = main(["fit", str(path), "--x", "x", "--y", "y", "--poly", "3"])
    captured = capsys.readouterr()
    assert code == 2
    assert "invalid_input" in captured.out
    assert "3 active observation(s) for 4 parameter(s)" in captured.out


def test_model_that_does_not_apply_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # log(x) with x = 0 in the data: the model does not apply, which is invalid input.
    path = tmp_path / "zero.csv"
    path.write_text("x,y\n0,1.0\n1,0.0\n2,-0.7\n3,-1.1\n4,-1.4\n")
    code = main(["fit", str(path), "--x", "x", "--y", "y", "--model", "log_natural"])
    captured = capsys.readouterr()
    assert code == 2
    assert "invalid_input" in captured.out
    assert "does not apply" in captured.out


def test_overflowing_y_is_a_failed_fit_not_invalid_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # y ~ 1e160: the sums of squares overflow. The statistics failure is a numerical
    # outcome of the fit (exit 3, Status: failed), not an invalid fit configuration.
    lines = ["x,y"]
    for i in range(20):
        y = "1e+160" if i % 2 else "9e+159"
        lines.append(f"{0.5 * i},{y}")
    path = tmp_path / "huge.csv"
    path.write_text("\n".join(lines) + "\n")
    code = main(["fit", str(path), "--x", "x", "--y", "y", "--poly", "1"])
    captured = capsys.readouterr()
    assert code == 3
    assert "Status: failed" in captured.out
    assert "overflow" in captured.out
    assert "invalid fit configuration" not in captured.err


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


# --- broken input files reach the user as messages, never tracebacks ---------------------


def _make_xlsx(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["x", "y"])
    sheet.append([1.0, 2.1])
    sheet.append([2.0, 3.9])
    workbook.save(path)


def _truncate_sheet_xml(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for info, data in members:
            cut = data[: len(data) // 2] if info.filename.endswith("sheet1.xml") else data
            archive.writestr(info, cut)


def _corrupt_sheet_deflate(path: Path) -> None:
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = next(i for i in archive.infolist() if i.filename.endswith("sheet1.xml"))
        start = info.header_offset
    name_len = int.from_bytes(raw[start + 26 : start + 28], "little")
    extra_len = int.from_bytes(raw[start + 28 : start + 30], "little")
    data_start = start + 30 + name_len + extra_len
    corrupted = bytearray(raw)
    for position in range(data_start + 10, min(data_start + 60, data_start + info.compress_size)):
        corrupted[position] ^= 0xFF
    path.write_bytes(bytes(corrupted))


def _fit_in_process(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "curvestudio.cli",
            "fit",
            str(path),
            "--x",
            "x",
            "--y",
            "y",
            "--poly",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )


def test_cli_truncated_sheet_xml_exits_2_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "truncated.xlsx"
    _make_xlsx(path)
    _truncate_sheet_xml(path)
    proc = _fit_in_process(path)
    assert proc.returncode == 2
    assert "PARSE_ERROR" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_corrupt_deflate_stream_exits_2_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.xlsx"
    _make_xlsx(path)
    _corrupt_sheet_deflate(path)
    proc = _fit_in_process(path)
    assert proc.returncode == 2
    assert "PARSE_ERROR" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="chmod is not a permission barrier on Windows")
def test_cli_permission_denied_exits_2_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "locked.xlsx"
    _make_xlsx(path)
    path.chmod(0o000)
    try:
        proc = _fit_in_process(path)
    finally:
        path.chmod(0o644)
    assert proc.returncode == 2
    assert "FILE_UNREADABLE" in proc.stderr
    assert "Traceback" not in proc.stderr
