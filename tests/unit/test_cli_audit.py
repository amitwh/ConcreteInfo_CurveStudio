"""``curvestudio audit`` and ``curvestudio methods``: the tools' findings and text, from a file.

Both commands fit through the workflow layer and read the fit through the tools layer, so
their numbers are the apps' numbers: the JSON each prints is compared, as parsed dicts, with
what ``audit_fit`` and ``methods_text`` return for the same data and specification. Every
test drives ``main`` through the real layers on the starter models (the installed catalogues
are hidden by the test fixture). ``--ranking`` runs Auto-fit with the ``autofit`` command's
defaults, worker processes included, so the tests that ask for one take a few seconds each.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from curvestudio.cli import _AUDIT_EXIT_CODES, _print_methods_text, main
from curvestudio.data import XY, read_dataset, validate
from curvestudio.discovery import AutoFitConfig, autofit
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio.tools import (
    FitAudit,
    MethodsText,
    Verdict,
    audit_fit,
    cross_validate,
    methods_text,
)
from curvestudio.workflow import FitSpec, PolynomialRef, RegisteredRef, fit_dataset

REPO = Path(__file__).resolve().parents[2]
DECAY = REPO / "examples" / "decay.csv"
QUADRATIC = FitSpec(model=PolynomialRef(degree=2))
TAGS = {"pass": "pass", "caution": "caution", "fail": "fail", "not_applicable": "n/a"}
"""The bracket tag the audit prints for each verdict."""
COMMANDS = ["audit", "methods"]


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register_starter_models()
    yield
    _clear_registry()
    register_starter_models()  # leave the registry as other tests (the CLI fit tests) expect it


# --- data, and what the tools return for it -------------------------------------------------


def quadratic() -> tuple[np.ndarray, np.ndarray]:
    """A quadratic on 40 points with seeded noise: fitted by the quadratic, every check passes."""
    x = np.linspace(0.0, 10.0, 40)
    y = 1.0 + 0.5 * x - 0.03 * x**2 + np.random.default_rng(0).normal(0.0, 0.1, x.size)
    return x, y


def noisy_line() -> tuple[np.ndarray, np.ndarray]:
    """A seeded straight line with noise: a cubic on it has parameters the data do not pin down."""
    x = np.linspace(1.0, 10.0, 30)
    y = 2.0 + 0.5 * x + np.random.default_rng(3).normal(0.0, 0.15, x.size)
    return x, y


def write_csv(path: Path, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> Path:
    columns = [x, y] if weights is None else [x, y, weights]
    header = "x,y" if weights is None else "x,y,w"
    rows = [",".join(repr(float(v)) for v in row) for row in zip(*columns, strict=True)]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def as_xy(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> XY:
    """The arrays as the data the workflow takes, built without the command line."""
    return XY(x, y, None if weights is None else weights, "x", "y", "cli-audit")


def read_xy(path: Path) -> XY:
    """The data of a file on disk, read the way the commands read it."""
    dataset = read_dataset(str(path))
    issues = validate(dataset, "x", "y", None)
    assert not [issue for issue in issues if issue.severity == "error"], issues
    return dataset.xy("x", "y", None)


def expected_audit(xy: XY, spec: FitSpec, *, folds=None, ranking: bool = False) -> FitAudit:
    """What the tools return for ``xy`` and ``spec``, given the inputs the flags stand for."""
    bundle = fit_dataset(xy, spec)
    cv = None if folds is None else cross_validate(bundle, spec, folds=folds)
    result = autofit(xy, AutoFitConfig()) if ranking else None
    return audit_fit(bundle, spec=spec, cv=cv, ranking=result)


def as_json(audit: FitAudit) -> dict:
    return json.loads(audit.model_dump_json())


def verdicts_of(audit: dict) -> dict[str, str]:
    return {check["id"]: check["verdict"] for check in audit["checks"]}


def audit_json(capsys: pytest.CaptureFixture[str], path: Path, *flags: str) -> tuple[int, dict]:
    """Run ``audit --json``; the whole of stdout must be one JSON document."""
    code = main(["audit", str(path), "--x", "x", "--y", "y", "--json", *flags])
    return code, json.loads(capsys.readouterr().out)


# --- audit ------------------------------------------------------------------------------------


def test_audit_of_the_quadratic_passes_and_lists_every_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv(tmp_path / "quadratic.csv", *quadratic())
    code = main(["audit", str(path), "--x", "x", "--y", "y", "--poly", "2"])
    out = capsys.readouterr().out
    expected = audit_fit(fit_dataset(as_xy(*quadratic()), QUADRATIC), spec=QUADRATIC)
    assert code == 0
    assert expected.overall is Verdict.PASS
    lines = out.splitlines()
    assert lines[:3] == ["Overall: PASS", expected.summary, ""]
    position = 3
    for check in expected.checks:
        assert lines[position : position + 2] == [
            f"[{TAGS[check.verdict]}] {check.title}",
            f"  {check.evidence}",
        ]
        position += 2
        if check.next_step is not None:
            assert lines[position] == f"    -> {check.next_step}"
            position += 1
    assert position == len(lines)  # every check in its own block, and nothing else


def test_the_checks_that_need_another_run_read_not_applicable_until_it_is_asked_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv(tmp_path / "quadratic.csv", *quadratic())
    code, audit = audit_json(capsys, path, "--poly", "2")
    assert code == 0
    unrun = {
        check_id for check_id, verdict in verdicts_of(audit).items() if verdict == "not_applicable"
    }
    assert unrun == {"cross_validation", "simpler_model"}


def test_audit_json_is_the_tools_audit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(tmp_path / "quadratic.csv", *quadratic())
    code, audit = audit_json(capsys, path, "--poly", "2")
    expected = audit_fit(fit_dataset(as_xy(*quadratic()), QUADRATIC), spec=QUADRATIC)
    assert code == 0
    assert audit == as_json(expected)


def test_cross_validation_decides_its_check_and_the_line_on_the_quadratic_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    x, y = quadratic()
    path = write_csv(tmp_path / "quadratic.csv", x, y)
    code, audit = audit_json(capsys, path, "--poly", "1", "--cv", "5")
    expected = expected_audit(as_xy(x, y), FitSpec(model=PolynomialRef(degree=1)), folds=5)
    assert expected.overall is Verdict.FAIL
    assert code == 2
    assert verdicts_of(audit)["cross_validation"] != "not_applicable"
    assert audit == as_json(expected)


def test_leave_one_out_cross_validation_decides_its_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    x, y = quadratic()
    path = write_csv(tmp_path / "quadratic.csv", x, y)
    code, audit = audit_json(capsys, path, "--poly", "2", "--cv", "loo")
    expected = expected_audit(as_xy(x, y), QUADRATIC, folds="loo")
    assert code == 0
    assert verdicts_of(audit)["cross_validation"] == "pass"
    assert audit == as_json(expected)


def test_ranking_decides_the_simpler_model_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    x, y = quadratic()
    path = write_csv(tmp_path / "quadratic.csv", x, y)
    code, audit = audit_json(capsys, path, "--poly", "2", "--cv", "5", "--ranking")
    expected = expected_audit(as_xy(x, y), QUADRATIC, folds=5, ranking=True)
    assert code == 0
    assert verdicts_of(audit)["simpler_model"] == "pass"
    assert audit["simpler_model"] is None
    assert audit == as_json(expected)


def test_a_simpler_model_is_named_on_the_last_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    x, y = noisy_line()
    path = write_csv(tmp_path / "line.csv", x, y)
    code = main(["audit", str(path), "--x", "x", "--y", "y", "--poly", "3", "--ranking"])
    out = capsys.readouterr().out
    expected = expected_audit(as_xy(x, y), FitSpec(model=PolynomialRef(degree=3)), ranking=True)
    assert expected.overall is Verdict.CAUTION
    simpler = expected.simpler_model
    assert simpler is not None
    assert code == 1  # a caution, and the exit code says so
    assert out.startswith("Overall: CAUTION\n")
    assert "[caution] A simpler model fits as well\n" in out
    assert out.splitlines()[-1] == (
        f"A simpler model fits as well: {simpler.name} ({simpler.model_id}), "
        f"{simpler.n_parameters} parameters"
    )


def test_a_registered_model_is_audited_through_its_specification(
    capsys: pytest.CaptureFixture[str],
) -> None:
    decay = FitSpec(model=RegisteredRef(id="exp_decay_offset"))
    code, audit = audit_json(capsys, DECAY, "--model", "exp_decay_offset", "--cv", "5")
    expected = expected_audit(read_xy(DECAY), decay, folds=5)
    assert code == _AUDIT_EXIT_CODES[expected.overall]
    assert audit == as_json(expected)


def test_weights_reach_the_fit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    x, y = quadratic()
    weights = np.full(x.size, 2.0)
    path = write_csv(tmp_path / "weighted.csv", x, y, weights=weights)
    weighted = FitSpec(model=PolynomialRef(degree=2), use_weights=True)
    code, audit = audit_json(capsys, path, "--poly", "2", "--weights", "w", "--cv", "loo")
    expected = expected_audit(as_xy(x, y, weights), weighted, folds="loo")
    assert code == 0
    assert audit == as_json(expected)
    # the same data without --weights are an unweighted fit: different numbers
    plain = expected_audit(as_xy(x, y), QUADRATIC, folds="loo")
    assert as_json(plain) != audit
    assert (
        main(["methods", str(path), "--x", "x", "--y", "y", "--weights", "w", "--poly", "2"]) == 0
    )
    assert "by weighted least squares" in capsys.readouterr().out


def test_a_fit_that_did_not_converge_fails_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sparse = write_csv(tmp_path / "three.csv", np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.1, 6.2]))
    cases = [
        # three points, four parameters
        (
            sparse,
            as_xy(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.1, 6.2])),
            ["--poly", "3"],
            FitSpec(model=PolynomialRef(degree=3)),
        ),
        # one symmetric peak cannot follow a monotone decay
        (
            DECAY,
            read_xy(DECAY),
            ["--model", "gaussian"],
            FitSpec(model=RegisteredRef(id="gaussian")),
        ),
    ]
    for path, xy, flags, spec in cases:
        code = main(["audit", str(path), "--x", "x", "--y", "y", *flags])
        out = capsys.readouterr().out
        result = fit_dataset(xy, spec).result
        assert result.status.value != "converged"
        assert code == 2
        assert out.startswith("Overall: FAIL\n")
        assert f"[fail] The fit did not converge\n  Status {result.status.value}" in out
        assert result.termination_reason in out

        # with --cv the cross-validation is skipped with a note on stderr: the audit is still
        # the answer, and it is not a usage error
        code = main(["audit", str(path), "--x", "x", "--y", "y", *flags, "--cv", "5"])
        captured = capsys.readouterr()
        assert code == 2
        assert captured.out.startswith("Overall: FAIL\n")
        assert "cross-validation was not run: the fit did not converge" in captured.err


def test_a_refused_cross_validation_exits_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Six points cannot be split into seven folds: the tool refuses, and nothing is audited.
    x = np.linspace(0.0, 10.0, 6)
    path = write_csv(tmp_path / "six.csv", x, np.array([1.0, 1.9, 3.2, 3.9, 5.1, 6.2]))
    code = main(["audit", str(path), "--x", "x", "--y", "y", "--poly", "1", "--cv", "7"])
    captured = capsys.readouterr()
    assert code == 3
    assert "INVALID_FOLDS" in captured.err
    assert captured.out == ""


def test_a_ranking_that_cannot_run_exits_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The fit of a flat line converges; Auto-fit has nothing to rank on a constant response.
    x = np.linspace(0.0, 5.0, 12)
    path = write_csv(tmp_path / "flat.csv", x, np.full(x.size, 4.0))
    code = main(["audit", str(path), "--x", "x", "--y", "y", "--poly", "1", "--ranking"])
    captured = capsys.readouterr()
    assert code == 3
    assert "CONSTANT_RESPONSE" in captured.err
    assert captured.out == ""


def test_data_warnings_go_to_stderr_and_leave_stdout_to_the_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    x, y = quadratic()
    path = write_csv(tmp_path / "repeated.csv", np.append(x, x[5]), np.append(y, y[5]))
    main(["audit", str(path), "--x", "x", "--y", "y", "--poly", "2", "--json"])
    captured = capsys.readouterr()
    assert "curvestudio: warning:" in captured.err
    assert "DUPLICATE_ROW" in captured.err
    assert json.loads(captured.out)["n_observations"] == 41


def test_exit_codes_follow_the_verdict() -> None:
    assert _AUDIT_EXIT_CODES == {
        Verdict.PASS: 0,
        Verdict.CAUTION: 1,
        Verdict.FAIL: 2,
        Verdict.NOT_APPLICABLE: 1,
    }


# --- methods ----------------------------------------------------------------------------------


def test_methods_prints_the_text_the_tools_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv(tmp_path / "quadratic.csv", *quadratic())
    code = main(["methods", str(path), "--x", "x", "--y", "y", "--poly", "2"])
    out = capsys.readouterr().out
    assert code == 0
    text = methods_text(fit_dataset(as_xy(*quadratic()), QUADRATIC), spec=QUADRATIC)
    assert text.missing == ()
    assert text.paragraph.startswith("A polynomial of degree 2 was fitted to 40 observations")
    assert text.equation.startswith("y = b0 + b1*x + b2*x**2")
    assert (
        out
        == "\n".join(
            [
                text.paragraph,
                "",
                text.equation,
                "",
                text.table,
                "",
                "Cite as:",
                text.citation_curvestudio,
                *text.citation_model,
                "",
                "For you to complete:",
                *text.to_complete,
            ]
        )
        + "\n"
    )


def test_methods_json_is_the_tools_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(tmp_path / "quadratic.csv", *quadratic())
    code = main(["methods", str(path), "--x", "x", "--y", "y", "--poly", "2", "--json"])
    out = capsys.readouterr().out
    expected = methods_text(fit_dataset(as_xy(*quadratic()), QUADRATIC), spec=QUADRATIC)
    assert code == 0
    assert json.loads(out) == json.loads(expected.model_dump_json())


def test_methods_style_markdown_and_latex(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv(tmp_path / "quadratic.csv", *quadratic())
    code = main(
        ["methods", str(path), "--x", "x", "--y", "y", "--poly", "2", "--style", "markdown"]
    )
    markdown = capsys.readouterr().out.splitlines()
    assert code == 0
    table = next(i for i, line in enumerate(markdown) if line.startswith("| Parameter |"))
    assert markdown[table + 1] == "| " + " | ".join(["---"] * 5) + " |"  # the rule row
    assert all(line.startswith("| b") for line in markdown[table + 2 : table + 5])
    assert (
        main(["methods", str(path), "--x", "x", "--y", "y", "--poly", "2", "--style", "latex"]) == 0
    )
    latex = capsys.readouterr().out
    assert "\\begin{tabular}{lrrrr}" in latex
    assert "\\end{tabular}" in latex


def test_an_unknown_style_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["methods", str(DECAY), "--x", "x", "--y", "y", "--poly", "2", "--style", "bogus"])
    captured = capsys.readouterr()
    assert code == 3
    assert "bogus" in captured.err
    assert captured.out == ""


def test_methods_text_lists_what_is_missing(capsys: pytest.CaptureFixture[str]) -> None:
    text = MethodsText(
        paragraph="A paragraph.",
        equation="y = b0",
        table="Parameter  Estimate",
        citation_curvestudio="CurveStudio 0. ConcreteInfo. https://example.invalid",
        citation_model=("A source.",),
        to_complete=("Instrument and measurement method",),
        missing=("CODE: a sentence.", "OTHER: another."),
    )
    _print_methods_text(text)
    out = capsys.readouterr().out
    assert "For you to complete:\nInstrument and measurement method\n\nNot known:\n" in out
    assert out.endswith("Not known:\nCODE: a sentence.\nOTHER: another.\n")


def test_methods_refuses_a_fit_that_did_not_converge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sparse = tmp_path / "three.csv"
    sparse.write_text("x,y\n1,2.0\n2,4.1\n3,6.2\n")
    code = main(["methods", str(sparse), "--x", "x", "--y", "y", "--poly", "3"])
    captured = capsys.readouterr()
    assert code == 3
    assert "FIT_NOT_CONVERGED_INPUT" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


# --- errors and help -------------------------------------------------------------------------


USAGE_ERRORS = [
    ["--y", "y", "--poly", "2"],  # no --x
    ["--x", "x", "--poly", "2"],  # no --y
    ["--x", "x", "--y", "y"],  # neither --model nor --poly
    ["--x", "x", "--y", "y", "--model", "exp_growth", "--poly", "2"],
    ["--x", "x", "--y", "y", "--poly", "11"],
    ["--x", "x", "--y", "y", "--bogus"],  # an option the command does not have
]


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("flags", USAGE_ERRORS)
def test_usage_errors_exit_3(
    command: str, flags: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([command, str(DECAY), *flags])
    captured = capsys.readouterr()
    assert code == 3
    assert captured.err.strip() != ""
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_audit_option_values_are_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    for flags in (["--cv", "1"], ["--cv", "five"], ["--poly", "2", "--style", "text"]):
        code = main(["audit", str(DECAY), "--x", "x", "--y", "y", *flags])
        captured = capsys.readouterr()
        assert code == 3, flags
        assert captured.out == ""


@pytest.mark.parametrize("command", COMMANDS)
def test_input_errors_exit_3_with_their_code(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [command, str(tmp_path / "absent.csv"), "--x", "x", "--y", "y", "--model", "exp_growth"]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert "FILE_NOT_FOUND" in captured.err
    assert captured.out == ""
    code = main([command, str(DECAY), "--x", "x", "--y", "y", "--model", "exp_decay_offse"])
    captured = capsys.readouterr()
    assert code == 3
    assert "UNKNOWN_MODEL" in captured.err
    assert "exp_decay_offset" in captured.err  # the registry's own near-match hint
    assert captured.out == ""
    code = main([command, str(DECAY), "--x", "nope", "--y", "y", "--poly", "2"])
    captured = capsys.readouterr()
    assert code == 3
    assert "UNKNOWN_COLUMN" in captured.err
    assert captured.out == ""


def test_help_lists_the_commands_and_explains_the_verdicts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "{fit,autofit,models,audit,methods}" in out
    assert main(["audit", "--help"]) == 0
    audit_help = " ".join(capsys.readouterr().out.split())  # argparse wraps its epilog
    assert "evidence, not proof" in audit_help
    assert (
        "Exit codes: 0 pass, 1 caution or not applicable, 2 fail, 3 usage or input error"
        in audit_help
    )
    assert main(["methods", "--help"]) == 0
    methods_help = " ".join(capsys.readouterr().out.split())
    assert "Exit codes: 0 text written, 3 usage or input error" in methods_help
    assert "evidence, not proof" in methods_help
