"""Tests for the NIST validation report generator (``scripts/nist_report.py``).

The renderer is a pure function from a hand-built report to Markdown, so its output is
checked character for character. ``build_report`` is exercised on the fast subset
(Misra1a nonlinear, Norris linear) only; the full 12+5 sweep is what
``uv run python scripts/nist_report.py`` writes to ``tests/numerical/VALIDATION.md``.
"""

import re
import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from nist_report import (  # noqa: E402
    LinearRow,
    MultistartRow,
    NonlinearRow,
    ReportData,
    Summary,
    _summary,
    build_report,
    main,
    render_markdown,
)

EXPECTED_TINY = """# NIST StRD validation report

Generated: 2026-11-16T17:00:00+05:30 (IST)
Platform: Linux-test
Versions: curvestudio 0.1.0 · numpy 2.0.0 · scipy 1.0.0 · lmfit 1.0.0 · sympy 1.0.0 · Python 3.14.0

Rule 1 — a CONVERGED fit must be right: minimum parameter LRE >= 5 and SSE LRE >= 7
(LRE = log relative error against the certified value, i.e. correct significant digits).

## Table 1 — nonlinear, single start

| Problem | Difficulty | Start | Status | Min param LRE | Min SE LRE | SSE LRE |
| --- | --- | --- | --- | --- | --- | --- |
| Misra1a | lower | start1 | converged | 9.0 | 4.4 | 9.5 |
| Misra1a | lower | start2 | converged | 8.8 | 4.3 | 9.4 |
| Gauss1 | lower | start2 | not_converged | n/a | n/a | n/a |
| Rat42 | higher | start1 | converged | 3.2 **VIOLATION** | n/a | 8.0 |

## Table 2 — nonlinear, multistart (20 starts, seed 0, bounds ±1000·|start|)

| Problem | Difficulty | Start | Status | Min param LRE | SSE LRE | Converged starts | Matching certified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Eckerle4 * | higher | start1 | converged | 8.8 | 9.1 | 17/20 | 17/20 |
| Gauss1 | lower | start1 | not_converged | n/a | n/a | 0/20 | 0/20 |

## Table 3 — linear (fit_polynomial)

| Problem | Status | Min coefficient LRE | Warnings |
| --- | --- | --- | --- |
| Norris | converged | 12.3 | - |

## Summary

- Nonlinear single-start CONVERGED with rule 1 satisfied: 2/4
- Nonlinear single-start NOT_CONVERGED: 1
- Rule-1 violations: 1 (must be 0)
- Lower-difficulty problems converged from both starts: 1/2
- Linear problems passing: 1/1

## Notes

- note one
"""  # noqa: E501 — the spec-mandated table rows are wider than 100 columns


def _tiny_report() -> ReportData:
    """A hand-built report: Misra1a clean both starts, Gauss1 not converged (with measured
    0.0 LREs that must display as n/a), Rat42 violating, and both multistart count shapes."""
    return ReportData(
        generated_at="2026-11-16T17:00:00+05:30",
        versions={
            "curvestudio": "0.1.0",
            "numpy": "2.0.0",
            "scipy": "1.0.0",
            "lmfit": "1.0.0",
            "sympy": "1.0.0",
            "Python": "3.14.0",
        },
        platform_info="Linux-test",
        single_rows=(
            NonlinearRow("Misra1a", "lower", "start1", "converged", 8.96, 4.41, 9.51),
            NonlinearRow("Misra1a", "lower", "start2", "converged", 8.8, 4.3, 9.4),
            NonlinearRow("Gauss1", "lower", "start2", "not_converged", 0.0, None, 0.0),
            NonlinearRow("Rat42", "higher", "start1", "converged", 3.21, None, 8.02),
        ),
        multistart_rows=(
            MultistartRow(
                "Eckerle4",
                "higher",
                "start1",
                "converged",
                8.8,
                9.1,
                converged_starts=17,
                matching_certified=17,
                mirrored=True,
            ),
            MultistartRow("Gauss1", "lower", "start1", "not_converged", 0.0, 0.0, 0, 0),
        ),
        linear_rows=(LinearRow("Norris", "converged", 12.3, ()),),
        summary=Summary(
            converged_rule1=2,
            single_total=4,
            not_converged=1,
            violations=1,
            lower_both_starts=1,
            lower_total=2,
            linear_passed=1,
            linear_total=1,
            errors=(),
        ),
        notes=("note one",),
    )


def test_renderer_produces_the_expected_markdown() -> None:
    assert render_markdown(_tiny_report()) == EXPECTED_TINY


def test_summary_lower_denominator_is_the_lower_difficulty_problem_count() -> None:
    """2 of 3 problems are lower difficulty, both converge from both starts: 2/2, not 2/3."""
    rows = (
        NonlinearRow("Misra1a", "lower", "start1", "converged", 8.96, 9.0, 10.5),
        NonlinearRow("Misra1a", "lower", "start2", "converged", 8.8, 8.9, 10.4),
        NonlinearRow("Gauss1", "lower", "start1", "converged", 7.2, 7.0, 10.6),
        NonlinearRow("Gauss1", "lower", "start2", "converged", 7.4, 7.1, 10.6),
        NonlinearRow("Rat42", "higher", "start1", "converged", 9.2, 9.0, 11.9),
    )
    s = _summary(rows, (), (), (), (), ())
    assert s.lower_both_starts == 2
    assert s.lower_total == 2
    assert s.violations == 0
    assert s.converged_rule1 == 5


def test_build_report_restricted_to_the_fast_subset() -> None:
    report = build_report(("Misra1a", "Norris"))
    text = render_markdown(report)

    assert "# NIST StRD validation report" in text
    assert "## Table 1 — nonlinear, single start" in text
    assert "## Table 2 — nonlinear, multistart" in text
    assert "## Table 3 — linear (fit_polynomial)" in text
    assert "| Misra1a |" in text and "| Norris |" in text
    assert report.summary.errors == ()
    assert report.summary.mismatches == ()
    assert report.summary.violations == 0
    assert report.summary.single_total == 2
    assert report.summary.linear_total == 1
    assert [row.status for row in report.single_rows] == ["converged", "converged"]
    assert any("scripts/nist_report.py" in note for note in report.notes)

    table2 = text.split("## Table 2")[1].split("## Table 3")[0]
    ms_rows = [line for line in table2.splitlines() if line.startswith("| Misra1a ")]
    assert len(ms_rows) == 2
    for line in ms_rows:
        assert re.search(r"\| \d+/20 \| \d+/20 \|$", line), line


def test_build_report_rejects_unknown_problem_names() -> None:
    try:
        build_report(("Misra1a", "NotAProblem"))
    except ValueError as exc:
        assert "NotAProblem" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown problem name")


def test_main_writes_the_report_and_exits_zero(tmp_path: Path) -> None:
    out = tmp_path / "subset.md"

    assert main(["--problems", "Misra1a", "Norris", "--out", str(out)]) == 0
    written = out.read_text(encoding="utf-8")
    assert written.startswith("# NIST StRD validation report")
    assert "| Misra1a |" in written and "| Norris |" in written
