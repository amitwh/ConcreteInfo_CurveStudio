"""Generate the NIST StRD validation report (tests/numerical/VALIDATION.md).

Every fit uses the exact configurations of the benchmark tests, through their shared
harness (``tests/numerical/nist_harness.py``): single-start fits from each NIST start,
multistart fits with 20 seeded starts inside +-1000 x |start value| bounds, and the five
linear problems via ``fit_polynomial``. The output grades each fit against the certified
NIST values with the LRE measure; a CONVERGED fit that is wrong (min parameter LRE < 5 or
SSE LRE < 7) is a rule-1 violation, printed as **VIOLATION** and reported through the
exit status. A fit that raises under the warnings-as-errors benchmark condition is
recorded as ``ERROR: <ExceptionType>`` and also makes the exit status 1.

Exit status: 0 when there are no violations and no errors, 1 otherwise.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _REPO_ROOT / "tests" / "numerical"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from nist_harness import (  # noqa: E402
    CONVERGED_MIN_PARAM_LRE,
    CONVERGED_MIN_SSE_LRE,
    LINEAR_ACCURATE_MIN_LRE,
    LINEAR_ILL_CONDITIONED_MIN_LRE,
    LINEAR_ILL_CONDITIONED_NAMES,
    LOWER_MIN_PARAM_LRE,
    MIRRORED_CASES,
    MULTISTART_N,
    MULTISTART_SEED,
    STARTS,
    coefficient_lres,
    linear_problem,
    mirror_reference,
    multistart_config,
    param_lres,
    problem,
    run_fit,
    run_linear_fit,
    se_lres,
    sse_lre,
    start_values,
)
from nist_models import LINEAR_NAMES, NONLINEAR_NAMES  # noqa: E402

from curvestudio.fitting import FitConfig, FitStatus  # noqa: E402
from curvestudio.fitting.result import IST  # noqa: E402

DEFAULT_OUT = _HARNESS_DIR / "VALIDATION.md"

_VERSION_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("curvestudio", "curvestudio-core"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("lmfit", "lmfit"),
    ("sympy", "sympy"),
)

# "Converged starts" / "Matching certified" are not reported per start: FitResult exposes
# only the winning run plus aggregate MULTISTART_* warnings, and the report does not
# fabricate what the engine does not say.
NOT_REPORTED = "n/a"


@dataclass(frozen=True)
class NonlinearRow:
    """One single-start fit: the run the benchmark test makes for this problem and start."""

    problem: str
    difficulty: str
    start: str
    status: str
    param_lre: float | None
    se_lre: float | None
    sse_lre: float | None


@dataclass(frozen=True)
class MultistartRow:
    """One 20-start fit. ``param_lre`` is mirror-aware for the Eckerle4 sign symmetry."""

    problem: str
    difficulty: str
    start: str
    status: str
    param_lre: float | None
    sse_lre: float | None
    converged_starts: str = NOT_REPORTED
    matching_certified: str = NOT_REPORTED
    mirrored: bool = False


@dataclass(frozen=True)
class LinearRow:
    """One polynomial fit; ``warnings`` holds the warning codes (text before the colon)."""

    problem: str
    status: str
    coef_lre: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Summary:
    """The counts printed in the Summary block."""

    converged_rule1: int
    single_total: int
    not_converged: int
    violations: int
    lower_both_starts: int
    nonlinear_total: int
    linear_passed: int
    linear_total: int
    errors: tuple[str, ...] = ()
    other_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportData:
    """Everything the renderer needs; built by :func:`build_report`, fully deterministic."""

    generated_at: str
    versions: dict[str, str]
    platform_info: str
    single_rows: tuple[NonlinearRow, ...]
    multistart_rows: tuple[MultistartRow, ...]
    linear_rows: tuple[LinearRow, ...]
    summary: Summary
    notes: tuple[str, ...]


def _versions() -> dict[str, str]:
    """Snapshot the tracked distributions plus Python (``"unknown"`` when not installed)."""
    versions: dict[str, str] = {}
    for label, distribution in _VERSION_DISTRIBUTIONS:
        try:
            versions[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[label] = "unknown"
    versions["Python"] = platform.python_version()
    return versions


def _resolve_problems(problems: Sequence[str] | None) -> tuple[list[str], list[str]]:
    """Split the requested names into (nonlinear, linear), preserving order.

    Raises:
        ValueError: a name is not one of the vendored NIST problems.
    """
    if problems is None:
        return list(NONLINEAR_NAMES), list(LINEAR_NAMES)
    nonlinear, linear, unknown = [], [], []
    for name in problems:
        if name in NONLINEAR_NAMES:
            nonlinear.append(name)
        elif name in LINEAR_NAMES:
            linear.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"unknown NIST problem name(s) {unknown}; valid names are "
            f"{list(NONLINEAR_NAMES)} and {list(LINEAR_NAMES)}"
        )
    return nonlinear, linear


def _param_violated(status: str, param_lre: float | None) -> bool:
    return status == FitStatus.CONVERGED.value and (
        param_lre is None or param_lre < CONVERGED_MIN_PARAM_LRE
    )


def _sse_violated(status: str, sse_lre: float | None) -> bool:
    return status == FitStatus.CONVERGED.value and (
        sse_lre is None or sse_lre < CONVERGED_MIN_SSE_LRE
    )


def _rule1_violation(status: str, param_lre: float | None, sse_lre: float | None) -> bool:
    """Rule 1: a CONVERGED fit must have min parameter LRE >= 5 and SSE LRE >= 7."""
    return _param_violated(status, param_lre) or _sse_violated(status, sse_lre)


def _linear_passes(name: str, row: LinearRow) -> bool:
    """The linear tests' acceptance rule, tier by tier."""
    if name in LINEAR_ILL_CONDITIONED_NAMES:
        accurate = (
            row.status == FitStatus.CONVERGED.value
            and row.coef_lre is not None
            and row.coef_lre >= LINEAR_ILL_CONDITIONED_MIN_LRE
        )
        return accurate or "ILL_CONDITIONED" in row.warnings
    return (
        row.status == FitStatus.CONVERGED.value
        and row.coef_lre is not None
        and row.coef_lre >= LINEAR_ACCURATE_MIN_LRE
    )


def _single_row(name: str, difficulty: str, start: str) -> NonlinearRow:
    p = problem(name)
    try:
        result = run_fit(name, FitConfig(initial=start_values(p, start)))
    except Exception as exc:  # recorded as ERROR, never swallowed silently
        return NonlinearRow(
            name, difficulty, start, f"ERROR: {type(exc).__name__}", None, None, None
        )
    param_min = min(param_lres(p, result).values()) if result.parameters else None
    se = se_lres(p, result)
    sse = None if result.stats is None else sse_lre(p, result)
    return NonlinearRow(
        name,
        difficulty,
        start,
        result.status.value,
        param_min,
        None if se is None else min(se.values()),
        sse,
    )


def _multistart_row(name: str, difficulty: str, start: str) -> MultistartRow:
    p = problem(name)
    try:
        result = run_fit(name, multistart_config(p, start))
    except Exception as exc:  # recorded as ERROR, never swallowed silently
        return MultistartRow(name, difficulty, start, f"ERROR: {type(exc).__name__}", None, None)
    mirrored = False
    if result.parameters:
        param_min = min(param_lres(p, result).values())
        if (name, start) in MIRRORED_CASES:
            mirror_min = min(param_lres(p, result, mirror_reference(p)).values())
            mirrored = mirror_min > param_min
            param_min = max(param_min, mirror_min)
    else:
        param_min = None
    sse = None if result.stats is None else sse_lre(p, result)
    return MultistartRow(
        name, difficulty, start, result.status.value, param_min, sse, mirrored=mirrored
    )


def _linear_row(name: str) -> LinearRow:
    p = linear_problem(name)
    try:
        result = run_linear_fit(p)
    except Exception as exc:  # recorded as ERROR, never swallowed silently
        return LinearRow(name, f"ERROR: {type(exc).__name__}", None, ())
    coef = min(coefficient_lres(p, result)) if result.parameters else None
    warnings = tuple(w.split(":", 1)[0] for w in result.warnings)
    return LinearRow(name, result.status.value, coef, warnings)


def _summary(
    single_rows: Sequence[NonlinearRow],
    multistart_rows: Sequence[MultistartRow],
    linear_rows: Sequence[LinearRow],
    nonlinear_total: int,
    errors: Sequence[str],
    other_statuses: Sequence[str],
) -> Summary:
    converged_rule1 = sum(
        1
        for r in single_rows
        if r.status == FitStatus.CONVERGED.value
        and not _rule1_violation(r.status, r.param_lre, r.sse_lre)
    )
    violations = sum(
        1 for r in single_rows if _rule1_violation(r.status, r.param_lre, r.sse_lre)
    ) + sum(1 for r in multistart_rows if _rule1_violation(r.status, r.param_lre, r.sse_lre))
    by_problem: dict[str, list[NonlinearRow]] = {}
    for r in single_rows:
        by_problem.setdefault(r.problem, []).append(r)
    lower_both_starts = sum(
        1
        for rows in by_problem.values()
        if rows[0].difficulty == "lower"
        and len(rows) == len(STARTS)
        and all(
            r.status == FitStatus.CONVERGED.value
            and r.param_lre is not None
            and r.param_lre >= LOWER_MIN_PARAM_LRE
            for r in rows
        )
    )
    return Summary(
        converged_rule1=converged_rule1,
        single_total=len(single_rows),
        not_converged=sum(1 for r in single_rows if r.status == FitStatus.NOT_CONVERGED.value),
        violations=violations,
        lower_both_starts=lower_both_starts,
        nonlinear_total=nonlinear_total,
        linear_passed=sum(1 for r in linear_rows if _linear_passes(r.problem, r)),
        linear_total=len(linear_rows),
        errors=tuple(errors),
        other_statuses=tuple(other_statuses),
    )


def _notes(multistart_rows: Sequence[MultistartRow]) -> tuple[str, ...]:
    notes: list[str] = []
    eckerle = [r for r in multistart_rows if r.problem == "Eckerle4"]
    if any(r.mirrored for r in eckerle):
        notes.append(
            "Eckerle4, (b1/b2) * exp(-0.5*((x-b3)/b2)**2), is invariant under "
            "(b1, b2) -> (-b1, -b2) with b3 unchanged: both sign choices fit the data "
            "equally well, so the data cannot tell them apart. The row marked * matched "
            "that mirror minimum rather than the certified signs, and its min param LRE is "
            "measured against the certified values or their mirror, whichever is closer."
        )
    elif eckerle:
        notes.append(
            "Eckerle4, (b1/b2) * exp(-0.5*((x-b3)/b2)**2), is invariant under "
            "(b1, b2) -> (-b1, -b2) with b3 unchanged; this run's Eckerle4 multistart "
            "rows matched the certified signs, so no mirror comparison was needed."
        )
    notes.append(
        "Fits use the engine's default ftol = 1e-14 rather than SciPy's 1e-10: ftol "
        "bounds the relative SSE change and the parameters are only resolved to about "
        "its square root, so 1e-10 let ill-conditioned fits report CONVERGED with about "
        "5 correct digits (NIST MGH09)."
    )
    notes.append(
        "Single-start caveat: a local optimiser run from one start can report CONVERGED "
        "at a stationary point that is not the certified minimum (a plateau or secondary "
        "minimum with a genuinely small gradient); the multistart runs in Table 2 inside "
        "wide bounds are the remedy."
    )
    notes.append(
        "The 'Converged starts' and 'Matching certified' columns are n/a: a FitResult "
        "reports only the winning run of a multistart fit (plus aggregate "
        "MULTISTART_DISAGREE / MULTISTART_BETTER_UNCONVERGED warnings), and per-start "
        "outcomes are not part of the result contract, so they are not invented here."
    )
    notes.append(
        "Generated by scripts/nist_report.py (uv run python scripts/nist_report.py); "
        "tests/numerical/test_nist_report.py checks the renderer and a restricted build."
    )
    return tuple(notes)


def build_report(problems: Sequence[str] | None = None) -> ReportData:
    """Run every fit and collect the report data.

    ``problems`` restricts the run to those NIST problem names (any mix of nonlinear and
    linear); the default is all 12 nonlinear and all 5 linear problems.
    """
    nonlinear, linear = _resolve_problems(problems)
    single_rows: list[NonlinearRow] = []
    multistart_rows: list[MultistartRow] = []
    linear_rows: list[LinearRow] = []
    errors: list[str] = []
    other: list[str] = []
    for name in nonlinear:
        p = problem(name)
        difficulty = p.difficulty or "unknown"
        for start in STARTS:
            single = _single_row(name, difficulty, start)
            single_rows.append(single)
            if single.status.startswith("ERROR:"):
                errors.append(f"{name}/{start}: {single.status}")
            elif single.status not in (
                FitStatus.CONVERGED.value,
                FitStatus.NOT_CONVERGED.value,
            ):
                other.append(f"{name}/{start}: {single.status}")
        for start in STARTS:
            ms_row = _multistart_row(name, difficulty, start)
            multistart_rows.append(ms_row)
            if ms_row.status.startswith("ERROR:"):
                errors.append(f"{name}/{start} (multistart): {ms_row.status}")
        print(f"nist_report: {name} done", file=sys.stderr)
    for name in linear:
        lin_row = _linear_row(name)
        linear_rows.append(lin_row)
        if lin_row.status.startswith("ERROR:"):
            errors.append(f"{name} (linear): {lin_row.status}")
        print(f"nist_report: {name} (linear) done", file=sys.stderr)
    summary = _summary(single_rows, multistart_rows, linear_rows, len(nonlinear), errors, other)
    return ReportData(
        generated_at=datetime.now(IST).isoformat(timespec="seconds"),
        versions=_versions(),
        platform_info=platform.platform(),
        single_rows=tuple(single_rows),
        multistart_rows=tuple(multistart_rows),
        linear_rows=tuple(linear_rows),
        summary=summary,
        notes=_notes(multistart_rows),
    )


def _lre_cell(value: float | None, violated: bool) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}" + (" **VIOLATION**" if violated else "")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(report: ReportData) -> str:
    """Render the report as Markdown. Pure: no fitting, no clock, no environment access."""
    lines: list[str] = [
        "# NIST StRD validation report",
        "",
        f"Generated: {report.generated_at} (IST)",
        f"Platform: {report.platform_info}",
        "Versions: " + " · ".join(f"{k} {v}" for k, v in report.versions.items()),
        "",
        "Rule 1 — a CONVERGED fit must be right: minimum parameter LRE >= 5 and SSE LRE >= 7",
        "(LRE = log relative error against the certified value, i.e. correct significant digits).",
        "",
        "## Table 1 — nonlinear, single start",
        "",
    ]
    lines.extend(
        _table(
            ("Problem", "Difficulty", "Start", "Status", "Min param LRE", "Min SE LRE", "SSE LRE"),
            [
                (
                    r.problem,
                    r.difficulty,
                    r.start,
                    r.status,
                    _lre_cell(r.param_lre, _param_violated(r.status, r.param_lre)),
                    _lre_cell(r.se_lre, False),
                    _lre_cell(r.sse_lre, _sse_violated(r.status, r.sse_lre)),
                )
                for r in report.single_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            f"## Table 2 — nonlinear, multistart "
            f"({MULTISTART_N} starts, seed {MULTISTART_SEED}, bounds ±1000·|start|)",
            "",
        ]
    )
    lines.extend(
        _table(
            (
                "Problem",
                "Difficulty",
                "Start",
                "Status",
                "Min param LRE",
                "SSE LRE",
                "Converged starts",
                "Matching certified",
            ),
            [
                (
                    r.problem + (" *" if r.mirrored else ""),
                    r.difficulty,
                    r.start,
                    r.status,
                    _lre_cell(r.param_lre, _param_violated(r.status, r.param_lre)),
                    _lre_cell(r.sse_lre, _sse_violated(r.status, r.sse_lre)),
                    r.converged_starts,
                    r.matching_certified,
                )
                for r in report.multistart_rows
            ],
        )
    )
    lines.extend(["", "## Table 3 — linear (fit_polynomial)", ""])
    lines.extend(
        _table(
            ("Problem", "Status", "Min coefficient LRE", "Warnings"),
            [
                (
                    r.problem,
                    r.status,
                    _lre_cell(r.coef_lre, False),
                    ", ".join(r.warnings) if r.warnings else "-",
                )
                for r in report.linear_rows
            ],
        )
    )
    s = report.summary
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Nonlinear single-start CONVERGED with rule 1 satisfied: "
            f"{s.converged_rule1}/{s.single_total}",
            f"- Nonlinear single-start NOT_CONVERGED: {s.not_converged}",
            f"- Rule-1 violations: {s.violations} (must be 0)",
            f"- Lower-difficulty problems converged from both starts: "
            f"{s.lower_both_starts}/{s.nonlinear_total}",
            f"- Linear problems passing: {s.linear_passed}/{s.linear_total}",
        ]
    )
    lines.extend(f"- Errors: {e}" for e in s.errors)
    lines.extend(f"- Other single-start statuses: {e}" for e in s.other_statuses)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output Markdown path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=None,
        metavar="NAME",
        help="restrict to these NIST problem names (default: all vendored problems)",
    )
    args = parser.parse_args(argv)
    report = build_report(args.problems)
    text = render_markdown(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.out}: {report.summary.violations} rule-1 violation(s), "
        f"{len(report.summary.errors)} error(s)"
    )
    return 1 if report.summary.violations or report.summary.errors else 0


if __name__ == "__main__":
    sys.exit(main())
