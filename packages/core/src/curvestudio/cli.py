"""Command-line interface: the vertical slice from a file to a fitted report.

``curvestudio fit FILE`` imports a table, validates the requested columns, fits one
curve (a registered model or a polynomial) and prints the equation with the fitted
numbers, a parameter table (value, standard error, 95 % CI), R², adjusted R², RMSE,
the status line and any warnings; ``--json`` additionally writes the full
:class:`~curvestudio.fitting.FitResult` machine-readably. ``curvestudio models``
lists the registry. ``curvestudio autofit FILE`` is the headline command: it fits every
applicable model (:func:`curvestudio.discovery.autofit`) and prints the converged fits ranked
best to worst, with the candidates that were skipped, did not converge or fit exactly listed
beside the ranking; ``--json`` prints the whole :class:`~curvestudio.discovery.AutoFitResult`.

``curvestudio audit FILE`` fits one curve the way the apps do, through the workflow layer
(:func:`curvestudio.workflow.fit_dataset`), and prints what its evidence shows, check by
check (:func:`curvestudio.tools.audit_fit`). ``--cv`` adds a cross-validation and
``--ranking`` an Auto-fit run with the ``autofit`` command's defaults, so that the two checks
that need them are decided; ``--json`` prints the :class:`~curvestudio.tools.FitAudit`.
``curvestudio methods FILE`` makes the same fit and prints its methods text
(:func:`curvestudio.tools.methods_text`): the paragraph, the equation, the table and the
citations, laid out in the ``--style`` asked for; ``--json`` prints the
:class:`~curvestudio.tools.MethodsText`.

This module is the outermost layer of the package: it may import the layers below it (data,
fitting, models, discovery, workflow and tools), and nothing imports it. A user mistake
(unreadable file, missing column, unknown model, data that fails validation, an impossible
method/loss combination) is reported as a message on stderr, never as a traceback, and drives
the exit code:

===========  =======================================================================
Command      Exit codes
===========  =======================================================================
``fit``      ``0`` the fit converged. ``2`` invalid input, including a fit result of
             ``invalid_input`` (too few observations, or a model that does not apply).
             ``3`` the fit ran but did not converge or failed.
``autofit``  ``0`` the run completed (whatever it found). ``1`` it could not run (an
             unreadable file, a missing column, data that fails validation, or an
             ``AutoFitError`` such as ``TOO_FEW_POINTS``). ``2`` a usage error, as argparse
             reports it.
``models``   ``0``.
``audit``    ``0`` every check passed. ``1`` a check reads ``caution``, or none had a
             number to read (``not_applicable``). ``2`` a check failed. ``3`` a usage or
             input error: bad arguments, an unreadable file, a specification the workflow
             refuses, or a cross-validation or Auto-fit run the tools refuse.
``methods``  ``0`` the text was written. ``3`` a usage or input error, or a fit that did
             not converge (the tool refuses it: there are no estimates to report).
===========  =======================================================================
"""

import argparse
import difflib
import math
import re
import sys
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from curvestudio.data import XY, DataError, Issue, read_dataset, validate
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitError,
    AutoFitResult,
    RankBy,
    RankedFit,
    autofit,
)
from curvestudio.fitting import (
    FitConfig,
    FitResult,
    FitStatus,
    ParameterEstimate,
    fit,
    fit_polynomial,
)
from curvestudio.fitting.methods import LOSSES, METHODS
from curvestudio.models import get_model, list_models
from curvestudio.tools import (
    FitAudit,
    MethodsText,
    ToolError,
    Verdict,
    audit_fit,
    cross_validate,
    methods_text,
)
from curvestudio.tools.methods import STYLES
from curvestudio.workflow import (
    FitBundle,
    FitSpec,
    PolynomialRef,
    RegisteredRef,
    WorkflowError,
    fit_dataset,
)

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_NOT_CONVERGED = 3
EXIT_AUTOFIT_ERROR = 1
"""``autofit`` could not run: bad file, columns or data, or an ``AutoFitError``."""

EXIT_USAGE = 2
"""A usage error, as argparse reports it for every command but ``audit`` and ``methods``."""

EXIT_CAUTION = 1
"""``audit``: a check reads ``caution``, or none had a number to read (``not_applicable``)."""

EXIT_AUDIT_FAIL = 2
"""``audit``: a check failed."""

EXIT_INPUT_ERROR = 3
"""``audit`` and ``methods``: a usage or input error. For ``audit`` the numbers below it are
verdicts, so its usage errors exit here too; for ``methods`` a fit that did not converge is
one, as it has no estimates to report."""

_AUDIT_EXIT_CODES = {
    Verdict.PASS: EXIT_OK,
    Verdict.CAUTION: EXIT_CAUTION,
    Verdict.FAIL: EXIT_AUDIT_FAIL,
    Verdict.NOT_APPLICABLE: EXIT_CAUTION,
}
"""The exit code of ``audit`` for each overall verdict."""

_USAGE_IS_INPUT_ERROR = frozenset({"audit", "methods"})
"""The commands whose usage errors exit ``3``: for ``audit`` argparse's ``2`` is a verdict, and
``methods`` gives its input errors the same code."""

MAX_STARTS = 1000
"""Upper bound for --starts: multistart cost and memory grow linearly in it."""

MAX_POLY_DEGREE = 10
"""Upper bound for --poly: the spec's polynomial family is degrees 1..10."""

DEFAULT_TOP = 10
"""Rows of the ranked table shown unless ``--top`` says otherwise."""

_ERROR_PREFIX = "curvestudio: error:"
_ROW_NOTE = (
    "Row numbers in messages are 1-based data rows: the first data row after the "
    "header is row 1; fully blank rows dropped at import are not counted."
)


# --- argument parsing -------------------------------------------------------------------


def _parse_int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None


def _non_negative_int(text: str) -> int:
    value = _parse_int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be an integer >= 0, got {text!r}")
    return value


def _bounded_int(low: int, high: int, flag: str) -> Callable[[str], int]:
    """An argparse ``type`` that enforces ``low <= value <= high`` for ``flag``."""

    def parse(text: str) -> int:
        value = _parse_int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(
                f"{flag} must be an integer between {low} and {high}, got {text!r}"
            )
        return value

    return parse


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {text!r}")
    return value


def _folds(text: str) -> int | Literal["loo"]:
    """An argparse ``type`` for ``--cv``: ``loo``, or a number of folds from 2 up."""
    if text == "loo":
        return "loo"
    value = _parse_int(text)
    if value < 2:
        raise argparse.ArgumentTypeError(f"--cv must be 'loo' or an integer >= 2, got {text!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    """The ``curvestudio`` argument parser: every command is a subcommand of it."""
    parser = argparse.ArgumentParser(
        prog="curvestudio",
        description="Fit curves to data tables.",
        epilog=(
            f"{_ROW_NOTE} Exit codes of fit: 0 converged, 2 invalid input, 3 not converged; "
            "each other command states its own in its help."
        ),
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="{fit,autofit,models,audit,methods}"
    )

    fit_command = commands.add_parser(
        "fit",
        help="import a file, validate the columns and fit one curve",
        description="Import FILE, validate the requested columns, fit one curve, print a report.",
        epilog=_ROW_NOTE,
    )
    fit_command.add_argument("file", help="input file: csv, tsv, txt, json or xlsx")
    fit_command.add_argument("--x", required=True, metavar="COL", help="column with the x values")
    fit_command.add_argument("--y", required=True, metavar="COL", help="column with the y values")
    target = fit_command.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model",
        metavar="ID",
        help="registered model id (see 'curvestudio models')",
    )
    target.add_argument(
        "--poly",
        type=_bounded_int(1, MAX_POLY_DEGREE, "--poly"),
        metavar="N",
        help=f"fit a degree-N polynomial by (weighted) least squares (1-{MAX_POLY_DEGREE})",
    )
    fit_command.add_argument("--weights", metavar="COL", help="column holding positive weights")
    fit_command.add_argument(
        "--method",
        choices=sorted(METHODS),
        metavar="M",
        help="optimisation method (default: auto)",
    )
    fit_command.add_argument(
        "--loss",
        choices=sorted(LOSSES),
        metavar="L",
        help="loss for residual down-weighting (default: linear)",
    )
    fit_command.add_argument(
        "--starts",
        type=_bounded_int(1, MAX_STARTS, "--starts"),
        metavar="N",
        help=f"number of starts for a multistart fit (1-{MAX_STARTS}; default: 1)",
    )
    fit_command.add_argument(
        "--seed",
        type=_non_negative_int,
        metavar="S",
        help="seed for the multistart restarts (default: 0)",
    )
    fit_command.add_argument(
        "--json",
        type=Path,
        metavar="OUT",
        help="also write the full fit result (FitResult.to_json) to OUT",
    )

    _add_autofit_command(commands)
    _add_audit_command(commands)
    _add_methods_command(commands)

    commands.add_parser(
        "models",
        help="list the registered models",
        description="List the registered models: id, name and category.",
    )
    return parser


def _add_autofit_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "autofit",
        help="fit every applicable model and rank them best to worst",
        description=(
            "Import FILE, fit every model the data can support (the model library, "
            "polynomials and generated families) and print the converged fits ranked best "
            "to worst, with the skipped, not converged and exact candidates listed after them."
        ),
        epilog=f"{_ROW_NOTE} Exit codes: 0 completed, 1 could not run, 2 usage error.",
    )
    command.add_argument("file", help="input file: csv, tsv, txt, json or xlsx")
    command.add_argument("--x", required=True, metavar="COL", help="column with the x values")
    command.add_argument("--y", required=True, metavar="COL", help="column with the y values")
    command.add_argument("--weights", metavar="COL", help="column holding non-negative weights")
    command.add_argument(
        "--rank-by",
        choices=[rank.value for rank in RankBy],
        default=RankBy.BEST_OVERALL.value,
        metavar="CRITERION",
        help="ordering: " + ", ".join(rank.value for rank in RankBy) + " (default: best_overall)",
    )
    command.add_argument(
        "--top",
        type=_bounded_int(1, 100_000, "--top"),
        default=DEFAULT_TOP,
        metavar="N",
        help=f"show the N best rows of the table (default: {DEFAULT_TOP})",
    )
    command.add_argument(
        "--workers",
        type=_bounded_int(1, 256, "--workers"),
        metavar="N",
        help="worker processes (default: all but one core; 1 runs in this process)",
    )
    command.add_argument(
        "--n-starts",
        type=_bounded_int(1, MAX_STARTS, "--n-starts"),
        metavar="N",
        help=f"starts per candidate (1-{MAX_STARTS}; default: {AutoFitConfig().n_starts})",
    )
    command.add_argument(
        "--seed",
        type=_non_negative_int,
        metavar="S",
        help="seed of the run (default: 0); the same seed reproduces the same result",
    )
    command.add_argument(
        "--time-limit",
        type=_positive_float,
        metavar="S",
        help=f"seconds allowed per candidate (default: {AutoFitConfig().time_limit:g})",
    )
    command.add_argument(
        "--exhaustive",
        action="store_true",
        help="also try the long tail of generated families and models of up to 10 parameters",
    )
    command.add_argument(
        "--json",
        action="store_true",
        help="print the whole result (AutoFitResult.to_json) instead of the table",
    )


def _add_audit_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "audit",
        help="fit one curve and report, check by check, what its evidence shows",
        description=(
            "Import FILE, fit one curve the way the apps do and print what the fit's evidence "
            "shows, check by check: the convergence, the residual tests, the parameters and "
            "the number of points, plus a cross-validation and an Auto-fit ranking when asked "
            "for."
        ),
        epilog=(
            "A verdict is evidence, not proof: pass means no check found anything, caution "
            "means look before you rely on the fit, fail means do not rely on the fit as it "
            "is, not applicable means the check had no number to read. "
            f"{_ROW_NOTE} Exit codes: 0 pass, 1 caution or not applicable, 2 fail, "
            "3 usage or input error."
        ),
    )
    _add_fit_target(command)
    command.add_argument(
        "--cv",
        type=_folds,
        metavar="K|loo",
        help="also cross-validate the fit, with K folds or leave-one-out (loo)",
    )
    command.add_argument(
        "--ranking",
        action="store_true",
        help=(
            "also run Auto-fit with the settings the autofit command uses by default, "
            "to look for a simpler model that fits as well"
        ),
    )
    command.add_argument(
        "--json",
        action="store_true",
        help="print the whole audit (FitAudit as JSON) instead of the text",
    )


def _add_methods_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "methods",
        help="fit one curve and write the methods text for a report",
        description=(
            "Import FILE, fit one curve the way the apps do and print the methods text: the "
            "paragraph, the equation, the parameter table and the citations."
        ),
        epilog=(
            "Exit codes: 0 text written, 3 usage or input error, or a fit that did not "
            "converge (there are no estimates to report). The text describes the fit; whether "
            "to rely on it is what 'curvestudio audit' reports, and a verdict there is "
            "evidence, not proof."
        ),
    )
    _add_fit_target(command)
    command.add_argument(
        "--style",
        choices=STYLES,
        default="text",
        help="layout of the equation and the table (default: text)",
    )
    command.add_argument(
        "--json",
        action="store_true",
        help="print the whole text (MethodsText as JSON) instead of printing it laid out",
    )


# --- entry point ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, run one subcommand and return the process exit code.

    ``argv`` defaults to ``sys.argv[1:]`` (the console-script behaviour). Argparse's
    own exits (usage errors, ``--help``) are converted into return codes so that
    ``main`` is fully testable in process: a usage error is ``2`` as argparse reports
    it, but ``3`` for ``audit`` and ``methods``. On Windows consoles the report's
    ``R²`` needs a UTF-8 stdout, so the stream is reconfigured when that is possible.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    tokens = sys.argv[1:] if argv is None else argv
    try:
        args = _build_parser().parse_args(tokens)
    except SystemExit as exc:
        return _argparse_exit_code(exc.code, tokens)
    if args.command == "fit":
        return _run_fit(args)
    if args.command == "autofit":
        return _run_autofit(args)
    if args.command == "audit":
        return _run_audit(args)
    if args.command == "methods":
        return _run_methods(args)
    return _run_models()


def _argparse_exit_code(code: object, argv: list[str]) -> int:
    """The exit code of argparse's own exit.

    A usage error is ``2`` as argparse reports it, except for ``audit`` and ``methods``, where
    it is ``3``: the code both give every input error (and for ``audit`` a verdict).
    """
    if code == EXIT_USAGE and _command_of(argv) in _USAGE_IS_INPUT_ERROR:
        return EXIT_INPUT_ERROR
    return code if isinstance(code, int) else 0


def _command_of(argv: list[str]) -> str | None:
    """The subcommand ``argv`` names, or ``None`` when it names none.

    The parser takes no option of its own but the help flag, so the first argument that is
    neither that flag nor ``--`` is the one argparse dispatches on.
    """
    return next((token for token in argv if token not in ("-h", "--help", "--")), None)


# --- fit ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReportContext:
    """The bits of the invocation a report header shows alongside the fit result."""

    model_label: str
    dataset_name: str
    n_rows: int
    x: str
    y: str
    weights: str | None


def _run_fit(args: argparse.Namespace) -> int:
    """Import, validate, fit and report; the return value is the exit code."""
    try:
        dataset = read_dataset(args.file)
    except DataError as exc:
        return _fail(_data_error(exc))

    try:
        issues = validate(dataset, args.x, args.y, args.weights)
    except DataError as exc:
        return _fail(_data_error(exc))
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        return _fail_lines([_issue_line(issue) for issue in errors])
    data_warnings = [issue for issue in issues if issue.severity == "warning"]

    try:
        xy = dataset.xy(args.x, args.y, args.weights)
    except DataError as exc:
        return _fail(_data_error(exc))

    if args.model is not None:
        try:
            spec = get_model(args.model)
        except KeyError:
            return _fail(_unknown_model(args.model))
        try:
            config = FitConfig(
                method=args.method if args.method is not None else "auto",
                loss=args.loss if args.loss is not None else "linear",
                n_starts=args.starts if args.starts is not None else 1,
                seed=args.seed if args.seed is not None else 0,
            )
            result = fit(spec, xy, config)
        except ValueError as exc:
            return _fail(f"invalid fit configuration: {exc}")
        model_label = f"{spec.id} ({spec.name})"
    else:
        rejected = [
            flag
            for flag, value in (
                ("--method", args.method),
                ("--loss", args.loss),
                ("--starts", args.starts),
                ("--seed", args.seed),
            )
            if value is not None
        ]
        if rejected:
            return _fail(f"{', '.join(rejected)} can only be used with --model, not with --poly")
        try:
            result = fit_polynomial(xy, args.poly)
        except ValueError as exc:
            return _fail(str(exc))
        model_label = f"polynomial of degree {args.poly}"

    context = _ReportContext(
        model_label=model_label,
        dataset_name=dataset.name,
        n_rows=int(xy.x.size),
        x=args.x,
        y=args.y,
        weights=args.weights,
    )
    _print_report(result, context, data_warnings)

    if args.json is not None:
        try:
            args.json.write_text(result.to_json() + "\n", encoding="utf-8")
        except OSError as exc:
            return _fail(f"cannot write {args.json}: {exc}")

    if result.status is FitStatus.CONVERGED:
        return EXIT_OK
    if result.status is FitStatus.INVALID_INPUT:
        return EXIT_INVALID  # the data cannot be fitted this way: invalid input
    return EXIT_NOT_CONVERGED


# --- autofit -----------------------------------------------------------------------------

_ORDERING_LABELS = {
    RankBy.BEST_OVERALL: (
        "best overall (lowest AICc first; models within 2 AICc are ordered by "
        "cross-validation, then fewer parameters)"
    ),
    RankBy.R2: "R² (highest first)",
    RankBy.ADJ_R2: "adjusted R² (highest first)",
    RankBy.RMSE: "RMSE (lowest first)",
    RankBy.AICC: "AICc (lowest first)",
    RankBy.BIC: "BIC (lowest first)",
    RankBy.CV_RMSE: "cross-validated RMSE (lowest first)",
}


def _load_xy(args: argparse.Namespace) -> tuple[XY, str, list[Issue]] | list[str]:
    """Read and validate the requested columns: ``(xy, dataset name, warnings)`` or error lines."""
    try:
        dataset = read_dataset(args.file)
        issues = validate(dataset, args.x, args.y, args.weights)
    except DataError as exc:
        return [_data_error(exc)]
    errors = [_issue_line(issue) for issue in issues if issue.severity == "error"]
    if errors:
        return errors
    try:
        xy = dataset.xy(args.x, args.y, args.weights)
    except DataError as exc:
        return [_data_error(exc)]
    return xy, dataset.name, [issue for issue in issues if issue.severity == "warning"]


def _run_autofit(args: argparse.Namespace) -> int:
    """Import, validate, run Auto-fit and print the ranked table (or the JSON result)."""
    loaded = _load_xy(args)
    if isinstance(loaded, list):
        return _fail_lines(loaded, EXIT_AUTOFIT_ERROR)
    xy, dataset_name, data_warnings = loaded

    overrides = {
        "workers": args.workers,
        "n_starts": args.n_starts,
        "seed": args.seed,
        "time_limit": args.time_limit,
    }
    try:
        config = AutoFitConfig(
            rank_by=RankBy(args.rank_by),
            exhaustive=args.exhaustive,
            **{name: value for name, value in overrides.items() if value is not None},
        )
        result = autofit(xy, config)
    except AutoFitError as exc:
        return _fail(f"{exc.code}: {exc}", EXIT_AUTOFIT_ERROR)
    except ValueError as exc:
        return _fail(f"invalid Auto-fit configuration: {exc}", EXIT_AUTOFIT_ERROR)

    if args.json:
        for issue in data_warnings:
            print(f"curvestudio: warning: {_issue_line(issue)}", file=sys.stderr)
        print(result.to_json())
        return EXIT_OK
    _print_autofit_report(result, args, dataset_name, data_warnings)
    return EXIT_OK


def _print_autofit_report(
    result: AutoFitResult, args: argparse.Namespace, dataset_name: str, data_warnings: list[Issue]
) -> None:
    """The ranked table, then the skipped, not converged and exact lists and the warnings."""
    weight_note = "" if args.weights is None else f", weights={args.weights!r}"
    print(
        f"Auto-fit: {dataset_name}, {result.n_active} active points, "
        f"x={args.x!r}, y={args.y!r}{weight_note}"
    )
    print(f"Ranked by: {_ORDERING_LABELS[result.config.rank_by]}")
    print(
        f"Candidates: {len(result.ranked)} ranked, {len(result.exact_fits)} exact, "
        f"{len(result.unconverged)} not converged, {len(result.skipped)} skipped "
        f"({result.elapsed_seconds:.1f} s)"
    )
    print()
    if result.ranked:
        _print_ranked_table(result.ranked, args.top)
    else:
        print("No candidate converged.")
    print()
    if result.overfitting_note:
        print(f"Overfitting warning: {result.overfitting_note}")
        print()
    if result.exact_fits:
        print(
            f"Exact fits ({len(result.exact_fits)}; the data are reproduced to rounding level, "
            "so AICc and BIC are not defined for them):"
        )
        _print_wrapped(", ".join(result.exact_fits))
        print()
    if result.unconverged:
        print(f"Not converged ({len(result.unconverged)}; never ranked):")
        by_reason: dict[str, list[str]] = defaultdict(list)
        for candidate_id, reason in result.unconverged.items():
            by_reason[reason].append(candidate_id)
        _print_groups(by_reason)
        print()
    if result.skipped:
        print(f"Skipped ({len(result.skipped)}; the data or the parameter cap rule them out):")
        by_cause: dict[str, list[str]] = defaultdict(list)
        for candidate_id, reasons in result.skipped.items():
            for cause in dict.fromkeys(_reason_head(reason) for reason in reasons):
                by_cause[cause].append(candidate_id)
        _print_groups(by_cause)
        print("  (the full reason of every candidate is in the --json output)")
        print()
    lines = [_issue_line(issue) for issue in data_warnings]
    lines.extend(result.warnings)
    if lines:
        print("Warnings:")
        for line in lines:
            print(f"  - {line}")
    else:
        print("Warnings: none")


def _print_ranked_table(ranked: tuple[RankedFit, ...], top: int) -> None:
    shown = ranked[:top]
    headers = ("Rank", "Model", "R²", "Adj. R²", "RMSE", "AICc", "BIC", "CV-RMSE", "k")
    rows = [
        (
            str(row.rank),
            row.model_id,
            _opt(row.metrics.r2),
            _opt(row.metrics.adj_r2),
            _opt(row.metrics.rmse),
            _opt(row.metrics.aicc),
            _opt(row.metrics.bic),
            _opt(row.metrics.cv_rmse),
            str(row.metrics.k),
        )
        for row in shown
    ]
    widths = [_column_width(h, [r[i] for r in rows]) for i, h in enumerate(headers)]

    def line(cells: tuple[str, ...]) -> str:
        left = f"{cells[0]:>{widths[0]}}  {cells[1]:<{widths[1]}}"
        right = "  ".join(
            f"{cell:>{width}}" for cell, width in zip(cells[2:], widths[2:], strict=True)
        )
        return f"{left}  {right}"

    pad = " " * widths[0]
    print(line(headers))
    for row, cells in zip(shown, rows, strict=True):
        print(line(cells))
        print(f"{pad}  {row.equation_fitted or '(no fitted equation)'}")
        if row.same_curve_as:
            print(f"{pad}  same curve as: {', '.join(row.same_curve_as)}")
        if row.redundant_with:
            print(f"{pad}  redundant with {row.redundant_with} (same curve, more parameters)")
        if row.status == "exact_fit":
            print(f"{pad}  exact fit: AICc and BIC are not defined")
        if row.degenerate:
            print(f"{pad}  degenerate: the optimum collapsed onto a simpler model")
        if row.overfitting_note:
            print(f"{pad}  note: {row.overfitting_note}")
    if len(ranked) > len(shown):
        print(f"(showing the top {len(shown)} of {len(ranked)} ranked fits; use --top to see more)")


def _opt(value: float | None) -> str:
    """A report number, or ``n/a`` for a quantity that is not defined."""
    return "n/a" if value is None else _g(value)


_REASON_HEAD = re.compile(r"\s+for\s|:|\s*\(")


def _reason_head(reason: str) -> str:
    """The leading clause of a skip reason, so many candidates share one group."""
    return _REASON_HEAD.split(reason, maxsplit=1)[0].strip() or reason


def _print_groups(groups: dict[str, list[str]]) -> None:
    """One block per distinct reason, with the ids that share it wrapped underneath."""
    for reason, ids in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        print(f"  {reason} ({len(ids)}):")
        _print_wrapped(", ".join(sorted(ids)), indent="    ")


def _print_wrapped(text: str, indent: str = "  ") -> None:
    print(textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent))


# --- audit and methods --------------------------------------------------------------------


@dataclass(frozen=True)
class _Fitted:
    """One fit made through the workflow, with what the tools need to read it."""

    xy: XY
    spec: FitSpec
    bundle: FitBundle
    data_warnings: list[Issue]


def _add_fit_target(command: argparse.ArgumentParser) -> None:
    """What is fitted, as ``fit`` takes it: the file, the columns, a model or a polynomial."""
    command.add_argument("file", help="input file: csv, tsv, txt, json or xlsx")
    command.add_argument("--x", required=True, metavar="COL", help="column with the x values")
    command.add_argument("--y", required=True, metavar="COL", help="column with the y values")
    command.add_argument("--weights", metavar="COL", help="column holding positive weights")
    target = command.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model",
        metavar="ID",
        help="registered model id (see 'curvestudio models')",
    )
    target.add_argument(
        "--poly",
        type=_bounded_int(1, MAX_POLY_DEGREE, "--poly"),
        metavar="N",
        help=f"fit a degree-N polynomial by (weighted) least squares (1-{MAX_POLY_DEGREE})",
    )


def _fit_for_tools(args: argparse.Namespace) -> _Fitted | int:
    """Read the file and fit what the arguments name, the way the apps do.

    Returns the fit, or the exit code of the failure, its message then being on stderr. The
    weights are used when a column was named for them.
    """
    loaded = _load_xy(args)
    if isinstance(loaded, list):
        return _fail_lines(loaded, EXIT_INPUT_ERROR)
    xy, _dataset_name, data_warnings = loaded
    model = (
        PolynomialRef(degree=args.poly) if args.poly is not None else RegisteredRef(id=args.model)
    )
    spec = FitSpec(model=model, use_weights=args.weights is not None)
    try:
        bundle = fit_dataset(xy, spec)
    except WorkflowError as exc:
        return _fail(str(exc), EXIT_INPUT_ERROR)
    return _Fitted(xy=xy, spec=spec, bundle=bundle, data_warnings=data_warnings)


def _run_audit(args: argparse.Namespace) -> int:
    """Fit through the workflow, audit and print; the return value is the exit code."""
    fitted = _fit_for_tools(args)
    if isinstance(fitted, int):
        return fitted
    spec, bundle = fitted.spec, fitted.bundle
    cv = None
    if args.cv is not None and bundle.result.status is FitStatus.CONVERGED:
        try:
            cv = cross_validate(bundle, spec, folds=args.cv)
        except ToolError as exc:
            return _fail(str(exc), EXIT_INPUT_ERROR)
    elif args.cv is not None:
        # Nothing to cross-validate. The audit's first check says why, and that is the answer.
        print(
            "curvestudio: note: cross-validation was not run: the fit did not converge "
            f"({bundle.result.termination_reason})",
            file=sys.stderr,
        )
    try:
        # AutoFitConfig() is what the autofit command builds from its default flags.
        ranking = autofit(fitted.xy, AutoFitConfig()) if args.ranking else None
        audit = audit_fit(bundle, spec=spec, ranking=ranking, cv=cv)
    except AutoFitError as exc:
        return _fail(f"{exc.code}: {exc}", EXIT_INPUT_ERROR)

    _warn(fitted.data_warnings)
    if args.json:
        print(audit.model_dump_json())
    else:
        _print_audit(audit)
    return _AUDIT_EXIT_CODES[audit.overall]


def _run_methods(args: argparse.Namespace) -> int:
    """Fit through the workflow and print the methods text; the return value is the exit code."""
    fitted = _fit_for_tools(args)
    if isinstance(fitted, int):
        return fitted
    try:
        text = methods_text(fitted.bundle, spec=fitted.spec, style=args.style)
    except ToolError as exc:
        return _fail(str(exc), EXIT_INPUT_ERROR)

    _warn(fitted.data_warnings)
    if args.json:
        print(text.model_dump_json())
    else:
        _print_methods_text(text)
    return EXIT_OK


_AUDIT_TAGS = {
    Verdict.PASS: "pass",
    Verdict.CAUTION: "caution",
    Verdict.FAIL: "fail",
    Verdict.NOT_APPLICABLE: "n/a",
}
"""The bracket tag ``audit`` prints for each verdict."""


def _print_audit(audit: FitAudit) -> None:
    """The overall verdict, the summary, then one block per check and the simpler model, if any."""
    print(f"Overall: {audit.overall.value.upper()}")
    print(audit.summary)
    print()
    for check in audit.checks:
        print(f"[{_AUDIT_TAGS[check.verdict]}] {check.title}")
        print(f"  {check.evidence}")
        if check.next_step is not None:
            print(f"    -> {check.next_step}")
    if audit.simpler_model is not None:
        simpler = audit.simpler_model
        print(
            f"A simpler model fits as well: {simpler.name} ({simpler.model_id}), "
            f"{simpler.n_parameters} parameters"
        )


def _print_methods_text(text: MethodsText) -> None:
    """The paragraph, the equation and the table, each apart; then the citations and the lists."""
    for part in (text.paragraph, text.equation, text.table):
        print(part)
        print()
    print("Cite as:")
    print(text.citation_curvestudio)
    for reference in text.citation_model:
        print(reference)
    print()
    print("For you to complete:")
    for item in text.to_complete:
        print(item)
    if text.missing:
        print()
        print("Not known:")
        for entry in text.missing:
            print(entry)


def _warn(warnings: list[Issue]) -> None:
    """Data warnings on stderr: they are about the input, not part of what the command prints."""
    for issue in warnings:
        print(f"curvestudio: warning: {_issue_line(issue)}", file=sys.stderr)


# --- models ------------------------------------------------------------------------------


def _run_models() -> int:
    """Print one line per registered model: id, name and category."""
    specs = list_models()
    if not specs:
        print("No models are registered.")
        return EXIT_OK
    id_width = _column_width("ID", [spec.id for spec in specs])
    name_width = _column_width("NAME", [spec.name for spec in specs])
    print(f"{'ID':<{id_width}}  {'NAME':<{name_width}}  CATEGORY")
    for spec in specs:
        print(f"{spec.id:<{id_width}}  {spec.name:<{name_width}}  {spec.category}")
    return EXIT_OK


def _column_width(header: str, values: list[str]) -> int:
    return max(len(header), *(len(value) for value in values))


# --- report ------------------------------------------------------------------------------


def _print_report(result: FitResult, context: _ReportContext, data_warnings: list[Issue]) -> None:
    """Print the human-readable report: equation, parameters, statistics, status."""
    print(f"Model: {context.model_label}")
    weight_note = "" if context.weights is None else f", weights={context.weights!r}"
    print(
        f"Data: {context.dataset_name}, {context.n_rows} rows, "
        f"x={context.x!r}, y={context.y!r}{weight_note}"
    )
    print()
    print(f"Equation: {result.equation_fitted or result.equation}")
    print()
    if result.parameters:
        name_width = max(len("Parameter"), *(len(p.name) for p in result.parameters))
        print(f"{'Parameter':<{name_width}}  {'Value':>12}  {'Std. err.':>12}  95% CI")
        for estimate in result.parameters:
            se = "n/a" if estimate.std_err is None else _g(estimate.std_err)
            value = _g(estimate.value)
            print(f"{estimate.name:<{name_width}}  {value:>12}  {se:>12}  {_ci(estimate)}")
        print()
    stats = result.stats
    r2 = "n/a" if stats is None or stats.r2 is None else _g(stats.r2)
    adj_r2 = "n/a" if stats is None or stats.adj_r2 is None else _g(stats.adj_r2)
    rmse = "n/a" if stats is None else _g(stats.rmse)
    print(f"R² = {r2}")
    print(f"Adjusted R² = {adj_r2}")
    print(f"RMSE = {rmse}")
    print()
    print(
        f"Status: {result.status.value} ({result.method}, loss={result.loss}, "
        f"starts={result.n_starts})"
    )
    print(f"Termination: {result.termination_reason}")
    lines = [_issue_line(issue) for issue in data_warnings]
    lines.extend(result.warnings)
    if lines:
        print("Warnings:")
        for line in lines:
            print(f"  - {line}")
    else:
        print("Warnings: none")


def _g(value: float) -> str:
    """Six significant digits, the report's number format throughout."""
    return f"{value:.6g}"


def _ci(estimate: ParameterEstimate) -> str:
    """The 95 % confidence interval, or ``n/a`` when it could not be determined."""
    if estimate.ci_low is None or estimate.ci_high is None:
        return "n/a"
    return f"[{_g(estimate.ci_low)}, {_g(estimate.ci_high)}]"


# --- error formatting --------------------------------------------------------------------


def _fail(message: str, code: int = EXIT_INVALID) -> int:
    print(f"{_ERROR_PREFIX} {message}", file=sys.stderr)
    return code


def _fail_lines(lines: list[str], code: int = EXIT_INVALID) -> int:
    """Print each validation error on its own line (with row numbers); exit ``code``."""
    for line in lines:
        print(f"{_ERROR_PREFIX} {line}", file=sys.stderr)
    return code


def _data_error(exc: DataError) -> str:
    return _with_rows(f"{exc.code}: {exc.args[0]}", exc.rows)


def _issue_line(issue: Issue) -> str:
    return _with_rows(f"{issue.code}: {issue.message}", issue.rows)


def _with_rows(message: str, rows: tuple[int, ...]) -> str:
    """Append 0-based table positions as 1-based data-row numbers."""
    if not rows:
        return message
    listed = ", ".join(str(row + 1) for row in rows)
    plural = "s" if len(rows) > 1 else ""
    return f"{message} (data row{plural} {listed})"


def _unknown_model(model_id: str) -> str:
    """A near-match hint over the registered ids, or how to list them all."""
    ids = [spec.id for spec in list_models()]
    close = difflib.get_close_matches(model_id, ids, n=3, cutoff=0.5)
    if close:
        hint = "did you mean " + ", ".join(repr(name) for name in close) + "?"
    else:
        hint = "run 'curvestudio models' to list the available models"
    return f"unknown model {model_id!r}; {hint}"


if __name__ == "__main__":
    raise SystemExit(main())
