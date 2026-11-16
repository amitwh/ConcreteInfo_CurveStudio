"""Command-line interface: the vertical slice from a file to a fitted report.

``curvestudio fit FILE`` imports a table, validates the requested columns, fits one
curve (a registered model or a polynomial) and prints the equation with the fitted
numbers, a parameter table (value, standard error, 95 % CI), R², adjusted R², RMSE,
the status line and any warnings; ``--json`` additionally writes the full
:class:`~curvestudio.fitting.FitResult` machine-readably. ``curvestudio models``
lists the registry.

This module is the outermost layer of the package: it may import the data, fitting
and model layers, and nothing imports it. A user mistake (unreadable file, missing
column, unknown model, data that fails validation, an impossible method/loss
combination) is reported as a message on stderr, never as a traceback, and drives
the exit code: ``0`` the fit converged, ``2`` invalid input (including a fit result
of ``invalid_input``: too few observations, or a model that does not apply),
``3`` the fit ran but did not converge or failed.
"""

import argparse
import difflib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from curvestudio.data import DataError, Issue, read_dataset, validate
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

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_NOT_CONVERGED = 3

MAX_STARTS = 1000
"""Upper bound for --starts: multistart cost and memory grow linearly in it."""

MAX_POLY_DEGREE = 10
"""Upper bound for --poly: the spec's polynomial family is degrees 1..10."""

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


def _build_parser() -> argparse.ArgumentParser:
    """The ``curvestudio`` argument parser with the ``fit`` and ``models`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="curvestudio",
        description="Fit curves to data tables.",
        epilog=f"{_ROW_NOTE} Exit codes: 0 converged, 2 invalid input, 3 not converged.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="{fit,models}")

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

    commands.add_parser(
        "models",
        help="list the registered models",
        description="List the registered models: id, name and category.",
    )
    return parser


# --- entry point ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, run one subcommand and return the process exit code.

    ``argv`` defaults to ``sys.argv[1:]`` (the console-script behaviour). Argparse's
    own exits (usage errors, ``--help``) are converted into return codes so that
    ``main`` is fully testable in process. On Windows consoles the report's ``R²``
    needs a UTF-8 stdout, so the stream is reconfigured when that is possible.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    if args.command == "fit":
        return _run_fit(args)
    return _run_models()


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


def _fail(message: str) -> int:
    print(f"{_ERROR_PREFIX} {message}", file=sys.stderr)
    return EXIT_INVALID


def _fail_lines(lines: list[str]) -> int:
    """Print each validation error on its own line (with row numbers); exit 2."""
    for line in lines:
        print(f"{_ERROR_PREFIX} {line}", file=sys.stderr)
    return EXIT_INVALID


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
