"""``import_dataset``: bytes or a file in, a validated :class:`~curvestudio.data.Dataset` out.

Problems with the *file* (too big, unreadable, unknown format) raise
:class:`WorkflowError`. Problems with the *data* (blank cells, text where numbers belong,
too few rows, an unknown column) never raise: they come back as :class:`ImportIssue` entries
and ``xy`` stays ``None`` while any of them is an error.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from curvestudio.data import XY, DataError, Dataset, Issue, read_dataset, validate
from curvestudio.data import ImportLimits as _DataImportLimits
from curvestudio.workflow.errors import WorkflowError

_TOO_LARGE_CODES = frozenset({"FILE_TOO_LARGE", "TOO_MANY_ROWS", "TOO_MANY_COLS"})
_DEFAULT_NAME = "dataset"
_MAX_NAME = 100


@dataclass(frozen=True)
class ImportLimits:
    """Upper bounds for an upload: file size in bytes and number of data rows."""

    max_bytes: int = 20_000_000
    max_rows: int = 1_000_000


class ImportIssue(BaseModel):
    """One finding about the imported data. ``rows`` are 0-based table positions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: Literal["error", "warning"]
    message: str
    rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class ImportOutcome:
    """What an import produced.

    ``xy`` is the fit input for the chosen columns, or ``None`` when the columns could not
    be chosen or the data has error-level issues (see ``issues``). ``x_column`` /
    ``y_column`` / ``weight_column`` name the columns ``xy`` was built from.
    """

    dataset: Dataset
    xy: XY | None
    issues: list[ImportIssue]
    columns: tuple[str, ...]
    n_rows: int
    x_column: str | None
    y_column: str | None
    weight_column: str | None


def import_dataset(
    source: Path | bytes,
    filename: str,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    weight_column: str | None = None,
    limits: ImportLimits = ImportLimits(),
) -> ImportOutcome:
    """Import a csv, tsv, txt, json or xlsx table and extract the fit input.

    ``filename`` supplies the format (by extension) and the dataset name (its stem); it is
    never used as a path. With neither ``x_column`` nor ``y_column`` given, the first two
    numeric columns are used; giving only one of them is reported as ``COLUMNS_REQUIRED``.

    Raises:
        WorkflowError: ``TOO_LARGE`` (bytes, rows or columns beyond ``limits``),
            ``UNSUPPORTED_FORMAT`` (unknown extension) or ``UNREADABLE`` (missing, empty,
            header-only, corrupt or non-UTF-8 content).
    """
    name, suffix = _name_and_suffix(filename)
    if not suffix:
        raise WorkflowError(
            "UNSUPPORTED_FORMAT",
            f"the file name {name!r} has no extension; use .csv, .tsv, .txt, .json or .xlsx",
        )
    data_limits = _DataImportLimits(max_bytes=limits.max_bytes, max_rows=limits.max_rows)
    try:
        if isinstance(source, bytes):
            dataset = _read_bytes(source, suffix, data_limits)
        else:
            dataset = read_dataset(source, fmt=suffix, limits=data_limits)
    except DataError as exc:
        raise _workflow_error(exc) from exc
    if dataset.name != name:
        dataset = Dataset(name, dataset.original, dataset.ops, dataset.import_notes)

    columns = tuple(str(c) for c in dataset.table().columns)
    n_rows = len(dataset.table())
    x, y, issues = _choose_columns(dataset, columns, x_column, y_column)
    xy: XY | None = None
    if x is not None and y is not None:
        xy, found = _extract(dataset, x, y, weight_column)
        issues.extend(found)
    return ImportOutcome(
        dataset=dataset,
        xy=xy,
        issues=issues,
        columns=columns,
        n_rows=n_rows,
        x_column=x,
        y_column=y,
        weight_column=weight_column,
    )


def _name_and_suffix(filename: str) -> tuple[str, str]:
    """Dataset name and format extension from a client-supplied name; directories are dropped."""
    base = PurePosixPath(filename.replace("\\", "/")).name
    suffix = PurePosixPath(base).suffix.lstrip(".")
    stem = PurePosixPath(base).stem if suffix else base
    return (stem[:_MAX_NAME] or _DEFAULT_NAME), suffix


def _read_bytes(raw: bytes, suffix: str, limits: _DataImportLimits) -> Dataset:
    if len(raw) > limits.max_bytes:
        raise DataError(
            "FILE_TOO_LARGE",
            f"Input is {len(raw):,} bytes; the limit is {limits.max_bytes:,} bytes",
        )
    with tempfile.TemporaryDirectory(prefix="curvestudio-import-") as directory:
        path = Path(directory) / "upload"  # fixed name: nothing from the client reaches the disk
        path.write_bytes(raw)
        return read_dataset(path, fmt=suffix, limits=limits)


def _workflow_error(exc: DataError) -> WorkflowError:
    message = f"{exc.code}: {exc.args[0]}"
    if exc.code in _TOO_LARGE_CODES:
        return WorkflowError("TOO_LARGE", message)
    if exc.code == "PARSE_ERROR" and "UTF-8" in str(exc.args[0]):
        return WorkflowError(
            "UNREADABLE",
            f"text must be UTF-8 (UTF-8 BOM accepted); re-save the file as UTF-8 ({exc.args[0]})",
        )
    if exc.code == "UNSUPPORTED_FORMAT":
        return WorkflowError("UNSUPPORTED_FORMAT", str(exc.args[0]))
    return WorkflowError("UNREADABLE", message)


def _choose_columns(
    dataset: Dataset, columns: tuple[str, ...], x: str | None, y: str | None
) -> tuple[str | None, str | None, list[ImportIssue]]:
    if x is not None and y is not None:
        return x, y, []
    if x is not None or y is not None:
        issue = ImportIssue(
            code="COLUMNS_REQUIRED",
            severity="error",
            message=(
                "give both x_column and y_column, or neither to use the first two numeric columns"
            ),
        )
        return None, None, [issue]
    table = dataset.table()
    numeric = [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(table[c]) and not pd.api.types.is_bool_dtype(table[c])
    ]
    if len(numeric) < 2:
        issue = ImportIssue(
            code="COLUMNS_REQUIRED",
            severity="error",
            message=(
                f"found {len(numeric)} numeric column(s) among {list(columns)}; "
                "choose the x and y columns"
            ),
        )
        return None, None, [issue]
    return numeric[0], numeric[1], []


def _extract(
    dataset: Dataset, x: str, y: str, weights: str | None
) -> tuple[XY | None, list[ImportIssue]]:
    try:
        found = validate(dataset, x, y, weights)
    except DataError as exc:  # an unknown column name
        return None, [ImportIssue(code=exc.code, severity="error", message=str(exc.args[0]))]
    issues = [_issue(i) for i in found]
    if any(i.severity == "error" for i in issues):
        return None, issues
    try:
        return dataset.xy(x, y, weights), issues
    except DataError as exc:
        issues.append(
            ImportIssue(code=exc.code, severity="error", message=str(exc.args[0]), rows=exc.rows)
        )
        return None, issues


def _issue(issue: Issue) -> ImportIssue:
    return ImportIssue(
        code=issue.code, severity=issue.severity, message=issue.message, rows=issue.rows
    )
