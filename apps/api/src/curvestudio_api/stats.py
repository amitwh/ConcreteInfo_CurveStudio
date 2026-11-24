"""Running a statistical tool for the web service, on the columns a dataset keeps.

The apps compute nothing: the request comes from :func:`build_request` and the numbers from
:func:`run_tool`. This module only supplies the column lookups a stored dataset can answer,
and keeps the result it returns.

A column is read from the fit's payload when it is the dataset's x, y or weight column, and
from the stored table otherwise. The stored table holds the fit's input rows in order, so a
column read from it lines up row for row with the fit's payload, and a group column gives the
same labels :func:`group_labels` gives on the desktop.
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from curvestudio.data import XY, Column, GroupColumn, coerce_numeric, group_labels
from curvestudio.tools import ToolError
from curvestudio.tools.stats import TOOLS, ToolInfo, ToolResult, build_request, run_tool
from curvestudio_api import audit, quotas
from curvestudio_api.deps import Context
from curvestudio_api.orm import Dataset, StatResult, User
from curvestudio_api.settings import STAT_RESULT_MAX_BYTES
from curvestudio_api.storage import decode_payload, decode_table

CellArray = NDArray[np.float64] | NDArray[np.object_]


class _NoData:
    """The column lookups of a tool that needs no dataset: it never asks for a column."""

    def column(self, name: str) -> Column:
        raise ToolError("UNKNOWN_COLUMN", f"this tool takes no column, not {name!r}")

    def group(self, name: str) -> GroupColumn:
        raise ToolError("UNKNOWN_COLUMN", f"this tool takes no column, not {name!r}")


class _StoredReads:
    """One dataset's columns, read on demand: the fit's from its payload, the rest from its
    table. The payload is decoded at most once per run."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._xy: XY | None = None

    def _fit_columns(self) -> set[str]:
        names = {self._dataset.x_column, self._dataset.y_column}
        if self._dataset.weight_column is not None:
            names.add(self._dataset.weight_column)
        return names

    def _fit(self) -> XY:
        if self._xy is None:
            self._xy = decode_payload(
                self._dataset.payload,
                x_name=self._dataset.x_column,
                y_name=self._dataset.y_column,
                fingerprint=self._dataset.fingerprint,
            )
        return self._xy

    def cells(self, name: str) -> CellArray:
        """The stored cells of column ``name``, exactly as the dataset keeps them."""
        dataset = self._dataset
        if name in self._fit_columns():
            xy = self._fit()
            if name == dataset.x_column:
                return xy.x
            if name == dataset.y_column:
                return xy.y
            if xy.weights is None:
                raise ToolError(
                    "UNKNOWN_COLUMN", f"column {name!r} is not one of this dataset's columns"
                )
            return xy.weights
        if name not in dataset.columns:
            raise ToolError(
                "UNKNOWN_COLUMN", f"column {name!r} is not one of this dataset's columns"
            )
        if dataset.table is None:
            raise ToolError(
                "TABLE_NOT_STORED",
                "this dataset was uploaded before its whole table was kept; upload it again "
                "to analyse its other columns",
            )
        decoded = decode_table(dataset.table, names=[name])
        return decoded[name]

    def column(self, name: str) -> Column:
        cells = self.cells(name)
        if cells.dtype.kind == "O":
            # A text column reads as the desktop reads it: a number where a cell parses as
            # one, and NaN everywhere else.
            values, _unparseable = coerce_numeric(pd.Series(cells))
        else:
            values = cast("NDArray[np.float64]", cells)
        return Column(name=name, values=values, rows=np.arange(cells.size))

    def group(self, name: str) -> GroupColumn:
        cells = self.cells(name)
        # ``group_labels`` reads a float array as numbers and an object array as text.
        labels, rows = group_labels(cast("Sequence[object]", cells))
        return GroupColumn(name=name, labels=labels, rows=rows)


def _check_rows(ctx: Context, dataset: Dataset, tool: str, info: ToolInfo) -> None:
    """Refuse, before anything is read, a dataset too big for the web service or the tool."""
    limit = ctx.settings.stats_max_rows
    if dataset.n_rows > limit:
        raise ToolError(
            "TOO_MANY_ROWS",
            f"this dataset holds {dataset.n_rows:,} rows, and the web service runs a tool on "
            f"at most {limit:,}; the desktop app and the command line have no such limit",
        )
    if info.max_rows is not None and dataset.n_rows > info.max_rows:
        raise ToolError(
            "TOO_MANY_ROWS",
            f"the {tool} tool is built for at most {info.max_rows:,} rows, and this dataset "
            f"holds {dataset.n_rows:,}; the desktop app and the command line have no such "
            "limit",
        )


def _result_bytes(dumped: Mapping[str, Any]) -> int:
    """The length of a result's JSON as UTF-8, the size the bytes quota counts."""
    return len(json.dumps(dumped, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def run_tool_for(
    db: Session,
    ctx: Context,
    user: User,
    dataset: Dataset | None,
    tool: str,
    choices: Mapping[str, Any],
) -> StatResult:
    """Run ``tool`` on ``choices`` and keep the result.

    Raises:
        ToolError: ``UNKNOWN_TOOL``; ``TOO_MANY_ROWS`` before any payload is read;
            ``INVALID_OPTION`` when the tool needs a dataset that was not given; whatever the
            request builder or the tool refuses; ``TOO_LARGE`` for a result over
            :data:`STAT_RESULT_MAX_BYTES`.
        ApiError: 413 ``QUOTA_EXCEEDED`` when the result would take the account's bytes over
            its limit.
    """
    info = TOOLS.info(tool)
    if dataset is None:
        if info.needs_data:
            raise ToolError("INVALID_OPTION", "this tool needs a dataset")
        reads: _StoredReads | _NoData = _NoData()
    else:
        _check_rows(ctx, dataset, tool, info)
        reads = _StoredReads(dataset)
    request = build_request(tool, choices, column=reads.column, group=reads.group)
    result = run_tool(tool, request)
    dumped = result.model_dump(mode="json")
    size = _result_bytes(dumped)
    if size > STAT_RESULT_MAX_BYTES:
        raise ToolError(
            "TOO_LARGE",
            f"the result is {size:,} bytes, over the {STAT_RESULT_MAX_BYTES:,}-byte limit of "
            "the web service",
        )
    quotas.check(db, ctx.settings, user.id, "dataset_bytes", adding_bytes=size)
    row = StatResult(
        id=uuid.uuid4(),
        user_id=user.id,
        dataset_id=None if dataset is None else dataset.id,
        tool=tool,
        request=dict(choices),
        result=dumped,
        size_bytes=size,
    )
    db.add(row)
    db.flush()
    audit.record(
        db,
        "stat.create",
        "stat_result",
        user_id=user.id,
        entity_id=row.id,
        detail={"tool": tool, "with_dataset": dataset is not None},
    )
    db.commit()
    return row


def result_out(row: StatResult) -> ToolResult:
    """A stored result, as the core's model (the route's response body)."""
    return ToolResult.model_validate(row.result)


def summary_title(row: StatResult) -> str:
    """The stored result's title, for a listing that sends no result body."""
    title = row.result.get("title") if isinstance(row.result, Mapping) else None
    return str(title) if title is not None else ""


__all__ = ["result_out", "run_tool_for", "summary_title"]
