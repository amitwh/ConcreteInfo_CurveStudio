"""The plain job functions the Statistics page submits to a ``JobRunner``.

Nothing here computes a number: the request comes from :func:`build_request` and the result
from :func:`run_tool`, exactly as on the command line and the web service. The one thing
this layer supplies is what a dataset-less tool does when its request names a column — it
refuses, in the tool's own code and words, rather than reaching for a column that cannot
exist.
"""

from collections.abc import Mapping
from typing import Any

from curvestudio.data import Column, Dataset, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools.stats import ToolResult, build_request, run_tool


def _no_column(name: str) -> Column:
    """The column lookup of a tool that needs no data: it never asks for one."""
    raise ToolError("UNKNOWN_COLUMN", f"this tool takes no data, so it cannot read column {name!r}")


def _no_group(name: str) -> GroupColumn:
    """The group lookup of a tool that needs no data: it never asks for one."""
    raise ToolError("UNKNOWN_COLUMN", f"this tool takes no data, so it cannot read group {name!r}")


def run_selected(dataset: Dataset | None, tool: str, choices: Mapping[str, Any]) -> ToolResult:
    """Run ``tool`` on ``choices``, read through ``dataset``'s columns.

    Args:
        dataset: the active dataset, or ``None`` for a tool that needs no data.
        tool: the tool's registered name.
        choices: what a person chose, by field name: column names, lists of names, options.

    Raises:
        ToolError: whatever :func:`build_request` or the tool refuses, unchanged.
    """
    column = dataset.column if dataset is not None else _no_column
    group = dataset.group if dataset is not None else _no_group
    request = build_request(tool, choices, column=column, group=group)
    return run_tool(tool, request)


__all__ = ["run_selected"]
