"""The standard statistical tools: descriptives, comparisons, relationships and counts.

Every tool is a pure function from a typed request to ONE :class:`ToolResult`, built from
blocks, so one renderer serves the command line, the desktop and the web. This package holds
what every family stands on: the result contract (:mod:`._base`), the blocks a result shows
(:mod:`._display`, :mod:`._tables`), the size harness (:mod:`._size`) and the registry
(:mod:`.registry`). A tool family registers its spec in :data:`TOOLS` when its module is
imported; :data:`curvestudio.tools.stats.registry.TOOL_MODULES` names those modules.
"""

from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    aligned,
    assumption,
    check_level,
    levels,
    list_excluded,
    p_text,
    pairs,
)
from curvestudio.tools.stats._display import (
    BoxPlot,
    HistogramPlot,
    MultiLinePlot,
    Plot,
    QQPlot,
    ScatterPlot,
    Table,
    TableColumn,
    box_plot,
    stride,
)
from curvestudio.tools.stats._size import assert_size, simulated_size
from curvestudio.tools.stats._tables import (
    confidence_interval,
    group_summary,
    group_table,
    holm,
    mean_ci,
    quantiles,
    table,
)
from curvestudio.tools.stats.registry import (
    FAMILIES,
    TOOLS,
    Registry,
    ToolInfo,
    ToolSpec,
    build_request,
    catalogue,
    run_tool,
)

__all__ = [
    "FAMILIES",
    "TOOLS",
    "AuditCheck",
    "BoxPlot",
    "Counts",
    "Estimate",
    "ExcludedRow",
    "HistogramPlot",
    "HypothesisTest",
    "MultiLinePlot",
    "Plot",
    "QQPlot",
    "Registry",
    "ScatterPlot",
    "Table",
    "TableColumn",
    "ToolError",
    "ToolInfo",
    "ToolResult",
    "ToolSpec",
    "Verdict",
    "aligned",
    "assert_size",
    "assumption",
    "box_plot",
    "build_request",
    "catalogue",
    "check_level",
    "confidence_interval",
    "group_summary",
    "group_table",
    "holm",
    "levels",
    "list_excluded",
    "mean_ci",
    "p_text",
    "pairs",
    "quantiles",
    "run_tool",
    "simulated_size",
    "stride",
    "table",
]
