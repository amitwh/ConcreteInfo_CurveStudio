"""A statistics result is kept in the project file and comes back without a run.

The desktop stores a tool's :class:`~curvestudio.tools.stats.ToolResult` and the choices
that produced it as one ``stat`` analysis. Save then open returns it, validates it into the
result model again, labels the row by the tool's title and renders the stored result — no
tool runs on the reopen.
"""

import pytest
from desktop_data import GOOD_CSV

from curvestudio.project import Analysis
from curvestudio.tools.stats import Counts, ToolResult
from curvestudio_desktop.controllers import analysis_store
from curvestudio_desktop.views.stats_result_view import StatsResultView

INTERPRETATION = "The mean of y is 4."


def _result() -> ToolResult:
    return ToolResult(
        tool="descriptives",
        title="Descriptive statistics of y",
        level=0.95,
        interpretation=INTERPRETATION,
        methods_fragment="The mean and standard deviation were taken.",
        counts=Counts(n_offered=5, n_used=5, n_excluded=0),
    )


def _with_stat(make_window, write_file):
    """A window holding one imported dataset and one stored statistics result."""
    window = make_window()
    assert window.import_file(write_file("stats.csv", GOOD_CSV))
    dataset_id = window.controller.project.datasets[0].id
    analysis = analysis_store.stat_analysis(dataset_id, "descriptives", {"values": "y"}, _result())
    assert analysis.kind == "stat"
    assert analysis.id.startswith("stat-")
    window.controller.add_or_replace_analysis(analysis)
    return window, dataset_id


def test_statistics_result_is_in_the_project_after_save_and_load(
    make_window, write_file, tmp_path, qtbot
) -> None:
    window, dataset_id = _with_stat(make_window, write_file)
    target = tmp_path / "statistics.cstudio"
    window.controller.paths.save_paths.append(target)
    assert window.controller.save() is True
    assert window.controller.open_project(target) is True

    loaded = next(
        analysis for analysis in window.controller.project.analyses if analysis.kind == "stat"
    )
    assert loaded.id == analysis_store.safe_analysis_id("stat-", dataset_id, "descriptives")
    assert isinstance(loaded.result, dict)
    assert analysis_store.stat_from_analysis(loaded) == _result()
    assert loaded.config == {"tool": "descriptives", "values": "y"}

    # the stored result renders as it was, without running the tool again
    view = StatsResultView()
    qtbot.addWidget(view)
    view.show_result(analysis_store.stat_from_analysis(loaded))
    assert INTERPRETATION in view.texts()


def test_a_stored_result_that_is_not_a_result_model_is_refused() -> None:
    """The project layer keeps plain JSON; the desktop revalidates it into the model."""
    analysis = Analysis(
        id="stat-ds1-descriptives",
        dataset_id="ds1",
        kind="stat",
        config={"tool": "descriptives"},
        result={
            "tool": "descriptives",
            "title": "Descriptive statistics",
            "level": "not a number",
            "interpretation": "The mean is unknown.",
            "methods_fragment": "",
            "counts": {"n_offered": 1, "n_used": 1, "n_excluded": 0},
        },
    )
    with pytest.raises(ValueError, match="does not hold a statistics result"):
        analysis_store.stat_from_analysis(analysis)


def test_tree_labels_a_statistics_result_by_the_tool_title(make_window, write_file) -> None:
    window, _ = _with_stat(make_window, write_file)

    rows = [
        child.text(0)
        for item in window.project_dock.tree.dataset_items()
        for index in range(item.childCount())
        if (child := item.child(index)) is not None
    ]
    assert "Statistics — Descriptive statistics" in rows
