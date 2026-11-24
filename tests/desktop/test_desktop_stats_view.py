"""The Statistics page, offscreen: the tool form and the run it submits.

The page is built from the real catalogue, never a fixture list: the picker groups the
tools under their families, choosing a tool builds one control per request field, and the
choices are read back into exactly what :func:`build_request` takes. A run is submitted to
a synchronous ``JobRunner`` with ``run_selected`` monkeypatched, so the test sees the
arguments the page sends and the result (or refusal) it shows, without a thread.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QListWidget,
    QSpinBox,
)

from curvestudio.tools import ToolError
from curvestudio.tools.stats import FAMILIES, Counts, ScatterPlot, ToolResult, catalogue
from curvestudio_desktop.controllers import stats_jobs
from curvestudio_desktop.controllers.selection import SelectionController
from curvestudio_desktop.jobs import JobRunner
from curvestudio_desktop.views.stats_view import StatsView

STATS_CSV = "x,y,group\n1,2.0,a\n2,4.1,a\n3,6.2,b\n4,7.9,b\n5,10.3,b\n"
"""Two numeric columns and one text column: the group combo holds what the column combo does not."""


@pytest.fixture()
def selection(controller, write_file) -> SelectionController:
    selection = SelectionController()
    assert controller.import_dataset_file(write_file("stats.csv", STATS_CSV))
    selection.set_project(controller.project)
    return selection


@pytest.fixture()
def view(qtbot, selection) -> StatsView:
    widget = StatsView(selection, jobs=JobRunner(sync=True))
    qtbot.addWidget(widget)
    return widget


def make_result(
    plots: tuple[object, ...] = (), title: str = "Descriptive statistics"
) -> ToolResult:
    return ToolResult(
        tool="descriptives",
        title=title,
        level=0.95,
        interpretation="The mean is 4.9.",
        methods_fragment="Descriptive statistics were computed.",
        counts=Counts(n_offered=5, n_used=5, n_excluded=0),
        plots=plots,  # type: ignore[arg-type]
    )


def item_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


class PausedRunner:
    """A runner that records the job and never runs it: its callbacks are invoked by hand."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[object, ...]]] = []
        self.on_result: object | None = None
        self.on_error: object | None = None
        self.on_settled: object | None = None

    def submit(self, fn: object, *args: object, **kwargs: object) -> object:
        self.calls.append((fn, args))
        self.on_result = kwargs.get("on_result")
        self.on_error = kwargs.get("on_error")
        self.on_settled = kwargs.get("on_settled")
        return object()


def test_the_tool_picker_lists_the_catalogue_grouped_by_family(view) -> None:
    picker = view.tool_picker
    texts = item_texts(picker)
    data = [picker.itemData(index) for index in range(picker.count())]
    titles = {info.name: info.title for info in catalogue()}
    # every tool is listed under its own title, carrying its name as the item's data
    for name, title in titles.items():
        assert texts[data.index(name)] == title
    # every family has one disabled heading, and it sits above its tools
    family_titles = dict(FAMILIES)
    for info in catalogue():
        heading = texts.index(family_titles[info.family])
        assert data[heading] is None
        assert not picker.model().item(heading).isEnabled()
        assert heading < data.index(info.name), info.name
    # the first tool is picked, so the form starts populated
    assert view.selected_tool() in titles


def test_choosing_a_tool_builds_one_control_per_field(view) -> None:
    assert view.select_tool("anova")
    values = view.field_widget("values")
    assert isinstance(values, QComboBox)
    assert item_texts(values) == ["x", "y"]  # the number columns alone
    groups = view.field_widget("groups")
    assert isinstance(groups, QComboBox)
    assert item_texts(groups) == ["x", "y", "group"]  # a group may be text
    level = view.field_widget("level")
    assert isinstance(level, QDoubleSpinBox)
    assert level.value() == pytest.approx(0.95)  # the schema's default
    post_hoc = view.field_widget("post_hoc")
    assert isinstance(post_hoc, QComboBox)
    assert post_hoc.currentText() == "tukey"
    control = view.field_widget("control")
    assert isinstance(control, QComboBox) and control.isEditable()
    assert control.currentText() == ""  # the nullable option starts unset


def test_option_controls_follow_their_schema(view) -> None:
    assert view.select_tool("power")
    design = view.field_widget("design")
    assert isinstance(design, QComboBox) and design.currentText() == "one_sample"
    alpha = view.field_widget("alpha")
    assert isinstance(alpha, QDoubleSpinBox) and alpha.value() == pytest.approx(0.05)
    n_groups = view.field_widget("n_groups")
    assert isinstance(n_groups, QSpinBox) and n_groups.value() == 2
    # a nullable number is a spin box at its minimum, which reads "not given"
    power = view.field_widget("power")
    assert isinstance(power, QDoubleSpinBox)
    assert power.value() == power.minimum()
    assert power.specialValueText() == "not given"


def test_a_boolean_option_is_a_check_box(view) -> None:
    assert view.select_tool("ttest")
    equal = view.field_widget("equal_variances")
    assert isinstance(equal, QCheckBox) and not equal.isChecked()
    null = view.field_widget("null")
    assert isinstance(null, QDoubleSpinBox) and null.value() == pytest.approx(0.0)


def test_a_text_column_is_absent_from_a_column_combo_and_present_in_a_group_combo(view) -> None:
    assert view.select_tool("descriptives")
    values = view.field_widget("values")
    assert isinstance(values, QComboBox)
    assert "group" not in item_texts(values)
    groups = view.field_widget("groups")
    assert isinstance(groups, QComboBox)
    assert groups.itemData(0) is None  # the blank first item of an optional group
    assert "group" in item_texts(groups)


def test_run_is_disabled_with_a_reason_for_a_missing_required_choice(view) -> None:
    assert view.select_tool("regression")
    assert not view.run_button.isEnabled()
    assert "predictors" in view.status_label.text().lower()
    predictors = view.field_widget("predictors")
    assert isinstance(predictors, QListWidget)
    predictors.item(0).setCheckState(Qt.CheckState.Checked)
    assert view.run_button.isEnabled()


def test_run_is_disabled_without_a_dataset_except_for_a_tool_that_needs_none(qtbot) -> None:
    selection = SelectionController()  # no project at all: nothing to run a column tool on
    view = StatsView(selection, jobs=JobRunner(sync=True))
    qtbot.addWidget(view)
    assert view.select_tool("descriptives")
    assert not view.run_button.isEnabled()
    assert "dataset" in view.status_label.text().lower()
    assert view.select_tool("power")  # needs no data: the form can run it
    assert view.run_button.isEnabled()


def test_run_is_disabled_while_a_job_runs(qtbot, selection) -> None:
    runner = PausedRunner()
    view = StatsView(selection, jobs=runner)  # type: ignore[arg-type]
    qtbot.addWidget(view)
    assert view.select_tool("descriptives")
    view.run_button.click()
    assert not view.run_button.isEnabled()
    assert len(runner.calls) == 1 and runner.calls[0][0] is stats_jobs.run_selected
    assert callable(runner.on_settled)
    runner.on_settled()
    assert view.run_button.isEnabled()


def test_a_submitted_run_calls_run_selected_with_the_choices_exactly(
    view, selection, monkeypatch
) -> None:
    calls: list[tuple[object, str, dict[str, object]]] = []
    result = make_result()

    def fake(dataset: object, tool: str, choices: dict[str, object]) -> ToolResult:
        calls.append((dataset, tool, dict(choices)))
        return result

    monkeypatch.setattr(stats_jobs, "run_selected", fake)
    assert view.select_tool("descriptives")
    values = view.field_widget("values")
    assert isinstance(values, QComboBox)
    values.setCurrentIndex(values.findData("y"))
    view.run_button.click()

    assert len(calls) == 1
    dataset, tool, choices = calls[0]
    assert dataset is selection.active_dataset()
    assert tool == "descriptives"
    # an unset optional group is left out; the level keeps its default
    assert choices == {"values": "y", "level": 0.95}
    assert view.result_view.result() is result


def test_a_tool_error_shows_its_code_and_message_and_keeps_the_previous_result(
    view, monkeypatch
) -> None:
    first = make_result(title="First result")
    monkeypatch.setattr(stats_jobs, "run_selected", lambda *_args: first)
    assert view.select_tool("descriptives")
    view.run_button.click()
    assert view.result_view.result() is first

    def refuse(*_args: object) -> ToolResult:
        raise ToolError("INVALID_OPTION", "the column holds no numbers")

    monkeypatch.setattr(stats_jobs, "run_selected", refuse)
    view.run_button.click()
    assert view.status_label.text() == "INVALID_OPTION: the column holds no numbers"
    assert view.result_view.result() is first  # nothing is cleared on a failure


def test_a_result_for_a_dataset_that_moved_away_is_not_shown(
    qtbot, selection, controller, write_file
) -> None:
    runner = PausedRunner()
    view = StatsView(selection, jobs=runner)  # type: ignore[arg-type]
    qtbot.addWidget(view)
    assert view.select_tool("descriptives")
    view.run_button.click()
    # the user activates another dataset before the tool lands
    assert controller.import_dataset_file(write_file("other.csv", STATS_CSV))
    selection.set_project(controller.project)
    selection.set_active(controller.project.datasets[-1].id)
    assert callable(runner.on_result)
    runner.on_result(make_result())
    assert view.result_view.result() is None  # the old dataset's result is not passed off


def test_a_result_emits_result_ready_and_fills_the_view(view, monkeypatch) -> None:
    result = make_result()
    monkeypatch.setattr(stats_jobs, "run_selected", lambda *_args: result)
    ready: list[ToolResult] = []
    view.result_ready.connect(ready.append)
    assert view.select_tool("descriptives")
    view.run_button.click()
    assert ready == [result]
    assert view.result_view.result() is result
    assert result.title in view.result_view.texts()


def test_the_plot_chooser_emits_the_selected_plot(view, monkeypatch) -> None:
    plots = (
        ScatterPlot(title="First", x_label="a", y_label="b", x=(1.0, 2.0), y=(2.0, 3.0)),
        ScatterPlot(title="Second", x_label="a", y_label="b", x=(1.0, 2.0), y=(3.0, 2.0)),
    )
    monkeypatch.setattr(stats_jobs, "run_selected", lambda *_args: make_result(plots))
    chosen: list[tuple[object, int]] = []
    view.plot_selected.connect(lambda emitted, index: chosen.append((emitted, index)))
    assert view.select_tool("descriptives")
    view.run_button.click()
    # the first plot is shown as soon as the result lands
    assert chosen and chosen[-1] == (plots, 0)
    assert item_texts(view.plot_picker) == ["First", "Second"]
    assert not view.plot_picker.isHidden()
    view.plot_picker.setCurrentIndex(1)
    assert chosen[-1] == (plots, 1)
    assert chosen[-1][0][chosen[-1][1]] is plots[1]


def test_a_result_without_plots_hides_the_chooser(view, monkeypatch) -> None:
    monkeypatch.setattr(stats_jobs, "run_selected", lambda *_args: make_result())
    assert view.select_tool("descriptives")
    view.run_button.click()
    assert view.plot_picker.isHidden()
