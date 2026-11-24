"""StatsView: the Statistics page — pick a tool, give it its columns, run it, read it.

The page is a vertical splitter: the form above and the tool's result below. The form is
built from the catalogue, never from a hard-coded list: the picker groups every tool under
its family's title, and choosing one builds a control per request field by the field's kind
— a combo of the dataset's number columns, a combo of all its columns for a group (blank
when the group is optional), a check list for several columns (at most twenty), or an
option's spin box, check box or combo by the option's schema, its default already set. The
fields' descriptions are the controls' tool tips.

The page computes nothing. Run submits :func:`stats_jobs.run_selected` to the runner the
dock gives it (the analysis controller's) with exactly the names and options the form holds,
so the tool runs off the GUI thread and every number is ``run_tool``'s. A refusal is shown
in the tool's own code and words, and the previous result stays on screen. A result fills
:class:`StatsResultView` and its plots become a chooser: one plot at a time, the chosen one
announced through :attr:`plot_selected`, so the window can draw it on the main chart.

With no dataset selected, a tool that needs data cannot run: the form says so and Run stays
disabled; a tool that needs no data still runs. Choosing another dataset clears the shown
result — it described the old one — and rebuilds the form's column choices.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from curvestudio.data import Dataset
from curvestudio.tools.stats import FAMILIES, Plot, ToolInfo, ToolResult, catalogue
from curvestudio.tools.stats.registry import RequestField
from curvestudio_desktop.controllers import stats_jobs
from curvestudio_desktop.controllers.selection import SelectionController
from curvestudio_desktop.jobs import JobRunner
from curvestudio_desktop.views.searchable_combo import SearchableComboBox
from curvestudio_desktop.views.stats_result_view import StatsResultView

UNSET_TEXT = "not given"
"""A nullable option's spin box reads this at its minimum: the option is left out."""
RUN_TEXT = "Run"
RUNNING_TEXT = "Running…"
MISSING_TOOL = "Pick a tool to run."
NEEDS_DATASET = "Select a dataset to run this tool."
CHECKED_COLUMNS_MAX = 20
"""The most columns the several-column check list offers."""
_UNSET_INT = -(2**31)
_UNSET_NUMBER = -1e12
_CHECK_LIST_HEIGHT = 160


@dataclass(frozen=True)
class _Control:
    """One field's control and how its value is read back."""

    field: RequestField
    widget: QWidget
    read: Callable[[], Any]


def numeric_columns(dataset: Dataset | None) -> list[str]:
    """The dataset's number columns by dtype kind (the selection's own rule), in order."""
    if dataset is None:
        return []
    frame = dataset.table()
    return [str(name) for name in frame.columns if str(frame[name].dtype.kind) in "ifu"]


def all_columns(dataset: Dataset | None) -> list[str]:
    """Every column of the dataset, in order (a group may be text)."""
    if dataset is None:
        return []
    return [str(name) for name in dataset.table().columns]


def _option_schema(field: RequestField) -> tuple[Mapping[str, Any], bool]:
    """The option's own schema, and whether a union with ``null`` lets it be left unset."""
    schema: Mapping[str, Any] = field.option_schema or {}
    members = schema.get("anyOf")
    if isinstance(members, list):
        remaining = [member for member in members if member != {"type": "null"}]
        if len(remaining) == 1 and isinstance(remaining[0], dict):
            return remaining[0], len(remaining) != len(members)
    return schema, False


def _text_control(core: Mapping[str, Any], default: Any) -> tuple[QWidget, Callable[[], Any]]:
    """A free string or a list option: an editable combo, blank meaning 'not given'."""
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItem("", None)
    if default not in (None, "", (), []):
        combo.setEditText(str(default))
    if core.get("type") != "array":
        return combo, lambda: combo.currentText().strip() or None

    items = core.get("items")
    if isinstance(items, dict) and items.get("type") == "string":

        def read_names() -> list[str] | None:
            parts = [part.strip() for part in combo.currentText().split(",")]
            return [part for part in parts if part] or None

        return combo, read_names

    def read_list() -> Any:
        text = combo.currentText().strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return text  # the request builder refuses it, in its own words

    return combo, read_list


def _option_control(field: RequestField) -> tuple[QWidget, Callable[[], Any]]:
    """The control an option's schema calls for, with its default already set."""
    core, nullable = _option_schema(field)
    schema: Mapping[str, Any] = field.option_schema or {}
    default = schema.get("default", core.get("default"))
    kind = core.get("type")
    if kind == "boolean":
        box = QCheckBox()
        box.setChecked(bool(default))
        return box, box.isChecked
    if kind == "integer":
        spin = QSpinBox()
        spin.setRange(_UNSET_INT, 2**31 - 1)
        if nullable:
            spin.setSpecialValueText(UNSET_TEXT)
            spin.setValue(_UNSET_INT if default is None else int(default))
            return spin, lambda: None if spin.value() == _UNSET_INT else spin.value()
        spin.setValue(int(default if default is not None else 0))
        return spin, spin.value
    if kind == "number":
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(_UNSET_NUMBER, 1e12)
        if nullable:
            spin.setSpecialValueText(UNSET_TEXT)
            spin.setValue(_UNSET_NUMBER if default is None else float(default))
            return spin, lambda: None if spin.value() == _UNSET_NUMBER else spin.value()
        spin.setValue(float(default if default is not None else 0.0))
        return spin, spin.value
    values = core.get("enum")
    if kind == "string" and values:
        combo = QComboBox()
        combo.addItems([str(value) for value in values])
        index = combo.findText(str(default)) if default is not None else -1
        combo.setCurrentIndex(index if index >= 0 else 0)
        return combo, combo.currentText
    return _text_control(core, default)


def _control_for(field: RequestField, dataset: Dataset | None) -> tuple[QWidget, Callable[[], Any]]:
    """The control a field's kind calls for, filled from the dataset's columns."""
    if field.kind == "column":
        combo = QComboBox()
        for name in numeric_columns(dataset):
            combo.addItem(name, name)
        return combo, combo.currentData
    if field.kind == "group":
        combo = QComboBox()
        if not field.required:
            combo.addItem("", None)
        for name in all_columns(dataset):
            combo.addItem(name, name)
        return combo, combo.currentData
    if field.kind == "columns":
        listing = QListWidget()
        listing.setMaximumHeight(_CHECK_LIST_HEIGHT)
        for name in numeric_columns(dataset)[:CHECKED_COLUMNS_MAX]:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            listing.addItem(item)

        def read_columns() -> list[str] | None:
            chosen = [
                listing.item(row).text()
                for row in range(listing.count())
                if listing.item(row).checkState() == Qt.CheckState.Checked
            ]
            return chosen or None

        return listing, read_columns
    return _option_control(field)


class StatsView(QWidget):
    """The Statistics page: a tool's form above, its result below, and its plots a chooser."""

    result_ready = Signal(object)
    """A tool returned: its :class:`~curvestudio.tools.stats.ToolResult`."""

    plot_selected = Signal(object, int)
    """A plot of the shown result was chosen: ``(plots, index)``."""

    def __init__(
        self,
        selection: SelectionController,
        *,
        jobs: JobRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("statsView")
        self._selection = selection
        self._jobs = jobs if jobs is not None else JobRunner(self)
        self._infos: dict[str, ToolInfo] = {info.name: info for info in catalogue()}
        self._controls: dict[str, _Control] = {}
        self._plots: tuple[Plot, ...] = ()
        self._running = False
        self._run_dataset: str | None = None
        self._message: str | None = None

        self.tool_picker = SearchableComboBox()
        self.tool_picker.setObjectName("statsToolPicker")
        self.run_button = QPushButton(RUN_TEXT)
        self.run_button.setObjectName("statsRunButton")
        self.status_label = QLabel("")
        self.status_label.setObjectName("statsStatusLabel")
        self.status_label.setWordWrap(True)
        self._fields = QFormLayout()
        self._fields.setContentsMargins(0, 0, 0, 0)
        fields_host = QWidget()
        fields_host.setLayout(self._fields)
        form = QWidget()
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(6, 6, 6, 6)
        form_layout.setSpacing(6)
        form_layout.addWidget(QLabel("Tool"))
        form_layout.addWidget(self.tool_picker)
        form_layout.addWidget(fields_host)
        form_layout.addWidget(self.run_button)
        form_layout.addWidget(self.status_label)
        form_layout.addStretch(1)
        form_scroll = QScrollArea()
        form_scroll.setObjectName("statsFormArea")
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        form_scroll.setWidget(form)

        self.plot_picker = QComboBox()
        self.plot_picker.setObjectName("statsPlotPicker")
        self.plot_picker.hide()
        self.result_view = StatsResultView()
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(4)
        result_layout.addWidget(self.plot_picker)
        result_layout.addWidget(self.result_view, stretch=1)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setObjectName("statsSplitter")
        self.splitter.addWidget(form_scroll)
        self.splitter.addWidget(result_panel)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setSizes([260, 460])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.splitter)

        self._fill_tools()
        self.tool_picker.currentIndexChanged.connect(self._on_tool_changed)
        self.run_button.clicked.connect(self._run)
        self.plot_picker.currentIndexChanged.connect(self._on_plot_chosen)
        selection.active_changed.connect(self._on_dataset_changed)
        self._rebuild_form()

    # -- the catalogue -----------------------------------------------------------------------

    def _fill_tools(self) -> None:
        """Fill the picker: a disabled heading per family, then its tools by title."""
        picker = self.tool_picker
        picker.blockSignals(True)
        picker.clear()
        infos = list(self._infos.values())
        for family, title in FAMILIES:
            tools = [info for info in infos if info.family == family]
            if not tools:
                continue
            heading = picker.count()
            picker.addItem(title, None)
            item = picker.model().item(heading)
            if item is not None:
                item.setEnabled(False)
            for info in tools:
                picker.addItem(info.title, info.name)
        first = next(
            (index for index in range(picker.count()) if isinstance(picker.itemData(index), str)),
            -1,
        )
        if first >= 0:
            picker.setCurrentIndex(first)
        picker.blockSignals(False)
        picker.set_search_entries([f"{info.title} ({info.name})" for info in infos])

    def selected_tool(self) -> str | None:
        """The chosen tool's registered name (``None`` on a family heading)."""
        data = self.tool_picker.currentData()
        return data if isinstance(data, str) else None

    def select_tool(self, name: str) -> bool:
        """Pick the tool registered as ``name``; ``False`` when there is none."""
        index = self.tool_picker.findData(name)
        if index < 0:
            return False
        self.tool_picker.setCurrentIndex(index)
        return True

    def field_widget(self, name: str) -> QWidget | None:
        """The control built for the field ``name`` (``None`` when the tool has no such field)."""
        control = self._controls.get(name)
        return control.widget if control is not None else None

    # -- the form ------------------------------------------------------------------------------

    def _info(self) -> ToolInfo | None:
        return self._infos.get(self.selected_tool() or "")

    def _rebuild_form(self) -> None:
        """Build one control per field of the chosen tool, from the active dataset."""
        self._controls.clear()
        while self._fields.rowCount():
            self._fields.removeRow(0)  # also deletes the row's label and control
        info = self._info()
        dataset = self._selection.active_dataset()
        if info is not None:
            for field in info.fields:
                widget, read = _control_for(field, dataset)
                widget.setToolTip(field.description or "")
                self._connect_choice(widget)
                self._fields.addRow(field.label, widget)
                self._controls[field.name] = _Control(field, widget, read)
        self._update_run_state()

    def _connect_choice(self, widget: QWidget) -> None:
        if isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_choice_changed)
            if widget.isEditable():
                widget.editTextChanged.connect(self._on_choice_changed)
        elif isinstance(widget, QAbstractSpinBox):
            widget.valueChanged.connect(self._on_choice_changed)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._on_choice_changed)
        elif isinstance(widget, QListWidget):
            widget.itemChanged.connect(self._on_choice_changed)

    def choices(self) -> dict[str, Any]:
        """What the form currently holds, by field name: names, lists of names, options.

        An unset optional field is left out; the request builder then applies the model's
        own default, as the command line does.
        """
        chosen: dict[str, Any] = {}
        for name, control in self._controls.items():
            value = control.read()
            if value is None or value == []:
                continue
            chosen[name] = value
        return chosen

    def _missing_reason(self) -> str | None:
        """Why Run cannot start now, in one line, or ``None`` when the form is complete."""
        info = self._info()
        if info is None:
            return MISSING_TOOL
        if info.needs_data and self._selection.active_dataset() is None:
            return NEEDS_DATASET
        for control in self._controls.values():
            value = control.read()
            if control.field.required and (value is None or value == []):
                return f"Choose the {control.field.label.lower()} to run this tool."
        return None

    def _update_run_state(self) -> None:
        if self._running:
            self.run_button.setEnabled(False)
            self.run_button.setText(RUNNING_TEXT)
            self.status_label.setText("")
            return
        reason = self._missing_reason()
        self.run_button.setEnabled(reason is None)
        self.run_button.setText(RUN_TEXT)
        self.status_label.setText(reason or self._message or "")

    def _on_tool_changed(self, _index: int = 0) -> None:
        self._message = None
        self._rebuild_form()

    def _on_choice_changed(self, *_args: object) -> None:
        self._message = None
        self._update_run_state()

    def _on_dataset_changed(self) -> None:
        """The active dataset moved: its old result is stale, and the columns are new."""
        self._message = None
        self.result_view.clear()
        self._set_plots(())
        self._rebuild_form()

    # -- running -------------------------------------------------------------------------------

    def _run(self) -> None:
        tool = self.selected_tool()
        if tool is None or self._running:
            return
        dataset = self._selection.active_dataset()
        choices = self.choices()
        self._message = None
        self._running = True
        self._run_dataset = self._selection.active_id
        self._update_run_state()
        self._jobs.submit(
            stats_jobs.run_selected,
            dataset,
            tool,
            choices,
            on_result=self._on_result,
            on_error=self._on_error,
            on_settled=self._on_settled,
        )

    def _on_result(self, result: ToolResult) -> None:
        if not self._still_current():
            return  # the dataset moved while the tool ran: the result describes the old one
        self.result_view.show_result(result)
        self._message = None
        self.result_ready.emit(result)
        self._set_plots(result.plots)

    def _on_error(self, error: BaseException) -> None:
        if not self._still_current():
            return
        # the tool's own words, and the previous result stays on screen untouched
        self._message = str(error)
        self._update_run_state()

    def _still_current(self) -> bool:
        """Whether the active dataset is still the one the running job was given."""
        return self._selection.active_id == self._run_dataset

    def _on_settled(self) -> None:
        self._running = False
        self._update_run_state()

    # -- the plots ------------------------------------------------------------------------------

    def _set_plots(self, plots: tuple[Plot, ...]) -> None:
        """Fill the chooser with the result's plots and show the first one, if any."""
        self._plots = plots
        picker = self.plot_picker
        picker.blockSignals(True)
        picker.clear()
        for plot in plots:
            picker.addItem(plot.title)
        picker.setCurrentIndex(0 if plots else -1)
        picker.blockSignals(False)
        picker.setVisible(bool(plots))
        if plots:
            self.plot_selected.emit(plots, 0)

    def _on_plot_chosen(self, index: int) -> None:
        if 0 <= index < len(self._plots):
            self.plot_selected.emit(self._plots, index)


__all__ = ["CHECKED_COLUMNS_MAX", "MISSING_TOOL", "NEEDS_DATASET", "RUN_TEXT", "StatsView"]
