"""FitToolsView: the three fit-centric tools — discernibility, feasibility, next measurement.

Each action is a short title, a small options form and a Run button; every run goes through
the window's :class:`~curvestudio_desktop.jobs.JobRunner`, never the GUI thread. One Cancel
button lends a :class:`threading.Event` to the core as its ``should_cancel`` token: the
tools are cooperative, so a cancel returns the part they finished, with its warning, and
that partial result is shown like any other. A run's :class:`ToolResult` opens in a
non-modal dialog hosting the shipped :class:`StatsResultView` and a plot chooser, with the
first plot drawn.

The three tools read what the open fit is: its bundle (and the full data snapshot it holds),
its specification, and — for the report — the Auto-fit ranking the fit was opened from. The
ranking is the one the fit audit also reads (``ResultsTabs.show_bundle`` passes it), so a
fit that did not come from an Auto-fit row has no ranking and that one action stays disabled
with the reason in its tool tip. The simulator's true values are the fit's own estimates:
the form states that, and nothing else could be said about an experiment not yet run.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from curvestudio.data import XY, DataError
from curvestudio.discovery import AutoFitResult
from curvestudio.tools.advisor import AdvisorRequest, advise
from curvestudio.tools.discernibility import DiscernibilityRequest, discernibility
from curvestudio.tools.feasibility import DesignSpec, NoiseSpec, SimulationRequest, simulate
from curvestudio.tools.stats import Plot, ToolResult
from curvestudio.workflow import FitBundle, FitSpec
from curvestudio_desktop.jobs import JobRunner, ProgressCallback
from curvestudio_desktop.views.stats_plots import draw_stats_plot
from curvestudio_desktop.views.stats_result_view import StatsResultView

MORE_TOOLS_TITLE = "More tools"
"""The Results tab's title (the Auto-fit view opens it by this name)."""
RUN_TEXT = "Run"
RUNNING_TEXT = "Running…"
CANCELLING_TEXT = "Cancelling… the partial result will be shown"
TRUTH_NOTE = "True values: the fit's own estimates"
_NO_FIT_REASON = "Open a fit first"
_NO_RANKING_REASON = (
    "This fit was not opened from an Auto-fit row, so there is no ranking to report on"
)
_NO_DATA_REASON = (
    "The fit's stored data cannot drive this tool (a subsample or a missing value); "
    "open the fit from its dataset"
)
_UNSET_NUMBER = -1e12
_MAX_SEED = 2**31 - 1
_DIALOG_SIZE = (900, 640)
RUNNING_MESSAGES = {
    "discernibility": "Running the model discernibility report…",
    "feasibility": "Simulating the design…",
    "advisor": "Advising where to measure next…",
}
_BUTTON_TIPS = {
    "discernibility": "Report whether these data can tell the leading models apart",
    "feasibility": "Simulate a design and measure how the estimates recover",
    "advisor": "Recommend the next measurement locations for this fit",
}


def xy_from_bundle(bundle: FitBundle) -> XY | None:
    """The fit's data as an :class:`XY`, or ``None`` when the snapshot cannot drive a tool.

    A compact bundle's snapshot is for plotting, not for running a tool: the tools read
    every row. A snapshot detail the data model refuses (a missing weight, say) also yields
    ``None``: the caller opens the fit from its dataset instead.
    """
    snapshot = bundle.data
    if snapshot.subsampled:
        return None
    try:
        return XY(
            np.asarray(snapshot.x, dtype=np.float64),
            np.asarray(snapshot.y, dtype=np.float64),
            None if snapshot.weights is None else np.asarray(snapshot.weights, dtype=np.float64),
            snapshot.x_name,
            snapshot.y_name,
            bundle.result.provenance.dataset_fingerprint,
        )
    except DataError:
        return None


class FitToolResultDialog(QDialog):
    """One fit tool's result, with the plot chooser and the first plot drawn."""

    plot_selected = Signal(object, int)
    """A plot of the shown result was chosen: ``(plots, index)``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("fitToolResultDialog")
        self.setWindowTitle(MORE_TOOLS_TITLE)
        self.setMinimumSize(*_DIALOG_SIZE)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self._plots: tuple[Plot, ...] = ()
        self._items: list[Any] = []
        self.plot_picker = QComboBox()
        self.plot_picker.setObjectName("fitToolsPlotPicker")
        self.plot_picker.hide()
        self.result_view = StatsResultView()
        self.plot = pg.PlotWidget()
        self.plot.setObjectName("fitToolsPlot")
        self.plot.setMinimumHeight(220)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.hide()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.result_view)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)
        outer.addWidget(self.plot_picker)
        outer.addWidget(splitter, stretch=1)
        self.plot_picker.currentIndexChanged.connect(self._on_plot_chosen)

    # -- filling -----------------------------------------------------------------------------

    def show_result(self, result: ToolResult) -> None:
        """Show every block of ``result``, draw its first plot, and raise the window."""
        self.result_view.show_result(result)
        self.setWindowTitle(result.title)
        self._set_plots(result.plots)
        self.show()
        self.raise_()
        self.activateWindow()

    def shown_result(self) -> ToolResult | None:
        """The result shown (``None`` before any run); not QDialog's ``exec`` result."""
        return self.result_view.result()

    def current_plot_index(self) -> int:
        """Which of the result's plots is drawn (-1 when there is none)."""
        return self.plot_picker.currentIndex()

    def _set_plots(self, plots: tuple[Plot, ...]) -> None:
        self._plots = plots
        picker = self.plot_picker
        picker.blockSignals(True)
        picker.clear()
        for plot in plots:
            picker.addItem(plot.title)
        picker.setCurrentIndex(0 if plots else -1)
        picker.blockSignals(False)
        picker.setVisible(bool(plots))
        self.plot.setVisible(bool(plots))
        self._clear_plot()
        if plots:
            self._draw(0)
            self.plot_selected.emit(plots, 0)

    def _on_plot_chosen(self, index: int) -> None:
        if 0 <= index < len(self._plots):
            self._draw(index)
            self.plot_selected.emit(self._plots, index)

    def _clear_plot(self) -> None:
        plot_item = self.plot.getPlotItem()
        for item in self._items:
            if item in plot_item.items:
                plot_item.removeItem(item)
            else:  # a note is parented to the item rather than added to its list
                item.setParentItem(None)
                item.deleteLater()
        self._items = []

    def _draw(self, index: int) -> None:
        self._clear_plot()
        self._items = draw_stats_plot(self.plot.getPlotItem(), self._plots[index])
        self.plot.autoRange()


@dataclass(frozen=True)
class _ShownFit:
    """The fit the three actions read, with the data and ranking they need."""

    bundle: FitBundle
    spec: FitSpec
    ranking: AutoFitResult | None
    data: XY | None
    truth: dict[str, float]
    converged: bool


class FitToolsView(QWidget):
    """The More tools tab: three cancellable runs, their forms and their results."""

    plot_selected = Signal(object, int)

    def __init__(self, runner: JobRunner | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fitToolsView")
        self._runner = runner if runner is not None else JobRunner(self)
        self._fit: _ShownFit | None = None
        self._running = False
        self._running_tool: str | None = None
        self._cancel_event: threading.Event | None = None
        self._last_error = ""
        self._dialog = FitToolResultDialog(self)
        self._dialog.plot_selected.connect(self.plot_selected.emit)
        self._run_buttons: dict[str, QPushButton] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)
        outer.addLayout(self._build_header())
        outer.addWidget(self._build_scroll(), stretch=1)
        self._sync()

    # -- construction ------------------------------------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.hint_label = QLabel("Open a fit to run any of these tools")
        self.hint_label.setObjectName("fitToolsHint")
        self.hint_label.setWordWrap(True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("fitToolsCancelButton")
        self.cancel_button.setToolTip("Stop the running tool; its partial result is shown")
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.hint_label, stretch=1)
        row.addWidget(self.cancel_button)
        return row

    def _build_scroll(self) -> QScrollArea:
        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(2, 2, 2, 2)
        column.setSpacing(8)
        self.message_label = QLabel("")
        self.message_label.setObjectName("bannerError")
        self.message_label.setWordWrap(True)
        self.message_label.hide()
        column.addWidget(self.message_label)
        self.progress = QProgressBar()
        self.progress.setObjectName("fitToolsProgress")
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.hide()
        column.addWidget(self.progress)
        column.addWidget(self._build_discernibility())
        column.addWidget(self._build_feasibility())
        column.addWidget(self._build_advisor())
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(body)
        return scroll

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 8)
        return group, form

    def _build_discernibility(self) -> QGroupBox:
        group, form = self._group("Model discernibility")
        self.discern_level = QDoubleSpinBox()
        self.discern_level.setDecimals(2)
        self.discern_level.setRange(0.5, 0.9999)
        self.discern_level.setSingleStep(0.01)
        self.discern_level.setValue(0.95)
        self.discern_grid = QSpinBox()
        self.discern_grid.setRange(2, 10_000)
        self.discern_grid.setValue(200)
        self.discern_max_models = QSpinBox()
        self.discern_max_models.setRange(1, 100)
        self.discern_max_models.setValue(12)
        self.discern_profile = QCheckBox("Profile-likelihood intervals")
        self.discern_profile.setChecked(True)
        self.discern_run = QPushButton(RUN_TEXT)
        self.discern_run.setObjectName("fitToolsDiscernButton")
        self.discern_run.setToolTip("Report whether these data can tell the leading models apart")
        self.discern_run.clicked.connect(self.run_discernibility)
        form.addRow("Confidence level", self.discern_level)
        form.addRow("Grid points", self.discern_grid)
        form.addRow("Most models", self.discern_max_models)
        form.addRow("", self.discern_profile)
        form.addRow(self.discern_run)
        self._run_buttons["discernibility"] = self.discern_run
        return group

    def _build_feasibility(self) -> QGroupBox:
        group, form = self._group("Experiment feasibility")
        self.feas_runs = QSpinBox()
        self.feas_runs.setRange(2, 5_000)
        self.feas_runs.setValue(200)
        self.feas_seed = QSpinBox()
        self.feas_seed.setRange(0, _MAX_SEED)
        self.feas_seed.setValue(0)
        noise = QWidget()
        noise_row = QHBoxLayout(noise)
        noise_row.setContentsMargins(0, 0, 0, 0)
        self.feas_noise_kind = QComboBox()
        self.feas_noise_kind.addItem("constant SD", "sd")
        self.feas_noise_kind.addItem("coefficient of variation", "cv")
        self.feas_noise_value = QDoubleSpinBox()
        self.feas_noise_value.setDecimals(4)
        self.feas_noise_value.setRange(0.0, 1e6)
        self.feas_noise_value.setValue(0.1)
        noise_row.addWidget(self.feas_noise_kind, stretch=1)
        noise_row.addWidget(self.feas_noise_value)
        self.feas_design_mode = QComboBox()
        self.feas_design_mode.addItem("range (low, high, count)", "range")
        self.feas_design_mode.addItem("explicit list", "list")
        self.feas_low = QDoubleSpinBox()
        self.feas_high = QDoubleSpinBox()
        for spin in (self.feas_low, self.feas_high):
            spin.setDecimals(6)
            spin.setRange(-1e12, 1e12)
        self.feas_high.setValue(1.0)
        self.feas_count = QSpinBox()
        self.feas_count.setRange(1, 200)
        self.feas_count.setValue(5)
        self.feas_locations = QLineEdit()
        self.feas_locations.setPlaceholderText("x values separated by commas, e.g. 0, 2.5, 5")
        self.feas_replicates = QSpinBox()
        self.feas_replicates.setRange(1, 20)
        self.feas_replicates.setValue(1)
        self.feas_difference = QLineEdit()
        self.feas_difference.setPlaceholderText("optional, e.g. a=0.5, b=0.1")
        self.feas_truth_label = QLabel(TRUTH_NOTE)
        self.feas_truth_label.setObjectName("fitToolsTruth")
        self.feas_truth_label.setWordWrap(True)
        self.feas_run = QPushButton(RUN_TEXT)
        self.feas_run.setObjectName("fitToolsFeasibilityButton")
        self.feas_run.setToolTip("Simulate a design and measure how the estimates recover")
        self.feas_run.clicked.connect(self.run_feasibility)
        form.addRow("Simulated runs", self.feas_runs)
        form.addRow("Seed", self.feas_seed)
        form.addRow("Noise", noise)
        form.addRow("Design", self.feas_design_mode)
        form.addRow("low", self.feas_low)
        form.addRow("high", self.feas_high)
        form.addRow("count", self.feas_count)
        form.addRow("locations", self.feas_locations)
        form.addRow("Replicates", self.feas_replicates)
        form.addRow("Difference", self.feas_difference)
        form.addRow(self.feas_truth_label)
        form.addRow(self.feas_run)
        self.feas_form = form
        self.feas_design_mode.currentIndexChanged.connect(self._sync_design_rows)
        self._run_buttons["feasibility"] = self.feas_run
        self._sync_design_rows()
        return group

    def _build_advisor(self) -> QGroupBox:
        group, form = self._group("Where to measure next")
        self.adv_points = QSpinBox()
        self.adv_points.setRange(1, 20)
        self.adv_points.setValue(3)
        self.adv_candidates = QSpinBox()
        self.adv_candidates.setRange(10, 1_000)
        self.adv_candidates.setValue(400)
        self.adv_low = QDoubleSpinBox()
        self.adv_high = QDoubleSpinBox()
        for spin in (self.adv_low, self.adv_high):
            spin.setDecimals(6)
            spin.setRange(_UNSET_NUMBER, 1e12)
            spin.setSpecialValueText("data range")
            spin.setValue(_UNSET_NUMBER)
        self.adv_replicates = QSpinBox()
        self.adv_replicates.setRange(1, 20)
        self.adv_replicates.setValue(1)
        self.adv_objective = QComboBox()
        self.adv_objective.addItem("D-optimal", "d_optimal")
        self.adv_objective.addItem("one parameter's SE", "parameter")
        self.adv_parameter = QComboBox()
        self.adv_parameter.setEditable(False)
        self.adv_run = QPushButton(RUN_TEXT)
        self.adv_run.setObjectName("fitToolsAdvisorButton")
        self.adv_run.setToolTip("Recommend the next measurement locations for this fit")
        self.adv_run.clicked.connect(self.run_advisor)
        form.addRow("Points", self.adv_points)
        form.addRow("Candidates", self.adv_candidates)
        form.addRow("low", self.adv_low)
        form.addRow("high", self.adv_high)
        form.addRow("Replicates", self.adv_replicates)
        form.addRow("Objective", self.adv_objective)
        form.addRow("Parameter", self.adv_parameter)
        form.addRow(self.adv_run)
        self._run_buttons["advisor"] = self.adv_run
        return group

    def _sync_design_rows(self, _index: int = 0) -> None:
        """The range's controls and the explicit list's field follow the design mode."""
        ranged = self.feas_design_mode.currentData() == "range"
        for field in (self.feas_low, self.feas_high, self.feas_count):
            field.setVisible(ranged)
            label = self.feas_form.labelForField(field)
            if label is not None:
                label.setVisible(ranged)
        self.feas_locations.setVisible(not ranged)
        label = self.feas_form.labelForField(self.feas_locations)
        if label is not None:
            label.setVisible(not ranged)

    # -- the fit -----------------------------------------------------------------------------

    def set_fit(
        self,
        bundle: FitBundle,
        spec: FitSpec,
        *,
        ranking: AutoFitResult | None = None,
        data: XY | None = None,
    ) -> None:
        """Point the three actions at ``bundle``/``spec``; ``ranking`` for the report only."""
        xy = data if data is not None else xy_from_bundle(bundle)
        names = [parameter.name for parameter in bundle.result.parameters]
        truth = {parameter.name: parameter.value for parameter in bundle.result.parameters}
        self._fit = _ShownFit(
            bundle=bundle,
            spec=spec,
            ranking=ranking,
            data=xy,
            truth=truth,
            converged=str(bundle.result.status) == "converged",
        )
        self._last_error = ""
        self._show_message("")
        self._refresh_choices(names, xy)
        self._sync()

    def clear(self) -> None:
        """No fit is open: the actions cannot run and the last result is put away."""
        self._fit = None
        self._running = False
        self._running_tool = None
        self._cancel_event = None
        self._last_error = ""
        self._show_message("")
        self._dialog.hide()
        self._sync()

    def _refresh_choices(self, names: list[str], xy: XY | None) -> None:
        """Fill the parameter choices and the design's default range from the fit."""
        current = self.adv_parameter.currentText()
        self.adv_parameter.clear()
        self.adv_parameter.addItems(names)
        index = self.adv_parameter.findText(current)
        if index >= 0:
            self.adv_parameter.setCurrentIndex(index)
        values = ", ".join(f"{name} = {value:.6g}" for name, value in self._truth_items())
        self.feas_truth_label.setText(TRUTH_NOTE + ("" if not values else f": {values}"))
        if xy is not None and xy.x.size:
            self.feas_low.setValue(float(np.min(xy.x)))
            self.feas_high.setValue(float(np.max(xy.x)))

    def _truth_items(self) -> list[tuple[str, float]]:
        return [] if self._fit is None else list(self._fit.truth.items())

    def shown_ranking(self) -> AutoFitResult | None:
        """The Auto-fit run the report will read (``None`` for a fit opened otherwise)."""
        return None if self._fit is None else self._fit.ranking

    def is_running(self) -> bool:
        return self._running

    # -- availability ------------------------------------------------------------------------

    def _reason(self, kind: str) -> str:
        """Why ``kind`` cannot run now ('' when it can); the button's tool tip."""
        fit = self._fit
        if fit is None:
            return _NO_FIT_REASON
        if kind == "discernibility":
            if fit.ranking is None:
                return _NO_RANKING_REASON
            if fit.data is None:
                return _NO_DATA_REASON
            return ""
        if kind == "feasibility":
            if not fit.truth:
                return "The fit has no parameter estimates to simulate from"
            return ""
        if fit.data is None:
            return _NO_DATA_REASON
        if not fit.converged:
            return "The fit did not converge, so there is no covariance to design from"
        return ""

    def _show_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def _sync(self) -> None:
        """Every control's state from what is shown and what is running."""
        has_fit = self._fit is not None
        self.setEnabled(has_fit)
        self.hint_label.setVisible(not has_fit)
        busy = self._running
        self.progress.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.cancel_button.setVisible(busy)
        for kind, button in self._run_buttons.items():
            reason = self._reason(kind)
            button.setEnabled(has_fit and not busy and not reason)
            button.setToolTip(reason or _BUTTON_TIPS[kind])
            button.setText(RUNNING_TEXT if busy and self._running_tool == kind else RUN_TEXT)

    # -- running -----------------------------------------------------------------------------

    def _start(self, kind: str, compute: Callable[[Callable[[], bool]], ToolResult]) -> None:
        """Submit ``compute`` off the GUI thread, with the token the Cancel button lends."""
        if self._fit is None or self._running or self._reason(kind):
            return
        event = threading.Event()
        self._cancel_event = event
        self._running = True
        self._running_tool = kind
        self._last_error = ""
        self._show_message(RUNNING_MESSAGES[kind])
        self._sync()

        def job(progress: ProgressCallback) -> ToolResult:
            progress(0, 1)
            try:
                return compute(event.is_set)
            finally:
                progress(1, 1)

        self._runner.submit(
            job,
            on_result=self._on_result,
            on_error=self._on_error,
            on_progress=self._on_progress,
            on_settled=self._on_settled,
        )

    def cancel(self) -> None:
        """Ask the running tool to stop; it returns the part it finished, with its warning."""
        event = self._cancel_event
        if self._running and event is not None and not event.is_set():
            event.set()
            self._show_message(CANCELLING_TEXT)

    def _on_result(self, result: ToolResult) -> None:
        self._dialog.show_result(result)

    def _on_error(self, error: BaseException) -> None:
        self._last_error = str(error)

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_settled(self) -> None:
        self._running = False
        self._running_tool = None
        self._cancel_event = None
        self._show_message(self._last_error)
        self._sync()

    # -- the three actions -------------------------------------------------------------------

    def run_discernibility(self) -> None:
        fit = self._fit
        if fit is None or fit.data is None or fit.ranking is None:
            return
        request = DiscernibilityRequest(
            level=self.discern_level.value(),
            grid_points=self.discern_grid.value(),
            max_models=self.discern_max_models.value(),
            profile=self.discern_profile.isChecked(),
        )
        data, ranking = fit.data, fit.ranking
        self._start(
            "discernibility",
            lambda should_cancel: discernibility(
                data, ranking, request=request, should_cancel=should_cancel
            ),
        )

    def run_feasibility(self) -> None:
        fit = self._fit
        if fit is None:
            return
        design = self._design()
        if design is None:
            return
        difference = self._difference()
        if difference is None:
            return
        kind: Literal["sd", "cv"] = "cv" if self.feas_noise_kind.currentIndex() == 1 else "sd"
        request = SimulationRequest(
            truth=dict(fit.truth),
            design=design,
            noise=NoiseSpec(kind=kind, value=self.feas_noise_value.value()),
            runs=self.feas_runs.value(),
            seed=self.feas_seed.value(),
            difference=difference,
        )
        spec = fit.spec
        self._start(
            "feasibility",
            lambda should_cancel: simulate(spec, request, should_cancel=should_cancel),
        )

    def run_advisor(self) -> None:
        fit = self._fit
        if fit is None or fit.data is None:
            return
        objective: Literal["d_optimal", "parameter"] = (
            "parameter" if self.adv_objective.currentIndex() == 1 else "d_optimal"
        )
        request = AdvisorRequest(
            points=self.adv_points.value(),
            candidates=self.adv_candidates.value(),
            low=self._optional(self.adv_low),
            high=self._optional(self.adv_high),
            replicates=self.adv_replicates.value(),
            objective=objective,
            parameter=self.adv_parameter.currentText() or None,
        )
        data, spec, result = fit.data, fit.spec, fit.bundle.result
        self._start(
            "advisor",
            lambda should_cancel: advise(
                data, spec, result, request=request, should_cancel=should_cancel
            ),
        )

    @staticmethod
    def _optional(spin: QDoubleSpinBox) -> float | None:
        value = spin.value()
        return None if value == _UNSET_NUMBER else float(value)

    def _design(self) -> DesignSpec | None:
        """The design the form holds, or ``None`` with the refusal shown in the form."""
        replicates = self.feas_replicates.value()
        if self.feas_design_mode.currentData() == "range":
            return DesignSpec(
                low=self.feas_low.value(),
                high=self.feas_high.value(),
                count=self.feas_count.value(),
                replicates=replicates,
            )
        text = self.feas_locations.text().replace(";", ",")
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            self._show_message("Give at least one design location, separated by commas")
            return None
        try:
            locations = tuple(float(part) for part in parts)
        except ValueError:
            self._show_message(f"not numbers: {text}")
            return None
        return DesignSpec(locations=locations, replicates=replicates)

    def _difference(self) -> dict[str, float] | None:
        """The differences to measure power for, or ``{}`` for none; refusals shown inline."""
        text = self.feas_difference.text().strip()
        if not text:
            return {}
        difference: dict[str, float] = {}
        for part in text.replace(";", ",").split(","):
            piece = part.strip()
            if not piece:
                continue
            name, separator, value = piece.partition("=")
            if not separator:
                self._show_message(f"a difference reads parameter=value, not {piece!r}")
                return None
            try:
                difference[name.strip()] = float(value)
            except ValueError:
                self._show_message(f"not a number: {value.strip()}")
                return None
        return difference or {}

    def dialog(self) -> FitToolResultDialog:
        """The result window this panel opens (one per panel, reused)."""
        return self._dialog


__all__ = [
    "MORE_TOOLS_TITLE",
    "FitToolResultDialog",
    "FitToolsView",
    "xy_from_bundle",
]
