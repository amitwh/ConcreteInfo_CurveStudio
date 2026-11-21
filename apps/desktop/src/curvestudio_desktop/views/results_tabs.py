"""ResultsTabs: the five result tabs every fit fills (Parameters, Statistics, Diagnostics,
Equation, Predict).

The widget is deliberately self-contained: the analysis dock no longer shows it, the
details window hosts it, and any other home could add it without re-implementing a tab.
Each tab keeps an honest empty state ("No fit yet"). A fit that did not converge is shown
as failed: a red banner with its reason, parameters with whatever uncertainty the engine
could not give shown as ``n/a``, and no curve anywhere. Undefined statistics are ``n/a``,
never 0 or NaN. The Statistics tab reuses the report layer's shared ``statistics_rows`` so
the tabs and every exported report state identical numbers, conventions and notes; the
diagnostic tests appear with their assumption and limitation text — none of them ever says
a model is "valid". Diagnostics are four small pyqtgraph panels (vs fitted, vs x, Q-Q,
histogram); an undefined panel says ``n/a`` instead of drawing empty axes.
"""

from collections.abc import Callable
from typing import Any, Literal

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from curvestudio.workflow import (
    MAX_PREDICT_POINTS,
    FitBundle,
    FitSpec,
    Prediction,
    WorkflowError,
    format_equation,
)
from curvestudio_desktop.theme import mono_font
from curvestudio_reporting.csv_report import statistics_rows

EMPTY_TEXT = "No fit yet"
_NA = "n/a"
PredictCaller = Callable[
    [FitBundle, FitSpec, list[float], Literal["confidence", "prediction"] | None],
    Prediction,
]


def _number(value: float | None, digits: int = 6) -> str:
    return _NA if value is None else f"{value:.{digits}g}"


def _placeholder(text: str = EMPTY_TEXT) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _size_columns(header: QHeaderView, last_column: int) -> None:
    """Content-size every column but the last, which stretches to fill the viewport.

    The details window is wide; leaving the tables content-sized would pin them to a
    fraction of the width and waste the rest of the window. Used where the last column is
    prose (the statistics note) so the stretch is where the text needs it.
    """
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(last_column, QHeaderView.ResizeMode.Stretch)


#: Horizontal padding in the generic QSS table cells (``QTableView::item``: ``3px 6px``)
#: plus the header's own margins and the delegate's focus-frame reservation. Measured from
#: ``views/styles.py``, so a value is never elided by an under-sized section.
_COLUMN_PADDING = 12 + 8 + 8


def _distribute_columns(table: QTableWidget) -> None:
    """Give every column a content-sized minimum, then share the slack by width.

    Stretching only the last column made the parameters table read as a small grid plus a
    mostly empty band: at 1150 px the "95% CI high" column absorbed ~700 px while the
    data stayed cramped left. Here the extra width is spread over every column in
    proportion to the content it holds, so the table stays one grid and still sums to the
    viewport (no horizontal scrollbar).
    """
    header = table.horizontalHeader()
    columns = header.count()
    viewport = table.viewport().width()
    if columns == 0 or viewport <= 0:
        return
    cell = table.fontMetrics()
    title_font = header.fontMetrics()
    minimums: list[int] = []
    for column in range(columns):
        heading = table.horizontalHeaderItem(column)
        widest = 0 if heading is None else title_font.horizontalAdvance(heading.text())
        for row in range(table.rowCount()):
            item = table.item(row, column)
            if item is not None:
                widest = max(widest, cell.horizontalAdvance(item.text()))
        minimums.append(widest + _COLUMN_PADDING)
    content = sum(minimums)
    if content <= 0:
        return
    if content < viewport:
        slack = viewport - content
        widths = [width + slack * width // content for width in minimums]
        widths[-1] += viewport - sum(widths)  # rounding: the sum is still the viewport
    else:
        widths = minimums
    for column, width in enumerate(widths):
        header.resizeSection(column, width)


class _ProportionalTable(QTableWidget):
    """A result table whose columns share the viewport proportionally, never one band."""

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        _distribute_columns(self)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        _distribute_columns(self)


class _Tab(QStackedLayout):
    """A tab page that is either a centred placeholder or its real content."""

    def __init__(self, content: QWidget, text: str = EMPTY_TEXT) -> None:
        super().__init__()
        self._placeholder = _placeholder(text)
        self.addWidget(self._placeholder)
        self.addWidget(content)
        self._content = content

    def show_content(self, enabled: bool) -> None:
        self.setCurrentWidget(self._content if enabled else self._placeholder)

    def set_placeholder(self, text: str) -> None:
        self._placeholder.setText(text)

    def placeholder_text(self) -> str:
        return self._placeholder.text()


class ResultsTabs(QWidget):
    """The tabbed results area of the analysis dock."""

    TAB_TITLES = ("Parameters", "Statistics", "Diagnostics", "Equation", "Predict")

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        predict_fn: PredictCaller | None = None,
    ) -> None:
        super().__init__(parent)
        self._predict_fn = predict_fn
        self._bundle: FitBundle | None = None
        self._spec: FitSpec | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._banner = QLabel("")
        self._banner.setObjectName("bannerError")
        self._banner.setWordWrap(True)
        self._banner.hide()
        outer.addWidget(self._banner)
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, stretch=1)
        self._build_parameters()
        self._build_statistics()
        self._build_diagnostics()
        self._build_equation()
        self._build_predict()
        for title, tab in zip(self.TAB_TITLES, self._tabs_of, strict=True):
            holder = QWidget()
            holder.setLayout(tab)
            self._tabs.addTab(holder, title)

    # -- construction ------------------------------------------------------------------------

    def _build_parameters(self) -> None:
        self._parameters = _ProportionalTable(0, 5)
        self._parameters.setObjectName("parametersTable")
        self._parameters.setHorizontalHeaderLabels(
            ["parameter", "value", "std. err.", "95% CI low", "95% CI high"]
        )
        self._parameters.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._parameters.setFont(mono_font(9))
        header = self._parameters.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self._parameters.verticalHeader().setVisible(True)
        self._parameters_tab = _Tab(self._parameters)
        self._tabs_of: list[_Tab] = [self._parameters_tab]

    def _build_statistics(self) -> None:
        self._statistics = QTableWidget(0, 3)
        self._statistics.setObjectName("statisticsTable")
        self._statistics.setHorizontalHeaderLabels(["statistic", "value", "note"])
        self._statistics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._statistics.setFont(mono_font(9))
        _size_columns(self._statistics.horizontalHeader(), 2)  # last: note
        self._statistics.verticalHeader().setVisible(False)
        self._statistics_tab = _Tab(self._statistics)
        self._tabs_of.append(self._statistics_tab)

    def _build_diagnostics(self) -> None:
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(2, 2, 2, 2)
        self._diag_plots: list[pg.PlotWidget] = []
        titles = ("Residuals vs fitted", "Residuals vs x", "Normal Q-Q", "Histogram")
        for index, title in enumerate(titles):
            plot = pg.PlotWidget()
            # a sane floor per panel: four panels share one two-column grid
            plot.setMinimumSize(140, 120)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            if index < 2:  # the two residual panels read against a zero line
                plot.addLine(y=0, pen=pg.mkPen("#b0b0b0", width=1))
            grid.addWidget(QLabel(title), index // 2, index % 2, Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(plot, index // 2 + 1, index % 2)
            self._diag_plots.append(plot)
        self._diag_note = QLabel("")
        self._diag_note.setWordWrap(True)
        grid.addWidget(self._diag_note, 3, 0, 1, 2)
        self._diagnostics_tab = _Tab(content)
        self._tabs_of.append(self._diagnostics_tab)

    def _build_equation(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        self._equation_label = _placeholder("")
        self._equation_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._equation_label.setWordWrap(True)
        # the details window's headline number: large, mono and selectable
        self._equation_label.setFont(mono_font(14))
        copy = QPushButton("Copy equation")
        copy.clicked.connect(self.copy_equation)
        layout.addWidget(self._equation_label, stretch=1)
        layout.addWidget(copy, alignment=Qt.AlignmentFlag.AlignRight)
        self._equation_tab = _Tab(content)
        self._tabs_of.append(self._equation_tab)

    def _build_predict(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self._predict_input = QPlainTextEdit()
        self._predict_input.setFont(mono_font(10))
        self._predict_input.setPlaceholderText("x values, one per line (e.g. 2.5)")
        self._predict_input.setMaximumHeight(90)
        self._band_combo = QComboBox()
        self._band_combo.addItems(("no band", "95% confidence band", "95% prediction band"))
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self._run_predict)
        row.addWidget(self._predict_input, stretch=1)
        side = QVBoxLayout()
        side.addWidget(QLabel("band"))
        side.addWidget(self._band_combo)
        side.addWidget(predict_button)
        row.addLayout(side)
        layout.addLayout(row)
        self._predict_error = QLabel("")
        self._predict_error.setObjectName("bannerError")
        self._predict_error.setWordWrap(True)
        self._predict_error.hide()
        layout.addWidget(self._predict_error)
        self._predict_table = QTableWidget(0, 5)
        self._predict_table.setObjectName("predictTable")
        self._predict_table.setHorizontalHeaderLabels(["x", "ŷ", "lower", "upper", "extrapolated"])
        self._predict_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._predict_table.setFont(mono_font(9))
        _size_columns(self._predict_table.horizontalHeader(), 4)  # last: extrapolated
        self._predict_table.verticalHeader().setVisible(False)
        layout.addWidget(self._predict_table, stretch=1)
        self._predict_tab = _Tab(content)
        self._tabs_of.append(self._predict_tab)

    # -- D1 compatibility (tests read the empty state through these) ------------------------

    def tab_titles(self) -> list[str]:
        return [self._tabs.tabText(i) for i in range(self._tabs.count())]

    def tab_text_at(self, index: int) -> str:
        return self._tabs_of[index].placeholder_text()

    def count(self) -> int:
        return self._tabs.count()

    # -- filling -----------------------------------------------------------------------------

    def clear_all(self) -> None:
        """Back to the honest empty state: no fit is shown."""
        self._bundle = None
        self._spec = None
        self._banner.hide()
        self._banner.setText("")
        for tab in self._tabs_of:
            tab.show_content(False)
        self._parameters.setRowCount(0)
        self._statistics.setRowCount(0)
        self._predict_table.setRowCount(0)
        self._predict_error.hide()
        self._equation_label.setText(EMPTY_TEXT)
        for plot in self._diag_plots:
            plot.clear()
        self._diag_note.setText("")

    def show_bundle(self, bundle: FitBundle, spec: FitSpec, meta: object) -> None:
        """Fill every tab from ``bundle`` (``spec`` drives the Predict tab)."""
        del meta  # the tabs show the bundle; who produced it does not change the numbers
        self._bundle = bundle
        self._spec = spec
        result = bundle.result
        converged = str(result.status) == "converged"
        if not converged:
            self._banner.setText(
                f"FIT DID NOT CONVERGE ({result.status.value}): {result.termination_reason}"
            )
            self._banner.show()
            # the tabs never say "no fit yet" for a fit that ran and failed
            for tab in self._tabs_of:
                tab.set_placeholder(f"No fit: the fit is {result.status.value}")
        else:
            self._banner.hide()
            for tab in self._tabs_of:
                tab.set_placeholder(EMPTY_TEXT)
        self._fill_parameters(bundle)
        self._fill_statistics(bundle)
        self._fill_diagnostics(bundle)
        self._fill_equation(bundle)
        self._reset_predict()

    def _fill_parameters(self, bundle: FitBundle) -> None:
        estimates = bundle.result.parameters
        table = self._parameters
        table.setRowCount(len(estimates))
        for row, estimate in enumerate(estimates):
            cells = (
                estimate.name,
                _number(estimate.value),
                _number(estimate.std_err),
                _number(estimate.ci_low),
                _number(estimate.ci_high),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if column and text == _NA:
                    item.setToolTip("not determined by this fit")
                table.setItem(row, column, item)
        _distribute_columns(table)
        self._parameters_tab.show_content(bool(estimates))

    def _fill_statistics(self, bundle: FitBundle) -> None:
        rows: list[tuple[str, Any, str]] = list(statistics_rows(bundle))
        summary = bundle.summary
        if summary is not None:
            for test in summary.diagnostics.tests:
                value = (
                    _NA
                    if test.statistic is None
                    else f"{test.statistic:.4g}"
                    + (f" (p = {test.p_value:.4g})" if test.p_value is not None else "")
                )
                note = "; ".join(
                    filter(
                        None,
                        (
                            test.reason,
                            f"H₀: {test.null_hypothesis}",
                            "assumes: " + "; ".join(test.assumptions) if test.assumptions else "",
                            "limitations: " + "; ".join(test.limitations)
                            if test.limitations
                            else "",
                        ),
                    )
                )
                rows.append((f"test: {test.name}", value, note))
        table = self._statistics
        table.setRowCount(len(rows))
        for row, (name, value, note) in enumerate(rows):
            text = value if isinstance(value, str) else _number(value)
            for column, cell in enumerate((name, text, note)):
                item = QTableWidgetItem(cell)
                item.setToolTip(note if column == 2 else cell)
                if column == 1:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, column, item)
        self._statistics_tab.show_content(True)

    def _fill_diagnostics(self, bundle: FitBundle) -> None:
        summary = bundle.summary
        residuals = None if summary is None else summary.residuals
        reason = bundle.summary_reason or "the residual diagnostics are not defined"
        for plot in self._diag_plots:
            plot.clear()
        if residuals is None:
            self._diag_note.setText(reason)
            self._diagnostics_tab.show_content(False)
            return
        self._diagnostics_tab.show_content(True)
        scatter = {
            "symbol": "o",
            "symbolSize": 4,
            "pen": None,
            "brush": pg.mkBrush("#1f77b4"),
        }
        plots = self._diag_plots
        if residuals.fitted:
            plots[0].plot(residuals.fitted, residuals.residuals, **scatter)
        if residuals.x:
            plots[1].plot(residuals.x, residuals.residuals, **scatter)
        if residuals.qq_theoretical and residuals.qq_ordered:
            plots[2].plot(
                residuals.qq_theoretical,
                residuals.qq_ordered,
                **scatter,
            )
            low = min(min(residuals.qq_theoretical), min(residuals.qq_ordered))
            high = max(max(residuals.qq_theoretical), max(residuals.qq_ordered))
            plots[2].plot([low, high], [low, high], pen=pg.mkPen("#999999", width=1))
        edges, counts = residuals.hist_edges, residuals.hist_counts
        if len(edges) >= 2:
            # the histogram's width must never depend on a data-bin width: the
            # grid's minimum size (set above) is what keeps the panels sane
            plots[3].plot(
                x=[(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)],
                y=list(counts),
                stepMode=False,
                fillLevel=0,
                fillOutline=True,
                brush=pg.mkBrush(255, 127, 14, 150),
                pen=pg.mkPen(None),
            )
        notes = [
            name
            for name, empty in (
                ("vs fitted", not residuals.fitted),
                ("vs x", not residuals.x),
                ("Q-Q", not (residuals.qq_theoretical and residuals.qq_ordered)),
                ("histogram", len(edges) < 2),
            )
            if empty
        ]
        self._diag_note.setText("n/a: " + ", ".join(notes) if notes else "")

    def _fill_equation(self, bundle: FitBundle) -> None:
        self._equation_label.setText(format_equation(bundle.equation_text))
        self._equation_tab.show_content(True)

    def equation_text(self) -> str:
        """The equation exactly as shown ('' when there is no bundle)."""
        return "" if self._bundle is None else format_equation(self._bundle.equation_text)

    def copy_equation(self) -> bool:
        """Copy the shown equation to the clipboard; ``False`` when there is none."""
        text = self.equation_text()
        if not text:
            return False
        QGuiApplication.clipboard().setText(text)
        return True

    # -- prediction --------------------------------------------------------------------------

    def _reset_predict(self) -> None:
        self._predict_table.setRowCount(0)
        self._predict_error.hide()
        self._predict_tab.show_content(self._bundle is not None)

    def _parsed_x(self) -> tuple[list[float] | None, str]:
        text_values = [
            line.strip() for line in self._predict_input.toPlainText().splitlines() if line.strip()
        ]
        values: list[float] = []
        bad: list[str] = []
        for text in text_values:
            try:
                values.append(float(text))
            except ValueError:
                bad.append(text)
        if bad:
            return None, "not numbers: " + ", ".join(bad[:5])
        return values, ""

    def _run_predict(self) -> None:
        bundle, spec, caller = self._bundle, self._spec, self._predict_fn
        if bundle is None or spec is None or caller is None:
            return
        values, problem = self._parsed_x()
        if values is None:
            self._show_predict_error(problem)
            return
        if len(values) > MAX_PREDICT_POINTS:
            self._show_predict_error(
                f"{len(values):,} x values requested; the limit is {MAX_PREDICT_POINTS:,}"
            )
            return
        band: Literal["confidence", "prediction"] | None = None
        if self._band_combo.currentIndex() == 1:
            band = "confidence"
        elif self._band_combo.currentIndex() == 2:
            band = "prediction"
        try:
            prediction = caller(bundle, spec, values, band)
        except WorkflowError as exc:
            self._show_predict_error(f"{exc.code}: {exc}")
            return
        self._predict_error.hide()
        table = self._predict_table
        table.setRowCount(len(values))
        lowers = prediction.lower
        uppers = prediction.upper
        for row in range(len(values)):
            cells = (
                _number(prediction.x[row]),
                _number(prediction.y[row]),
                _number(lowers[row]) if lowers is not None else _NA,
                _number(uppers[row]) if uppers is not None else _NA,
                "yes" if prediction.extrapolated[row] else "",
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if 1 <= column <= 3:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, column, item)
        note = prediction.band_reason or ""
        if note:
            self._show_predict_error(note)

    def _show_predict_error(self, message: str) -> None:
        self._predict_error.setText(message)
        self._predict_error.show()

    # -- helpers -----------------------------------------------------------------------------

    def banner_text(self) -> str:
        """The red banner's text ('' when the fit converged)."""
        return self._banner.text()

    def parameters_text(self, row: int, column: int) -> str:
        item = self._parameters.item(row, column)
        return "" if item is None else item.text()

    def predict_rows(self) -> int:
        return self._predict_table.rowCount()
