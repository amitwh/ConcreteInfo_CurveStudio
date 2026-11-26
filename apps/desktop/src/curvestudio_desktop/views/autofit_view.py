"""AutoFitView: the Auto-fit half of the analysis dock — the primary workflow.

One place shows everything a run produces: a determinate progress bar with counts and
elapsed time while candidates stream in, the live top-5 ranking (replaced by the full
table when the run ends), the ``RankBy`` selector that re-sorts instantly without
refitting, the collapsible exact-fit / not-converged / skipped lists with reasons, the
run's warnings, an overfitting note banner when the ranking raises one, and the options
disclosure (families, exhaustive, parameter cap, starts, CV folds, seed). A cancelled run
keeps its partial ranking behind a clear notice.
"""

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QPaintEvent, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from curvestudio_desktop.controllers.analysis import AnalysisController
from curvestudio_desktop.views.collapsible import CollapsibleSection
from curvestudio_desktop.views.ranking_table import RankingTableView

if TYPE_CHECKING:
    from curvestudio.discovery import AutoFitConfig, AutoFitResult, RankedFit
    from curvestudio_desktop.controllers.analysis_jobs import AutoFitLive

RANK_LABELS: tuple[tuple[str, str], ...] = (
    ("best_overall", "Best overall (recommended)"),
    ("r2", "R²"),
    ("adj_r2", "adjusted R²"),
    ("rmse", "RMSE"),
    ("aicc", "AICc"),
    ("bic", "BIC"),
    ("cv_rmse", "CV-RMSE"),
)
"""The criteria as their stored strings (``RankBy`` values), so the combo builds with no
import; a window-first test pins them to the enum's members."""
_MAX_SEED = 2**31 - 1
#: Smallest usable height of the disclosure scroll. The region can always shrink this far
#: and scroll, so its content never raises the page's minimum past the dock's height.
_DISCLOSURE_MIN_HEIGHT = 32


class ElidingComboBox(QComboBox):
    """A closed combo that elides its label with an ellipsis instead of clipping a word.

    ``QComboBox`` paints the current item's text into whatever width the layout gives it,
    so a label like "Best overall (recommended)" was cut mid-word at the dock's default
    width. The popup keeps the full item labels; only the closed field elides, and the
    full label stays in the widget's tooltip. :meth:`display_text` returns exactly what
    the field paints, so tests can assert the visible string rather than the full label.
    """

    def _option(self) -> QStyleOptionComboBox:
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        return option

    def _text_rect(self) -> QRect:
        option = self._option()
        return self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )

    def display_text(self) -> str:
        """The closed field's text: the full label when it fits, else an elided version."""
        option = self._option()
        return self.fontMetrics().elidedText(
            option.currentText,
            Qt.TextElideMode.ElideRight,
            max(0, self._text_rect().width() - 2),
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        # draw the frame/arrow with an empty label, then the elided text in the field
        option = self._option()
        text = option.currentText
        option.currentText = ""
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        flags = self.style().visualAlignment(
            self.layoutDirection(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        painter.drawItemText(
            rect,
            int(flags),
            option.palette,
            self.isEnabled(),
            self.fontMetrics().elidedText(
                text, Qt.TextElideMode.ElideRight, max(0, rect.width() - 2)
            ),
            QPalette.ColorRole.Text,
        )


class AutoFitView(QWidget):
    """Everything about the current dataset's Auto-fit run."""

    rowSelected = Signal(str)
    """A ranking row was selected: its candidate id (the plot follows)."""

    openRequested = Signal(str)
    """A row's full result was requested (double-click, Enter, button): its candidate id."""

    discernRequested = Signal(str)
    """A row's discernibility report was requested: its candidate id.

    The owner opens the row (refitting it exactly as the run fitted it) and shows the
    details window's More tools tab; the report itself is the user's to run there.
    """

    top3Changed = Signal(bool)
    """The "Show top 3" option changed (the shell persists it and redraws the plot)."""

    def __init__(self, analysis: AnalysisController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._analysis = analysis
        self._result: AutoFitResult | None = None
        self._running = False
        self._live_rows: tuple[RankedFit, ...] = ()
        self._started = 0.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self._build_run_row(layout)
        self._build_ranking(layout)
        self._build_sections(layout)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_elapsed)
        analysis.rank_by_changed.connect(self._on_rank_changed)
        self.refresh_order()  # the closed combo's tooltip carries its full label from the start

    # -- construction ------------------------------------------------------------------------

    def _build_run_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setObjectName("autofitProgress")
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.counts_label = QLabel("")
        self.counts_label.setObjectName("autofitCounts")
        self.elapsed_label = QLabel("")
        self.run_button = QPushButton("Run again")
        self.run_button.setObjectName("autofitRunButton")
        self.run_button.setProperty("role", "primary")
        self.run_button.setToolTip("Fit every applicable model to the current data")
        self.run_button.clicked.connect(self._run_again)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("autofitCancelButton")
        self.cancel_button.clicked.connect(self._analysis.cancel_autofit)
        self.cancel_button.hide()
        row.addWidget(self.progress, stretch=1)
        row.addWidget(self.counts_label)
        row.addWidget(self.elapsed_label)
        row.addWidget(self.cancel_button)
        row.addWidget(self.run_button)
        layout.addLayout(row)

    def _build_ranking(self, layout: QVBoxLayout) -> None:
        self.notice = QLabel("")
        self.notice.setObjectName("bannerWarning")
        self.notice.setWordWrap(True)  # a long notice wraps instead of being clipped
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        self.notice.hide()
        layout.addWidget(self.notice)
        order_row = QHBoxLayout()
        self.order_label = QLabel("Order by")
        self.order_label.setObjectName("orderByLabel")
        self.rank_combo = ElidingComboBox()
        self.rank_combo.setObjectName("rankBySelector")
        # long catalogue-style labels stay on the items, but must never force the dock
        # wide: the combo clips instead of resizing the panel
        self.rank_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.rank_combo.setMinimumContentsLength(10)
        for rank_value, label in RANK_LABELS:
            self.rank_combo.addItem(label, rank_value)
        self.rank_combo.setCurrentIndex(
            next(
                i
                for i, (value, _label) in enumerate(RANK_LABELS)
                if value == self._analysis.rank_by_value
            )
        )
        self.rank_combo.currentIndexChanged.connect(self._rank_combo_changed)
        order_row.addWidget(self.order_label)
        order_row.addWidget(self.rank_combo, stretch=1)
        layout.addLayout(order_row)
        # The row actions get their own line: all four controls in one row set a ~490 px
        # minimum width for the whole dock, which stopped the user shrinking the panel
        # (and so enlarging the plot). Split, the floor is the wider of the two rows.
        action_row = QHBoxLayout()
        self.top3_check = QCheckBox("Show top 3")
        self.top3_check.setObjectName("showTop3Check")
        self.top3_check.setToolTip("Overlay the three best-ranked models on the plot")
        self.top3_check.toggled.connect(self.top3Changed)
        self.open_button = QPushButton("Open full results")
        self.open_button.setToolTip("Refit the selected row and open its details window")
        self.open_button.clicked.connect(self._open_selected)
        self.discern_button = QPushButton("Discernibility report")
        self.discern_button.setObjectName("autofitDiscernButton")
        self.discern_button.setToolTip(
            "Refit the selected row and open its More tools tab, where the report runs"
        )
        self.discern_button.clicked.connect(self._emit_discern)
        action_row.addWidget(self.top3_check)
        action_row.addStretch(1)
        action_row.addWidget(self.discern_button)
        action_row.addWidget(self.open_button)
        layout.addLayout(action_row)
        self.table = RankingTableView()
        # The ranking is the primary result and never collapses; its *minimum* is a small
        # usable window (three rows — everything more scrolls), because that floor is
        # what decides whether the user can give the plot and the data table their space.
        # The default splitter share still shows eight-plus rows on a normal screen.
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        row_height = self.table.verticalHeader().defaultSectionSize()
        header_height = self.table.horizontalHeader().sizeHint().height()
        self.table.setMinimumHeight(3 * row_height + header_height + 4)
        self.table.selectionModel().currentRowChanged.connect(
            lambda _current, _previous: self._emit_selected()
        )
        self.table.openRequested.connect(self._emit_open)
        layout.addWidget(self.table, stretch=1)
        self.overfit_banner = QLabel("")
        self.overfit_banner.setObjectName("bannerWarning")
        self.overfit_banner.setWordWrap(True)
        self.overfit_banner.setTextFormat(Qt.TextFormat.PlainText)
        self.overfit_banner.hide()
        layout.addWidget(self.overfit_banner)

    def _build_sections(self, layout: QVBoxLayout) -> None:
        self.exact_section = CollapsibleSection("Exact fits")
        self.unconverged_section = CollapsibleSection("Not converged")
        self.skipped_section = CollapsibleSection("Skipped")
        self.warnings_label = QLabel("")
        self.warnings_label.setObjectName("bannerWarning")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setTextFormat(Qt.TextFormat.PlainText)
        self.warnings_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # a bounded, scrollable banner: several warnings can never overlap or push
        # the ranking table off the dock
        self.warnings_scroll = QScrollArea()
        self.warnings_scroll.setObjectName("warningsScroll")
        self.warnings_scroll.setWidgetResizable(True)
        self.warnings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.warnings_scroll.setMaximumHeight(96)
        self.warnings_scroll.setWidget(self.warnings_label)
        self.warnings_scroll.hide()
        # The disclosures below the ranking can be tall (a long skipped list, a stack of
        # warnings, the open options form). A tall set must scroll inside its own region
        # instead of forcing the page's minimum past the dock: the splitter would then
        # overflow into the neighbouring Data-table dock and cover the fit summary. The
        # ranking table stays outside this scroll so the primary surface keeps its own
        # space and its own scrolling.
        self.disclosure_body = QWidget()
        disclosure = QVBoxLayout(self.disclosure_body)
        disclosure.setContentsMargins(0, 0, 0, 0)
        disclosure.setSpacing(4)
        for section in (self.exact_section, self.unconverged_section, self.skipped_section):
            section.set_items([])
            disclosure.addWidget(section)
        disclosure.addWidget(self.warnings_scroll)
        self._build_options(disclosure)
        disclosure.addStretch(1)
        self.disclosure_scroll = QScrollArea()
        self.disclosure_scroll.setObjectName("autofitDisclosures")
        self.disclosure_scroll.setWidgetResizable(True)
        self.disclosure_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.disclosure_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.disclosure_scroll.setMinimumHeight(_DISCLOSURE_MIN_HEIGHT)
        self.disclosure_scroll.setWidget(self.disclosure_body)
        layout.addWidget(self.disclosure_scroll)

    def _build_options(self, layout: QVBoxLayout) -> None:
        self.options_button = QToolButton()
        self.options_button.setText("Options")
        self.options_button.setCheckable(True)
        self.options_button.setArrowType(Qt.ArrowType.RightArrow)
        self.options_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.options_button.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        layout.addWidget(self.options_button)
        self.options_body = QFrame()
        self.options_body.setFrameShape(QFrame.Shape.StyledPanel)
        form = QFormLayout(self.options_body)
        self.families_edit = QLineEdit()
        self.families_edit.setPlaceholderText("all families (blank)")
        self.families_edit.editingFinished.connect(self._apply_options)
        self.exhaustive_check = QCheckBox(
            "Exhaustive (include the long tail of generated families)"
        )
        self.exhaustive_check.toggled.connect(self._apply_options)
        self.max_params_spin = QSpinBox()
        self.max_params_spin.setRange(0, 20)
        self.max_params_spin.setSpecialValueText("auto")
        self.max_params_spin.valueChanged.connect(self._apply_options)
        self.n_starts_spin = QSpinBox()
        self.n_starts_spin.setRange(1, 64)
        self.n_starts_spin.setValue(8)
        self.n_starts_spin.valueChanged.connect(self._apply_options)
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(0, 20)
        self.cv_folds_spin.setSpecialValueText("off")
        self.cv_folds_spin.setValue(5)
        self.cv_folds_spin.valueChanged.connect(self._apply_options)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, _MAX_SEED)
        self.seed_spin.setValue(0)
        self.seed_spin.valueChanged.connect(self._apply_options)
        form.addRow("families (categories)", self.families_edit)
        form.addRow("", self.exhaustive_check)
        form.addRow("max parameters", self.max_params_spin)
        form.addRow("multi-starts", self.n_starts_spin)
        form.addRow("CV folds", self.cv_folds_spin)
        form.addRow("seed", self.seed_spin)
        self.options_body.setVisible(False)
        self.options_button.toggled.connect(self._toggle_options)
        layout.addWidget(self.options_body)

    def _toggle_options(self, open: bool) -> None:
        self.options_body.setVisible(open)
        self.options_button.setArrowType(
            Qt.ArrowType.DownArrow if open else Qt.ArrowType.RightArrow
        )

    # -- options -----------------------------------------------------------------------------

    def set_config(self, config: AutoFitConfig) -> None:
        """Set the options the next run uses (tests and the selftest inject tiny configs)."""
        self._analysis.set_autofit_options(config)
        families = "" if config.families is None else ", ".join(config.families)
        self.families_edit.setText(families)
        self.exhaustive_check.setChecked(config.exhaustive)
        self.max_params_spin.setValue(config.max_params or 0)
        self.n_starts_spin.setValue(config.n_starts)
        self.cv_folds_spin.setValue(config.cv_folds or 0)
        self.seed_spin.setValue(config.seed % (_MAX_SEED + 1))

    def _apply_options(self) -> None:
        text = self.families_edit.text()
        families = tuple(part.strip() for part in text.replace(";", ",").split(",") if part.strip())
        base = self._analysis.autofit_options()
        self._analysis.set_autofit_options(
            base.model_copy(
                update={
                    "families": families or None,
                    "exhaustive": self.exhaustive_check.isChecked(),
                    "max_params": self.max_params_spin.value() or None,
                    "n_starts": self.n_starts_spin.value(),
                    "cv_folds": self.cv_folds_spin.value() or None,
                    "seed": self.seed_spin.value(),
                }
            )
        )

    # -- view options ------------------------------------------------------------------------

    def show_top3(self) -> bool:
        """Whether the plot overlays the three best-ranked models."""
        return self.top3_check.isChecked()

    def set_show_top3(self, enabled: bool) -> None:
        """Set the option without re-emitting (restoring the persisted choice)."""
        self.top3_check.blockSignals(True)
        self.top3_check.setChecked(enabled)
        self.top3_check.blockSignals(False)

    # -- run control -------------------------------------------------------------------------

    def _run_again(self) -> None:
        dataset_id = self._analysis.running_dataset or self._analysis.selection_active_id()
        if dataset_id is not None:
            self._analysis.run_autofit(dataset_id)

    def _rank_combo_changed(self, index: int) -> None:
        from curvestudio.discovery import RankBy

        self._analysis.set_rank_by(RankBy(str(self.rank_combo.itemData(index))))

    def _on_rank_changed(self, rank_by: object) -> None:
        value = getattr(rank_by, "value", rank_by)
        index = next((i for i, (known, _label) in enumerate(RANK_LABELS) if known == value), None)
        if index is not None:
            self.rank_combo.setCurrentIndex(index)
        self.refresh_order()

    def _emit_selected(self) -> None:
        row = self.table.selected_row()
        if row is not None:
            self.rowSelected.emit(row.model_id)

    def _emit_open(self) -> None:
        row = self.table.selected_row()
        if row is not None:
            self.openRequested.emit(row.model_id)

    def _open_selected(self) -> None:
        self._emit_open()

    def _emit_discern(self) -> None:
        row = self.table.selected_row()
        if row is not None:
            self.discernRequested.emit(row.model_id)

    def current_candidate(self) -> str | None:
        row = self.table.selected_row()
        return None if row is None else row.model_id

    # -- display -----------------------------------------------------------------------------

    def run_started(self, dataset_id: str) -> None:
        """The run for ``dataset_id`` began: progress on, live ranking incoming."""
        del dataset_id
        self._running = True
        self._live_rows = ()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        self.run_button.hide()
        if self._result is not None:
            # the rows below belong to the previous run until live rows replace them
            self._set_notice("Previous run — new run in progress")
        else:
            self.notice.hide()
        self.overfit_banner.hide()
        self._started = time.monotonic()
        self._timer.start()
        self._tick_elapsed()

    def show_live(self, live: AutoFitLive) -> None:
        """A streaming snapshot: top rows so far plus counts and progress.

        The live ranking is what the table shows while the run is in progress; the
        previous result's rows stay behind only until the first snapshot lands.
        """
        self.progress.setRange(0, max(1, live.total))
        self.progress.setValue(live.done)
        self.counts_label.setText(f"{live.done}/{live.total} fitted · {live.counts_text()}")
        if live.rows:
            self._live_rows = live.rows
            self.table.set_rows(live.rows)
            self.notice.hide()  # live rows arrived: the stale marker is done

    def run_ended(self) -> None:
        self._running = False
        self._started = 0.0  # no stale start time may read as an elapsed value later
        self._timer.stop()
        self.progress.hide()
        self.cancel_button.hide()
        self.run_button.show()
        self.elapsed_label.setText("")

    def show_result(self, result: AutoFitResult) -> None:
        """A finished (or cancelled, or restored) run: full table, sections, banners."""
        self._result = result
        self._live_rows = ()
        self.run_ended()
        if result.cancelled:
            self._set_notice(
                f"Run cancelled — partial ranking ({len(result.ranked)} fitted, "
                f"{len(result.skipped)} never attempted)."
            )
        else:
            self.notice.hide()
        self.refresh_order()
        if self.table.rows():
            # Auto-fit first: the best row (and its curve on the plot) show immediately
            self.table.select_row(0)
        self.exact_section.set_items(
            [
                (
                    cid,
                    "reproduces the data exactly; information criteria are not defined",
                )
                for cid in result.exact_fits
            ]
        )
        self.unconverged_section.set_items(sorted(result.unconverged.items()))
        self.skipped_section.set_items(
            sorted((cid, "; ".join(reasons)) for cid, reasons in result.skipped.items())
        )
        self.warnings_label.setText("\n\n".join(result.warnings))
        self.warnings_scroll.setVisible(bool(result.warnings))

    def show_failure(self, message: str) -> None:
        """The run could not start at all (too few points, no columns): say why inline."""
        self.run_ended()
        self._set_notice(f"Auto-fit could not run: {message}")

    def _set_notice(self, text: str) -> None:
        """Show one banner line; the full text stays available as its tooltip."""
        self.notice.setText(text)
        self.notice.setToolTip(text)
        self.notice.show()

    def refresh_order(self) -> None:
        """Re-render the rows under the active ordering (no refit)."""
        rank_value = self._analysis.rank_by_value  # the enum would import the engine
        label = next((text for value, text in RANK_LABELS if value == rank_value), rank_value)
        # the combo keeps the long label; the tooltip spells out the recommendation
        self.rank_combo.setToolTip(
            f"Order by {label}"
            + (" — the engine's recommended default" if rank_value == RANK_LABELS[0][0] else "")
        )
        if self._running:
            # the live ranking is the truth while a run is in progress; a RankBy
            # change re-sorts those rows, never the previous result's
            if self._live_rows:
                from curvestudio.discovery import rank

                self.table.set_rows(rank(self._live_rows, self._analysis.rank_by))
        elif self._result is not None:
            self._render_rows(self._result.ranked_by(self._analysis.rank_by))
        self._refresh_overfit_banner()

    def _render_rows(self, rows: Sequence[RankedFit]) -> None:
        """Replace the table's rows, keeping the selected row selected by its id."""
        previous = self.table.selected_row()
        keep = None if previous is None else previous.model_id
        self.table.set_rows(rows)
        if keep is not None:
            index = next((i for i, row in enumerate(rows) if row.model_id == keep), None)
            if index is not None:
                self.table.select_row(index)

    def _refresh_overfit_banner(self) -> None:
        if self._running:  # the previous run's banner says nothing about the live rows
            self.overfit_banner.hide()
            return
        result = self._result
        if result is None:
            self.overfit_banner.hide()
            return
        notes: list[str] = []
        if result.overfitting_note:
            notes.append(result.overfitting_note)
        rows = self.table.rows()
        if rows and rows[0].overfitting_note:
            notes.append(rows[0].overfitting_note)
        self.overfit_banner.setText("\n".join(notes))
        self.overfit_banner.setToolTip("\n".join(notes))
        self.overfit_banner.setVisible(bool(notes))

    def clear(self) -> None:
        """No run to show (no dataset, or Manual mode never ran)."""
        self._result = None
        self._live_rows = ()
        self.run_ended()
        self.notice.hide()
        self.table.set_rows([])
        for section in (self.exact_section, self.unconverged_section, self.skipped_section):
            section.set_items([])
        self.warnings_label.setText("")
        self.warnings_scroll.hide()
        self.overfit_banner.hide()
        self.counts_label.setText("")

    def _tick_elapsed(self) -> None:
        # a missing start time means the elapsed value is unknown, not zero
        if not self._started:
            self.elapsed_label.setText("—")
            return
        self.elapsed_label.setText(f"{time.monotonic() - self._started:.0f} s")
