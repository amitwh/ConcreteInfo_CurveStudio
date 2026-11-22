"""FitSummaryPanel: the compact fit summary at the foot of the analysis dock.

The ranking table is the dock's primary result, so the area below it shows one fit's
numbers in a few lines — model name, category, status, dataset, R² / adj. R² / RMSE / k —
plus two actions: open the full details window and export the result. Everything long
(equation, parameters, diagnostics, prediction) lives in the details window; the dock
never repeats it. With no selection the panel is a quiet hint, not an empty frame.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from curvestudio.discovery import RankedFit
from curvestudio.workflow import FitBundle, WorkflowError, get_model
from curvestudio_desktop.theme import mono_font
from curvestudio_desktop.views.ranking_table import format_metric, format_precise

HINT_TEXT = "Select a ranked row to see its numbers"
METRIC_TITLES = ("R²", "adj. R²", "RMSE", "k")
#: Floor height that always shows the header row, metric strip and actions. The content's
#: own minimum size hint is used when a theme's fonts make it larger.
MINIMUM_HEIGHT = 112


def apply_status_badge(label: QLabel, status: str) -> None:
    """Text plus the semantic state the QSS colours (green for converged, red otherwise)."""
    label.setText(status)
    label.setProperty("state", "ok" if status == "converged" else "failed")
    style = label.style()
    if style is not None:
        style.unpolish(label)
        style.polish(label)


def model_identity(model_id: str) -> tuple[str, str]:
    """The catalogue name and category for ``model_id``; the id itself when unknown."""
    if model_id.startswith("poly"):
        return "Polynomial", "polynomial"
    try:
        info = get_model(model_id)
    except WorkflowError:  # a custom or generated candidate is not in the catalogue
        return model_id, ""
    return info.name, info.category


class FitSummaryPanel(QWidget):
    """One fit's headline numbers and the actions that need the full result."""

    openDetailsRequested = Signal()
    """The user asked for the details window (the current selection or fit)."""

    exportRequested = Signal()
    """The user asked to export the fit's report (enabled once a bundle is shown)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fitSummaryPanel")
        self._bundle: FitBundle | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)
        self.hint = QLabel(HINT_TEXT)
        self.hint.setObjectName("summaryHint")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)
        self.content = QWidget(self)
        content = QVBoxLayout(self.content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(3)
        content.addLayout(self._build_header())
        self.category_dataset = QLabel("")
        self.category_dataset.setObjectName("summaryMeta")
        self.category_dataset.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content.addWidget(self.category_dataset)
        content.addLayout(self._build_metrics())
        content.addLayout(self._build_actions())
        outer.addWidget(self.content)
        # the summary's own floor: the splitter must never crush it to a sliver
        margins = outer.contentsMargins()
        content_min = self.content.minimumSizeHint().height() + margins.top() + margins.bottom()
        self.setMinimumHeight(max(MINIMUM_HEIGHT, content_min))
        self.clear()

    # -- construction ------------------------------------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.model_label = QLabel("")
        self.model_label.setObjectName("summaryModel")
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusBadge")
        row.addWidget(self.model_label, stretch=1)
        row.addWidget(self.status_label)
        return row

    def _build_metrics(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        # Full-precision R² strings are long; a tighter gap keeps the strip readable
        # at the dock's default width.
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(1)
        self._metric_values: dict[str, QLabel] = {}
        for column, title in enumerate(METRIC_TITLES):
            caption = QLabel(title)
            caption.setObjectName("summaryMetricCaption")
            value = QLabel("n/a")
            value.setObjectName("summaryMetricValue")
            value.setFont(mono_font(10))
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(caption, 0, column)
            grid.addWidget(value, 1, column)
            self._metric_values[title] = value
        grid.setColumnStretch(len(METRIC_TITLES), 1)
        return grid

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.open_button = QPushButton("Open details…")
        self.open_button.setObjectName("summaryOpenButton")
        self.open_button.clicked.connect(lambda: self.openDetailsRequested.emit())
        self.export_button = QPushButton("Export…")
        self.export_button.setObjectName("summaryExportButton")
        self.export_button.clicked.connect(lambda: self.exportRequested.emit())
        row.addStretch(1)
        row.addWidget(self.open_button)
        row.addWidget(self.export_button)
        return row

    # -- filling -----------------------------------------------------------------------------

    def show_row(self, row: RankedFit, dataset_name: str) -> None:
        """Summary of a ranked row (the full bundle arrives when its details are opened)."""
        self._bundle = None
        self._fill(
            name=row.name,
            category=row.category,
            status=row.status,
            dataset_name=dataset_name,
            r2=row.metrics.r2,
            adj_r2=row.metrics.adj_r2,
            rmse=row.metrics.rmse,
            k=row.metrics.k,
        )
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("Open the details first to export a report")
        self.open_button.setToolTip("Refit this row and open its details window")

    def show_bundle(
        self,
        bundle: FitBundle,
        *,
        dataset_name: str,
        name: str | None = None,
        category: str | None = None,
    ) -> None:
        """Summary of a complete fit (manual, opened row or restored)."""
        self._bundle = bundle
        stats = bundle.result.stats
        if name is None or category is None:
            fallback_name, fallback_category = model_identity(bundle.result.model_id)
            name = name if name is not None else fallback_name
            category = category if category is not None else fallback_category
        self._fill(
            name=name,
            category=category,
            status=bundle.result.status.value,
            dataset_name=dataset_name,
            r2=None if stats is None else stats.r2,
            adj_r2=None if stats is None else stats.adj_r2,
            rmse=None if stats is None else stats.rmse,
            k=len(bundle.result.parameters) if stats is None else stats.k,
        )
        self.export_button.setEnabled(True)
        self.export_button.setToolTip("Export this fit as JSON, XLSX, PDF or CSV")
        self.open_button.setToolTip("Open this fit's full details window")

    def clear(self) -> None:
        """Nothing is selected: a quiet hint instead of an empty frame."""
        self._bundle = None
        self.hint.show()
        self.content.hide()

    def _fill(
        self,
        *,
        name: str,
        category: str,
        status: str,
        dataset_name: str,
        r2: float | None,
        adj_r2: float | None,
        rmse: float | None,
        k: int,
    ) -> None:
        self.hint.hide()
        self.content.show()
        self.model_label.setText(name)
        apply_status_badge(self.status_label, status)
        meta = category or "—"
        self.category_dataset.setText(f"{meta} · {dataset_name}")
        # The tooltip carries the full value even if a narrow dock clips the label.
        for title, text in (
            ("R²", format_precise(r2)),
            ("adj. R²", format_precise(adj_r2)),
            ("RMSE", format_metric(rmse)),
            ("k", str(k)),
        ):
            label = self._metric_values[title]
            label.setText(text)
            label.setToolTip(text)

    # -- accessors (tests and the status bar) -------------------------------------------------

    def bundle(self) -> FitBundle | None:
        return self._bundle

    def hint_visible(self) -> bool:
        return not self.hint.isHidden()

    def model_text(self) -> str:
        return self.model_label.text()

    def status_text(self) -> str:
        return self.status_label.text()

    def dataset_text(self) -> str:
        return self.category_dataset.text()

    def metric_text(self, title: str) -> str:
        return self._metric_values[title].text()

    def metrics_text(self) -> str:
        return " ".join(f"{title}={self.metric_text(title)}" for title in METRIC_TITLES)


__all__ = ["HINT_TEXT", "FitSummaryPanel", "apply_status_badge", "model_identity"]
