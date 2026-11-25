"""DetailsDialog: one fit's complete result in a reusable, non-modal window.

The dock stays a ranking plus a headline summary; the full result — equation, parameters,
statistics, diagnostics, fit audit and prediction — opens here. The dialog is a single
instance per main window: asking for another fit updates the same window and raises it, so
a pile of stale result windows can never form. It remembers its geometry in QSettings
(``details/geometry``), is resizable and is explicitly non-modal, so the user can keep
comparing rows behind it.

The seven views are a :class:`ResultsTabs`, the same widget the fit path has always filled;
nothing is re-implemented here. The equation is also shown large in JetBrains Mono and
selectable, with Copy in both the tab and the footer; the footer also offers the fit's
methods text for the clipboard, as the Equation tab does, and shows the reason when the
tool that writes it refuses the fit.
"""

from typing import cast

from PySide6.QtCore import QByteArray, QSettings, Qt, Signal
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from curvestudio.discovery import AutoFitResult
from curvestudio.tools import Style
from curvestudio.workflow import FitBundle, FitSpec, format_equation
from curvestudio_desktop.jobs import JobRunner
from curvestudio_desktop.theme import mono_font
from curvestudio_desktop.views.fit_audit_view import CrossValidateCaller
from curvestudio_desktop.views.fit_summary import apply_status_badge
from curvestudio_desktop.views.ranking_table import format_metric, format_precise
from curvestudio_desktop.views.results_tabs import (
    PredictCaller,
    ResultsTabs,
    methods_text_menu,
    sync_methods_button,
)

GEOMETRY_KEY = "details/geometry"
MINIMUM_SIZE = (900, 640)
TITLE = "Fit details"


class DetailsDialog(QDialog):
    """The full result of one fit, reused for every subsequent fit."""

    exportRequested = Signal()
    """The footer's Export… was pressed (the window exports the current bundle)."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: QSettings | None = None,
        predict_fn: PredictCaller | None = None,
        cross_validate_fn: CrossValidateCaller | None = None,
        jobs: JobRunner | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._settings = settings if settings is not None else QSettings()
        self._bundle: FitBundle | None = None
        self._spec: FitSpec | None = None
        self.setObjectName("detailsDialog")
        self.setWindowTitle(TITLE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self.results = ResultsTabs(
            predict_fn=predict_fn, cross_validate_fn=cross_validate_fn, jobs=jobs
        )
        self._build()
        self._restore_geometry()
        self.clear()

    # -- construction ------------------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        outer.addWidget(self._build_header())
        outer.addWidget(self.results, stretch=1)
        # the one place either Copy methods text control reports a refusal; hidden otherwise
        self.refusal_label = QLabel("")
        self.refusal_label.setObjectName("bannerError")
        self.refusal_label.setWordWrap(True)
        self.refusal_label.hide()
        self.results.methodsTextRefused.connect(self._show_refusal)
        outer.addWidget(self.refusal_label)
        outer.addLayout(self._build_footer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("detailsHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(2, 2, 2, 4)
        layout.setSpacing(3)
        row = QHBoxLayout()
        self.model_label = QLabel("")
        self.model_label.setObjectName("detailsModel")
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusBadge")
        row.addWidget(self.model_label, stretch=1)
        row.addWidget(self.status_label)
        layout.addLayout(row)
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("detailsMeta")
        self.meta_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.meta_label)
        self.metrics_label = QLabel("")
        self.metrics_label.setObjectName("detailsMetrics")
        self.metrics_label.setFont(mono_font(10))
        self.metrics_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.metrics_label)
        return header

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.copy_button = QPushButton("Copy equation")
        self.copy_button.setObjectName("detailsCopyButton")
        self.copy_button.clicked.connect(self.copy_equation)
        self.methods_button = QPushButton("Copy methods text")
        self.methods_button.setObjectName("detailsCopyMethodsButton")
        self.methods_button.setMenu(methods_text_menu(self, self._copy_methods_text))
        self.export_button = QPushButton("Export…")
        self.export_button.setObjectName("detailsExportButton")
        self.export_button.clicked.connect(lambda: self.exportRequested.emit())
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("detailsCloseButton")
        self.close_button.clicked.connect(self.close)
        row.addWidget(self.copy_button)
        row.addWidget(self.methods_button)
        row.addStretch(1)
        row.addWidget(self.export_button)
        row.addWidget(self.close_button)
        return row

    # -- filling -----------------------------------------------------------------------------

    def show_bundle(
        self,
        bundle: FitBundle,
        spec: FitSpec | None,
        *,
        dataset_name: str,
        name: str | None = None,
        category: str | None = None,
        meta: object = None,
        ranking: AutoFitResult | None = None,
    ) -> None:
        """Fill every view from ``bundle``; the same window is reused for later fits.

        ``ranking`` is the Auto-fit run the fit was opened from, for the fit audit's
        simpler-model check; ``None`` for any other fit.
        """
        self._bundle = bundle
        self._spec = spec
        result = bundle.result
        stats = result.stats
        if name is None:
            name = result.model_id if not result.model_id.startswith("poly") else "Polynomial"
        if category is None:
            category = ""
        self.model_label.setText(name)
        apply_status_badge(self.status_label, result.status.value)
        meta_text = category or "—"
        self.meta_label.setText(f"{meta_text} · {dataset_name}")
        self.metrics_label.setText(
            "   ".join(
                (
                    f"R² {format_precise(None if stats is None else stats.r2)}",
                    f"adj. R² {format_precise(None if stats is None else stats.adj_r2)}",
                    f"RMSE {format_metric(None if stats is None else stats.rmse)}",
                    f"k {len(result.parameters) if stats is None else stats.k}",
                )
            )
        )
        self.results.show_bundle(bundle, cast("FitSpec", spec), meta, ranking=ranking)
        self.copy_button.setEnabled(bool(format_equation(bundle.equation_text)))
        sync_methods_button(self.methods_button, bundle)
        self.export_button.setEnabled(True)
        self._hide_refusal()  # a refusal belongs to the fit that was shown before

    def clear(self) -> None:
        """No fit to show: empty views, as if the window had never been filled."""
        self._bundle = None
        self._spec = None
        self.model_label.setText("No fit")
        self.status_label.setText("")
        self.meta_label.setText("")
        self.metrics_label.setText("")
        self.results.clear_all()
        self.copy_button.setEnabled(False)
        sync_methods_button(self.methods_button, None)
        self.export_button.setEnabled(False)
        self._hide_refusal()

    # -- actions -----------------------------------------------------------------------------

    def reveal(self) -> None:
        """Show, raise and focus the single shared window."""
        self.show()
        self.raise_()
        self.activateWindow()

    def select_tab(self, title: str) -> bool:
        """Show the results tab called ``title``; ``False`` when there is no such tab."""
        return self.results.select_tab(title)

    def bundle(self) -> FitBundle | None:
        return self._bundle

    def spec(self) -> FitSpec | None:
        return self._spec

    def equation_text(self) -> str:
        return "" if self._bundle is None else format_equation(self._bundle.equation_text)

    def copy_equation(self) -> bool:
        """Copy the shown equation to the clipboard; ``False`` when there is none."""
        text = self.equation_text()
        if not text:
            return False
        QGuiApplication.clipboard().setText(text)
        return True

    def _copy_methods_text(self, style: Style) -> None:
        """The footer's Copy methods text: the tabs hold the fit, so they do the copying."""
        self.results.copy_methods_text(style)

    def _show_refusal(self, message: str) -> None:
        self.refusal_label.setText(message)
        self.refusal_label.show()

    def _hide_refusal(self) -> None:
        self.refusal_label.setText("")
        self.refusal_label.hide()

    def refusal_text(self) -> str:
        """Why the methods tool refused the last copy ('' when it did not)."""
        return self.refusal_label.text()

    def model_text(self) -> str:
        return self.model_label.text()

    def status_text(self) -> str:
        return self.status_label.text()

    def meta_text(self) -> str:
        return self.meta_label.text()

    def metrics_text(self) -> str:
        return self.metrics_label.text()

    # -- geometry persistence ----------------------------------------------------------------

    def _restore_geometry(self) -> None:
        stored = self._settings.value(GEOMETRY_KEY, "")
        if isinstance(stored, str) and stored:
            self.restoreGeometry(QByteArray.fromBase64(bytes(stored, encoding="ascii")))

    def save_geometry(self) -> None:
        """Remember the window's geometry (also called when the main window saves)."""
        data = bytes(self.saveGeometry().toBase64().data()).decode("ascii")
        self._settings.setValue(GEOMETRY_KEY, data)
        self._settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        # closeEvent, never hideEvent: hide() runs after the window manager may have
        # shrunk the window, which would overwrite the real size with the minimum
        self.save_geometry()
        super().closeEvent(event)


__all__ = ["GEOMETRY_KEY", "TITLE", "DetailsDialog"]
