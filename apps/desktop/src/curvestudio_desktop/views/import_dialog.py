"""Import dialogs: file picker with column pickers and preview, and clipboard paste.

The source is *parsed once* — in the background through a
:class:`~curvestudio_desktop.jobs.JobRunner` — and every later column-combo change
remaps :meth:`~curvestudio.data.Dataset.xy` from the cached dataset without
touching the file again (a 1M-row CSV is never re-read because the user retargeted
a combo box). What the dialog previews is exactly what lands in the project;
validation issues are listed with row numbers (displayed 1-based, like the table's
row gutter); any error-level issue disables OK, warnings do not.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from curvestudio.data import DataError, Dataset, validate
from curvestudio.workflow import ImportIssue, ImportOutcome, import_dataset
from curvestudio_desktop.jobs import JobHandle, JobRunner

PREVIEW_ROWS = 50
_FILE_FILTER = "Data files (*.csv *.tsv *.txt *.json *.xlsx);;All files (*)"
_NO_WEIGHT = "(no weights)"
_CLIPBOARD_BASENAME = "clipboard"


def remap_outcome(
    dataset: Dataset, x: str | None, y: str | None, weight: str | None
) -> ImportOutcome:
    """Rebuild an :class:`ImportOutcome` for new columns over an already-parsed dataset.

    The same semantics as the workflow's extract step — validation issues instead
    of exceptions, ``xy`` None while any error stands — using the public data API,
    so switching columns costs one in-memory pass, never a re-parse.
    """
    columns = tuple(str(name) for name in dataset.table().columns)
    n_rows = len(dataset.table())
    issues: list[ImportIssue] = []
    xy = None
    if x is None or y is None:
        issues.append(
            ImportIssue(
                code="COLUMNS_REQUIRED",
                severity="error",
                message="choose both the x and the y columns",
            )
        )
    else:
        try:
            found = validate(dataset, x, y, weight)
        except DataError as exc:  # an unknown column name
            issues.append(ImportIssue(code=exc.code, severity="error", message=str(exc.args[0])))
        else:
            issues.extend(
                ImportIssue(
                    code=issue.code, severity=issue.severity, message=issue.message, rows=issue.rows
                )
                for issue in found
            )
            if not any(issue.severity == "error" for issue in issues):
                try:
                    xy = dataset.xy(x, y, weight)
                except DataError as exc:
                    issues.append(
                        ImportIssue(
                            code=exc.code,
                            severity="error",
                            message=str(exc.args[0]),
                            rows=exc.rows,
                        )
                    )
    return ImportOutcome(
        dataset=dataset,
        xy=xy,
        issues=issues,
        columns=columns,
        n_rows=n_rows,
        x_column=x,
        y_column=y,
        weight_column=weight,
    )


class _ImportForm(QDialog):
    """Shared machinery: parse the source once, remap columns, gate the OK button."""

    def __init__(self, parent: QWidget | None = None, jobs: JobRunner | None = None) -> None:
        super().__init__(parent)
        self._jobs = jobs if jobs is not None else JobRunner(sync=True)
        self._outcome: ImportOutcome | None = None
        self._failure: str | None = None
        self._columns: tuple[str, ...] = ()
        self._parsed_source: str | None = None
        self._parsed_dataset: Dataset | None = None
        self._parse_job: JobHandle | None = None
        self._parsing = False
        self._x_box = QComboBox()
        self._y_box = QComboBox()
        self._weight_box = QComboBox()
        for box in (self._x_box, self._y_box):
            box.currentTextChanged.connect(lambda _text: self.rerun())
        self._weight_box.currentTextChanged.connect(lambda _text: self.rerun())
        self._issues = QListWidget()
        self._preview = QTableWidget()
        self._preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._refresh_ok()  # nothing imported yet: OK stays off until a clean import

    # -- to implement -----------------------------------------------------------------------

    def _current_source(self) -> tuple[str | None, str | None, Path | bytes | None]:
        """(cache key, filename, source) for the current selection — or Nones."""
        raise NotImplementedError

    def _no_source_message(self) -> str:
        raise NotImplementedError

    def rerun(self) -> None:
        """Show the current selection: remap the cached parse, or parse a new source."""
        key, filename, source = self._current_source()
        if key is None or filename is None or source is None:
            self._set_failure(self._no_source_message())
            return
        if self._parsed_source == key and self._parsed_dataset is not None:
            self._apply_outcome(remap_outcome(self._parsed_dataset, *self._chosen_columns()))
            return
        self._start_parse(source, filename, key)

    # -- parsing -----------------------------------------------------------------------------

    def _start_parse(self, source: Path | bytes, filename: str, key: str) -> None:
        if self._parse_job is not None and not self._parse_job.done:
            self._parse_job.cancel()  # superseded: its callbacks are dropped
        self._parsing = True
        self._issues.clear()
        self._issues.addItem("Reading the data…")
        self._refresh_ok()

        def parsed(outcome: ImportOutcome, k: str = key) -> None:
            self._parsing = False
            self._parsed_dataset = outcome.dataset
            self._parsed_source = k
            if self._columns != outcome.columns:
                self._columns = outcome.columns
                self._refill_boxes(outcome)
            self._apply_outcome(remap_outcome(outcome.dataset, *self._chosen_columns()))

        def failed(exc: BaseException) -> None:
            self._parsing = False
            self._parsed_source = None
            self._parsed_dataset = None
            self._set_failure(str(exc))

        self._parse_job = self._jobs.submit(
            import_dataset,
            source,
            filename,
            on_result=parsed,
            on_error=failed,
        )

    def reject(self) -> None:
        if self._parse_job is not None and not self._parse_job.done:
            self._parse_job.cancel()  # the dialog is going away: drop the callbacks
        super().reject()

    # -- shared behaviour -------------------------------------------------------------------

    def _chosen_columns(self) -> tuple[str | None, str | None, str | None]:
        """The current combo selection (the boxes are filled by the first parse)."""
        return self._chosen("x"), self._chosen("y"), self._chosen("weight")

    def _apply_outcome(self, outcome: ImportOutcome) -> None:
        self._outcome = outcome
        self._failure = None
        self._show_issues(outcome)
        self._show_preview(outcome)

    def _set_failure(self, message: str) -> None:
        self._outcome = None
        self._failure = message
        self._issues.clear()
        self._issues.addItem(message)
        self._preview.setRowCount(0)
        self._preview.setColumnCount(0)
        self._refresh_ok()

    def _chosen(self, which: str) -> str | None:
        box = {"x": self._x_box, "y": self._y_box, "weight": self._weight_box}[which]
        text = box.currentText()
        if not text or text == _NO_WEIGHT:
            return None
        return text

    def _refill_boxes(self, defaults: ImportOutcome | None = None) -> None:
        for box in (self._x_box, self._y_box, self._weight_box):
            box.blockSignals(True)
            box.clear()
        self._weight_box.addItem(_NO_WEIGHT)
        for column in self._columns:
            self._x_box.addItem(column)
            self._y_box.addItem(column)
            self._weight_box.addItem(column)
        if defaults is not None and defaults.x_column is not None:
            self._x_box.setCurrentText(defaults.x_column)
            self._y_box.setCurrentText(defaults.y_column or defaults.x_column)
        for box in (self._x_box, self._y_box, self._weight_box):
            box.blockSignals(False)

    def _show_issues(self, outcome: ImportOutcome) -> None:
        self._issues.clear()
        for issue in outcome.issues:
            rows = ""
            if issue.rows:
                shown = ", ".join(str(row + 1) for row in issue.rows[:8])
                more = "" if len(issue.rows) <= 8 else ", …"
                rows = f" (row{'s' if len(issue.rows) > 1 else ''} {shown}{more})"
            self._issues.addItem(f"{issue.severity.upper()}: {issue.code}: {issue.message}{rows}")
        self._refresh_ok()

    def _show_preview(self, outcome: ImportOutcome) -> None:
        table = outcome.dataset.table()
        self._preview.setRowCount(min(PREVIEW_ROWS, len(table)))
        self._preview.setColumnCount(len(table.columns))
        self._preview.setHorizontalHeaderLabels([str(name) for name in table.columns])
        self._preview.setVerticalHeaderLabels([str(i + 1) for i in range(self._preview.rowCount())])
        for row in range(self._preview.rowCount()):
            for column in range(self._preview.columnCount()):
                self._preview.setItem(
                    row, column, QTableWidgetItem(_cell_text(table.iat[row, column]))
                )

    def _refresh_ok(self) -> None:
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(
                not self._parsing and self._outcome is not None and self._outcome.xy is not None
            )

    # -- test/commit accessors ---------------------------------------------------------------

    def outcome(self) -> ImportOutcome | None:
        return self._outcome

    def issue_texts(self) -> list[str]:
        return [self._issues.item(row).text() for row in range(self._issues.count())]

    def preview_row_count(self) -> int:
        return self._preview.rowCount()

    def ok_enabled(self) -> bool:
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        return bool(ok is not None and ok.isEnabled())


def _cell_text(value: object) -> str:
    """A preview cell: blank for missing markers, readable text otherwise."""
    text = "" if value is None else str(value)
    return "" if text in {"nan", "None", "<NA>", "NaT"} else text


class ImportDialog(_ImportForm):
    """File import: picker for CSV/TSV/TXT/XLSX/JSON, column pickers, preview, issues."""

    def __init__(
        self,
        path: Path | None = None,
        parent: QWidget | None = None,
        jobs: JobRunner | None = None,
    ) -> None:
        super().__init__(parent, jobs=jobs)
        self.setWindowTitle("Import data")
        self.setMinimumSize(760, 520)
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(QLabel("File:"))
        row.addWidget(self._path_edit, 1)
        row.addWidget(browse)
        pickers = QGridLayout()
        pickers.addWidget(QLabel("x column:"), 0, 0)
        pickers.addWidget(self._x_box, 0, 1)
        pickers.addWidget(QLabel("y column:"), 0, 2)
        pickers.addWidget(self._y_box, 0, 3)
        pickers.addWidget(QLabel("weights:"), 1, 0)
        pickers.addWidget(self._weight_box, 1, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addLayout(pickers)
        layout.addWidget(QLabel("Issues (rows are 1-based, excluding the header):"))
        layout.addWidget(self._issues, 1)
        layout.addWidget(QLabel(f"Preview (first {PREVIEW_ROWS} rows):"))
        layout.addWidget(self._preview, 2)
        layout.addWidget(self._buttons)
        if path is not None:
            self._set_path(path)

    # -- _ImportForm hooks ------------------------------------------------------------------

    def _current_source(self) -> tuple[str | None, str | None, Path | bytes | None]:
        text = self._path_edit.text().strip()
        if not text:
            return (None, None, None)
        return (text, Path(text).name, Path(text))

    def _no_source_message(self) -> str:
        return "Choose a file to import."

    # -- extras -----------------------------------------------------------------------------

    def _browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a data file", "", _FILE_FILTER)
        if chosen:
            self._set_path(Path(chosen))

    def _set_path(self, path: Path) -> None:
        self._path_edit.setText(str(path))
        self.rerun()

    def column_choices(self) -> list[str]:
        return [self._x_box.itemText(i) for i in range(self._x_box.count())]

    def selected_columns(self) -> tuple[str | None, str | None, str | None]:
        return (self._chosen("x"), self._chosen("y"), self._chosen("weight"))

    def select_columns(self, *, x: str | None, y: str | None, weight: str | None) -> None:
        self._x_box.setCurrentText(x or "")
        self._y_box.setCurrentText(y or "")
        self._weight_box.setCurrentText(weight or _NO_WEIGHT)
        self.rerun()

    def total_rows(self) -> int:
        return self._outcome.n_rows if self._outcome is not None else 0


class ClipboardImportDialog(_ImportForm):
    """Clipboard import: pasted tab- or comma-separated text via ``import_dataset``."""

    def __init__(
        self,
        initial_text: str = "",
        parent: QWidget | None = None,
        jobs: JobRunner | None = None,
    ) -> None:
        super().__init__(parent, jobs=jobs)
        self.setWindowTitle("Import from clipboard")
        self.setMinimumSize(640, 520)
        self._text = QPlainTextEdit()
        self._text.setPlaceholderText("Paste tab- or comma-separated data (with a header row)")
        self._text.setPlainText(initial_text)
        refresh = QPushButton("Refresh preview")
        refresh.clicked.connect(self.rerun)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Pasted data:"))
        layout.addWidget(self._text, 1)
        layout.addWidget(refresh)
        layout.addWidget(QLabel("Issues (rows are 1-based, excluding the header):"))
        layout.addWidget(self._issues)
        layout.addWidget(QLabel(f"Preview (first {PREVIEW_ROWS} rows):"))
        layout.addWidget(self._preview, 2)
        layout.addWidget(self._buttons)
        self.rerun()

    @classmethod
    def from_clipboard(
        cls, parent: QWidget | None = None, jobs: JobRunner | None = None
    ) -> ClipboardImportDialog:
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        text = clipboard.text() if clipboard is not None else ""
        return cls(initial_text=text, parent=parent, jobs=jobs)

    # -- _ImportForm hooks ------------------------------------------------------------------

    def _current_source(self) -> tuple[str | None, str | None, Path | bytes | None]:
        text = self._text.toPlainText()
        if not text.strip():
            return (None, None, None)
        # Tab-separated paste goes in as .tsv: the workflow's csv sniffer looks at
        # comma/semicolon only, so a .csv name would collapse it to one column.
        filename = (
            f"{_CLIPBOARD_BASENAME}.tsv"
            if text.count("\t") > text.count(",")
            else f"{_CLIPBOARD_BASENAME}.csv"
        )
        return (text, filename, text.encode("utf-8"))

    def _no_source_message(self) -> str:
        return "Paste some data first."
