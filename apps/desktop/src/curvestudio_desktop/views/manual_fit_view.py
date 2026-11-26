"""ManualFitView: the secondary workflow — a fit only when the user asks for one.

A searchable model picker (by id, name, category or equation, grouped by category), the
polynomial degree, and a custom-equation entry with live inline validation: a debounced
dry-run through the workflow layer reports ``INVALID_EQUATION`` (and every other
specification problem) as text under the entry — never a dialog, and Fit stays unusable
while the current draft is known-invalid. Bounds and initial values are prefilled from the
model's own information and reset when the model changes. The weights checkbox exists only
when the data carry weights. Fitting runs through the JobRunner (Cancel drops the result);
a fit that does not converge is a red banner with its reason, never a curve.

The search does not cover model aliases: the workflow catalogue (the only model layer the
desktop may import) does not expose them.
"""

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from curvestudio_desktop.controllers.analysis import AnalysisController

if TYPE_CHECKING:
    from curvestudio.data import XY
    from curvestudio.workflow import (
        CustomRef,
        FitSpec,
        ModelInfo,
        PolynomialRef,
        RegisteredRef,
    )
from curvestudio_desktop.theme import mono_font
from curvestudio_desktop.views.searchable_combo import SearchableComboBox

POLYNOMIAL_ID = "polynomial"
METHODS = ("auto", "lm", "trf", "dogbox", "nelder", "powell", "bfgs", "lbfgsb")
LOSSES = ("linear", "soft_l1", "huber", "cauchy", "arctan")
DEBOUNCE_MS = 400
_FORM_SCROLL_MIN_HEIGHT = 140
"""The form scrolls rather than pinning the dock: a short viewport of the form is
still usable, and the Analysis dock must be able to give its height to the plot."""
_MAX_SEED = 2**31 - 1
_ID_IN_TEXT = re.compile(r"^.*?\(([^()]+)\)")

XyProvider = Callable[[], "XY | None"]


class ManualFitView(QWidget):
    """The manual fit form and its Fit/Cancel controls."""

    def __init__(
        self,
        analysis: AnalysisController,
        xy_provider: XyProvider,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._analysis = analysis
        self._xy_provider = xy_provider
        self._models: list[ModelInfo] = []
        self._validation_token = -1
        self._last_error: str | None = None
        self._fitting = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        # The form is taller than a short dock at 720-820 px: inside a scroll area it
        # scrolls instead of pinning the whole Analysis dock's minimum height (which
        # used to stop the chart/data-table divider moving at all). Fit stays pinned
        # below, so the action is always reachable.
        body = QWidget()
        self.form_body = body
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)
        self._build_form(body_layout)
        self._build_parameters(body_layout)
        body_layout.addStretch(1)
        # a form never scrolls sideways: this floor keeps its rows readable
        body.setMinimumWidth(300)
        self.form_scroll = QScrollArea()
        self.form_scroll.setObjectName("manualFormScroll")
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.form_scroll.setWidget(body)
        self.form_scroll.setMinimumHeight(_FORM_SCROLL_MIN_HEIGHT)
        layout.addWidget(self.form_scroll, stretch=1)
        self._build_actions(layout)
        self._banner = QLabel("")
        self._banner.setObjectName("bannerError")
        self._banner.setWordWrap(True)
        self._banner.hide()
        layout.addWidget(self._banner)
        layout.addStretch(1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._validate_now)
        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._check_cancelled)
        analysis.validation_ready.connect(self._on_validation)
        analysis.fit_started.connect(lambda _dataset_id: self.set_fitting(True))
        analysis.fit_finished.connect(lambda *_args: self.set_fitting(False))
        analysis.fit_failed.connect(lambda _message: self.set_fitting(False))
        from curvestudio_desktop import model_library

        if model_library.is_ready():
            self._fill_models()
        self._on_model_changed()

    # -- construction ------------------------------------------------------------------------

    def _build_form(self, layout: QVBoxLayout) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.model_combo = SearchableComboBox()
        self.model_combo.setObjectName("modelPicker")
        self.model_combo.setMinimumWidth(150)
        self.model_combo.setToolTip(
            "Type to search the catalogue, or open the list to pick a model"
        )
        editor = self.model_combo.lineEdit()
        assert editor is not None  # an editable combo always has an editor
        self.model_edit: QLineEdit = editor
        self.model_combo.currentIndexChanged.connect(lambda _index: self._on_model_changed())
        self.model_edit.textEdited.connect(lambda _text: self._schedule_validation())
        completer = self.model_combo.completer()
        if completer is not None:
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            completer.activated.connect(self._on_completion)
        # a picker, not a text box: an abandoned search falls back to the chosen model
        self.model_edit.editingFinished.connect(self._on_search_finished)
        form.addRow("model", self.model_combo)
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 10)
        self.degree_spin.setValue(2)
        self.degree_spin.valueChanged.connect(lambda _value: self._schedule_validation())
        form.addRow("polynomial degree", self.degree_spin)
        self.equation_edit = QLineEdit()
        self.equation_edit.setPlaceholderText("y = a*exp(-r*x)  (or just the right-hand side)")
        self.equation_edit.textEdited.connect(lambda _text: self._schedule_validation())
        self.parameters_edit = QLineEdit()
        self.parameters_edit.setPlaceholderText("parameters to fit, e.g. a, r")
        # textChanged (not textEdited): the rows follow the parameter list itself,
        # however it changed — typed, pasted or set programmatically
        self.parameters_edit.textChanged.connect(self._custom_parameters_edited)
        self.variable_edit = QLineEdit("x")
        self.variable_edit.setMaximumWidth(80)
        self.variable_edit.textEdited.connect(lambda _text: self._schedule_validation())
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.parameters_edit, stretch=1)
        custom_row.addWidget(QLabel("variable"))
        custom_row.addWidget(self.variable_edit)
        self._equation_box = QWidget()
        equation_form = QFormLayout(self._equation_box)
        equation_form.setContentsMargins(0, 0, 0, 0)
        equation_form.addRow("equation", self.equation_edit)
        equation_form.addRow("parameters", custom_row)
        form.addRow(self._equation_box)
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        self.method_combo.currentIndexChanged.connect(lambda _index: self._schedule_validation())
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(LOSSES)
        self.loss_combo.currentIndexChanged.connect(lambda _index: self._schedule_validation())
        optimisation = QHBoxLayout()
        optimisation.addWidget(self.method_combo)
        optimisation.addWidget(self.loss_combo)
        form.addRow("method / loss", optimisation)
        self.weights_check = QCheckBox("use the weights column")
        self.weights_check.toggled.connect(lambda _checked: self._schedule_validation())
        self.seed_check = QCheckBox("fixed seed")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, _MAX_SEED)
        self.seed_spin.setValue(0)
        self.seed_spin.setMaximumWidth(100)
        self.seed_spin.setEnabled(False)
        self.seed_check.toggled.connect(self.seed_spin.setEnabled)
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.0, 3600.0)
        self.time_spin.setDecimals(1)
        self.time_spin.setSingleStep(5.0)
        self.time_spin.setValue(0.0)
        self.time_spin.setSpecialValueText("no limit")
        self.time_spin.setMaximumWidth(100)
        self.time_spin.valueChanged.connect(lambda _value: self._schedule_validation())
        # stacked rows, not one very wide line: the dock's minimum width must stay
        # small enough for the plot to keep the centre of the window
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed_check)
        seed_row.addWidget(self.seed_spin)
        seed_row.addStretch(1)
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("time limit (s)"))
        time_row.addWidget(self.time_spin)
        time_row.addStretch(1)
        form.addRow("", self.weights_check)
        form.addRow("", seed_row)
        form.addRow("", time_row)
        holder = QWidget()
        holder.setLayout(form)
        layout.addWidget(holder)

    def _build_parameters(self, layout: QVBoxLayout) -> None:
        self._parameters_label = QLabel(
            "parameters — initial values and bounds (blank: the engine's choice)"
        )
        self._parameters_label.setWordWrap(True)
        self.parameters_table = QTableWidget(0, 4)
        self.parameters_table.setObjectName("manualParameters")
        self.parameters_table.setFont(mono_font(9))
        self.parameters_table.setHorizontalHeaderLabels(
            ["parameter", "initial", "lower bound", "upper bound"]
        )
        self.parameters_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.parameters_table.verticalHeader().setVisible(False)
        self.parameters_table.setMaximumHeight(140)
        self.parameters_table.itemChanged.connect(lambda _item: self._schedule_validation())
        layout.addWidget(self._parameters_label)
        layout.addWidget(self.parameters_table)

    def _build_actions(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.fit_button = QPushButton("Fit")
        self.fit_button.setObjectName("manualFitButton")
        self.fit_button.setProperty("role", "primary")
        self.fit_button.clicked.connect(self.run_fit)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("manualCancelButton")
        self.cancel_button.clicked.connect(self._cancel_fit)
        self.cancel_button.hide()
        self.validation_label = QLabel("")
        self.validation_label.setObjectName("equationValidation")
        self.validation_label.setWordWrap(True)
        row.addWidget(self.validation_label, stretch=1)
        row.addWidget(self.cancel_button)
        row.addWidget(self.fit_button)
        layout.addLayout(row)

    # -- model catalogue ---------------------------------------------------------------------

    def _fill_models(self) -> None:
        """Fill the picker. Every item carries its model **id** as data, never a position:
        the category separators occupy item indices, so index arithmetic against the
        catalogue list is wrong by the number of separators above the row."""
        from curvestudio.workflow import list_models

        combo = self.model_combo
        combo.blockSignals(True)
        combo.clear()
        self._models = list(list_models())
        category = None
        for info in self._models:
            if info.category != category:
                category = info.category
                combo.insertSeparator(combo.count())
            combo.addItem(f"{info.name} ({info.id})", info.id)
        # the custom entry sits last, after the catalogue, as it always has
        combo.addItem("Custom equation…", None)
        # the polynomial family is the picker's default: always present, always safe
        default = combo.findData(POLYNOMIAL_ID)
        combo.setCurrentIndex(default if default >= 0 else 0)
        combo.blockSignals(False)
        combo.set_search_entries([f"{m.name} ({m.id}) — {m.category}" for m in self._models])

    def select_model(self, model_id: str) -> bool:
        """Pick the catalogue entry ``model_id`` (tests and the search path)."""
        return self.model_combo.select_by_id(model_id)

    def select_custom_equation(self) -> None:
        """Pick the custom-equation entry (the catalogue's models are not applicable)."""
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def _on_completion(self, text: str) -> None:
        """A search completion: pick the model whose id the string carries."""
        match = _ID_IN_TEXT.match(text)
        if match is not None:
            self.select_model(match.group(1))

    def _on_search_finished(self) -> None:
        """Abandoned text never becomes the value: fall back to the chosen model."""
        editor = self.model_edit
        text = editor.text().strip()
        current = self.model_combo.itemText(self.model_combo.currentIndex())
        if text == current:
            return
        match = _ID_IN_TEXT.match(text)
        if match is not None and self.select_model(match.group(1)):
            return
        self.model_combo.restore_text()
        self.model_edit.setCursorPosition(0)

    def current_model(self) -> ModelInfo | None:
        model_id = self.model_combo.selected_id()
        if model_id is None:
            return None
        return next((info for info in self._models if info.id == model_id), None)

    def model_count(self) -> int:
        """How many catalogue models the picker holds (0 while the library loads)."""
        return len(self._models)

    def fill_models(self) -> None:
        """(Re)fill the picker from the catalogue — the window calls this once the model
        library has loaded, and again never; before that the picker waits empty."""
        self._fill_models()

    def _is_custom(self) -> bool:
        return self.model_combo.selected_id() is None

    def _is_polynomial(self) -> bool:
        model = self.current_model()
        return model is not None and model.is_family and model.id == POLYNOMIAL_ID

    def _on_model_changed(self) -> None:
        """A different model: reset the per-parameter rows from the new model's info."""
        self.parameters_table.setRowCount(0)
        info = self.current_model()
        custom = self._is_custom()
        self._equation_box.setVisible(custom)
        self.degree_spin.setVisible(self._is_polynomial())
        if custom:
            self._sync_custom_parameters()
        elif info is not None and not self._is_polynomial():
            for parameter in info.parameters:
                self._add_parameter_row(
                    parameter.name,
                    "",
                    "" if parameter.lower is None else repr(parameter.lower),
                    "" if parameter.upper is None else repr(parameter.upper),
                )
        # show the selected model from its start, not the scrolled-to-end tail
        self.model_edit.setCursorPosition(0)
        self._schedule_validation()

    def _custom_parameters_edited(self, _text: str) -> None:
        self._sync_custom_parameters()
        self._schedule_validation()

    def _custom_parameter_names(self) -> list[str]:
        return [part.strip() for part in self.parameters_edit.text().split(",") if part.strip()]

    def _sync_custom_parameters(self) -> None:
        """One editable initial/bounds row per typed custom parameter (blank: default).

        Values already entered follow their parameter's name, so reordering or
        extending the list never loses them.
        """
        if not self._is_custom():
            return
        previous = {
            self._cell(row, 0): (self._cell(row, 1), self._cell(row, 2), self._cell(row, 3))
            for row in range(self.parameters_table.rowCount())
        }
        self.parameters_table.setRowCount(0)
        for name in self._custom_parameter_names():
            initial, lower, upper = previous.get(name, ("", "", ""))
            self._add_parameter_row(name, initial, lower, upper)

    def _add_parameter_row(self, name: str, initial: str, lower: str, upper: str) -> None:
        row = self.parameters_table.rowCount()
        self.parameters_table.insertRow(row)
        for column, text in enumerate((name, initial, lower, upper)):
            item = QTableWidgetItem(text)
            if column == 0:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.parameters_table.setItem(row, column, item)

    def refresh_data_options(self) -> None:
        """Weights exist only when the active data carry a weights column."""
        xy = self._xy_provider()
        self.weights_check.setEnabled(xy is not None and xy.weights is not None)
        if not self.weights_check.isEnabled():
            self.weights_check.setChecked(False)
        self._refresh_fit_enabled()

    # -- spec building -----------------------------------------------------------------------

    def current_spec(self) -> tuple[FitSpec | None, str]:
        """The specification the form describes, or ``(None, why it cannot be built)``."""
        from curvestudio.workflow import CustomRef, FitSpec, PolynomialRef, RegisteredRef

        model: CustomRef | PolynomialRef | RegisteredRef
        if self._is_custom():
            equation = self.equation_edit.text().strip()
            if not equation:
                return None, "type an equation to fit"
            names = [part.strip() for part in self.parameters_edit.text().split(",")]
            names = [name for name in names if name]
            if not names:
                return None, "list the parameters to fit, e.g. a, r"
            model = CustomRef(
                equation=equation,
                parameters=names,
                variable=self.variable_edit.text().strip() or "x",
            )
        elif self._is_polynomial():
            model = PolynomialRef(degree=self.degree_spin.value())
        else:
            info = self.current_model()
            if info is None:
                return None, "choose a model"
            model = RegisteredRef(id=info.id)
        bounds: dict[str, tuple[float | None, float | None]] = {}
        initial: dict[str, float] = {}
        if not self._is_polynomial():
            for row in range(self.parameters_table.rowCount()):
                name = self._cell(row, 0)
                if not name:
                    continue
                parsed = [
                    self._number(row, column, label)
                    for column, label in (
                        (1, "initial"),
                        (2, "lower bound"),
                        (3, "upper bound"),
                    )
                ]
                for _parsed_value, problem in parsed:
                    if problem:
                        return None, f"parameter {name}: {problem}"
                start, low, high = (value for value, _problem in parsed)
                if low is not None or high is not None:
                    bounds[name] = (low, high)
                if start is not None:
                    initial[name] = start
        return (
            FitSpec(
                model=model,
                method=self.method_combo.currentText(),
                loss=self.loss_combo.currentText(),
                bounds=bounds,
                initial=initial,
                use_weights=self.weights_check.isChecked(),
                seed=self.seed_spin.value() if self.seed_check.isChecked() else None,
                time_limit=self.time_spin.value() or None,
            ),
            "",
        )

    def _cell(self, row: int, column: int) -> str:
        item = self.parameters_table.item(row, column)
        return "" if item is None else item.text().strip()

    def _number(self, row: int, column: int, label: str) -> tuple[float | None, str]:
        text = self._cell(row, column)
        if not text:
            return None, ""
        try:
            return float(text), ""
        except ValueError:
            return None, f"the {label} is not a number: {text!r}"

    # -- validation --------------------------------------------------------------------------

    def _schedule_validation(self) -> None:
        self._timer.start()

    def _validate_now(self) -> None:
        """Debounced dry-run of the current draft through the workflow layer."""
        spec, problem = self.current_spec()
        xy = self._xy_provider()
        self._refresh_fit_enabled()
        if spec is None:
            self._set_validation(problem or "nothing to fit yet", error=False)
            return
        self._validation_token = self._analysis.validate_spec(
            spec, xy is not None and xy.weights is not None
        )

    def _on_validation(self, token: int, error: object) -> None:
        from curvestudio.workflow import WorkflowError

        if token != self._validation_token:
            return  # a newer draft was submitted meanwhile
        if isinstance(error, WorkflowError):
            self._last_error = f"{error.code}: {error}"
            self._set_validation(self._last_error, error=True)
        else:
            self._last_error = None
            self._set_validation("specification runs — press Fit", error=False)
        self._refresh_fit_enabled()

    def _set_validation(self, text: str, *, error: bool) -> None:
        self.validation_label.setText(text)
        # dynamic property, styled by the QSS ``#validationLabel[state=...]`` rules
        self.validation_label.setProperty("state", "error" if error else "ok")
        style = self.validation_label.style()
        style.unpolish(self.validation_label)
        style.polish(self.validation_label)

    def validation_text(self) -> str:
        return self.validation_label.text()

    # -- fitting -----------------------------------------------------------------------------

    def run_fit(self) -> None:
        """Submit the current specification through the JobRunner (no job when invalid)."""
        if self._fitting or self._last_error is not None:
            return
        spec, problem = self.current_spec()
        xy = self._xy_provider()
        dataset = self._analysis.selection_active_id()
        if spec is None or xy is None or dataset is None:
            self._set_validation(problem or "no data to fit", error=bool(problem))
            return
        self._banner.hide()
        self._analysis.submit_fit(dataset, xy, spec, source={"kind": "manual"})

    def _cancel_fit(self) -> None:
        """Drop the running fit's result (cooperative: the computation ends on its own)."""
        self._analysis.cancel_fit()
        self._poll.start()

    def _check_cancelled(self) -> None:
        if not self._analysis.is_fit_running():
            self._poll.stop()
            self.set_fitting(False)
            self._set_validation("fit cancelled — no result was taken", error=False)

    def set_fitting(self, fitting: bool) -> None:
        self._fitting = fitting
        self.cancel_button.setVisible(fitting)
        self._refresh_fit_enabled()

    def _refresh_fit_enabled(self) -> None:
        xy = self._xy_provider()
        spec_ready = self.current_spec()[0] is not None
        self.fit_button.setEnabled(
            not self._fitting and self._last_error is None and spec_ready and xy is not None
        )

    def show_failure(self, message: str) -> None:
        """A fit job failed outright (a non-converged fit is a result, not a failure)."""
        self._banner.setText(message)
        self._banner.show()

    def clear_banner(self) -> None:
        self._banner.hide()
        self._banner.setText("")
