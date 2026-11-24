"""FitAuditView: what a finished fit's evidence shows, check by check — the Fit audit tab.

The audit is ``curvestudio.tools.audit_fit``: a pure, fast reading of the bundle, computed in
place each time a bundle is shown, so the view keeps no numbers of its own and rewords
nothing. At the top sit the overall verdict and the audit's summary, with the counts it
rests on and the rows that took no part. Then comes one block per check, in the audit's
order: a badge, the title, the evidence, the statistic and the threshold a verdict turned
on (each when the check has it), the explanation, and the next step when there is one. A
check that had no number to read (``not_applicable``) is quiet: its title, the evidence
that says why, and a next step when the tool set one. A verdict is a statement of evidence,
never of proof — the tools' wording obeys this, and none of it is reworded here.

The content is a scroll area: eleven blocks outrun the details window.

Cross-validation is the one check that is not a reading: it refits the model once per
fold, so it runs in a background job. The tab cannot reach a window, so its owner injects
the callable that runs the job (as the Predict tab is given ``predict_fn``); the tab calls
it with the bundle, the spec and the chosen folds, and audits again with the result when it
returns. While the job runs the button says so and cannot start another. A refusal is
shown in the tool's words and the tab is left as it was; so is a cancelled job, which
reports nothing but its ending. A result stays with the fit it was run for: it is not
saved with the project, and showing another fit drops it, together with the tab's
interest in any job still running for the fit that was.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from curvestudio.discovery import AutoFitResult
from curvestudio.tools import (
    AuditCheck,
    CrossValidation,
    ExcludedRow,
    FitAudit,
    SimplerModel,
    ToolError,
    Verdict,
    audit_fit,
)
from curvestudio.workflow import FitBundle, FitSpec
from curvestudio_desktop.views.fit_summary import apply_status_badge, apply_verdict_badge

EXCLUDED_LISTED_MAX = 8
"""How many excluded rows the line names; the rest are counted."""
NEXT_STEP_INDENT = 12
"""Pixels the next step sits in from the evidence above it."""
CROSS_VALIDATION = "cross_validation"
"""The id of the one check the tab can decide by running something."""
FOLD_CHOICES: tuple[tuple[str, int | Literal["loo"]], ...] = (
    ("5 folds", 5),
    ("10 folds", 10),
    ("Leave one out", "loo"),
)
RUN_TEXT = "Run cross-validation check"
RUN_AGAIN_TEXT = "Run again"
RUNNING_TEXT = "Running cross-validation…"


class CrossValidateCaller(Protocol):
    """How the tab reaches a cancellable background job for one cross-validation.

    The owner runs ``cross_validate(bundle, spec, folds)`` off the GUI thread and, on the
    GUI thread, calls ``on_result`` or ``on_error`` — never both — and then ``on_settled``:
    always, and on its own after a cancel, whose result and error are dropped.
    """

    def __call__(
        self,
        bundle: FitBundle,
        spec: FitSpec,
        folds: int | Literal["loo"],
        *,
        on_result: Callable[[CrossValidation], None],
        on_error: Callable[[BaseException], None],
        on_settled: Callable[[], None],
    ) -> None: ...


def en_in_digits(count: int) -> str:
    """``count`` as ``en-IN`` groups its digits: the last three alone, then pairs (``1,00,000``)."""
    digits = str(count)
    head, groups = digits[:-3], [digits[-3:]]
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(groups)


def _counted(count: int, noun: str) -> str:
    return f"{en_in_digits(count)} {noun}" if count == 1 else f"{en_in_digits(count)} {noun}s"


def _number(value: float) -> str:
    """A number for a line: three significant digits, as the tools write them."""
    return f"{value:.3g}"


def _counts_text(n_observations: int, n_parameters: int) -> str:
    """``40 observations, 3 parameters``."""
    return f"{_counted(n_observations, 'observation')}, {_counted(n_parameters, 'parameter')}"


def _excluded_text(rows: Sequence[ExcludedRow]) -> str:
    """The rows that took no part, each with its reason; ``''`` when there are none.

    A row is named by its 1-based row in the dataset (the tool keeps 0-based positions).
    """
    if not rows:
        return ""
    named = ", ".join(f"row {row.index + 1} ({row.reason})" for row in rows[:EXCLUDED_LISTED_MAX])
    rest = len(rows) - EXCLUDED_LISTED_MAX
    if rest > 0:
        named += f", and {en_in_digits(rest)} more"
    return f"{_counted(len(rows), 'row')} took no part: {named}"


def _simpler_text(model: SimplerModel) -> str:
    """The simpler model the audit points at, with what says it fits as well."""
    text = (
        f"Simpler model that fits as well: {model.name} ({model.model_id}), "
        f"{_counted(model.n_parameters, 'parameter')}"
    )
    if model.delta_aicc is not None:
        text += f", {_number(model.delta_aicc)} AICc from the best"
    if model.cv_rmse is not None:
        text += f"; cross-validation RMSE {_number(model.cv_rmse)}"
    return text


def _line(name: str, text: str = "") -> QLabel:
    """A wrapping, selectable line of the audit."""
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _text(label: QLabel | None) -> str:
    return "" if label is None else label.text()


def _labelled_line(name: str, label: str, value: str | None) -> QLabel | None:
    """``Label: value`` when the check has the value; no line at all when it does not."""
    return None if value is None else _line(name, f"{label}: {value}")


@dataclass(frozen=True)
class AuditBlockView:
    """What one check's block shows: its badge state and the text of each of its lines.

    A line the block does not have reads ``''``: the statistic or threshold of a check
    without one, the explanation of a quiet check, a next step that was not set.
    """

    check_id: str
    state: str
    title: str
    evidence: str
    statistic: str
    threshold: str
    explanation: str
    next_step: str


@dataclass(frozen=True)
class _Block:
    """The labels of one check's block; a line the check does not have is ``None``."""

    widget: QWidget
    check_id: str
    badge: QLabel
    title: QLabel
    evidence: QLabel
    statistic: QLabel | None
    threshold: QLabel | None
    explanation: QLabel | None
    next_step: QLabel | None

    def view(self) -> AuditBlockView:
        return AuditBlockView(
            check_id=self.check_id,
            state=str(self.badge.property("state")),
            title=self.title.text(),
            evidence=self.evidence.text(),
            statistic=_text(self.statistic),
            threshold=_text(self.threshold),
            explanation=_text(self.explanation),
            next_step=_text(self.next_step),
        )


def _build_block(check: AuditCheck) -> _Block:
    """One check as a block, in the audit's own words."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    head = QHBoxLayout()
    badge = QLabel("")
    badge.setObjectName("statusBadge")
    title = _line("auditTitle", check.title)
    head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
    head.addWidget(title, 1)
    layout.addLayout(head)
    evidence = _line("auditEvidence", check.evidence)
    layout.addWidget(evidence)
    quiet = check.verdict is Verdict.NOT_APPLICABLE
    statistic = _labelled_line("auditStatistic", "Statistic", check.statistic)
    threshold = _labelled_line("auditThreshold", "Threshold", check.threshold)
    explanation = None if quiet else _line("auditExplanation", check.explanation)
    for label in (statistic, threshold, explanation):
        if label is not None:
            layout.addWidget(label)
    next_step = None
    if check.next_step is not None:
        next_step = _line("auditNextStep", check.next_step)
        next_step.setContentsMargins(NEXT_STEP_INDENT, 0, 0, 0)
        layout.addWidget(next_step)
    apply_verdict_badge(badge, check.verdict)
    return _Block(
        widget=widget,
        check_id=check.id,
        badge=badge,
        title=title,
        evidence=evidence,
        statistic=statistic,
        threshold=threshold,
        explanation=explanation,
        next_step=next_step,
    )


class FitAuditView(QWidget):
    """The Fit audit tab's content: an audit laid out as the tools give it."""

    def __init__(
        self, parent: QWidget | None = None, *, cross_validate_fn: CrossValidateCaller | None = None
    ) -> None:
        super().__init__(parent)
        self._cross_validate_fn = cross_validate_fn
        self._bundle: FitBundle | None = None
        self._spec: FitSpec | None = None
        self._ranking: AutoFitResult | None = None
        self._audit: FitAudit | None = None
        self._blocks: list[_Block] = []
        self._cv: CrossValidation | None = None
        self._cv_serial = 0
        """Counts runs and shown fits: a callback carries the number it was started under."""
        self._cv_running = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._build_scroll())
        self._sync_cross_validation_button()

    # -- construction ------------------------------------------------------------------------

    def _build_scroll(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        self._column = QVBoxLayout(body)
        self._column.setContentsMargins(8, 8, 8, 8)
        self._column.setSpacing(4)
        head = QHBoxLayout()
        self.overall_badge = QLabel("")
        self.overall_badge.setObjectName("statusBadge")
        self.summary_label = _line("auditSummary")
        head.addWidget(self.overall_badge, 0, Qt.AlignmentFlag.AlignTop)
        head.addWidget(self.summary_label, 1)
        self._column.addLayout(head)
        self.counts_label = _line("auditCounts")
        self._column.addWidget(self.counts_label)
        self.excluded_label = _line("auditExcluded")
        self._column.addWidget(self.excluded_label)
        self._checks = QVBoxLayout()
        self._checks.setSpacing(12)
        self._column.addLayout(self._checks)
        self.simpler_label = _line("auditSimplerModel")
        self._column.addWidget(self.simpler_label)
        self._column.addLayout(self._build_cross_validation_control())
        self.error_label = QLabel("")
        self.error_label.setObjectName("bannerError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self._column.addWidget(self.error_label)
        self._column.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def _build_cross_validation_control(self) -> QHBoxLayout:
        control = QHBoxLayout()
        self.cross_validate_combo = QComboBox()
        self.cross_validate_combo.setObjectName("auditFolds")
        for label, folds in FOLD_CHOICES:
            self.cross_validate_combo.addItem(label, folds)
        self.cross_validate_button = QPushButton(RUN_TEXT)
        self.cross_validate_button.setObjectName("auditCrossValidateButton")
        self.cross_validate_button.clicked.connect(self._run_cross_validation)
        control.addWidget(self.cross_validate_combo)
        control.addWidget(self.cross_validate_button)
        control.addStretch(1)
        return control

    # -- filling -----------------------------------------------------------------------------

    def show_audit(
        self, bundle: FitBundle, spec: FitSpec | None, ranking: AutoFitResult | None
    ) -> None:
        """Audit ``bundle`` in place: with the spec it ran with and, when there is one, its run.

        A cross-validation result belongs to the fit it was run for: showing another fit
        drops it, and ends the tab's interest in a job still running for the fit that was.
        """
        self._bundle, self._spec, self._ranking = bundle, spec, ranking
        self._reset_cross_validation()
        self._audit_and_lay_out()
        self._sync_cross_validation_button()

    def clear(self) -> None:
        """No fit is shown: nothing of the fit that was stays on the tab."""
        self._bundle = self._spec = self._ranking = None
        self._reset_cross_validation()
        self._lay_out(None)
        self._sync_cross_validation_button()

    def _audit_and_lay_out(self) -> None:
        """Audit the shown fit with whatever cross-validation it has, then lay the audit out."""
        bundle = self._bundle
        if bundle is None:
            return
        try:
            audit = audit_fit(bundle, spec=self._spec, ranking=self._ranking, cv=self._cv)
        except ToolError as exc:  # a bundle with no observations: the tool's own words say so
            self._lay_out(None)
            apply_status_badge(self.overall_badge, "Not applicable", state="not_applicable")
            self._show(self.summary_label, str(exc))
            return
        self._lay_out(audit)

    def _lay_out(self, audit: FitAudit | None) -> None:
        self._audit = audit
        self._clear_blocks()
        self.overall_badge.setText("")
        self._show(self.summary_label, "")
        self._show(self.counts_label, "")
        self._show(self.excluded_label, "")
        self._show(self.simpler_label, "")
        if audit is None:
            return
        apply_verdict_badge(self.overall_badge, audit.overall)
        self._show(self.summary_label, audit.summary)
        self._show(self.counts_label, _counts_text(audit.n_observations, audit.n_parameters))
        self._show(self.excluded_label, _excluded_text(audit.excluded))
        for check in audit.checks:
            block = _build_block(check)
            self._checks.addWidget(block.widget)
            self._blocks.append(block)
        simpler = audit.simpler_model
        self._show(self.simpler_label, "" if simpler is None else _simpler_text(simpler))

    def _clear_blocks(self) -> None:
        while self._checks.count():
            widget = self._checks.takeAt(0).widget()
            if widget is not None:
                widget.hide()  # off the view at once: no layout places it any more
                widget.deleteLater()
        self._blocks.clear()

    @staticmethod
    def _show(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setVisible(bool(text))

    # -- the cross-validation check ----------------------------------------------------------

    def _run_cross_validation(self) -> None:
        """Run the check for the shown fit with the chosen folds, in a job the owner provides."""
        bundle, spec, caller = self._bundle, self._spec, self._cross_validate_fn
        if bundle is None or spec is None or caller is None or self._cv_running:
            return
        folds = self.cross_validate_combo.currentData()
        self._cv_serial += 1
        serial = self._cv_serial
        self._cv_running = True
        self._show(self.error_label, "")
        self._sync_cross_validation_button()
        # In the runner's synchronous mode the callbacks run inside this very call, so
        # nothing below it may touch the button again.
        caller(
            bundle,
            spec,
            folds,
            on_result=lambda result: self._on_cross_validated(serial, result),
            on_error=lambda error: self._on_cross_validation_failed(serial, error),
            on_settled=lambda: self._on_cross_validation_settled(serial),
        )

    def _reset_cross_validation(self) -> None:
        """Forget the result and the running job of the fit that was; the error line with them."""
        self._cv = None
        self._cv_running = False
        self._cv_serial += 1
        self._show(self.error_label, "")

    def _on_cross_validated(self, serial: int, result: CrossValidation) -> None:
        if serial != self._cv_serial:
            return  # a run for the fit that was shown before: not this fit's number
        self._cv = result
        self._audit_and_lay_out()

    def _on_cross_validation_failed(self, serial: int, error: BaseException) -> None:
        if serial != self._cv_serial:
            return
        self._show(self.error_label, str(error))  # a ToolError reads "CODE: message"

    def _on_cross_validation_settled(self, serial: int) -> None:
        if serial != self._cv_serial:
            return
        self._cv_running = False
        self._sync_cross_validation_button()

    def _cross_validation_decided(self) -> bool:
        """Whether the check has a verdict; it reads not applicable until a run completes."""
        audit = self._audit
        if audit is None:
            return False
        check = next((check for check in audit.checks if check.id == CROSS_VALIDATION), None)
        return check is not None and check.verdict is not Verdict.NOT_APPLICABLE

    def _cross_validation_availability(self) -> tuple[bool, str]:
        """Whether the check can be run for the shown fit, and, when not, why (its tool tip)."""
        bundle = self._bundle
        if bundle is None or self._cross_validate_fn is None:
            return False, ""
        if str(bundle.result.status) != "converged":
            reason = bundle.result.termination_reason
            head = f"The fit did not converge ({reason})" if reason else "The fit did not converge"
            return False, f"{head}, so there is no model to cross-validate"
        if self._audit is None:
            return False, "There is nothing in this fit to cross-validate"
        if self._spec is None:
            return False, "The fit has no specification, so it cannot be cross-validated"
        return True, ""

    def _sync_cross_validation_button(self) -> None:
        """The button's text, state and tool tip, from what is shown and what is running."""
        if self._cv_running:
            self.cross_validate_button.setText(RUNNING_TEXT)
            self.cross_validate_button.setEnabled(False)
            self.cross_validate_button.setToolTip("")
            return
        decided = self._cross_validation_decided()
        self.cross_validate_button.setText(RUN_AGAIN_TEXT if decided else RUN_TEXT)
        available, reason = self._cross_validation_availability()
        self.cross_validate_button.setEnabled(available)
        self.cross_validate_button.setToolTip("" if available else reason)

    # -- reading -----------------------------------------------------------------------------

    def audit(self) -> FitAudit | None:
        """The audit shown (``None`` when no fit is shown or the tool refused the bundle)."""
        return self._audit

    def blocks(self) -> list[AuditBlockView]:
        """What each check's block shows, in the audit's order."""
        return [block.view() for block in self._blocks]

    def cross_validation(self) -> CrossValidation | None:
        """The cross-validation run for the shown fit (``None`` until one completes)."""
        return self._cv


__all__ = ["AuditBlockView", "CrossValidateCaller", "FitAuditView", "en_in_digits"]
