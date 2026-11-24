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

The content is a scroll area: eleven blocks outrun the details window. Nothing here reaches
up to a window; whatever the tab needs from outside is injected into it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from curvestudio.discovery import AutoFitResult
from curvestudio.tools import (
    AuditCheck,
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._audit: FitAudit | None = None
        self._blocks: list[_Block] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._build_scroll())

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
        self._column.addStretch(1)
        scroll.setWidget(body)
        return scroll

    # -- filling -----------------------------------------------------------------------------

    def show_audit(
        self, bundle: FitBundle, spec: FitSpec | None, ranking: AutoFitResult | None
    ) -> None:
        """Audit ``bundle`` in place: with the spec it ran with and, when there is one, its run."""
        try:
            audit = audit_fit(bundle, spec=spec, ranking=ranking)
        except ToolError as exc:  # a bundle with no observations: the tool's own words say so
            self._lay_out(None)
            apply_status_badge(self.overall_badge, "Not applicable", state="not_applicable")
            self.summary_label.setText(str(exc))
            return
        self._audit = audit
        self._lay_out(audit)

    def clear(self) -> None:
        """No fit is shown: nothing of the fit that was stays on the tab."""
        self._lay_out(None)

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
        self.summary_label.setText(audit.summary)
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

    # -- reading -----------------------------------------------------------------------------

    def audit(self) -> FitAudit | None:
        """The audit shown (``None`` when no fit is shown or the tool refused the bundle)."""
        return self._audit

    def blocks(self) -> list[AuditBlockView]:
        """What each check's block shows, in the audit's order."""
        return [block.view() for block in self._blocks]


__all__ = ["AuditBlockView", "FitAuditView", "en_in_digits"]
