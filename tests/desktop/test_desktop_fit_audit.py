"""The Fit audit tab, its cross-validation check and the methods text in the details window.

Every fit here is real: the bundles come from ``fit_dataset`` and every expected verdict is
``audit_fit`` on that bundle, so the tests say what the tab shows of the tools' own words.
The cross-validation job is driven at two levels: at the tab, through a caller that keeps
its callbacks until the test delivers an outcome; and through a real window, where the job
runs on the thread pool, is cancelled from the status bar and settles back into the tab.
"""

import threading
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel
from test_desktop_autofit_run import NOISY_QUADRATIC, TINY, wait_stored

from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, AutoFitResult, CandidateMetrics, RankedFit
from curvestudio.tools import (
    CrossValidation,
    ToolError,
    Verdict,
    all_methods_texts,
    audit_fit,
    cross_validate,
)
from curvestudio.workflow import CustomRef, FitBundle, FitSpec, PolynomialRef, fit_dataset
from curvestudio_desktop.theme import LIGHT_TOKENS
from curvestudio_desktop.views import main_window as main_window_module
from curvestudio_desktop.views.details_dialog import DetailsDialog
from curvestudio_desktop.views.fit_audit_view import AuditBlockView, en_in_digits
from curvestudio_desktop.views.fit_summary import apply_status_badge
from curvestudio_desktop.views.results_tabs import ResultsTabs
from curvestudio_desktop.views.styles import stylesheet

X = np.linspace(0.0, 10.0, 40)
NOISE = np.random.default_rng(0).normal(0.0, 0.1, 40)
"""A quadratic on 40 points with the generator's first ordinary draw for its noise."""
QUADRATIC = FitSpec(model=PolynomialRef(degree=2))
UNIDENTIFIABLE = FitSpec(
    model=CustomRef(equation="a*x + b*x", parameters=["a", "b"]), initial={"a": 1.0, "b": 2.0}
)
"""Two parameters along one direction (``a*x + b*x``): the fit can never converge."""
STATE = {
    "pass": "ok",
    "caution": "caution",
    "fail": "failed",
    "not_applicable": "not_applicable",
}
STYLES = ("text", "markdown", "latex")
STYLE_LABELS = ("Text", "Markdown", "LaTeX")


def quadratic(x: np.ndarray) -> np.ndarray:
    return 1.0 + 0.5 * x - 0.03 * x**2


QUADRATIC_CSV = "x,y\n" + "".join(
    f"{x:.10g},{y:.10g}\n" for x, y in zip(X, quadratic(X) + NOISE, strict=True)
)


def fit(x, y, spec: FitSpec, weights=None) -> FitBundle:
    xy = XY(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        None if weights is None else np.asarray(weights, dtype=float),
        "x",
        "y",
        "fp-audit",
    )
    return fit_dataset(xy, spec)


def check_of(audit, check_id: str):
    return next(check for check in audit.checks if check.id == check_id)


@pytest.fixture()
def clean() -> FitBundle:
    return fit(X, quadratic(X) + NOISE, QUADRATIC)


@pytest.fixture()
def unconverged() -> FitBundle:
    return fit(X, quadratic(X) + NOISE, UNIDENTIFIABLE)


@pytest.fixture()
def exact() -> FitBundle:
    """Six points that lie exactly on the quadratic: nothing is left to estimate uncertainty."""
    x = np.linspace(0.0, 10.0, 6)
    return fit(x, quadratic(x), QUADRATIC)


def make_tabs(qtbot, **kwargs) -> ResultsTabs:
    tabs = ResultsTabs(**kwargs)
    qtbot.addWidget(tabs)
    tabs.show()
    return tabs


def block(view, check_id: str) -> AuditBlockView:
    return next(item for item in view.blocks() if item.check_id == check_id)


def _ranking(*rows: RankedFit) -> AutoFitResult:
    return AutoFitResult(
        config=AutoFitConfig(),
        dataset_fingerprint="fp-audit",
        n_active=40,
        ranked=rows,
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


def _row(model_id: str, name: str, parameters: int, delta: float, rank: int) -> RankedFit:
    return RankedFit(
        rank=rank,
        model_id=model_id,
        name=name,
        metrics=CandidateMetrics(k=parameters, n=40, aicc=100.0 + delta, cv_rmse=None),
        degenerate=False,
        delta_aicc=delta,
        redundant_with=None,
    )


class PendingCrossValidation:
    """Stands in for the window's job: keeps the callbacks until the test delivers an outcome."""

    def __init__(self) -> None:
        self.calls: list[tuple[FitBundle, FitSpec, int | Literal["loo"]]] = []
        self.deliver: Callable[[CrossValidation], None] | None = None
        self.refuse: Callable[[BaseException], None] | None = None
        self.settle: Callable[[], None] | None = None

    def __call__(
        self,
        bundle: FitBundle,
        spec: FitSpec,
        folds: int | Literal["loo"],
        *,
        on_result: Callable[[CrossValidation], None],
        on_error: Callable[[BaseException], None],
        on_settled: Callable[[], None],
    ) -> None:
        self.calls.append((bundle, spec, folds))
        self.deliver, self.refuse, self.settle = on_result, on_error, on_settled


def fit_quadratic(window, qtbot, write_file) -> None:
    """Import the 40 points into ``window`` (Manual mode) and run the default quadratic fit."""
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("quad.csv", QUADRATIC_CSV))
    window.analysis_panel.manual_view.run_fit()
    qtbot.waitUntil(lambda: window.details_dialog.bundle() is not None, timeout=30000)


# -- the tab ---------------------------------------------------------------------------------


def test_the_audit_tab_is_fourth_and_shows_its_content_for_any_fit(
    qtbot, clean, unconverged
) -> None:
    tabs = make_tabs(qtbot)
    index = tabs.TAB_TITLES.index("Fit audit")
    assert index == 3  # after Diagnostics, before Equation
    assert tabs.tab_text_at(index).strip() == "No fit yet"
    for bundle, spec in ((clean, QUADRATIC), (unconverged, UNIDENTIFIABLE)):
        tabs.show_bundle(bundle, spec, None)
        # a fit that ran and failed is audited; the tab never falls back to "No fit"
        assert tabs._tabs_of[index].content_shown()
    tabs.clear_all()
    assert tabs.tab_text_at(index).strip() == "No fit yet"
    assert not tabs._tabs_of[index].content_shown()


def test_a_clean_fit_shows_pass_badges_the_summary_and_nothing_to_do(qtbot, clean) -> None:
    tabs = make_tabs(qtbot)
    tabs.show_bundle(clean, QUADRATIC, None)
    view = tabs._audit
    audit = audit_fit(clean, spec=QUADRATIC)
    assert audit.overall is Verdict.PASS  # the sample this test relies on

    assert view.audit() == audit
    assert (view.overall_badge.text(), view.overall_badge.property("state")) == ("Pass", "ok")
    assert view.summary_label.text() == audit.summary
    # Shown, not merely written: a hidden label left the overall badge to fill the row.
    assert not view.summary_label.isHidden()
    blocks = view.blocks()
    assert [item.check_id for item in blocks] == [check.id for check in audit.checks]
    ran = [item for item in blocks if item.state == "ok"]
    assert len(ran) == 9
    for item, check in zip(blocks, audit.checks, strict=True):
        assert (item.title, item.evidence) == (check.title, check.evidence)
        assert item.next_step == (check.next_step or "")
        # the numbers a verdict turned on are shown, each under its own label, when it has them
        assert item.statistic == (f"Statistic: {check.statistic}" if check.statistic else "")
        assert item.threshold == (f"Threshold: {check.threshold}" if check.threshold else "")
        if check.verdict is Verdict.PASS:
            assert item.explanation == check.explanation
    # nothing that passed asks for anything; only the check not yet run says what to do
    assert not any(item.next_step for item in ran)
    assert [item.check_id for item in blocks if item.next_step] == ["cross_validation"]
    assert block(view, "cross_validation").next_step == "Run cross-validation to check this."
    # the two checks with nothing to read yet are quiet: no explanation
    quiet = [item for item in blocks if item.state == "not_applicable"]
    assert [item.check_id for item in quiet] == ["cross_validation", "simpler_model"]
    assert all(item.explanation == "" for item in quiet)


def test_a_fit_that_did_not_converge_shows_a_fail_badge_and_quiet_blocks(
    qtbot, unconverged
) -> None:
    tabs = make_tabs(qtbot)
    tabs.show_bundle(unconverged, UNIDENTIFIABLE, None)
    view = tabs._audit
    reason = unconverged.result.termination_reason
    assert unconverged.result.status.value == "not_converged" and reason

    assert view.audit() == audit_fit(unconverged, spec=UNIDENTIFIABLE)
    assert (view.overall_badge.text(), view.overall_badge.property("state")) == ("Fail", "failed")
    assert view.summary_label.text() == view.audit().summary
    assert view.summary_label.text().startswith(f"The fit did not converge ({reason}),")
    first = view.blocks()[0]
    assert (first.check_id, first.state) == ("converged", "failed")
    assert first.title == "The fit did not converge"
    assert f"Status not_converged: {reason}" == first.evidence
    assert (first.statistic, first.threshold) == ("", "Threshold: status converged")
    assert first.next_step  # the audit's own advice on what to try next
    rest = view.blocks()[1:]
    assert len(rest) == 10
    for item in rest:
        assert item.state == "not_applicable"
        assert item.evidence == "the fit did not converge"
        assert (item.statistic, item.threshold, item.explanation, item.next_step) == (
            "",
            "",
            "",
            "",
        )


def test_a_check_shows_its_statistic_and_its_threshold_when_it_has_them(qtbot, clean) -> None:
    tabs = make_tabs(qtbot)
    tabs.show_bundle(clean, QUADRATIC, None)
    view = tabs._audit
    audit = audit_fit(clean, spec=QUADRATIC)
    both = check_of(audit, "residual_normality")
    assert both.statistic and both.threshold
    # one quiet, labelled line each, under the evidence
    assert block(view, "residual_normality").statistic == f"Statistic: {both.statistic}"
    assert block(view, "residual_normality").threshold == f"Threshold: {both.threshold}"
    # a check with only one of them shows that one only
    converged = check_of(audit, "converged")
    assert converged.statistic is None and converged.threshold
    shown = block(view, "converged")
    assert (shown.statistic, shown.threshold) == ("", f"Threshold: {converged.threshold}")
    # a check with neither shows neither label
    quiet = [item.check_id for item in view.blocks() if item.state == "not_applicable"]
    assert len(quiet) == 2
    assert all(
        (block(view, check_id).statistic, block(view, check_id).threshold) == ("", "")
        for check_id in quiet
    )
    labels = {
        name: [label.text() for label in view.findChildren(QLabel, name)]
        for name in ("auditStatistic", "auditThreshold")
    }
    assert labels["auditStatistic"] == [
        f"Statistic: {c.statistic}" for c in audit.checks if c.statistic
    ]
    assert labels["auditThreshold"] == [
        f"Threshold: {c.threshold}" for c in audit.checks if c.threshold
    ]


def test_the_counts_sit_under_the_summary(qtbot, clean) -> None:
    tabs = make_tabs(qtbot)
    tabs.show_bundle(clean, QUADRATIC, None)
    view = tabs._audit
    assert view.counts_label.text() == "40 observations, 3 parameters"
    assert not view.counts_label.isHidden()
    # nothing was left out of the audit, so no line says so
    assert view.excluded_label.isHidden()
    assert view.excluded_label.text() == ""


@pytest.mark.parametrize(
    ("count", "text"), [(0, "0"), (40, "40"), (999, "999"), (1000, "1,000"), (100000, "1,00,000")]
)
def test_counts_are_written_as_en_in_writes_them(count: int, text: str) -> None:
    assert en_in_digits(count) == text


def _with_blank_weight(bundle: FitBundle, row: int) -> FitBundle:
    """The same bundle with the weight at ``row`` (0-based) blanked out."""
    blanked: list[float | None] = list(bundle.data.weights or ())
    blanked[row] = None
    return bundle.model_copy(update={"data": bundle.data.model_copy(update={"weights": blanked})})


def test_rows_that_took_no_part_are_listed_with_their_reason(qtbot) -> None:
    weights = np.ones(40)
    weights[[4, 30]] = 0.0  # rows 5 and 31
    spec = FitSpec(model=PolynomialRef(degree=2), use_weights=True)
    bundle = fit(X, quadratic(X) + NOISE, spec, weights)
    assert [row.index for row in audit_fit(bundle, spec=spec).excluded] == [4, 30]

    tabs = make_tabs(qtbot)
    tabs.show_bundle(bundle, spec, None)
    view = tabs._audit
    assert view.counts_label.text() == "38 observations, 3 parameters"
    assert view.excluded_label.text() == (
        "2 rows took no part: row 5 (weight is zero), row 31 (weight is zero)"
    )
    assert not view.excluded_label.isHidden()


def test_a_row_with_a_missing_weight_is_named_as_such(qtbot) -> None:
    """A blank weight never survives an import; the bundle here carries one as the tool reads it."""
    weights = np.ones(40)
    weights[4] = 0.0
    spec = FitSpec(model=PolynomialRef(degree=2), use_weights=True)
    bundle = _with_blank_weight(fit(X, quadratic(X) + NOISE, spec, weights), 30)
    excluded = audit_fit(bundle, spec=spec).excluded
    assert [(row.index, row.reason) for row in excluded] == [
        (4, "weight is zero"),
        (30, "weight is missing"),
    ]

    tabs = make_tabs(qtbot)
    tabs.show_bundle(bundle, spec, None)
    assert tabs._audit.excluded_label.text() == (
        "2 rows took no part: row 5 (weight is zero), row 31 (weight is missing)"
    )


def test_a_long_list_of_excluded_rows_names_the_first_eight_and_counts_the_rest(qtbot) -> None:
    weights = np.ones(40)
    weights[1:11] = 0.0  # rows 2 to 11
    spec = FitSpec(model=PolynomialRef(degree=2), use_weights=True)
    tabs = make_tabs(qtbot)
    tabs.show_bundle(fit(X, quadratic(X) + NOISE, spec, weights), spec, None)
    view = tabs._audit
    named = ", ".join(f"row {row} (weight is zero)" for row in range(2, 10))
    assert view.excluded_label.text() == f"10 rows took no part: {named}, and 2 more"
    assert view.counts_label.text() == "30 observations, 3 parameters"


def test_one_excluded_row_reads_in_the_singular(qtbot) -> None:
    weights = np.ones(40)
    weights[39] = 0.0
    spec = FitSpec(model=PolynomialRef(degree=2), use_weights=True)
    tabs = make_tabs(qtbot)
    tabs.show_bundle(fit(X, quadratic(X) + NOISE, spec, weights), spec, None)
    assert tabs._audit.excluded_label.text() == "1 row took no part: row 40 (weight is zero)"


def test_a_caution_and_a_fail_show_their_next_steps(qtbot, exact) -> None:
    tabs = make_tabs(qtbot)
    tabs.show_bundle(exact, QUADRATIC, None)
    view = tabs._audit
    audit = audit_fit(exact, spec=QUADRATIC)
    caution = check_of(audit, "perfect_fit")
    failed = check_of(audit, "points_per_parameter")
    assert (caution.verdict, failed.verdict) == (Verdict.CAUTION, Verdict.FAIL)

    caution_block, failed_block = block(view, "perfect_fit"), block(view, "points_per_parameter")
    assert caution_block.state == "caution"
    assert failed_block.state == "failed"
    assert view.overall_badge.property("state") == "failed"  # the worst of them
    for item, check in ((caution_block, caution), (failed_block, failed)):
        assert item.title == check.title
        assert item.evidence == check.evidence
        assert item.explanation == check.explanation
        assert item.next_step == check.next_step
        assert item.next_step  # what to do about it is part of the block


def test_every_badge_state_has_a_stylesheet_rule_and_the_fit_statuses_keep_theirs(
    qtbot, clean, unconverged, exact
) -> None:
    """Every badge state has its colour; a fit status still maps to the two states it had."""
    qss = stylesheet("light")
    colours = {"ok": "success", "caution": "warning", "failed": "danger", "not_applicable": "muted"}
    for state, token in colours.items():
        rule = f'QLabel#statusBadge[state="{state}"]'
        assert rule in qss
        body = qss.split(rule, 1)[1].split("}", 1)[0]
        assert LIGHT_TOKENS[token] in body  # the rule is rendered, not left as a template

    # each verdict's badge on a real fit carries its state
    seen = set()
    for bundle, spec in ((clean, QUADRATIC), (unconverged, UNIDENTIFIABLE), (exact, QUADRATIC)):
        tabs = make_tabs(qtbot)
        tabs.show_bundle(bundle, spec, None)
        seen.update((item.check_id, item.state) for item in tabs._audit.blocks())
    assert {
        ("converged", "ok"),
        ("perfect_fit", "caution"),
        ("points_per_parameter", "failed"),
        ("residual_normality", "not_applicable"),
    } <= seen

    # the two states that existed are untouched: a fit status still maps to ok or failed
    label = QLabel()
    apply_status_badge(label, "converged")
    assert (label.text(), label.property("state")) == ("converged", "ok")
    apply_status_badge(label, "not_converged")
    assert (label.text(), label.property("state")) == ("not_converged", "failed")


def test_a_simpler_model_that_fits_as_well_is_named_below_the_checks(qtbot, clean) -> None:
    ranking = _ranking(_row("poly2", "Quadratic", 3, 0.0, 1), _row("poly1", "Linear", 2, 1.0, 2))
    tabs = make_tabs(qtbot)
    tabs.show_bundle(clean, QUADRATIC, None, ranking=ranking)
    view = tabs._audit
    audit = audit_fit(clean, spec=QUADRATIC, ranking=ranking)
    assert audit.simpler_model is not None and audit.simpler_model.model_id == "poly1"
    assert view.audit() == audit
    assert block(view, "simpler_model").state == "caution"
    assert view.simpler_label.text() == (
        "Simpler model that fits as well: Linear (poly1), 2 parameters, 1 AICc from the best"
    )
    assert not view.simpler_label.isHidden()

    # with no ranking to compare against there is no such line
    tabs.show_bundle(clean, QUADRATIC, None)
    assert view.simpler_label.isHidden()
    assert view.simpler_label.text() == ""


def test_an_audit_the_tool_refuses_shows_its_message(qtbot) -> None:
    spec = FitSpec(model=PolynomialRef(degree=1))
    empty = fit([], [], spec)
    with pytest.raises(ToolError) as caught:
        audit_fit(empty)
    assert caught.value.code == "NO_DATA"

    tabs = make_tabs(qtbot)
    tabs.show_bundle(empty, spec, None)
    view = tabs._audit
    assert view.audit() is None
    assert view.summary_label.text() == f"{caught.value.code}: {caught.value.message}"
    assert not view.summary_label.isHidden()
    assert view.blocks() == []
    assert view.overall_badge.property("state") == "not_applicable"


# -- the cross-validation check --------------------------------------------------------------


def test_cross_validation_stays_disabled_when_the_audit_itself_was_refused(qtbot) -> None:
    """The bundle with no observations: there is nothing to cross-validate either."""
    spec = FitSpec(model=PolynomialRef(degree=1))
    tabs = make_tabs(qtbot, cross_validate_fn=PendingCrossValidation())
    tabs.show_bundle(fit([], [], spec), spec, None)
    assert tabs._audit.audit() is None
    assert not tabs._audit.cross_validate_button.isEnabled()


@pytest.mark.parametrize(("index", "folds"), [(0, 5), (1, 10), (2, "loo")])
def test_the_button_runs_the_injected_caller_with_the_chosen_folds(
    qtbot, clean, index, folds
) -> None:
    caller = PendingCrossValidation()
    tabs = make_tabs(qtbot, cross_validate_fn=caller)
    tabs.show_bundle(clean, QUADRATIC, None)
    view = tabs._audit
    assert block(view, "cross_validation").state == "not_applicable"
    assert view.cross_validate_button.text() == "Run cross-validation check"

    view.cross_validate_combo.setCurrentIndex(index)
    view.cross_validate_button.click()
    assert caller.calls == [(clean, QUADRATIC, folds)]
    # while it runs: the button says so and cannot start a second job
    assert not view.cross_validate_button.isEnabled()
    assert view.cross_validate_button.text() == "Running cross-validation…"
    view.cross_validate_button.click()
    assert len(caller.calls) == 1
    assert block(view, "cross_validation").state == "not_applicable"

    cv = cross_validate(clean, QUADRATIC, folds=folds)
    caller.deliver(cv)
    caller.settle()
    expected = check_of(audit_fit(clean, spec=QUADRATIC, cv=cv), "cross_validation")
    shown = block(view, "cross_validation")
    assert shown.state != "not_applicable"
    assert shown.state == STATE[expected.verdict.value]
    assert (shown.title, shown.evidence, shown.explanation) == (
        expected.title,
        expected.evidence,
        expected.explanation,
    )
    assert view.audit() == audit_fit(clean, spec=QUADRATIC, cv=cv)
    assert view.summary_label.text() == view.audit().summary
    assert view.cross_validation() is cv
    # decided now: the same choice can be run again, and nothing is reported as wrong
    assert view.cross_validate_button.isEnabled()
    assert view.cross_validate_button.text() == "Run again"
    assert view.error_label.isHidden()


def test_a_refused_cross_validation_shows_its_message_and_leaves_every_block_as_it_was(
    qtbot, clean
) -> None:
    caller = PendingCrossValidation()
    tabs = make_tabs(qtbot, cross_validate_fn=caller)
    tabs.show_bundle(clean, QUADRATIC, None)
    view = tabs._audit
    before = view.blocks()
    summary = view.summary_label.text()

    view.cross_validate_button.click()
    caller.refuse(ToolError("INVALID_FOLDS", "folds must lie between 2 and the number of points"))
    caller.settle()

    assert view.error_label.text() == (
        "INVALID_FOLDS: folds must lie between 2 and the number of points"
    )
    assert not view.error_label.isHidden()
    assert view.blocks() == before
    assert view.summary_label.text() == summary
    assert view.cross_validation() is None
    # still undecided, so it is a first run again, not "Run again"
    assert view.cross_validate_button.isEnabled()
    assert view.cross_validate_button.text() == "Run cross-validation check"


def test_a_cancelled_cross_validation_leaves_the_tab_as_it_was(qtbot, clean) -> None:
    caller = PendingCrossValidation()
    tabs = make_tabs(qtbot, cross_validate_fn=caller)
    tabs.show_bundle(clean, QUADRATIC, None)
    view = tabs._audit
    before = view.blocks()

    view.cross_validate_button.click()
    caller.settle()  # cancelled: the job ends and nothing is delivered

    assert view.blocks() == before
    assert view.cross_validation() is None
    assert view.error_label.isHidden()
    assert view.error_label.text() == ""
    assert view.cross_validate_button.isEnabled()
    assert view.cross_validate_button.text() == "Run cross-validation check"


def test_a_result_that_arrives_after_another_fit_was_shown_is_ignored(
    qtbot, clean, unconverged
) -> None:
    caller = PendingCrossValidation()
    tabs = make_tabs(qtbot, cross_validate_fn=caller)
    tabs.show_bundle(clean, QUADRATIC, None)
    view = tabs._audit
    view.cross_validate_button.click()
    first = block(view, "cross_validation")

    tabs.show_bundle(unconverged, UNIDENTIFIABLE, None)  # another fit while the job runs
    second = block(view, "cross_validation")
    assert first.state != second.state or first.evidence != second.evidence

    caller.deliver(cross_validate(clean, QUADRATIC, folds=5))
    caller.refuse(ToolError("INVALID_FOLDS", "late"))
    caller.settle()
    assert block(view, "cross_validation") == second  # the old job's result is not this fit's
    assert view.error_label.isHidden()
    assert not view.cross_validate_button.isEnabled()  # this fit's own state stands


def test_cross_validation_is_disabled_for_a_fit_that_did_not_converge(qtbot, unconverged) -> None:
    caller = PendingCrossValidation()
    tabs = make_tabs(qtbot, cross_validate_fn=caller)
    tabs.show_bundle(unconverged, UNIDENTIFIABLE, None)
    view = tabs._audit
    button = view.cross_validate_button
    assert not button.isEnabled()
    assert button.toolTip() == (
        "The fit did not converge "
        f"({unconverged.result.termination_reason}), so there is no model to cross-validate"
    )
    button.click()
    assert caller.calls == []
    assert block(view, "cross_validation").state == "not_applicable"


# -- the methods text ------------------------------------------------------------------------


@pytest.mark.parametrize(("index", "style"), list(enumerate(STYLES)))
def test_copy_methods_text_puts_the_paragraph_the_equation_and_the_table_on_the_clipboard(
    qtbot, settings, clean, index, style
) -> None:
    dialog = DetailsDialog(settings=settings)
    qtbot.addWidget(dialog)
    dialog.show_bundle(clean, QUADRATIC, dataset_name="quad")
    text = all_methods_texts(clean, spec=QUADRATIC)[style]
    expected = f"{text.paragraph}\n\n{text.equation}\n\n{text.table}"
    # one control in the footer, beside Copy equation, and one in the Equation tab
    footer, equation = dialog.methods_button, dialog.results._methods_button
    assert footer.text() == equation.text() == "Copy methods text"
    for button in (footer, equation):
        assert button.isEnabled()
        assert [action.text() for action in button.menu().actions()] == list(STYLE_LABELS)
        QGuiApplication.clipboard().clear()
        button.menu().actions()[index].trigger()
        assert QGuiApplication.clipboard().text() == expected


def test_methods_text_is_disabled_for_a_fit_that_did_not_converge(
    qtbot, settings, unconverged
) -> None:
    dialog = DetailsDialog(settings=settings)
    qtbot.addWidget(dialog)
    dialog.show_bundle(unconverged, UNIDENTIFIABLE, dataset_name="quad")
    for button in (dialog.methods_button, dialog.results._methods_button):
        assert not button.isEnabled()
        assert button.toolTip() == "No methods text for a fit that did not converge"


def test_a_refused_methods_text_shows_the_tools_message_and_copies_nothing(
    qtbot, settings, clean
) -> None:
    dialog = DetailsDialog(settings=settings)
    qtbot.addWidget(dialog)
    other = FitSpec(model=PolynomialRef(degree=3))  # a specification that is not the fit's
    dialog.show_bundle(clean, other, dataset_name="quad")
    with pytest.raises(ToolError) as caught:
        all_methods_texts(clean, spec=other)
    assert caught.value.code == "MODEL_MISMATCH"
    assert dialog.refusal_label.isHidden()

    QGuiApplication.clipboard().setText("untouched")
    for button in (dialog.methods_button, dialog.results._methods_button):
        for action in button.menu().actions():
            action.trigger()
    assert QGuiApplication.clipboard().text() == "untouched"
    assert dialog.refusal_label.text() == f"{caught.value.code}: {caught.value.message}"
    assert not dialog.refusal_label.isHidden()

    dialog.show_bundle(clean, QUADRATIC, dataset_name="quad")  # the next fit starts clean
    assert dialog.refusal_label.isHidden()
    assert dialog.refusal_label.text() == ""
    dialog.methods_button.menu().actions()[0].trigger()
    text = all_methods_texts(clean, spec=QUADRATIC)["text"]
    assert (
        QGuiApplication.clipboard().text() == f"{text.paragraph}\n\n{text.equation}\n\n{text.table}"
    )


# -- the window: a restored fit, the real job, the ranking ------------------------------------


def test_a_fit_restored_from_a_saved_project_shows_its_audit(
    make_window, qtbot, controller, write_file
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    fit_quadratic(window, qtbot, write_file)
    shown = window.details_dialog.bundle()
    assert shown is not None

    target = write_file("fit.cstudio", "")
    controller.paths.save_paths.append(target)
    assert controller.save() is True
    controller.paths.open_paths.append(target)
    assert controller.open_project(target) is True
    assert window.details_dialog.results._audit.audit() is None  # nothing is shown after a reopen
    fit_id = next(a.id for a in controller.project.analyses if a.kind == "fit")
    assert window.analysis.restore_analysis(fit_id) is True
    dialog = window.details_dialog
    restored, spec = dialog.bundle(), dialog.spec()
    assert restored is not None and spec is not None
    assert restored.result.parameters == shown.result.parameters

    view = dialog.results._audit
    assert view.audit() == audit_fit(restored, spec=spec)
    assert view.overall_badge.property("state") == "ok"
    assert view.summary_label.text() == view.audit().summary
    assert block(view, "cross_validation").state == "not_applicable"  # a result is never stored
    assert window.analysis.run_count == 0  # nothing re-ran anywhere in this test


def test_the_window_runs_the_cross_validation_as_a_background_job(
    make_window, qtbot, write_file
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    fit_quadratic(window, qtbot, write_file)
    dialog = window.details_dialog
    bundle, spec = dialog.bundle(), dialog.spec()
    assert bundle is not None and spec is not None
    view = dialog.results._audit
    dialog.results._tabs.setCurrentIndex(dialog.results.TAB_TITLES.index("Fit audit"))
    QApplication.processEvents()
    assert view.isVisible()

    view.cross_validate_combo.setCurrentIndex(1)  # 10 folds
    before = view.blocks()
    view.cross_validate_button.click()
    assert window.statusBar().currentMessage() == "Cross-validating… (Cancel to stop)"
    assert not view.cross_validate_button.isEnabled()
    assert not window._cancel_button.isHidden()

    qtbot.waitUntil(lambda: view.cross_validation() is not None, timeout=30000)
    cv = view.cross_validation()
    assert cv is not None and cv.folds == 10 and cv.status == "ok"
    expected = check_of(audit_fit(bundle, spec=spec, cv=cv), "cross_validation")
    shown = block(view, "cross_validation")
    assert shown.state == STATE[expected.verdict.value]
    assert (shown.title, shown.evidence) == (expected.title, expected.evidence)
    assert [item.check_id for item in before] == [item.check_id for item in view.blocks()]
    assert view.cross_validate_button.text() == "Run again"
    assert view.cross_validate_button.isEnabled()
    assert window.statusBar().currentMessage() == "Cross-validation finished"
    qtbot.waitUntil(lambda: window._cancel_button.isHidden(), timeout=10000)


def test_a_cross_validation_cancelled_from_the_status_bar_leaves_the_tab_as_it_was(
    make_window, qtbot, write_file, monkeypatch
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    fit_quadratic(window, qtbot, write_file)
    view = window.details_dialog.results._audit
    started, release = threading.Event(), threading.Event()

    def slow(bundle: FitBundle, spec: FitSpec, folds: int | Literal["loo"]) -> CrossValidation:
        started.set()
        release.wait(timeout=10)
        return cross_validate(bundle, spec, folds=folds)

    monkeypatch.setattr(main_window_module.analysis_jobs, "cross_validate_job", slow)
    before = view.blocks()
    view.cross_validate_button.click()
    qtbot.waitUntil(started.is_set, timeout=10000)
    assert not view.cross_validate_button.isEnabled()
    cancel = window._cancel_button
    assert not cancel.isHidden() and cancel.isEnabled()

    cancel.click()
    assert not cancel.isEnabled()
    assert window.statusBar().currentMessage() == "Cancelling…"
    release.set()
    qtbot.waitUntil(lambda: view.cross_validate_button.isEnabled(), timeout=10000)
    assert view.blocks() == before
    assert view.cross_validation() is None
    assert view.error_label.isHidden()
    assert view.cross_validate_button.text() == "Run cross-validation check"
    qtbot.waitUntil(lambda: cancel.isHidden(), timeout=10000)


def test_an_autofit_rows_audit_is_given_the_run_and_a_manual_fits_is_not(
    make_window, qtbot, write_file
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    dataset_id = window.selection.active_id
    run = wait_stored(qtbot, window, dataset_id)
    assert run is not None

    ranking = window.analysis_panel.autofit_view.table
    assert ranking.select_row(0) is not None
    ranking.openRequested.emit()
    dialog = window.details_dialog
    qtbot.waitUntil(lambda: dialog.bundle() is not None, timeout=60000)
    bundle, spec = dialog.bundle(), dialog.spec()
    assert bundle is not None and spec is not None
    view = dialog.results._audit
    # the run the row was opened from is what the simpler-model check compares against
    assert view.audit() == audit_fit(bundle, spec=spec, ranking=run)
    assert block(view, "simpler_model").state != "not_applicable"

    window.analysis.set_mode("manual")
    fit_quadratic(window, qtbot, write_file)
    assert dialog.bundle() is not None
    # a manual fit is not given a ranking, although the dataset has one
    assert window.details_dialog.results._audit.audit() == audit_fit(
        dialog.bundle(), spec=dialog.spec()
    )
    simpler = block(window.details_dialog.results._audit, "simpler_model")
    assert (simpler.state, simpler.evidence) == (
        "not_applicable",
        "no Auto-fit ranking for this dataset",
    )
