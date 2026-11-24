"""The desktop's one view for a statistical result, offscreen.

The result is a real :class:`ToolResult` built from every block a tool can return: two
estimates, a test that ran and one that could not, two assumption checks with their
verdicts, a warning, a table with a p-value column and a missing number, the readings, the
methods fragment, and 250 excluded rows of which only the first 200 are listed. The tests
read the words through ``texts()`` and the structured pieces through the view's own
accessors, so nothing depends on how a widget chooses to paint.
"""

import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QPushButton, QTableWidget, QToolButton, QWidget

from curvestudio.tools import ExcludedRow, Verdict
from curvestudio.tools.stats import (
    Counts,
    Estimate,
    HypothesisTest,
    Table,
    TableColumn,
    ToolResult,
    assumption,
    p_text,
)
from curvestudio_desktop.views.stats_result_view import EMPTY_TEXT, MISSING, StatsResultView

LEVEL = 0.95
P_VALUE = 0.0324
REASON = "the second group holds too few rows"


def _result() -> ToolResult:
    excluded = tuple(ExcludedRow(index=i, reason=f"missing in y ({i})") for i in range(200))
    return ToolResult(
        tool="descriptives",
        title="Descriptive statistics",
        level=LEVEL,
        estimates=(
            Estimate(label="Mean", value=12.5, se=None, lower=10.0, upper=15.0),
            Estimate(label="SD", value=2.25, se=0.5, lower=None, upper=None),
        ),
        tests=(
            HypothesisTest(
                name="Shapiro-Wilk",
                method="Shapiro-Wilk test of normality, two-sided",
                null_hypothesis="the values are normally distributed",
                statistic=0.97,
                p_value=P_VALUE,
                p_value_text=p_text(P_VALUE),
            ),
            HypothesisTest(
                name="Levene",
                method="Levene's test for equal variances",
                null_hypothesis="the groups have equal variances",
                statistic=None,
                p_value=None,
                p_value_text=p_text(None),
                reason=REASON,
            ),
        ),
        assumptions=(
            assumption(
                "normality",
                "The values are consistent with normality",
                Verdict.PASS,
                "Shapiro-Wilk p = 0.0324",
                statistic="W = 0.97",
                threshold="p >= 0.05 passes",
            ),
            assumption(
                "variance",
                "The groups' spreads differ",
                Verdict.CAUTION,
                "Levene's test could not be computed",
                next_step="Report the spread group by group.",
            ),
        ),
        warnings=("Three values were left out.",),
        interpretation="The mean strength is 12.5 MPa.",
        methods_fragment="A one-sample interval was computed at 95% confidence.",
        counts=Counts(n_offered=250, n_used=50, n_excluded=250),
        excluded=excluded,
        excluded_unlisted=50,
        tables=(
            Table(
                title="Summary by group",
                note="SD is the sample standard deviation.",
                columns=(
                    TableColumn(key="group", label="Group", kind="text"),
                    TableColumn(key="p", label="P value", kind="p_value"),
                    TableColumn(key="mean", label="Mean", kind="number"),
                ),
                rows=(("Plant A", P_VALUE, 12.5), ("Plant B", None, None)),
            ),
        ),
    )


@pytest.fixture()
def view(qtbot) -> StatsResultView:
    widget = StatsResultView()
    qtbot.addWidget(widget)
    widget.show()
    return widget


@pytest.fixture()
def shown(view: StatsResultView) -> StatsResultView:
    view.show_result(_result())
    return view


def test_an_empty_result_says_what_to_do(view: StatsResultView) -> None:
    assert view.texts() == [EMPTY_TEXT]


def test_the_title_the_level_and_the_estimates_are_shown(shown: StatsResultView) -> None:
    texts = shown.texts()
    assert "Descriptive statistics" in texts
    assert "95% confidence" in texts
    assert "Estimates" in texts
    assert "Mean" in texts and "12.5" in texts
    assert "SD" in texts and "2.25" in texts
    # a number the tool could not give is an em dash, never 0
    estimates = shown.findChild(QTableWidget, "statsEstimatesTable")
    assert estimates is not None
    assert estimates.item(0, 2).text() == MISSING  # the mean's standard error
    assert estimates.item(0, 2).text() != "0"


def test_the_tests_table_and_the_reason_under_a_test_that_could_not_run(
    shown: StatsResultView,
) -> None:
    texts = shown.texts()
    assert "Tests" in texts
    assert "Shapiro-Wilk" in texts
    assert p_text(P_VALUE) in texts
    assert "Levene" in texts
    assert REASON in texts
    table = shown.findChild(QTableWidget, "statsTestsTable")
    assert table is not None
    # the reason row is under the test's own row, spanning its columns
    assert table.item(2, 0).text() == REASON
    assert table.columnSpan(2, 0) == table.columnCount()


def test_each_assumption_is_a_card_with_its_verdict_badge(shown: StatsResultView) -> None:
    blocks = shown.assumption_blocks()
    assert [block.check_id for block in blocks] == ["normality", "variance"]
    assert [block.state for block in blocks] == ["ok", "caution"]
    texts = shown.texts()
    assert "The values are consistent with normality" in texts
    assert "Shapiro-Wilk p = 0.0324" in texts
    assert "W = 0.97" in texts
    assert "Pass" in texts and "Caution" in texts
    assert "Report the spread group by group." in texts


def test_a_table_shows_its_title_its_note_and_every_cell(shown: StatsResultView) -> None:
    texts = shown.texts()
    assert "Summary by group" in texts
    assert "SD is the sample standard deviation." in texts
    table = shown.findChild(QTableWidget, "statsBlockTable")
    assert table is not None
    assert [table.horizontalHeaderItem(i).text() for i in range(3)] == [
        "Group",
        "P value",
        "Mean",
    ]
    # a p-value column reads through p_text; a missing number is an em dash
    assert table.item(0, 1).text() == p_text(P_VALUE)
    assert table.item(1, 1).text() == MISSING
    assert table.item(0, 2).text() == "12.5"
    assert table.item(1, 2).text() == MISSING


def test_the_warning_the_interpretation_and_the_methods_fragment_are_shown(
    shown: StatsResultView,
) -> None:
    texts = shown.texts()
    assert "Three values were left out." in texts
    assert "The mean strength is 12.5 MPa." in texts
    assert "A one-sample interval was computed at 95% confidence." in texts


def test_copy_methods_text_puts_the_fragment_on_the_clipboard(shown: StatsResultView) -> None:
    QGuiApplication.clipboard().setText("")
    button = shown.findChild(QPushButton, "statsCopyMethodsButton")
    assert button is not None
    button.click()
    assert QGuiApplication.clipboard().text() == _result().methods_fragment


def test_the_counts_and_excluded_rows_are_folded_until_opened(shown: StatsResultView) -> None:
    body = shown.findChild(QWidget, "statsDetailsBody")
    assert body is not None
    assert shown.details_open() is False
    assert body.isHidden()

    toggle = shown.findChild(QToolButton, "statsDetailsToggle")
    assert toggle is not None
    toggle.click()

    assert shown.details_open() is True
    assert not body.isHidden()
    texts = shown.texts()
    assert "250 offered, 50 used, 250 excluded" in texts
    # the first 200 excluded rows, then how many were not listed
    assert "row 1: missing in y (0)" in texts
    assert "row 200: missing in y (199)" in texts
    assert "and 50 more" in texts
