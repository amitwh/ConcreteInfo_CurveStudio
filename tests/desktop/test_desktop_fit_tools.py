"""The More tools tab: the three fit-centric tools, run off the GUI thread.

The fits are real (``fit_dataset``) and the ranking's rows name polynomial candidates, so
the discernibility report refits them exactly as it would in the app. Every run uses
``JobRunner(sync=True)`` except the cancel case, whose runner keeps the submitted job
until the test drains it — that is how a token can be set before the core reads it.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from PySide6.QtCore import Qt
from test_desktop_autofit_run import NOISY_QUADRATIC, TINY, wait_stored

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    CandidateMetrics,
    RankedFit,
)
from curvestudio.workflow import FitBundle, FitSpec, PolynomialRef, fit_dataset
from curvestudio_desktop.jobs import JobRunner, ProgressCallback
from curvestudio_desktop.views.fit_tools_view import MORE_TOOLS_TITLE, FitToolsView
from curvestudio_desktop.views.results_tabs import ResultsTabs

X = np.linspace(0.0, 10.0, 25)
NOISE = np.random.default_rng(0).normal(0.0, 0.05, X.size)
Y = 1.0 + 0.5 * X - 0.03 * X**2 + NOISE
SPEC = FitSpec(model=PolynomialRef(degree=2))
FINGERPRINT = "fp-fit-tools"


def data() -> XY:
    return XY(X, Y, None, "x", "y", FINGERPRINT)


def bundle() -> FitBundle:
    return fit_dataset(data(), SPEC)


def ranking() -> AutoFitResult:
    """Two polynomial rows within 2 AICc: the report's leading group has both."""

    def row(model_id: str, name: str, parameters: int, aicc: float, rank: int) -> RankedFit:
        return RankedFit(
            rank=rank,
            model_id=model_id,
            name=name,
            metrics=CandidateMetrics(k=parameters, n=X.size, aicc=aicc, cv_rmse=None),
        )

    return AutoFitResult(
        config=AutoFitConfig(families=("polynomial",)),
        dataset_fingerprint=FINGERPRINT,
        n_active=X.size,
        ranked=(row("poly2", "Quadratic", 3, 100.0, 1), row("poly1", "Linear", 2, 101.5, 2)),
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


def make_view(qtbot, *, ranked: AutoFitResult | None = None) -> FitToolsView:
    view = FitToolsView(JobRunner(sync=True))
    qtbot.addWidget(view)
    view.set_fit(bundle(), SPEC, ranking=ranked)
    return view


def test_each_action_runs_and_the_dialog_shows_its_title(qtbot) -> None:
    view = make_view(qtbot, ranked=ranking())
    view.feas_runs.setValue(5)
    dialog = view.dialog()

    view.discern_run.click()
    result = dialog.shown_result()
    assert result is not None and result.tool == "discernibility"
    assert result.title == "Model discernibility"
    assert dialog.isVisible() and dialog.current_plot_index() == 0

    view.feas_run.click()
    result = dialog.shown_result()
    assert result is not None and result.tool == "feasibility"
    assert result.title.startswith("Experiment feasibility")
    assert dialog.current_plot_index() == 0

    view.adv_run.click()
    result = dialog.shown_result()
    assert result is not None and result.tool == "advisor"
    assert result.title.startswith("Where to measure next")
    assert dialog.current_plot_index() == 0


def test_the_discernibility_action_needs_the_ranking(qtbot) -> None:
    view = make_view(qtbot)
    assert view.isEnabled()
    assert not view.discern_run.isEnabled()
    assert "Auto-fit" in view.discern_run.toolTip()
    view.set_fit(bundle(), SPEC, ranking=ranking())
    assert view.discern_run.isEnabled()


def test_a_large_simulation_returns_every_run(qtbot) -> None:
    view = make_view(qtbot, ranked=ranking())
    view.feas_runs.setValue(300)
    view.feas_run.click()
    result = view.dialog().shown_result()
    assert result is not None
    runs = {estimate.label: estimate.value for estimate in result.estimates}
    assert runs["Runs"] == 300.0


class _DeferredRunner:
    """Keeps the submitted job; the test starts it after setting the token."""

    def __init__(self) -> None:
        self.pending: (
            tuple[
                Callable[..., Any],
                Callable[[Any], None],
                Callable[[BaseException], None],
                ProgressCallback | None,
                Callable[[], None] | None,
            ]
            | None
        ) = None

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None],
        on_error: Callable[[BaseException], None],
        on_progress: ProgressCallback | None = None,
        on_settled: Callable[[], None] | None = None,
    ) -> None:
        self.pending = (fn, on_result, on_error, on_progress, on_settled)

    def drain(self) -> None:
        assert self.pending is not None
        fn, on_result, on_error, on_progress, on_settled = self.pending
        try:
            result = fn(progress=on_progress)
        except BaseException as exc:  # the runner's own contract, reproduced
            on_error(exc)
        else:
            on_result(result)
        finally:
            if on_settled is not None:
                on_settled()


def test_cancel_returns_the_partial_result_with_the_warning(qtbot) -> None:
    runner = _DeferredRunner()
    view = FitToolsView(cast("JobRunner", runner))
    qtbot.addWidget(view)
    view.set_fit(bundle(), SPEC, ranking=ranking())
    view.feas_runs.setValue(6)
    view.feas_run.click()
    assert view.is_running()
    assert view.cancel_button.isEnabled()
    view.cancel_button.click()
    runner.drain()
    result = view.dialog().shown_result()
    assert result is not None and result.tool == "feasibility"
    assert any("cancelled" in warning for warning in result.warnings)
    assert not view.is_running()
    assert not view.cancel_button.isEnabled()


def test_the_panel_is_disabled_without_a_fit(qtbot) -> None:
    view = FitToolsView(JobRunner(sync=True))
    qtbot.addWidget(view)
    assert not view.isEnabled()
    view.set_fit(bundle(), SPEC, ranking=ranking())
    assert view.isEnabled()
    view.clear()
    assert not view.isEnabled()


def test_results_tabs_has_the_more_tools_tab_and_fills_it(qtbot) -> None:
    tabs = ResultsTabs(jobs=JobRunner(sync=True))
    qtbot.addWidget(tabs)
    assert tabs.tab_titles()[-1] == MORE_TOOLS_TITLE
    assert not tabs.more_tools().isEnabled()
    tabs.show_bundle(bundle(), SPEC, None, ranking=ranking())
    assert tabs.more_tools().isEnabled()
    assert tabs.more_tools().shown_ranking() is not None
    tabs.clear_all()
    assert not tabs.more_tools().isEnabled()


def test_the_autofit_row_opens_the_more_tools_tab(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    wait_stored(qtbot, window, window.selection.active_id)
    view = window.analysis_panel.autofit_view
    view.table.select_row(0)
    qtbot.mouseClick(view.discern_button, Qt.MouseButton.LeftButton)
    dialog = window.details_dialog
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=60000)
    assert dialog.results.current_tab_title() == MORE_TOOLS_TITLE
    assert dialog.results.more_tools().isEnabled()
    assert dialog.results.more_tools().shown_ranking() is not None
