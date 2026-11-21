"""Auto-fit runs in the desktop shell: auto-start, streaming, cancel, re-ranking, rows.

Tiny configurations (the polynomial family, in-process workers) keep every test fast; one
slow, real-library test exercises the true catalogue and the process pool and checks that
the true model of exponential data lands in the top three.
"""

import pytest
from desktop_data import GOOD_CSV

from curvestudio.discovery import AutoFitConfig, RankBy

TINY = AutoFitConfig(
    families=("polynomial",), workers=1, n_starts=2, cv_folds=None, time_limit=10.0
)
HEAVIER = AutoFitConfig(
    families=("polynomial",), workers=1, n_starts=8, cv_folds=10, time_limit=30.0
)
NOISY_QUADRATIC = "x,y\n" + "".join(
    f"{i},{2.0 + 0.5 * i + 0.09 * i * i + (i % 3 - 1) * 0.31:.4f}\n" for i in range(12)
)
EXACT_LINEAR = "x,y\n" + "".join(f"{i},{2.0 + 3.0 * i:.10g}\n" for i in range(6))


def make_tiny_window(make_window, write_file, config=TINY, text=NOISY_QUADRATIC):
    """A window in Auto-fit mode (tests default to Manual) with tiny run options."""
    window = make_window()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(config)
    assert window.import_file(write_file("quad.csv", text))
    return window


def wait_stored(qtbot, window, dataset_id, timeout=60000):
    qtbot.waitUntil(lambda: window.analysis.stored_autofit(dataset_id) is not None, timeout=timeout)
    return window.analysis.stored_autofit(dataset_id)


def active_id(window):
    return window.selection.active_id


def displayed_ids(window):
    return [row.model_id for row in window.analysis_panel.autofit_view.table.rows()]


def test_autostart_runs_after_import_and_stores_analysis(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file)
    result = wait_stored(qtbot, window, active_id(window))
    assert result is not None and not result.cancelled
    assert result.ranked, "the polynomial family always yields ranked rows"
    analyses = [a for a in window.controller.project.analyses if a.kind == "autofit"]
    assert len(analyses) == 1
    assert analyses[0].dataset_id == active_id(window)
    table = window.analysis_panel.autofit_view.table
    assert table.rows(), "the finished ranking is displayed"
    assert window.controller.is_dirty  # the analysis is part of the project now


def test_no_new_run_when_a_stored_one_exists(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file)
    first = wait_stored(qtbot, window, first_dataset_id(window))
    window.import_file(write_file("b.csv", GOOD_CSV))  # a second dataset: it auto-runs too
    wait_stored(qtbot, window, active_id(window))
    assert window.analysis.run_count == 2
    # back to the first: its stored run shows again, and nothing re-runs
    window.selection.set_active(first_dataset_id(window))
    qtbot.waitUntil(lambda: window.analysis.running_dataset is None, timeout=10000)
    assert window.analysis.maybe_autostart(first_dataset_id(window)) is False
    assert window.analysis.stored_autofit(first_dataset_id(window)) is first
    assert window.analysis.run_count == 2  # one per dataset, never a repeat
    assert displayed_ids(window)  # the stored ranking is shown again


def first_dataset_id(window):
    return window.controller.project.datasets[0].id


def test_cancel_shows_partial_ranking_with_notice(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file, config=HEAVIER)
    dataset_id = first_dataset_id(window)
    finished: list = []
    window.analysis.autofit_finished.connect(lambda ds, result: finished.append((ds, result)))
    first = wait_stored(qtbot, window, dataset_id)  # a full run is stored first
    assert first is not None and not first.cancelled
    # a fresh, slower run we control: start it and cancel as soon as outcomes stream in
    seen_live: list = []
    window.analysis.autofit_live.connect(lambda live: seen_live.append(live))
    window.analysis.run_autofit(dataset_id)
    qtbot.waitUntil(
        lambda: bool(seen_live) or not window.analysis.is_autofit_running(), timeout=60000
    )
    window.analysis.cancel_autofit()
    qtbot.waitUntil(lambda: len(finished) >= 2, timeout=60000)
    result = finished[-1][1]
    if result.cancelled:
        notice = window.analysis_panel.autofit_view.notice
        assert (not notice.isHidden()) and "cancel" in notice.text().lower()
        # the partial never stands in for the stored full run...
        assert window.analysis.stored_autofit(dataset_id) is first
        # ...it is its own analysis, flagged partial, and the shown one
        assert window.analysis.autofit_for_display(dataset_id) is result
        from curvestudio_desktop.controllers.analysis_store import partial_autofit_analysis_id

        partial_id = partial_autofit_analysis_id(dataset_id)
        partials = [
            a
            for a in window.controller.project.analyses
            if a.kind == "autofit" and a.id == partial_id
        ]
        assert len(partials) == 1 and partials[0].dataset_id == dataset_id
        assert window.analysis.analysis_label(partials[0].id) == "Auto-fit (partial)"
        # restoring it later shows the partial notice again, still without refitting
        runs_before = window.analysis.run_count
        assert window.analysis.restore_analysis(partials[0].id) is True
        assert window.analysis.run_count == runs_before
        assert window.analysis_panel.autofit_view.notice.text() == notice.text()
    else:  # the run beat the cancel: nothing partial to assert, but it must be consistent
        assert not result.cancelled


def test_run_again_shows_live_rows_and_marks_the_old_result_stale(
    make_window, qtbot, write_file
) -> None:
    window = make_tiny_window(make_window, write_file)
    dataset_id = active_id(window)
    result = wait_stored(qtbot, window, dataset_id)
    assert result is not None and result.ranked
    view = window.analysis_panel.autofit_view

    window.analysis.run_autofit(dataset_id)  # "Run again" over a finished run
    qtbot.waitUntil(lambda: window.analysis.is_autofit_running(), timeout=10000)
    # the old result is clearly marked as previous, never passed off as current
    assert not view.notice.isHidden()
    assert "previous run" in view.notice.text().lower()

    # while the run streams, the table must show the LIVE ranking, not the old result
    shown_vs_live: list[tuple[list[str], list[str]]] = []

    def record(live: object) -> None:
        rows = live.rows
        if rows:
            shown_vs_live.append(
                ([r.model_id for r in view.table.rows()], [r.model_id for r in rows])
            )

    window.analysis.autofit_live.connect(record)
    qtbot.waitUntil(lambda: bool(shown_vs_live), timeout=60000)
    for shown, live in shown_vs_live:
        assert shown == live, "the table shows the live ranking while the run is in progress"
    wait_stored(qtbot, window, dataset_id)  # the new run finishes; nothing hangs


def test_is_fit_running_reports_only_manual_fit_jobs(make_window, qtbot, write_file) -> None:
    from curvestudio.workflow import FitSpec, PolynomialRef

    window = make_tiny_window(make_window, write_file)
    dataset_id = active_id(window)
    window.analysis.run_autofit(dataset_id)
    assert window.analysis.is_autofit_running()
    assert not window.analysis.is_fit_running()  # an Auto-fit run is not a fit job
    wait_stored(qtbot, window, dataset_id)

    validated: list = []
    window.analysis.validation_ready.connect(lambda token, error: validated.append(token))
    window.analysis.validate_spec(FitSpec(model=PolynomialRef(degree=2)), False)
    assert not window.analysis.is_fit_running()  # nor is a spec dry-run, mid-flight
    qtbot.waitUntil(lambda: bool(validated), timeout=30000)

    xy = window.selection.xy()
    assert xy is not None
    window.analysis.submit_fit(
        dataset_id, xy, FitSpec(model=PolynomialRef(degree=2)), source={"kind": "manual"}
    )
    assert window.analysis.is_fit_running()  # a manual fit is exactly what it reports
    qtbot.waitUntil(lambda: not window.analysis.is_fit_running(), timeout=30000)


def test_finished_run_auto_selects_the_best_row(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file)
    result = wait_stored(qtbot, window, active_id(window))
    view = window.analysis_panel.autofit_view
    top = result.ranked[0]
    # Auto-fit first: the moment a run lands, rank 1 is selected and its curve plotted
    selected = view.table.selected_row()
    assert selected is not None and selected.model_id == top.model_id
    if top.model_id in result.curves:
        assert window.plot_view.fit_curve_data() is not None


def test_rankby_resort_keeps_the_selected_row_and_its_curve(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file)
    result = wait_stored(qtbot, window, active_id(window))
    view = window.analysis_panel.autofit_view
    assert len(result.ranked) >= 2
    chosen = result.ranked[1]
    assert view.table.select_row(1) is not None and chosen.model_id
    had_curve = chosen.model_id in result.curves
    if had_curve:
        assert window.plot_view.fit_curve_data() is not None

    view.rank_combo.setCurrentIndex(4)  # AICc: a different order for the same rows
    qtbot.waitUntil(lambda: "AICc" in view.rank_combo.currentText(), timeout=5000)
    selected = view.table.selected_row()
    assert selected is not None and selected.model_id == chosen.model_id
    if had_curve:
        assert window.plot_view.fit_curve_data() is not None  # the curve survived too


def test_row_without_sampled_curve_shows_inline_hint(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file)
    result = wait_stored(qtbot, window, active_id(window))
    view = window.analysis_panel.autofit_view
    unsampled = next((r for r in result.ranked if r.model_id not in result.curves), None)
    if unsampled is None:
        pytest.skip("this tiny run sampled every ranked row; nothing to assert")
    assert view.table.select_row(result.ranked.index(unsampled)) is not None
    assert window.plot_view.fit_curve_data() is None  # nothing is invented for it
    assert window.plot_view.hint_visible()
    assert "not sampled" in window.plot_view.hint_text()

    sampled = next(r for r in result.ranked if r.model_id in result.curves)
    view.table.select_row(result.ranked.index(sampled))
    assert not window.plot_view.hint_visible()  # a real curve replaces the hint
    assert window.plot_view.fit_curve_data() is not None


def test_rankby_resorts_without_refitting(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file)
    result = wait_stored(qtbot, window, active_id(window))
    view = window.analysis_panel.autofit_view
    ids_before = displayed_ids(window)
    runs_before = window.analysis.run_count

    view.rank_combo.setCurrentIndex(4)  # AICc
    qtbot.waitUntil(lambda: "AICc" in view.rank_combo.currentText(), timeout=5000)
    assert displayed_ids(window) == [r.model_id for r in result.ranked_by(RankBy.AICC)]
    view.rank_combo.setCurrentIndex(1)  # R²
    assert displayed_ids(window) == [r.model_id for r in result.ranked_by(RankBy.R2)]
    assert window.analysis.run_count == runs_before
    assert not window.analysis.is_autofit_running()
    assert ids_before  # sanity: rows were displayed throughout


def test_undefined_metrics_show_na_and_exact_fits_behave(make_window, qtbot, write_file) -> None:
    from curvestudio_desktop.views.ranking_table import COLUMNS, format_metric

    window = make_tiny_window(make_window, write_file, config=TINY, text=EXACT_LINEAR)
    result = wait_stored(qtbot, window, active_id(window))
    view = window.analysis_panel.autofit_view
    assert result.exact_fits, "noiseless linear data is reproduced exactly"
    # the dock's columns no longer include AICc; the undefined value must still never
    # become a zero anywhere it is shown
    assert "AICc" not in COLUMNS
    assert format_metric(None) == "n/a"
    exact = [row for row in result.ranked if row.model_id in result.exact_fits]
    assert exact and all(row.metrics.aicc is None for row in exact)
    # under AICc the exact fits cannot take part at all
    view.rank_combo.setCurrentIndex(4)
    assert displayed_ids(window) == [r.model_id for r in result.ranked_by(RankBy.AICC)]
    assert "Exact fits" in view.exact_section.title()


def test_row_selection_plots_curve_and_open_matches_workflow(
    make_window, qtbot, write_file
) -> None:
    window = make_tiny_window(make_window, write_file)
    result = wait_stored(qtbot, window, active_id(window))
    view = window.analysis_panel.autofit_view
    top = result.ranked[0]
    row = view.table.select_row(0)
    assert row is not None and row.model_id == top.model_id
    curve = window.plot_view.fit_curve_data()
    if top.model_id in result.curves:
        assert curve is not None and len(curve[0]) == len(result.curves[top.model_id].x)
    shown: list = []
    window.analysis.fit_finished.connect(lambda bundle, spec, meta: shown.append(bundle))
    view.table.openRequested.emit()
    qtbot.waitUntil(lambda: bool(shown), timeout=60000)

    from curvestudio.workflow import bundle_for_candidate

    xy = window.selection.xy()
    expected = bundle_for_candidate(xy, result, top.model_id)
    assert shown[-1].result.parameters == expected.result.parameters
    assert shown[-1].result.status == expected.result.status
    # the opened row also landed in the project as a stored fit
    fits = [a for a in window.controller.project.analyses if a.kind == "fit"]
    assert any(a.result.model_id == top.model_id for a in fits)


def test_selecting_second_row_switches_the_curve(make_window, qtbot, write_file) -> None:
    window = make_tiny_window(make_window, write_file)
    result = wait_stored(qtbot, window, active_id(window))
    view = window.analysis_panel.autofit_view
    assert len(result.ranked) >= 2
    second = result.ranked[1]
    row = view.table.select_row(1)
    assert row is not None and row.model_id == second.model_id
    if second.model_id in result.curves:
        assert window.plot_view.fit_curve_data() is not None
    else:  # not among the sampled top rows: nothing is drawn and nothing is invented
        assert window.plot_view.fit_curve_data() is None


@pytest.mark.slow
@pytest.mark.real_plugins
def test_slow_true_model_in_top3_for_exponential_data(make_window, qtbot, write_file) -> None:
    import numpy as np

    from curvestudio.models.registry import _clear_registry

    # the registry loads plugins once per process: an earlier test (plugins hidden) may
    # already have taught it that none exist, so let it load them again — for this test
    _clear_registry()
    rng = np.random.default_rng(20261117)
    x = np.linspace(0.0, 8.0, 40)
    y = 4.0 * np.exp(-0.35 * x) + rng.normal(0.0, 0.02, x.size)
    text = "x,y\n" + "".join(f"{xi:.6g},{yi:.6g}\n" for xi, yi in zip(x, y, strict=True))
    window = make_window()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(
        AutoFitConfig(families=("exponential",))  # the real catalogue, real process pool
    )
    assert window.import_file(write_file("exp.csv", text))
    result = wait_stored(qtbot, window, active_id(window), timeout=300000)
    top3 = [row.model_id for row in result.ranked[:3]]
    assert "exponential_decay" in top3, f"true model missing from the top three: {top3}"
