"""Persistence of analyses (save → open → restore without refitting) and exports."""

import csv
import json

from desktop_data import GOOD_CSV
from test_desktop_autofit_run import NOISY_QUADRATIC, TINY, wait_stored

from curvestudio.discovery import AutoFitConfig, AutoFitResult
from curvestudio_desktop.controllers.exports import ranking_csv

QUAD_CSV = NOISY_QUADRATIC


def tiny_window(make_window, write_file, text=QUAD_CSV):
    window = make_window()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", text))
    return window


def test_autofit_analysis_round_trips_through_save_and_open(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    window = tiny_window(make_window, write_file)
    dataset_id = window.controller.project.datasets[0].id
    result = wait_stored(qtbot, window, dataset_id)
    ranked_ids = [row.model_id for row in result.ranked]
    target = tmp_path / "analyses.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save() is True

    controller.paths.open_paths.append(target)
    assert controller.open_project(target) is True  # saved: no unsaved-changes prompt
    reopened_id = controller.project.datasets[0].id
    assert reopened_id == dataset_id  # archive ids survive the round trip
    runs_before = window.analysis.run_count

    stored = window.analysis.stored_autofit(reopened_id)
    assert stored is not None
    assert [row.model_id for row in stored.ranked] == ranked_ids
    assert window.analysis.maybe_autostart(reopened_id) is False  # current run exists: no refit
    assert window.analysis.run_count == runs_before

    # selecting the analysis in the tree restores the ranking without running anything
    analysis_id = next(a.id for a in controller.project.analyses if a.kind == "autofit")
    assert window.analysis.restore_analysis(analysis_id) is True
    assert window.analysis.run_count == runs_before
    rows = window.analysis_panel.autofit_view.table.rows()
    assert [row.model_id for row in rows] == [
        row.model_id for row in stored.ranked_by(window.analysis.rank_by)
    ]


def test_manual_fit_analysis_round_trips(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    window = make_window()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("good.csv", GOOD_CSV))
    panel = window.analysis_panel
    panel.manual_view.degree_spin.setValue(2)
    shown: list = []
    window.analysis.fit_finished.connect(lambda bundle, spec, meta: shown.append(bundle))
    panel.manual_view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)
    original = shown[0]
    assert str(original.result.status) == "converged"

    target = tmp_path / "fit.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save() is True
    controller.paths.open_paths.append(target)
    assert controller.open_project(target) is True
    fit_id = next(a.id for a in controller.project.analyses if a.kind == "fit")
    assert window.analysis.restore_analysis(fit_id) is True
    restored = window.analysis.current_bundle
    assert restored is not None
    assert restored.result.parameters == original.result.parameters
    assert restored.equation_text == original.equation_text
    assert window.plot_view.fit_curve_data() is not None  # the plot restores without refitting
    assert window.analysis.run_count == 0  # nothing re-ran anywhere in this test


def test_mode_is_remembered_in_settings_and_manual_does_not_autostart(
    make_window, qtbot, write_file, settings, tmp_path
) -> None:
    settings.remove("analysis/mode")  # the autouse seed: this test checks the product default
    settings.sync()
    first = make_window()
    qtbot.addWidget(first)
    assert first.analysis_panel.mode() == "autofit"  # the product default
    first.analysis.set_mode("manual")
    assert settings.fileName().startswith(str(tmp_path))  # isolated: nothing else written
    assert settings.value("analysis/mode") == "manual"

    second = make_window()
    qtbot.addWidget(second)
    assert second.analysis.mode == "manual"
    assert second.analysis_panel.mode() == "manual"
    assert second.import_file(write_file("good.csv", GOOD_CSV))
    qtbot.wait(300)
    assert second.analysis.run_count == 0  # manual mode never starts anything by itself
    assert not second.analysis.is_autofit_running()


def _export_paths(paths, tmp_path, names):
    for name in names:
        paths.export_paths.append(tmp_path / name)


def test_export_results_creates_parseable_files(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    window = make_window()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("good.csv", GOOD_CSV))
    panel = window.analysis_panel
    shown: list = []
    window.analysis.fit_finished.connect(lambda bundle, spec, meta: shown.append(bundle))
    panel.manual_view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)

    names = ("fit.json", "fit.xlsx", "fit.pdf", "fit.csv")
    _export_paths(controller.paths, tmp_path, names)
    for name in names:
        window.export_results()
        qtbot.waitUntil(lambda name=name: (tmp_path / name).is_file(), timeout=30000)
        qtbot.waitUntil(lambda: not window.jobs.is_busy(), timeout=30000)

    parsed = json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))
    assert parsed["fit"]["result"]["status"] == "converged"
    assert (tmp_path / "fit.xlsx").read_bytes()[:2] == b"PK"  # a real zip workbook
    assert (tmp_path / "fit.pdf").read_bytes().lstrip()[:4] == b"%PDF"
    with (tmp_path / "fit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["parameter", "value", "std_err", "ci_low", "ci_high"]
    with (tmp_path / "fit_statistics.csv").open(newline="", encoding="utf-8") as handle:
        stats = list(csv.reader(handle))
    assert stats[0] == ["statistic", "value", "note"]
    with (tmp_path / "fit_observations.csv").open(newline="", encoding="utf-8") as handle:
        observations = list(csv.reader(handle))
    assert len(observations) == 6  # header + five rows


def test_export_ranking_csv_is_formula_safe(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    window = tiny_window(make_window, write_file)
    dataset_id = window.controller.project.datasets[0].id
    result = wait_stored(qtbot, window, dataset_id)
    target = tmp_path / "ranking.csv"
    controller.paths.export_paths.append(target)
    window.export_ranking()
    qtbot.waitUntil(lambda: target.is_file(), timeout=30000)
    qtbot.waitUntil(lambda: not window.jobs.is_busy(), timeout=30000)

    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "rank"
    assert len(rows) == 1 + len(result.ranked)
    # neutralisation applies to text columns only: numbers must stay numbers
    text_columns = (1, 2, 3, 11, 12)  # model, category, equation, status, notes
    number_columns = (4, 5, 6, 7, 8, 9, 10)  # metrics and k
    dangerous = ("=", "+", "-", "@")
    for row in rows[1:]:
        for column in text_columns:
            cell = row[column]
            assert not cell.startswith(dangerous) or cell.startswith("'"), (
                f"unneutralised formula-like text cell {cell!r}"
            )
        for column in number_columns:
            cell = row[column]
            if cell:
                float(cell)  # a metric is a number, never a quoted or prefixed string


def _single_row_result(name: str, equation_fitted: str, aicc: float | None) -> AutoFitResult:
    from curvestudio.discovery import CandidateMetrics, CandidateOutcome
    from curvestudio.fitting import FitResult, FitStatus, collect_provenance

    fit = FitResult(
        model_id="row-model",
        equation="y = x",
        equation_fitted=equation_fitted,
        parameters=(),
        covariance=None,
        correlation=None,
        stats=None,
        fitted=[1.0] * 5,
        residuals=[0.0] * 5,
        status=FitStatus.CONVERGED,
        termination_reason="",
        method="auto",
        loss="linear",
        initial_params={},
        n_starts=1,
        warnings=(),
        provenance=collect_provenance("f" * 64, 0),
    )
    return AutoFitResult(
        config=AutoFitConfig(),
        dataset_fingerprint="f" * 64,
        n_active=5,
        ranked=(),
        outcomes=(
            CandidateOutcome(
                candidate_id="row-model",
                name=name,
                category="test",
                outcome="converged",
                fit=fit,
                metrics=CandidateMetrics(k=1, n=5, r2=0.5, aicc=aicc, bic=None),
            ),
        ),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )


def _parsed_rows(result: AutoFitResult) -> list[list[str]]:
    import io

    return list(csv.reader(io.StringIO(ranking_csv(result, AutoFitConfig().rank_by))))


def test_ranking_csv_neutralises_a_formula_named_model() -> None:
    result = _single_row_result("=SUM(A1:A9)", "=1+1", None)
    rows = _parsed_rows(result)
    assert len(rows) == 2  # header + the one row, whatever the equation contained
    assert rows[1][1] == "'=SUM(A1:A9)"  # the name carries the neutralising apostrophe
    assert rows[1][3] == "'=1+1"  # and so does the equation cell


def test_ranking_csv_keeps_negative_numbers_quoted_text_and_commas() -> None:
    result = _single_row_result('quo"te, name', "y = a*exp(-r*x), honest", -248.4)
    rows = _parsed_rows(result)
    assert len(rows) == 2  # the embedded quote and commas split no fields
    assert rows[1][1] == 'quo"te, name'  # quotes survive csv escaping untouched
    assert rows[1][3] == "y = a*exp(-r*x), honest"  # a comma inside a field moves nothing
    assert float(rows[1][7]) == -248.4  # a negative metric round-trips as a number
    assert not rows[1][7].startswith("'"), "a negative number is never a formula"
