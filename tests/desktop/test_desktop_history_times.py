"""History timestamps (design item 13): stored fits and Auto-fit runs show their time.

Every list of them is the project tree's analysis rows, labelled by the analysis
controller; the time is `DD/MM/YYYY HH:mm` IST (the stored ISO string already carries
the +05:30 offset). Analyses recorded before the field existed show no time rather than
a guessed one.
"""

import re

from desktop_data import GOOD_CSV

from curvestudio_desktop.controllers.analysis import history_time

TIME = re.compile(r" — \d{2}/\d{2}/\d{4} \d{2}:\d{2}$")


def test_history_time_formats_a_fixed_ist_instant() -> None:
    assert history_time("2026-11-25T14:35:00+05:30") == " — 25/11/2026 14:35"


def test_history_time_converts_a_utc_instant_to_ist() -> None:
    # 09:05Z is 14:35 the same day at +05:30
    assert history_time("2026-11-25T09:05:00+00:00") == " — 25/11/2026 14:35"


def test_history_time_is_empty_when_nothing_was_recorded() -> None:
    assert history_time(None) == ""


def test_history_time_ignores_an_unparseable_stamp() -> None:
    assert history_time("not a date") == ""


def make_manual_window(make_window, write_file):
    window = make_window()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("good.csv", GOOD_CSV))
    return window


def analysis_labels(window) -> list[str]:
    """The analysis rows under the first dataset, as the tree shows them."""
    tree = window.project_tree
    root = tree.topLevelItem(0)
    assert root is not None
    labels: list[str] = []
    for index in range(root.childCount()):
        dataset = root.child(index)
        for row in range(dataset.childCount()):
            labels.append(dataset.child(row).text(0))
    return labels


def test_a_stored_autofit_run_and_fit_show_their_ist_time(make_window, qtbot, write_file) -> None:
    from test_desktop_autofit_run import TINY, make_tiny_window, wait_stored

    window = make_tiny_window(make_window, write_file, config=TINY)
    dataset_id = window.controller.project.datasets[0].id
    wait_stored(qtbot, window, dataset_id)

    labels = analysis_labels(window)
    autofit_label = next(label for label in labels if label.startswith("Auto-fit"))
    assert TIME.search(autofit_label), f"no IST time on {autofit_label!r}"

    # the stored analysis carries the same time the label shows
    analysis = next(a for a in window.controller.project.analyses if a.kind == "autofit")
    assert history_time(analysis.created_at).removeprefix(" — ") in autofit_label

    # a manual fit on top: its row is the model id plus the fit's own time
    window.analysis.set_mode("manual")
    shown: list = []
    window.analysis.fit_finished.connect(lambda bundle, spec, meta: shown.append(bundle))
    window.analysis_panel.manual_view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)
    fit = next(a for a in window.controller.project.analyses if a.kind == "fit")
    fit_label = next(
        label for label in analysis_labels(window) if label.startswith(fit.result.model_id)
    )
    assert TIME.search(fit_label), f"no IST time on {fit_label!r}"
    # a fit's time is its result's provenance timestamp
    assert history_time(fit.created_at) == history_time(fit.result.provenance.created_at)


def test_the_time_survives_save_and_open(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    """The timestamp is part of the stored analysis, not a view-time guess."""
    from test_desktop_autofit_run import TINY, make_tiny_window, wait_stored

    window = make_tiny_window(make_window, write_file, config=TINY)
    dataset_id = window.controller.project.datasets[0].id
    wait_stored(qtbot, window, dataset_id)
    before = next(a for a in controller.project.analyses if a.kind == "autofit")
    assert before.created_at is not None

    target = tmp_path / "times.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save() is True
    controller.paths.open_paths.append(target)
    assert controller.open_project(target) is True

    reopened = next(a for a in controller.project.analyses if a.kind == "autofit")
    assert reopened.created_at == before.created_at
    assert TIME.search(window.analysis.analysis_label(reopened.id))
