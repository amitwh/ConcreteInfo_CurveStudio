"""Manual fit panel: spec building, inline validation, jobs, weights, prediction."""

import math

import pytest
from desktop_data import GOOD_CSV

from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _REGISTRY
from curvestudio.workflow import MAX_PREDICT_POINTS, WorkflowError, fit_dataset

WEIGHTED_CSV = "x,y,w\n1,2.0,1.0\n2,4.1,1.5\n3,6.2,1.0\n4,7.9,2.0\n5,10.3,1.0\n"
EXPONENTIAL_CSV = "x,y\n" + "".join(
    f"{i / 5:.2f},{4.0 * math.exp(-0.35 * i / 5):.6f}\n" for i in range(25)
)


@pytest.fixture()
def test_model():
    """One extra model for the picker, registered *additively* and removed afterwards.

    Clearing the shared registry here (the core tests' habit) would poison the whole
    process: with the plugin entry points hidden for ordinary tests, the next registry
    use would reload it empty, and the later in-process CLI tests would find no models.
    """
    register(
        ModelSpec(
            id="test_exp",
            name="Test exponential",
            category="test",
            expression="a*exp(-r*x)",
            parameters=(
                Parameter("a", "Amplitude"),
                Parameter("r", "Rate (positive)", lower=0.0),
            ),
            initial_guess=lambda x, y: {"a": float(y[0]), "r": 0.5},
        )
    )
    yield "test_exp"
    _REGISTRY.pop("test_exp", None)


def make_manual_window(make_window, write_file, text=GOOD_CSV, name="good.csv"):
    window = make_window()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file(name, text))
    view = window.analysis_panel.manual_view
    view.select_model("polynomial")  # explicit: never rely on the registry's starter models
    return window


def import_weighted(window, write_file):
    from curvestudio.workflow import import_dataset

    outcome = import_dataset(
        write_file("w.csv", WEIGHTED_CSV),
        "w.csv",
        x_column="x",
        y_column="y",
        weight_column="w",
    )
    return window.controller.add_imported(outcome)


def wait_fit(qtbot, window):
    shown: list = []
    window.analysis.fit_finished.connect(lambda bundle, spec, meta: shown.append(bundle))
    return shown


def test_manual_polynomial_fit_equals_fit_dataset(make_window, qtbot, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    panel = window.analysis_panel
    assert panel.manual_view._is_polynomial()  # the default is the polynomial family
    panel.manual_view.degree_spin.setValue(2)
    shown = wait_fit(qtbot, window)
    panel.manual_view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)
    spec = window.analysis.current_spec
    assert spec is not None
    expected = fit_dataset(window.selection.xy(), spec)
    assert shown[-1].result.parameters == expected.result.parameters
    assert shown[-1].result.stats is not None and expected.result.stats is not None
    assert shown[-1].result.stats.sse == expected.result.stats.sse
    # the result went to the same result views an opened row uses (now in the details window)
    assert window.details_dialog.results.parameters_text(0, 0) == "b0"
    assert window.plot_view.fit_curve_data() is not None
    fits = [a for a in window.controller.project.analyses if a.kind == "fit"]
    assert len(fits) == 1 and fits[0].result.model_id.startswith("poly2")


def test_registered_model_fit_converges_and_prefills_bounds(
    make_window, qtbot, write_file, test_model
) -> None:
    window = make_manual_window(make_window, write_file, EXPONENTIAL_CSV, "exp.csv")
    panel = window.analysis_panel
    view = panel.manual_view
    view._fill_models()  # the registry changed after the panel was built
    assert view.select_model(test_model)
    table = view.parameters_table
    assert table.rowCount() == 2
    assert table.item(0, 0).text() == "a" and table.item(0, 2).text() == ""
    assert table.item(1, 0).text() == "r" and table.item(1, 2).text() == "0.0"  # prefilled bound
    shown = wait_fit(qtbot, window)
    view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)
    bundle = shown[-1]
    assert str(bundle.result.status) == "converged"
    values = {p.name: p.value for p in bundle.result.parameters}
    assert values["a"] == pytest.approx(4.0, rel=1e-3)
    assert values["r"] == pytest.approx(0.35, rel=1e-3)
    # switching to another model resets the parameter rows
    assert view.select_model("polynomial")
    assert table.rowCount() == 0


def test_custom_equation_rows_offer_initial_and_bounds_editing(
    make_window, qtbot, write_file
) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    view.model_combo.setCurrentIndex(view.model_combo.count() - 1)  # "Custom equation…"
    view.equation_edit.setText("y = a*exp(-r*x)")
    view.parameters_edit.setText("a, r")  # the user typed the parameter list
    rows = view.parameters_table.rowCount()
    names = [view._cell(row, 0) for row in range(rows)]
    assert names == ["a", "r"]  # one editable row per typed parameter, blank = default

    view.parameters_table.item(0, 1).setText("4.0")  # initial a
    view.parameters_table.item(1, 2).setText("0.0")  # lower bound on r
    spec, problem = view.current_spec()
    assert spec is not None and problem == ""
    assert spec.initial == {"a": 4.0}
    assert spec.bounds == {"r": (0.0, None)}

    # retyping the list keeps values entered for names that survive, on their rows
    view.parameters_edit.setText("r, a")
    names = [view._cell(row, 0) for row in range(view.parameters_table.rowCount())]
    assert names == ["r", "a"]
    assert view._cell(0, 2) == "0.0"  # r's lower bound followed r
    assert view._cell(1, 1) == "4.0"  # a's initial followed a
    spec, problem = view.current_spec()
    assert spec is not None and problem == ""
    assert spec.initial == {"a": 4.0}
    assert spec.bounds == {"r": (0.0, None)}


def test_invalid_custom_equation_shows_inline_error_and_fits_nothing(
    make_window, qtbot, write_file
) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    view.model_combo.setCurrentIndex(view.model_combo.count() - 1)  # "Custom equation…"
    view.equation_edit.setText("y = a*x**")
    view.parameters_edit.setText("a")
    view._validate_now()  # the debounced path, run immediately
    qtbot.waitUntil(lambda: view._last_error is not None, timeout=30000)
    assert "INVALID_EQUATION" in view.validation_text()
    assert not view.fit_button.isEnabled()
    fits_before = window.analysis.fit_count
    view.run_fit()
    qtbot.wait(200)
    assert window.analysis.fit_count == fits_before  # no job was ever submitted


def test_nonconverged_fit_shows_banner_and_draws_no_curve(make_window, qtbot, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    view.model_combo.setCurrentIndex(view.model_combo.count() - 1)
    view.equation_edit.setText("y = a*exp(-r*x)")
    view.parameters_edit.setText("a, r")
    view.time_spin.setDecimals(6)
    view.time_spin.setValue(0.000001)  # the deadline passes before the optimiser's first step
    shown = wait_fit(qtbot, window)
    view._validate_now()
    qtbot.waitUntil(lambda: view._last_error is None, timeout=30000)
    view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)
    bundle = shown[-1]
    assert str(bundle.result.status) != "converged"
    banner = window.details_dialog.results.banner_text()
    assert "DID NOT CONVERGE" in banner
    assert "time limit" in banner
    assert window.plot_view.fit_curve_data() is None  # never a curve for a failed fit


def test_weights_checkbox_follows_the_data(make_window, qtbot, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    assert not view.weights_check.isEnabled()  # GOOD_CSV has no weights
    assert import_weighted(window, write_file) is not None
    assert view.weights_check.isEnabled()
    assert not view.weights_check.isChecked()  # off by default: an explicit choice


def test_weighted_manual_fit_uses_the_weights(make_window, qtbot, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    assert import_weighted(window, write_file) is not None
    view = window.analysis_panel.manual_view
    view.weights_check.setChecked(True)
    shown = wait_fit(qtbot, window)
    view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)
    spec = window.analysis.current_spec
    assert spec is not None and spec.use_weights
    expected = fit_dataset(window.selection.xy(), spec)
    assert shown[-1].result.parameters == expected.result.parameters


def test_predict_tab_reports_values_and_refuses_oversized_requests(
    make_window, qtbot, write_file
) -> None:
    from curvestudio.workflow import predict

    window = make_manual_window(make_window, write_file)
    panel = window.analysis_panel
    shown = wait_fit(qtbot, window)
    panel.manual_view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)
    bundle = shown[-1]
    spec = window.analysis.current_spec
    assert spec is not None
    predict_tab = window.details_dialog.results
    predict_tab._predict_input.setPlainText("1\n2\n3.5\n")
    predict_tab._run_predict()
    assert predict_tab.predict_rows() == 3
    expected = predict(bundle, spec, [1.0, 2.0, 3.5])
    assert expected.y is not None
    table_y = [float(predict_tab._predict_table.item(row, 1).text()) for row in range(3)]
    # the table shows six significant digits; the comparison must allow for that rounding
    assert table_y == pytest.approx([v for v in expected.y if v is not None], rel=1e-5)

    with pytest.raises(WorkflowError) as too_large:
        window.analysis.predict(bundle, spec, [1.0] * (MAX_PREDICT_POINTS + 1), None)
    assert too_large.value.code == "TOO_LARGE"
    predict_tab._predict_input.setPlainText("not-a-number\n")
    predict_tab._run_predict()
    assert not predict_tab._predict_error.isHidden()  # shown (the window itself is offscreen)
    assert "not numbers" in predict_tab._predict_error.text()


def test_model_picker_search_finds_by_id_and_equation(
    make_window, qtbot, write_file, test_model
) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    view._fill_models()
    assert any(info.id == test_model for info in view._models)
    view._on_completion("Test exponential (test_exp) test y = a*exp(-r*x)")
    assert view.current_model() is not None and view.current_model().id == test_model
    view._on_completion("Polynomial (polynomial) polynomial y = b0 + b1*x")
    assert view.current_model() is not None and view.current_model().id == "polynomial"
    assert view._is_polynomial()


def test_cancel_button_appears_while_fitting(make_window, qtbot, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    assert view.cancel_button.isHidden()
    window.analysis.fit_started.connect(lambda _ds: None)
    view.set_fitting(True)
    assert not view.cancel_button.isHidden()
    assert not view.fit_button.isEnabled()
    view.set_fitting(False)
    assert view.cancel_button.isHidden()


# -- the model picker: searchable dropdown, id-keyed ---------------------------------------


def test_the_picker_maps_the_displayed_model_to_the_fitted_one(make_window, write_file) -> None:
    """Every item must resolve to the model its text names.

    The picker groups the catalogue with separator items; separators occupy item indices,
    so the old index-into-the-model-list lookup silently shifted by the separator count —
    the field said "Einstein heat capacity" while the form fitted `polynomial`.
    """
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    combo = view.model_combo
    checked = 0
    for index in range(combo.count()):
        model_id = combo.itemData(index)
        if not isinstance(model_id, str):
            continue
        combo.setCurrentIndex(index)
        model = view.current_model()
        assert model is not None and model.id == model_id, (
            f"item {index} shows {combo.itemText(index)!r} but resolves to "
            f"{None if model is None else model.id}"
        )
        assert model.name in combo.itemText(index)
        checked += 1
    assert checked == len(view._models), "every catalogue model must be reachable"


def test_the_picker_cannot_hold_free_text(make_window, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    combo = view.model_combo
    from PySide6.QtWidgets import QComboBox

    assert combo.insertPolicy() is QComboBox.InsertPolicy.NoInsert
    chosen = view.current_model()
    assert chosen is not None

    editor = view.model_edit
    editor.setText("some model nobody has heard of")
    editor.editingFinished.emit()
    assert view.current_model() is not None and view.current_model().id == chosen.id
    assert combo.currentText() == combo.itemText(combo.currentIndex())


def test_the_picker_filters_by_any_word_and_offers_the_catalogue(make_window, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    combo = view.model_combo
    completer = combo.completer()
    assert completer is not None
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCompleter

    assert completer.filterMode() is Qt.MatchFlag.MatchContains
    assert completer.caseSensitivity() is Qt.CaseSensitivity.CaseInsensitive
    assert completer.completionMode() is QCompleter.CompletionMode.PopupCompletion

    # use a model that this test registry actually carries
    target = view._models[-1]
    prefix = target.name.split()[0].lower()
    completer.setCompletionPrefix(prefix)
    model = completer.completionModel()
    assert model.rowCount() >= 1
    first = str(model.index(0, 0).data())
    # the completion carries the id, so activating it picks the right model
    view._on_completion(first)
    assert view.current_model() is not None
    assert view.current_model().id == target.id


def test_an_anonymous_search_restores_the_chosen_model(make_window, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    chosen = view.current_model()
    assert chosen is not None
    view.model_edit.setText("zzzz")
    view._on_search_finished()
    assert view.current_model() is not None and view.current_model().id == chosen.id


def test_the_custom_entry_still_selects_custom_mode(make_window, write_file) -> None:
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    view.select_custom_equation()
    assert view.current_model() is None
    assert view._is_custom() is True
    assert view._equation_box.isVisible() or not window.isVisible()  # shown in a live window


def test_parameters_grid_stacks_when_the_form_is_narrow(
    make_window, qtbot, write_file, test_model
) -> None:
    """Design item 14: below its width threshold the parameter grid stacks — each
    parameter becomes its name over three labelled value rows — and nothing entered is
    lost, whatever way the panel is resized."""
    window = make_manual_window(make_window, write_file, EXPONENTIAL_CSV, "exp.csv")
    view = window.analysis_panel.manual_view
    view._fill_models()
    assert view.select_model(test_model)
    table = view.parameters_table
    assert table.columnCount() == 4  # wide: parameter | initial | lower | upper
    table.item(0, 1).setText("4.0")  # initial for a
    table.item(1, 2).setText("0.0")  # lower bound for r (prefilled, made explicit)

    view._set_parameters_stacked(True)

    assert table.columnCount() == 2
    assert not view.parameters_table.horizontalHeader().isVisible()  # labels are per row now
    # each parameter spans four rows: its name, then one labelled row per field
    assert table.rowCount() == 8
    assert view._param_names == ["a", "r"]
    assert view._param_text(0, 1) == "4.0"  # a's initial survived the switch
    assert view._param_text(1, 2) == "0.0"  # r's lower bound survived
    spec, problem = view.current_spec()
    assert spec is not None and problem == ""
    assert spec.initial == {"a": 4.0}
    assert spec.bounds == {"r": (0.0, None)}

    # widening the panel lays the four columns out again, values still in place
    view._set_parameters_stacked(False)
    assert table.columnCount() == 4
    assert table.rowCount() == 2
    assert view._param_text(0, 1) == "4.0"
    assert view._param_text(1, 2) == "0.0"

    # the switch follows the form's real width, not a manual call
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent
    from PySide6.QtWidgets import QApplication

    viewport = view.form_scroll.viewport()
    QApplication.sendEvent(viewport, QResizeEvent(QSize(300, 400), QSize(700, 400)))
    assert view._parameters_stacked is True
    QApplication.sendEvent(viewport, QResizeEvent(QSize(500, 400), QSize(300, 400)))
    assert view._parameters_stacked is False


def test_stacked_grid_labels_each_field(make_window, qtbot, write_file) -> None:
    """A stacked row says what each value is: the field labels are in the grid."""
    window = make_manual_window(make_window, write_file)
    view = window.analysis_panel.manual_view
    view.model_combo.setCurrentIndex(view.model_combo.count() - 1)  # custom equation
    view.equation_edit.setText("y = a*x")
    view.parameters_edit.setText("a")
    view._set_parameters_stacked(True)
    table = view.parameters_table
    texts = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert texts == ["a", "initial", "lower bound", "upper bound"]
    # the label cells are read-only; only the value column is editable
    from PySide6.QtCore import Qt

    assert not bool(table.item(1, 0).flags() & Qt.ItemFlag.ItemIsEditable)
    assert bool(table.item(1, 1).flags() & Qt.ItemFlag.ItemIsEditable)
