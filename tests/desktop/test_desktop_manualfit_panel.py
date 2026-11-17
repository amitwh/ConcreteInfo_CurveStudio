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
    # the result went to the same results tabs an opened row uses
    assert panel.results.parameters_text(0, 0) == "b0"
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
    banner = window.analysis_panel.results.banner_text()
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
    predict_tab = panel.results
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
