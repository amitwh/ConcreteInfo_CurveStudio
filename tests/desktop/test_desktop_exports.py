"""The fit export's PDF report and workbook carry the fit's audit and its methods paragraph.

``export_fit`` is the function the window's Export results action submits to its job
runner, so these tests call it as it is; the reporting package's own tests cover what the
two writers do with what they are given.
"""

from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

from curvestudio.data import XY
from curvestudio.fitting import FitStatus
from curvestudio.tools import ToolError, audit_fit, methods_text
from curvestudio.workflow import CustomRef, FitBundle, FitSpec, PolynomialRef, fit_dataset
from curvestudio_desktop.controllers import exports

LINEAR = FitSpec(model=PolynomialRef(degree=1))


def _fit(x: np.ndarray, y: np.ndarray, spec: FitSpec) -> FitBundle:
    data = XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="fp-export",
    )
    return fit_dataset(data, spec)


def _line_bundle() -> FitBundle:
    """A straight line through 30 noisy points, and the specification it ran with."""
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 30)
    return _fit(x, 2.0 * x + 1.0 + rng.normal(0.0, 0.1, 30), LINEAR)


@pytest.fixture
def given(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, object]]:
    """What the export hands each report writer, by writer, while the real file is written."""
    handed: dict[str, dict[str, object]] = {}
    for name in ("fit_to_pdf", "fit_to_xlsx"):
        real = getattr(exports, name)

        def recording(
            bundle: FitBundle, target: Path, _real=real, _name=name, **options: object
        ) -> None:
            handed[_name] = options
            _real(bundle, target, **options)

        monkeypatch.setattr(exports, name, recording)
    return handed


def _summary(target: Path) -> dict[str, object]:
    sheet = load_workbook(target)["Summary"]
    return {row[0].value: row[1].value for row in sheet.iter_rows() if row[0].value is not None}


def test_the_exports_hand_the_writers_the_audit_and_the_methods_text(
    given: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    bundle = _line_bundle()
    audit = audit_fit(bundle, spec=LINEAR)
    texts = methods_text(bundle, spec=LINEAR, style="text")

    exports.export_fit(tmp_path / "fit.pdf", bundle, byline="Amit Haridas", spec=LINEAR)
    exports.export_fit(tmp_path / "fit.xlsx", bundle, byline="Amit Haridas", spec=LINEAR)

    expected = {"byline": "Amit Haridas", "audit": audit, "methods": texts}
    assert given == {"fit_to_pdf": expected, "fit_to_xlsx": expected}
    assert (tmp_path / "fit.pdf").read_bytes().startswith(b"%PDF")
    summary = _summary(tmp_path / "fit.xlsx")
    assert summary["methods paragraph"] == f"{texts.paragraph} {audit.methods_fragment}"
    assert summary["Fit audit"] == audit.summary


def test_without_a_specification_the_export_reads_the_fit_alone(
    given: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    """The window does not always have the specification; the export then audits without it."""
    bundle = _line_bundle()
    exports.export_fit(tmp_path / "fit.xlsx", bundle)
    assert given["fit_to_xlsx"] == {
        "byline": None,
        "audit": audit_fit(bundle),
        "methods": methods_text(bundle, style="text"),
    }


def test_a_fit_that_did_not_converge_exports_with_its_audit_and_no_methods_text(
    given: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    spec = FitSpec(
        model=CustomRef(equation="a*x + b*x", parameters=["a", "b"]), initial={"a": 1.0, "b": 2.0}
    )
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 30)
    bundle = _fit(x, 2.0 * x + 1.0 + rng.normal(0.0, 0.1, 30), spec)
    assert bundle.result.status is not FitStatus.CONVERGED
    with pytest.raises(ToolError) as refused:  # the tool writes no methods text for such a fit
        methods_text(bundle, spec=spec)
    assert refused.value.code == "FIT_NOT_CONVERGED_INPUT"

    exports.export_fit(tmp_path / "fit.pdf", bundle, spec=spec)  # the export must not fail
    exports.export_fit(tmp_path / "fit.xlsx", bundle, spec=spec)

    audit = audit_fit(bundle, spec=spec)
    assert audit.overall == "fail"
    for name in ("fit_to_pdf", "fit_to_xlsx"):
        assert given[name] == {"byline": None, "audit": audit, "methods": None}
    summary = _summary(tmp_path / "fit.xlsx")
    assert "methods paragraph" not in summary and "Fit audit" not in summary
    assert summary["status"] == "not_converged"


def test_the_window_hands_the_export_the_fit_s_settings(
    make_window, qtbot, monkeypatch, tmp_path
) -> None:
    """Export results gives ``export_fit`` the fit and the specification it ran with."""
    window = make_window()
    qtbot.addWidget(window)
    bundle, spec = _line_bundle(), LINEAR
    analysis = type(window.analysis)
    monkeypatch.setattr(analysis, "current_bundle", property(lambda self: bundle))
    monkeypatch.setattr(analysis, "current_spec", property(lambda self: spec))
    window.controller.paths.export_paths.append(tmp_path / "report.pdf")
    handed: dict[str, object] = {}

    def recording(target: Path, given: FitBundle, *, byline=None, spec=None) -> None:
        handed["bundle"] = given
        handed["spec"] = spec

    monkeypatch.setattr(exports, "export_fit", recording)
    monkeypatch.setattr(window, "_run_job", lambda label, work, **callbacks: work())

    window.export_results()

    assert handed == {"bundle": bundle, "spec": spec}
