"""Application wiring: QApplication factory, theme, crash guard and the selftest.

``run_app`` is the real entry point; ``run_selftest`` is the headless smoke used
by CI and packaging (D4): it builds the whole main window offscreen, loads the
bundled example, runs one fit and one Auto-fit (through the spawn process pool,
which a frozen build only survives with ``freeze_support()``) via the workflow
layer and reports PASS/FAIL on stdout. It never shows a window and exits 0 only
when everything worked.
"""

import io
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from curvestudio_desktop import APP_NAME, ORG_NAME, __version__

if TYPE_CHECKING:
    from curvestudio.data import XY
    from curvestudio_desktop.views.main_window import MainWindow

Shower = Callable[[str], None]

SELFTEST_TIMEOUT_SECONDS = 60.0
"""Bound on the selftest's Auto-fit wait: a tiny run finishes in seconds, so a hang
(a run that never started, or never delivered) fails fast with a clear reason."""


@contextmanager
def selftest_settings() -> Iterator[QSettings]:
    """Throwaway INI settings, so the selftest never touches the user's configuration.

    A stored preference (a manual mode, say) must not steer what the smoke test
    exercises: the selftest forces Auto-fit mode and its own run options on top of
    this clean slate, and nothing it writes survives it.
    """
    with tempfile.TemporaryDirectory(prefix="curvestudio-selftest-") as directory:
        yield QSettings(str(Path(directory) / "settings.ini"), QSettings.Format.IniFormat)


def create_application(argv: list[str] | None = None) -> QApplication:
    """The one QApplication, with identity, DPI policy and the Fusion base style."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)
    app.setDesktopFileName("curvestudio-desktop")
    # Fusion first: icons render against the final style/palette
    app.setStyle("Fusion")
    from curvestudio_desktop.theme import ensure_fonts_loaded
    from curvestudio_desktop.views.icons import app_icon

    ensure_fonts_loaded()
    app.setWindowIcon(app_icon())
    return app


def install_crash_guard(shower: Shower | None = None) -> None:
    """Route unhandled exceptions — from any thread — to a dialog with copyable details.

    Main-thread exceptions (``sys.excepthook``) show the dialog immediately: we are
    already on the GUI thread. Exceptions raised inside ``threading.Thread`` workers
    (``threading.excepthook``) are marshalled to the GUI thread through a queued
    signal first — opening a Qt dialog off the GUI thread is undefined behaviour —
    and are always printed to stderr so logs keep them. While a dialog is up, a
    second crash goes to stderr only (no re-entrant dialogs). Whether the process
    exits afterwards depends on where the exception escaped: a top-level one still
    terminates the interpreter through the default hook, while one caught inside
    the Qt event loop leaves the application running.
    """
    from PySide6.QtCore import QObject, QThread, Signal

    app = QApplication.instance()

    class _GuiRelay(QObject):
        """Lives on the GUI thread; worker threads emit into it (queued delivery)."""

        show_requested = Signal(str)

        def deliver(self, details: str) -> None:
            _display(details)

    relay: _GuiRelay | None = None
    if app is not None:
        relay = _GuiRelay()
        # connecting to a slot of `relay` itself makes cross-thread emissions
        # queued: `deliver` always runs on relay's thread (the GUI thread)
        relay.show_requested.connect(relay.deliver)

    showing: list[bool] = [False]

    def _display(details: str) -> None:
        if showing[0]:  # a crash while showing the crash dialog: stderr, not recursion
            print(details, file=sys.stderr, flush=True)
            return
        showing[0] = True
        try:
            from curvestudio_desktop.views.dialogs import show_crash_dialog

            (shower or show_crash_dialog)(details)
        except Exception:
            print(details, file=sys.stderr, flush=True)
        finally:
            showing[0] = False

    def _hook(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
    ) -> None:
        from curvestudio_desktop.views.dialogs import format_exception

        _display(format_exception(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        from curvestudio_desktop.views.dialogs import format_exception

        if args.exc_value is None:  # a manual call with no exception object
            print(f"exception in thread {args.thread}: {args.exc_type!r}", file=sys.stderr)
            return
        details = format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        if relay is None or app is None or QThread.currentThread() is app.thread():
            _display(details)  # no event loop to relay through (or already GUI)
            return
        print(details, file=sys.stderr, flush=True)
        relay.show_requested.emit(details)

    sys.excepthook = _hook
    threading.excepthook = _thread_hook


def build_main_window(settings: QSettings | None = None) -> MainWindow:
    """The main window wired to real (dialog-backed) providers.

    ``settings`` overrides where window/controller preferences live — the selftest
    injects its throwaway INI so it neither reads nor writes the user's config.
    """
    from curvestudio_desktop.controllers.project import ProjectController
    from curvestudio_desktop.theme import apply_theme, theme_preference
    from curvestudio_desktop.views.dialogs import (
        DialogConfirmationProvider,
        DialogPathProvider,
        MessageBoxMessenger,
    )
    from curvestudio_desktop.views.main_window import MainWindow

    store = settings if settings is not None else QSettings()
    app = QApplication.instance()
    if app is not None:  # the remembered look applies before any icon is built
        apply_theme(cast("QApplication", app), preference=theme_preference(store))
    controller = ProjectController(
        confirmation=DialogConfirmationProvider(),
        paths=DialogPathProvider(),
        messenger=MessageBoxMessenger(),
        settings=store,
    )
    # the window shares the controller's settings: a split (injected controller
    # settings, default window settings) would read two different stores
    return MainWindow(controller, settings=store)


def run_app(argv: list[str] | None = None) -> int:
    """Start the application: theme, crash guard, main window, event loop."""
    from curvestudio_desktop.theme import apply_theme

    app = create_application(argv)
    apply_theme(app)
    install_crash_guard()
    window = build_main_window()
    window.show()
    return int(app.exec())


def _selftest_autofit(xy: XY) -> tuple[bool, str]:
    """One tiny Auto-fit through the spawn process pool.

    Two workers force the multiprocessing pool even for this small run. In a
    frozen (PyInstaller) build every worker re-executes the program, so this
    returning proves ``freeze_support()`` runs before the app. The exponential
    family only yields candidates when the model-library entry points were
    found, so a plain ``importlib.metadata`` scan also gets proven here.

    Returns ``(ok, summary)``; the summary is printed in the SELFTEST OK line.
    """
    from curvestudio.workflow import AutoFitConfig, run_autofit

    result = run_autofit(
        xy,
        AutoFitConfig(
            families=("exponential",),
            n_starts=2,
            time_limit=10.0,
            cv_folds=None,
            workers=2,  # the pool, not an in-process run
            curve_top_n=0,
            staged_starts=False,
        ),
    )
    if result.cancelled:
        return False, "the Auto-fit run was cancelled"
    catalogue = [o for o in result.outcomes if not o.candidate_id.startswith("poly")]
    if not catalogue:
        return False, "Auto-fit saw no registered models (the entry points did not load)"
    if not result.ranked:
        return False, "no Auto-fit candidate converged"
    best = result.ranked[0]
    return True, f"autofit ({len(result.outcomes)} candidates, 2-worker pool, best {best.model_id})"


def run_selftest(argv: list[str] | None = None) -> int:
    """Run the smoke test, optionally logging to ``CURVESTUDIO_SELFTEST_OUT``.

    A windowed Windows build has no console, so the packaging driver asks for the log in
    a file; with a console (Linux/macOS, CI) the same text still reaches stdout.
    """
    out_path = os.environ.get("CURVESTUDIO_SELFTEST_OUT")
    if out_path is None:
        return _run_selftest(argv)
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        code = _run_selftest(argv)
    text = sink.getvalue()
    Path(out_path).write_text(text, encoding="utf-8")
    sys.stdout.write(text)  # a silent no-op when the frozen app has no console
    return code


def _run_selftest(argv: list[str] | None = None) -> int:
    """Headless smoke test: window + bundled example + one fit + one Auto-fit.

    Exit 0 is pass. Used by CI and the packaging build. Prints each step so a failure
    says where. Runs against throwaway settings with Auto-fit forced on: the user's
    real preferences must never steer (or break) the smoke test.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # selftest is headless by design
    from importlib import resources

    from curvestudio.discovery import AutoFitConfig
    from curvestudio.workflow import FitSpec, fit_dataset
    from curvestudio.workflow.spec import PolynomialRef
    from curvestudio_desktop.controllers.analysis import MODE_AUTOFIT

    app = create_application(argv if argv is not None else [sys.argv[0]])
    from curvestudio_desktop.theme import apply_theme

    apply_theme(app)
    with selftest_settings() as settings:
        window = build_main_window(settings)
        if window.analysis.mode != MODE_AUTOFIT:
            # a stored mode must not steer the smoke test: Auto-fit is what it checks
            window.analysis.set_mode(MODE_AUTOFIT)
        # the selftest's Auto-fit uses a tiny configuration: the point is that the whole
        # Auto-fit-first path works headless, not that it exercises the full catalogue
        window.analysis_panel.autofit_view.set_config(
            AutoFitConfig(
                families=("polynomial",), workers=1, n_starts=2, cv_folds=None, time_limit=10.0
            )
        )

        example = resources.files("curvestudio_desktop") / "resources" / "example.csv"
        with resources.as_file(example) as path:
            if not window.import_file(path):
                print("SELFTEST FAIL: importing the bundled example was refused", file=sys.stderr)
                return 1
        model = window.table_view.model()
        if model is None or model.rowCount() < 10:
            print("SELFTEST FAIL: the data table did not populate", file=sys.stderr)
            return 1
        x_plot, _ = window.plot_view.scatter_data()
        if x_plot is None or len(x_plot) != model.rowCount():
            print("SELFTEST FAIL: the plot did not populate", file=sys.stderr)
            return 1

        xy = window.selection.xy()
        if xy is None:
            print("SELFTEST FAIL: no fit input for the example", file=sys.stderr)
            return 1
        bundle = fit_dataset(xy, FitSpec(model=PolynomialRef(degree=2)))
        if str(bundle.result.status) != "converged":
            print(f"SELFTEST FAIL: fit did not converge ({bundle.result.status})", file=sys.stderr)
            return 1
        pool_ok, pool_summary = _selftest_autofit(xy)
        if not pool_ok:
            print(f"SELFTEST FAIL: {pool_summary}", file=sys.stderr)
            return 1
        r2 = bundle.summary.result_summary.get("r2", "n/a") if bundle.summary else "n/a"

        # Auto-fit starts by itself on import (the product rule); wait for its result
        # without an exec() loop: pump the event loop until it lands in the project
        dataset_id = window.selection.active_id
        deadline = time.monotonic() + SELFTEST_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            app.processEvents()
            if dataset_id is not None and window.analysis.stored_autofit(dataset_id) is not None:
                break
            time.sleep(0.02)
        result = None if dataset_id is None else window.analysis.stored_autofit(dataset_id)
        if result is None:
            print(
                "SELFTEST FAIL: the auto-started Auto-fit produced no result within "
                f"{SELFTEST_TIMEOUT_SECONDS:.0f} s (mode={window.analysis.mode}, "
                f"run in progress={window.analysis.is_autofit_running()})",
                file=sys.stderr,
            )
            return 1
        top = result.ranked[0] if result.ranked else None
        top_text = (
            f", autofit ({len(result.ranked)} fitted, top: {top.model_id} R² {top.metrics.r2:.4f})"
            if top is not None and top.metrics.r2 is not None
            else f", autofit ({len(result.ranked)} fitted)"
        )
        print(
            f"SELFTEST OK window, table ({model.rowCount()} rows), plot "
            f"({len(x_plot)} points), fit ({bundle.result.status}, R² {r2}){top_text}, "
            f"{pool_summary}"
        )
        return 0
