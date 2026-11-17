"""Application wiring: QApplication factory, theme, crash guard and the selftest.

``run_app`` is the real entry point; ``run_selftest`` is the headless smoke used
by CI and packaging (D4): it builds the whole main window offscreen, loads the
bundled example, runs one fit through the workflow layer and reports PASS/FAIL
on stdout. It never shows a window and exits 0 only when everything worked.
"""

import os
import sys
import threading
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from curvestudio_desktop import APP_NAME, ORG_NAME, __version__

if TYPE_CHECKING:
    from curvestudio_desktop.views.main_window import MainWindow

Shower = Callable[[str], None]


def create_application(argv: list[str] | None = None) -> QApplication:
    """The one QApplication, with identity, DPI policy and a sensible app name."""
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


def build_main_window() -> MainWindow:
    """The main window wired to real (dialog-backed) providers."""
    from PySide6.QtCore import QSettings

    from curvestudio_desktop.controllers.project import ProjectController
    from curvestudio_desktop.views.dialogs import (
        DialogConfirmationProvider,
        DialogPathProvider,
        MessageBoxMessenger,
    )
    from curvestudio_desktop.views.main_window import MainWindow

    controller = ProjectController(
        confirmation=DialogConfirmationProvider(),
        paths=DialogPathProvider(),
        messenger=MessageBoxMessenger(),
        settings=QSettings(),
    )
    return MainWindow(controller)


def run_app(argv: list[str] | None = None) -> int:
    """Start the application: theme, crash guard, main window, event loop."""
    from curvestudio_desktop.theme import apply_theme

    app = create_application(argv)
    apply_theme(app)
    install_crash_guard()
    window = build_main_window()
    window.show()
    return int(app.exec())


def run_selftest(argv: list[str] | None = None) -> int:
    """Headless smoke test: window + bundled example + one fit. Exit 0 is pass.

    Used by CI and the packaging build. Prints each step so a failure says where.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # selftest is headless by design
    from importlib import resources

    from curvestudio.workflow import FitSpec, fit_dataset
    from curvestudio.workflow.spec import PolynomialRef

    app = create_application(argv if argv is not None else [sys.argv[0]])
    from curvestudio_desktop.theme import apply_theme

    apply_theme(app)
    window = build_main_window()

    with resources.as_file(
        resources.files("curvestudio_desktop") / "resources" / "example.csv"
    ) as example:
        if not window.import_file(example):
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
    r2 = bundle.summary.result_summary.get("r2", "n/a") if bundle.summary else "n/a"
    print(
        f"SELFTEST OK window, table ({model.rowCount()} rows), plot "
        f"({len(x_plot)} points), fit ({bundle.result.status}, R² {r2})"
    )
    return 0
