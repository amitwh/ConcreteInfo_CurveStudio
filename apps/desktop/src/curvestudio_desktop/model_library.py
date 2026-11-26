"""The model library: loaded once, in the background, after the window is up.

The window is built without the fitting stack (NumPy, SciPy, SymPy, pandas, the model
catalogue — the imports that held the first paint back for seconds), so it appears as soon
as Qt can draw it. Everything the stack provides arrives here instead: one load per
process, started when the first window is shown, publishing readiness through callbacks.
Qt owners marshal the callback to the GUI thread themselves (a queued signal); callbacks
registered after the load ran are called inline, on the caller's thread.

A command that needs the library before it is ready either waits (an import the loader is
already running blocks on the same module locks and finishes with them) or asks
:func:`is_ready` and says "still loading" — nothing crashes, nothing silently no-ops. The
load also warms the catalogues the pickers are built from and pre-starts the Auto-fit pool
(:func:`prewarm_autofit_pool`), so the first fit pays for none of it.
"""

import threading
from collections.abc import Callable

__all__ = [
    "is_ready",
    "load_now",
    "prewarm_autofit_pool",
    "when_ready",
]

LOADING_MESSAGE = "Loading the model library…"
"""What a window shows while the load runs (cleared when it lands)."""

READY = "the model library is ready"
"""The status message that replaces :data:`LOADING_MESSAGE` (also a test probe)."""

_MODULES = (
    "curvestudio.workflow.catalog",
    "curvestudio.discovery",
    "curvestudio.project",
    "curvestudio.tools.stats",
    "curvestudio_desktop.controllers.analysis_jobs",
    "curvestudio_desktop.controllers.analysis_store",
)
"""What the window's deferred paths share: the fitting stack and both catalogues. The
export and statistics writers stay on first use, exactly as before — they pull Matplotlib
and the reporting layer, which no start-up path ever needed."""

_lock = threading.Lock()
_loaded = False
_thread: threading.Thread | None = None
_callbacks: list[Callable[[], None]] = []
_done = threading.Event()
"""Set the moment the load completes: :func:`wait_until_ready` blocks on it."""


def _load() -> None:
    """Import the stack and warm both catalogues; idempotent."""
    global _loaded
    if _loaded:
        return
    from importlib import import_module

    catalogue = import_module("curvestudio.workflow.catalog")
    stats = import_module("curvestudio.tools.stats")
    for name in _MODULES:
        import_module(name)
    catalogue.list_models()  # the plugins load here, once
    stats.catalogue()
    with _lock:
        _loaded = True
        callbacks, _callbacks[:] = list(_callbacks), []
    _done.set()
    for callback in callbacks:
        callback()


def is_ready() -> bool:
    """Whether the library is loaded (a window built now needs no loading state)."""
    return _loaded


def when_ready(callback: Callable[[], None]) -> None:
    """Run ``callback`` once the library is ready — inline when it already is.

    Before that it runs on the loader's thread at the moment the load completes: Qt owners
    marshal to the GUI thread (a queued signal), as the main window does.
    """
    with _lock:
        if not _loaded:
            _callbacks.append(callback)
            return
    callback()


def load_now() -> None:
    """Load synchronously. Tests (the stack is imported around them anyway) and any caller
    that must have the library before it proceeds."""
    _load()


def wait_until_ready(timeout_s: float | None = None) -> bool:
    """Block until the library is ready, loading it here when no load was started.

    Returns ``False`` only on a timeout; a caller that never armed
    :func:`begin_loading` still gets a loaded library, just synchronously.
    """
    if not _thread_started():
        load_now()
    return _done.wait(timeout_s)


def _thread_started() -> bool:
    return _thread is not None and _thread.is_alive()


def prewarm_autofit_pool() -> None:
    """Start the shared Auto-fit pool's workers now, so the first run pays no start-up.

    The same pool :func:`curvestudio_desktop.controllers.analysis_jobs.run_autofit` uses,
    at the count its default configuration resolves to. Runs on the caller's thread; the
    app calls it from its own background thread, after the library has loaded, off the
    GUI thread.
    """
    from curvestudio.discovery import get_shared_pool

    get_shared_pool().prewarm()


def begin_loading() -> bool:
    """Start the one background load; True when the library is ready already.

    The app calls this when the first window is shown. A second call does nothing (the
    first load owns the thread); ready callbacks registered meanwhile still run.
    """
    global _thread
    with _lock:
        if _loaded:
            return True
        if _thread is not None and _thread.is_alive():
            return False
        _thread = threading.Thread(target=_load, name="curvestudio-model-library", daemon=True)
        _thread.start()
        return False
