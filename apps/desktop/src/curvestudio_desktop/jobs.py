"""JobRunner: move slow work (import, save, open) off the GUI thread.

Generic API, designed to also host Auto-fit jobs later (progress + cancel +
result signals)::

    handle = runner.submit(fn, *args, on_result=..., on_error=..., on_progress=...)
    handle.cancel()

``fn`` runs on a QThreadPool worker; ``on_result`` / ``on_error`` / ``on_progress``
always run on the GUI thread (queued signal delivery, because the signals object
lives on the runner's thread). A job function may declare a ``progress`` parameter
— it receives a ``(done, total)`` reporter, injected only when ``on_progress`` was
given. Cancellation is cooperative: ``cancel()`` marks the handle and the runner
drops the job's callbacks; a job already running is never interrupted mid-flight
(these jobs are side-effect-free reads — parsing, serialising — so dropping the
result is always safe).

``JobRunner(sync=True)`` runs everything on the calling thread and invokes the
callbacks inline — the deterministic mode tests and the selftest use.
"""

import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, SignalInstance

ProgressCallback = Callable[[int, int], None]


def _as_handle(handle: object) -> JobHandle:
    """Signals carry the handle as ``object``; it always comes back unchanged."""
    return cast("JobHandle", handle)


class JobHandle:
    """The caller's grip on one submitted job."""

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._done_event = threading.Event()

    def cancel(self) -> None:
        """Ask the runner to drop this job's callbacks (cooperative)."""
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    def _finish(self) -> None:
        self._done_event.set()


class _JobSignals(QObject):
    """Emitted from the worker thread; delivered queued on the GUI thread."""

    succeeded = Signal(object, object)  # (handle, result)
    failed = Signal(object, object)  # (handle, exception)
    progressed = Signal(object, int, int)  # (handle, done, total)
    settled = Signal(object)  # (handle,) — always, also for cancelled jobs


@dataclass
class _JobSpec:
    handle: JobHandle
    on_result: Callable[[Any], None]
    on_error: Callable[[BaseException], None]
    on_progress: ProgressCallback | None


class _Runnable(QRunnable):
    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        signals: _JobSignals,
        handle: JobHandle,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._signals = signals
        self._handle = handle

    def _emit(self, signal: SignalInstance, *args: object) -> None:
        """Emit from the worker thread, tolerating a torn-down signal source.

        A worker can still be finishing when the application tears its Qt objects
        down (closing the window mid-fit, or a headless run ending). Emitting on the
        already-deleted C++ object raises ``RuntimeError`` inside the thread, which
        the thread hook reports as a crash *after* the work itself succeeded. The
        delivery is pointless by then — the receiver is gone — so the emit is
        best-effort, and only that exact condition is swallowed.
        """
        try:
            signal.emit(*args)
        except RuntimeError as exc:
            if "has been deleted" not in str(exc):
                raise

    def run(self) -> None:  # QThreadPool worker thread
        try:
            result = self._fn(*self._args, **self._kwargs)
        except BaseException as exc:  # delivered on the GUI thread, never re-raised here
            if not self._handle.cancelled:
                self._emit(self._signals.failed, self._handle, exc)
        else:
            if not self._handle.cancelled:
                self._emit(self._signals.succeeded, self._handle, result)
        finally:
            self._handle._finish()
            self._emit(self._signals.settled, self._handle)


def _declares_progress(fn: Callable[..., Any]) -> bool:
    """Whether ``fn`` can accept an injected ``progress`` keyword."""
    try:
        signature = inspect.signature(fn)
    except TypeError, ValueError:
        return False
    if "progress" in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


class JobRunner(QObject):
    """Runs jobs off the GUI thread and delivers their outcomes on it."""

    busy_changed = Signal(bool)
    """Flips True when the first job starts and False when the last one settles."""

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        sync: bool = False,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(parent)
        self._sync = sync
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._jobs: dict[JobHandle, _JobSpec] = {}

    def is_busy(self) -> bool:
        return bool(self._jobs)

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None],
        on_error: Callable[[BaseException], None],
        on_progress: ProgressCallback | None = None,
    ) -> JobHandle:
        """Run ``fn(*args)`` off-thread; deliver its outcome on the GUI thread."""
        handle = JobHandle()
        signals = _JobSignals()
        kwargs: dict[str, Any] = {}
        if on_progress is not None and _declares_progress(fn):
            kwargs["progress"] = lambda done, total: signals.progressed.emit(handle, done, total)
        self._jobs[handle] = _JobSpec(handle, on_result, on_error, on_progress)
        # slots of `self` (a QObject on the GUI thread): worker emissions queue
        signals.succeeded.connect(self._on_succeeded)
        signals.failed.connect(self._on_failed)
        signals.progressed.connect(self._on_progress)
        signals.settled.connect(self._on_settled)
        if len(self._jobs) == 1:
            self.busy_changed.emit(True)
        if self._sync:
            _Runnable(fn, args, kwargs, signals, handle).run()
        else:
            self._pool.start(_Runnable(fn, args, kwargs, signals, handle))
        return handle

    # -- GUI-thread slots --------------------------------------------------------------------

    def _on_succeeded(self, handle: object, result: object) -> None:
        spec = self._spec_of(handle)
        if spec is not None and not spec.handle.cancelled:
            spec.on_result(result)

    def _on_failed(self, handle: object, error: object) -> None:
        spec = self._spec_of(handle)
        if spec is not None and not spec.handle.cancelled and isinstance(error, BaseException):
            spec.on_error(error)

    def _on_progress(self, handle: object, done: int, total: int) -> None:
        spec = self._spec_of(handle)
        if spec is not None and spec.on_progress is not None and not spec.handle.cancelled:
            spec.on_progress(done, total)

    def _on_settled(self, handle: object) -> None:
        self._jobs.pop(_as_handle(handle), None)
        if not self._jobs:
            self.busy_changed.emit(False)

    # -- helpers -----------------------------------------------------------------------------

    def _spec_of(self, handle: object) -> _JobSpec | None:
        return self._jobs.get(_as_handle(handle))
