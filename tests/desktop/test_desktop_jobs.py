"""JobRunner: off-thread work with GUI-thread callbacks, progress and cancel."""

import threading

import pytest

from curvestudio_desktop.jobs import JobRunner


def test_sync_runner_runs_inline_and_delivers_result() -> None:
    runner = JobRunner(sync=True)
    seen: dict[str, object] = {}
    handle = runner.submit(
        lambda a, b: a + b,
        2,
        3,
        on_result=lambda result: seen.__setitem__("result", result),
        on_error=lambda error: seen.__setitem__("error", error),
    )
    assert seen == {"result": 5}
    assert handle.done and not handle.cancelled


def test_sync_runner_routes_exceptions_to_on_error() -> None:
    runner = JobRunner(sync=True)

    def boom() -> int:
        raise ValueError("nope")

    seen: dict[str, object] = {}
    runner.submit(
        boom,
        on_result=lambda result: seen.__setitem__("result", result),
        on_error=lambda error: seen.__setitem__("error", error),
    )
    assert isinstance(seen.get("error"), ValueError)
    assert "result" not in seen


def test_async_runner_delivers_result_on_gui_thread(qtbot) -> None:
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    runner = JobRunner()
    results: list[object] = []
    errors: list[object] = []
    threads: list[object] = []
    runner.submit(
        lambda: 21 * 2,
        on_result=lambda result: (results.append(result), threads.append(QThread.currentThread())),
        on_error=errors.append,
    )
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    assert results == [42] and errors == []
    app = QApplication.instance()
    assert app is not None and threads[0] is app.thread()


def test_async_runner_delivers_errors_on_gui_thread(qtbot) -> None:
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    runner = JobRunner()

    def boom() -> int:
        raise RuntimeError("worker boom")

    errors: list[object] = []
    threads: list[object] = []
    runner.submit(
        boom,
        on_result=lambda _result: None,
        on_error=lambda error: (errors.append(error), threads.append(QThread.currentThread())),
    )
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)
    assert isinstance(errors[0], RuntimeError)
    app = QApplication.instance()
    assert app is not None and threads[0] is app.thread()


def test_async_progress_reaches_callback_in_order(qtbot) -> None:
    runner = JobRunner()

    def work(progress=None):  # type: ignore[no-untyped-def]
        for step in (1, 2, 3):
            progress(step, 3)
        return "done"

    results: list[object] = []
    progress_seen: list[tuple[int, int]] = []
    runner.submit(
        work,
        on_result=results.append,
        on_error=lambda _error: None,
        on_progress=lambda done, total: progress_seen.append((done, total)),
    )
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    assert progress_seen == [(1, 3), (2, 3), (3, 3)]


def test_no_progress_callback_means_no_injection(qtbot) -> None:
    """fn declares `progress` but on_progress is None: no kwarg is injected."""
    runner = JobRunner()

    def work(progress=None):  # type: ignore[no-untyped-def]
        assert progress is None
        return "ok"

    results: list[object] = []
    runner.submit(work, on_result=results.append, on_error=lambda _error: None)
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    assert results == ["ok"]


def test_cancel_drops_the_callbacks_of_a_finished_job(qtbot) -> None:
    runner = JobRunner()
    release = threading.Event()

    def slow() -> str:
        release.wait(timeout=5)
        return "late"

    results: list[object] = []
    errors: list[object] = []
    handle = runner.submit(slow, on_result=results.append, on_error=errors.append)
    handle.cancel()
    release.set()
    qtbot.waitUntil(lambda: handle.done, timeout=5000)
    assert results == [] and errors == []  # cancelled: nothing delivered


def test_busy_signal_flips_true_then_false(qtbot) -> None:
    runner = JobRunner()
    states: list[bool] = []
    runner.busy_changed.connect(states.append)
    results: list[object] = []
    runner.submit(lambda: "x", on_result=results.append, on_error=lambda _error: None)
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    assert states == [True, False]


def test_qthreadpool_exception_does_not_bypass_the_runner(qtbot) -> None:
    """A BaseException in the worker is delivered, never swallowed nor printed."""
    runner = JobRunner()

    def boom() -> int:
        raise KeyError("missing")

    errors: list[object] = []
    runner.submit(boom, on_result=lambda _r: None, on_error=errors.append)
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)
    assert isinstance(errors[0], KeyError)


@pytest.mark.parametrize("sync", [True, False])
def test_generic_api_shape(sync: bool, qtbot) -> None:
    """submit returns a JobHandle with cancel(); the documented signature holds."""
    runner = JobRunner(sync=sync)
    results: list[object] = []
    handle = runner.submit(
        str.upper,
        "ab",
        on_result=results.append,
        on_error=lambda _error: None,
    )
    if sync:
        assert results == ["AB"]
    else:
        qtbot.waitUntil(lambda: bool(results), timeout=5000)
        assert results == ["AB"]
    assert hasattr(handle, "cancel")
    handle.cancel()
    assert handle.cancelled
