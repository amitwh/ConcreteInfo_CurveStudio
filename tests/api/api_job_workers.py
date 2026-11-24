"""Job workers used by the API tests. Top-level so a spawned process can import them."""

import os
import time
from typing import Any

from curvestudio_api.audit_worker import run_cross_validation_job
from curvestudio_api.worker import run_fit_job


def sleeping_worker(**_arguments: Any) -> dict[str, Any]:
    time.sleep(600)
    return {"ok": False, "code": "NEVER", "message": "unreachable"}


def crashing_worker(**_arguments: Any) -> dict[str, Any]:
    os._exit(13)


def crash_on_boom_worker(**arguments: Any) -> dict[str, Any]:
    """Dies hard for a dataset whose x column is called ``boom``; otherwise a normal fit
    (after a short wait, so a neighbouring job is still in flight when the crash happens)."""
    if arguments["x_name"] == "boom":
        time.sleep(0.5)
        os._exit(13)
    time.sleep(1.5)
    return run_fit_job(**arguments)


def slow_worker(**arguments: Any) -> dict[str, Any]:
    time.sleep(3)
    return run_fit_job(**arguments)


def victim_or_sleeper_worker(**arguments: Any) -> dict[str, Any]:
    """A 6 s fit for the dataset whose x column is ``victim``; everything else hangs."""
    if arguments["x_name"] == "victim":
        time.sleep(6)
        return run_fit_job(**arguments)
    time.sleep(600)
    return {"ok": False, "code": "NEVER", "message": "unreachable"}


def big_result_worker(**_arguments: Any) -> dict[str, Any]:
    return {"ok": True, "bundle": {"padding": "x" * 50_000}}


# -- import workers: signature (connection, memory_mb, arguments) ----------------------------


def sleeping_import_worker(connection: Any, _memory_mb: int, _arguments: dict[str, Any]) -> None:
    from curvestudio_api.worker import STARTED

    connection.send_bytes(STARTED)
    time.sleep(600)


def crashing_import_worker(connection: Any, _memory_mb: int, _arguments: dict[str, Any]) -> None:
    from curvestudio_api.worker import STARTED

    connection.send_bytes(STARTED)
    os._exit(9)


def memory_hog_import_worker(connection: Any, memory_mb: int, _arguments: dict[str, Any]) -> None:
    """Asks for far more memory than the cap allows, as a decompression bomb would."""
    import json

    from curvestudio_api.import_worker import limit_address_space
    from curvestudio_api.worker import STARTED

    connection.send_bytes(STARTED)
    assert limit_address_space(memory_mb), "this platform cannot cap the address space"
    try:
        hog = bytearray(8 * 1024**3)
        meta: dict[str, Any] = {"status": "ok", "size": len(hog)}
    except MemoryError:
        meta = {"status": "too_much_memory"}
    connection.send_bytes(json.dumps(meta).encode())
    connection.send_bytes(b"")


# -- Auto-fit workers ------------------------------------------------------------------------------


def sleeping_autofit_worker(**_arguments: Any) -> dict[str, Any]:
    """An Auto-fit run that never ends (it accepts the ``report=`` the real one gets)."""
    time.sleep(600)
    return {"ok": False, "code": "NEVER", "message": "unreachable"}


def child_spawning_worker(*, pid_file: str, **_arguments: Any) -> dict[str, Any]:
    """Starts a long-lived grandchild process, records its pid, then sleeps."""
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(pid_file, "w") as handle:
        handle.write(str(child.pid))
    time.sleep(600)
    return {"ok": False, "code": "NEVER", "message": "unreachable"}


# -- The audit worker ------------------------------------------------------------------------------


def slow_audit_worker(*, seconds: float = 1.5, **arguments: Any) -> dict[str, Any]:
    """The real cross-validation after a wait, so a neighbouring request finds the job in flight."""
    time.sleep(seconds)
    return run_cross_validation_job(**arguments)
