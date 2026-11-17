"""Job workers used by the API tests. Top-level so a spawned process can import them."""

import os
import time
from typing import Any

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
