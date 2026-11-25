"""Warm job workers: one long-lived process per worker slot, one fresh process per job.

A worker is started once (``spawn``, so it is a clean interpreter) and imports the whole
fitting stack; after that it only ever *forks*. Every job runs in a child of the worker,
and the worker itself never executes job code, so each job starts from exactly the
post-import state a freshly spawned process would have — nothing a job does (globals it
sets, environment changes, working-directory moves, files it leaves) can reach the next
job, because all of it dies with the child. The per-job interpreter cost disappears; the
per-job isolation does not.

The wire protocol, over one duplex pipe between the manager and each worker:

* manager → worker (``Connection.send``, multiprocessing' own serialisation):
  ``(TICKET, worker function, arguments, progress?, job id)`` — one job to run — or
  ``(RETIRE,)`` — stop; you will get no more tickets.
* job child → manager (raw bytes): ``STARTED`` + its pid (it is up and about to work; the
  manager counts the job's wall-clock budget from here and can kill the child's process
  group), ``PROGRESS`` + JSON snapshots, then the result as raw JSON.
* worker → manager (raw bytes): ``EXITED`` + the reaped child's exit status, the jobs it
  has run, its own resident size and the job id — the manager's signal that the child is
  gone, and where it decides whether this worker has reached its recycle limits.

Only the worker deserialises, and only what the manager sent it; everything the manager
reads is raw bytes or JSON. Only the worker (after reaping a dead child) and the child
ever write on the pipe, and never at the same time, so messages cannot interleave. A
child leads its own session (``setsid``), so one ``killpg`` ends it and everything it
started — an Auto-fit run's worker pool included — without touching the worker. A child
also dies with its worker (``PR_SET_PDEATHSIG`` where Linux has it) and both die with the
API process, so a hard kill of the server cannot leak a fitter.
"""

import contextlib
import json
import multiprocessing
import os
import signal
import struct
import sys
import threading
import traceback
from multiprocessing.connection import Connection
from typing import Any

from curvestudio_api.audit_worker import run_cross_validation_job
from curvestudio_api.autofit_worker import run_autofit_job, run_open_row_job
from curvestudio_api.fit_tools_worker import (
    run_advice_job,
    run_discernibility_job,
    run_simulation_job,
)
from curvestudio_api.worker import PROGRESS, STARTED, JobWorker, run_fit_job

__all__ = [
    "EXITED",
    "RETIRE",
    "TICKET",
    "WORKER_FUNCTIONS",
    "child_pid_of_hello",
    "parse_exited",
    "send_ticket",
    "warm_worker_main",
]

TICKET = "ticket"
"""Manager → worker: the marker of a job to run (see :func:`send_ticket`)."""

RETIRE = "retire"
"""Manager → worker: stop; the worker exits and its slot is refilled on demand."""

EXITED = b"X"
"""Worker → manager: ``EXITED`` + exit status, jobs run, resident KB and the job id."""

_PID = struct.Struct("<i")
_EXIT = struct.Struct("<iiq")
"""The reaped child's exit status (negative: the signal), jobs run and resident KB."""

# Importing these functions is the point of a warm worker: a forked child finds the whole
# stack already imported, whichever kind of job it is handed. The manager takes this as its
# map of job kinds to workers (tests override entries through the app factory).
WORKER_FUNCTIONS = {
    "fit": run_fit_job,
    "autofit": run_autofit_job,
    "open_row": run_open_row_job,
    "audit": run_cross_validation_job,
    "discern": run_discernibility_job,
    "simulate": run_simulation_job,
    "advise": run_advice_job,
}


def send_ticket(
    connection: Connection,
    worker: JobWorker,
    arguments: dict[str, Any],
    reports_progress: bool,
    job_id: Any,
) -> bool:
    """Hand one job to a warm worker. False when the worker's pipe is already gone.

    ``Connection.send`` — multiprocessing' own serialisation, the same one a spawned
    process' arguments cross with — carries a top-level function reference and the job's
    own arguments between two processes of ours. Only the worker deserialises, and only
    what the manager sent it: results and reports come back as raw JSON or packed bytes,
    so nothing the API process reads is ever dynamically loaded (the upload parser's
    rule — compare :mod:`curvestudio_api.import_worker` — is kept).
    """
    try:
        connection.send((TICKET, worker, arguments, reports_progress, job_id))
        return True
    except OSError, ValueError:
        return False


def child_pid_of_hello(data: bytes) -> int | None:
    """The pid a job child announced in its hello, or ``None`` when it is not a hello."""
    if not data.startswith(STARTED) or len(data) < 1 + _PID.size:
        return None
    (pid,) = _PID.unpack_from(data, 1)
    return int(pid)


def parse_exited(data: bytes) -> tuple[int, int, int, bytes] | None:
    """``(exit status, jobs run, resident KB, job id bytes)`` of an ``EXITED`` report."""
    if not data.startswith(EXITED) or len(data) < 1 + _EXIT.size + 16:
        return None
    status, jobs_done, rss_kb = _EXIT.unpack_from(data, 1)
    return status, jobs_done, rss_kb, data[1 + _EXIT.size :]


def _lead_own_process_group() -> bool:
    """Make this process the leader of a new session, so the parent can stop it and everything
    it started (the Auto-fit worker pool) with one ``killpg``. POSIX only; false elsewhere."""
    setsid = getattr(os, "setsid", None)
    if setsid is None:
        return False
    try:
        setsid()
    except OSError:
        return False  # the parent falls back to killing this process alone
    return True


def _exit_with_parent(leads_group: bool) -> None:
    """Die, together with every child, when the API process disappears without a clean stop.

    A job child that has children is not a daemon, so nothing else would end it (or its
    pool) after a hard kill of the API process.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()  # returns when the parent has exited
        kill_group = getattr(os, "killpg", None)
        if leads_group and kill_group is not None:
            kill_group(os.getpgrp(), signal.SIGKILL)
        os._exit(1)

    threading.Thread(target=watch, name="curvestudio-parent-watch", daemon=True).start()


def _die_with_the_worker(parent_pid: int) -> None:
    """Ask the kernel to kill this child the moment its warm worker dies (Linux
    ``PR_SET_PDEATHSIG``). The check afterwards closes the window where the worker could
    have died between the fork and the request, which would leave no signal to come."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:
        return  # not Linux, or no prctl: the pipe breaking still ends the child
    if os.getppid() != parent_pid:
        os._exit(1)


def _send(connection: Connection, data: bytes) -> bool:
    """Best-effort send: false when the manager's end is gone (it stopped, or the result
    was refused). A job whose messages cannot be delivered is lost either way."""
    try:
        connection.send_bytes(data)
        return True
    except OSError, ValueError:
        return False


def _job_child(
    connection: Connection,
    worker: JobWorker,
    arguments: dict[str, Any],
    reports_progress: bool,
) -> int:
    """The forked child of a warm worker: announce, run the job, report, exit.

    Returns the exit status for ``os._exit`` — the child must never unwind into the
    worker's loop or run its atexit handlers. An exception in ``worker`` is reported as
    ``INTERNAL_ERROR`` (the traceback goes to stderr); a hard crash shows up to the
    manager as the worker's ``EXITED`` report with no result before it.
    """
    leads_group = _lead_own_process_group()
    _die_with_the_worker(os.getppid())
    _exit_with_parent(leads_group)
    try:
        if not _send(connection, STARTED + _PID.pack(os.getpid())):
            return 1
        if reports_progress:
            arguments = {
                **arguments,
                "report": lambda snapshot: _send(
                    connection, PROGRESS + json.dumps(snapshot, allow_nan=False).encode()
                ),
            }
        try:
            result: dict[str, Any] = worker(**arguments)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            result = {
                "ok": False,
                "code": "INTERNAL_ERROR",
                "message": "the fit job failed unexpectedly",
            }
        return 0 if _send(connection, json.dumps(result, allow_nan=False).encode()) else 1
    finally:
        with contextlib.suppress(OSError):
            connection.close()


def _exit_status(wait_status: int) -> int:
    """The conventional status of ``os.waitpid``: the exit code, or ``-signal``."""
    if os.WIFEXITED(wait_status):
        return os.WEXITSTATUS(wait_status)
    if os.WIFSIGNALED(wait_status):
        return -os.WTERMSIG(wait_status)
    return -255  # stopped or otherwise unrecognised: report it as an abnormal death


def _rss_kb() -> int:
    """This process's resident size in KB (0 where it cannot be read)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError, ValueError, IndexError:
        return 0
    return 0


def warm_worker_main(connection: Connection) -> None:
    """Entry point of one warm worker: import the stack once, then fork a child per job.

    The worker never runs job code itself, so what a job leaves behind cannot outlive it.
    After reaping each child it reports the exit status, the number of jobs it has run and
    its own resident size, from which the manager decides when to recycle it. It exits on
    ``RETIRE``, when the pipe closes, or with the API process (see ``_exit_with_parent``).
    """
    leads_group = _lead_own_process_group()
    _exit_with_parent(leads_group)
    jobs_done = 0
    while True:
        try:
            message = connection.recv()
        except EOFError, OSError:
            return  # the manager is gone or stopped this worker
        if not isinstance(message, tuple) or not message:
            continue  # not a message shaped for the worker: nothing to do
        if message[0] == RETIRE:
            return
        if message[0] != TICKET:
            continue
        _tag, worker, arguments, reports_progress, job_id = message
        pid = os.fork()
        if pid == 0:  # the child: everything from here on dies with this job
            code = 1
            try:
                code = _job_child(connection, worker, arguments, reports_progress)
            except BaseException:
                traceback.print_exc(file=sys.stderr)
            finally:
                os._exit(code)
        jobs_done += 1
        _reaped, wait_status = os.waitpid(pid, 0)
        _send(
            connection,
            EXITED + _EXIT.pack(_exit_status(wait_status), jobs_done, _rss_kb()) + job_id.bytes,
        )
