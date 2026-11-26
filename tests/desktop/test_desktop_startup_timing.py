"""``--startup-timing``: one line per phase, seconds from process start (P6 Task 6).

With the flag the app prints ``<phase> <seconds>`` per start-up phase to stdout, flushed,
then runs normally; without it nothing is printed and behaviour is the ordinary run's.
The phased run is the real app (event loop and all), read until the last phase lands and
then stopped — the flag must never change what the app does, only report it.
"""

import os
import subprocess
import sys
import time

PHASES = ("t0", "app_module", "qapp", "window_built", "window_shown", "model_list", "pool_warm")


def parse_phases(stdout: str) -> dict[str, float]:
    phases: dict[str, float] = {}
    for line in stdout.splitlines():
        name, _, seconds = line.partition(" ")
        if name and seconds and _is_number(seconds):
            phases[name] = float(seconds)
    return phases


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def run_phased_app(timeout_s: float = 240.0) -> str:
    """Start the real app with the flag; collect stdout until every phase printed."""
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    process = subprocess.Popen(
        [sys.executable, "-m", "curvestudio_desktop", "--startup-timing"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    collected = ""
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            assert process.poll() is None, "the app exited before its phases printed"
            line = process.stdout.readline()
            if line:
                collected += line
                if parse_phases(collected).keys() >= set(PHASES):
                    break
            else:
                time.sleep(0.05)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    collected += process.stdout.read() or ""
    return collected


def test_the_flag_prints_one_line_per_phase_and_the_app_keeps_running() -> None:
    collected = run_phased_app()
    phases = parse_phases(collected)
    for phase in PHASES:
        assert phase in phases, f"no {phase} line in {collected!r}"
    # the sequential phases come in order; pool_warm runs beside the library load, so
    # only its being after the window is asserted
    ordered = [phases[p] for p in PHASES if p != "pool_warm"]
    assert ordered == sorted(ordered)
    assert phases["pool_warm"] >= phases["window_shown"]
    assert all(t >= 0.0 for t in phases.values())
    assert "Traceback" not in collected


def test_without_the_flag_no_phase_lines_are_printed() -> None:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-m", "curvestudio_desktop", "--selftest"],
        capture_output=True,
        text=True,
        env=environment,
        timeout=240,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    assert result.returncode == 0, result.stderr
    assert not parse_phases(result.stdout), result.stdout
    assert "SELFTEST OK" in result.stdout
