"""``--startup-timing``: one line per start-up phase, seconds from process start.

Enabled by the console entry point before anything else is imported; every phase the app
passes prints ``<name> <seconds>`` on stdout, flushed, and the app then runs normally.
With the flag absent every call is a no-op — behaviour is byte for byte the ordinary run's.
The names may grow; one phase per line is the contract the measurement scripts parse.
"""

import sys
import time

__all__ = ["enable", "mark"]

_enabled = False
_t0 = time.perf_counter()
"""The earliest moment this module was imported — the entry point imports it first, so
this is process start to within the interpreter's own start-up."""


def enable() -> None:
    """Turn the phase lines on (once, before the app is imported)."""
    global _enabled
    _enabled = True


def mark(name: str) -> None:
    """Print one phase line: ``name`` and seconds since process start."""
    if _enabled:
        print(f"{name} {time.perf_counter() - _t0:.6f}", file=sys.stdout, flush=True)
