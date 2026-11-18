"""Console entry: ``curvestudio-desktop`` and ``python -m curvestudio_desktop``."""

import argparse
import sys
from multiprocessing import freeze_support

from curvestudio_desktop import __version__


def main(argv: list[str] | None = None) -> int:
    # First call, before anything else runs. The Auto-fit engine drives a spawn
    # process pool, and a frozen (PyInstaller) build re-executes this program
    # for every worker: freeze_support() intercepts that re-execution and runs
    # the worker loop instead of the app — without it the pool would fork-bomb.
    # Unfrozen it is a no-op.
    freeze_support()
    parser = argparse.ArgumentParser(
        prog="curvestudio-desktop", description="ConcreteInfo CurveStudio desktop app"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="headless smoke test: build the window, load the bundled example, "
        "run one fit and one Auto-fit through the process pool",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        from curvestudio_desktop.app import run_selftest

        return run_selftest()
    from curvestudio_desktop.app import run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
