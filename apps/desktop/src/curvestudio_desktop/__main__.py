"""Console entry: ``curvestudio-desktop`` and ``python -m curvestudio_desktop``."""

import argparse
import sys

from curvestudio_desktop import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="curvestudio-desktop", description="ConcreteInfo CurveStudio desktop app"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="headless smoke test: build the window, load the bundled example, run one fit",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        from curvestudio_desktop.app import run_selftest

        return run_selftest()
    from curvestudio_desktop.app import run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
