"""PyInstaller entry script: run the installed console entry point's ``main``.

This is what the frozen executable executes, so it behaves exactly like
``curvestudio-desktop``. The spawn guard lives in ``main`` itself
(``multiprocessing.freeze_support()`` before anything else): a frozen Auto-fit
worker process re-executes this file's program, and only that call stops the
child from re-running the application instead of the worker loop.
"""

import sys

from curvestudio_desktop.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
