"""Smoke test: the curvestudio package is importable and exposes its version."""

import curvestudio


def test_version_is_exposed() -> None:
    assert curvestudio.__version__ == "0.1.0.dev0"
