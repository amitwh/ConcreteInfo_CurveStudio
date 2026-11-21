"""Smoke test: the curvestudio package is importable and exposes its version."""

from importlib.metadata import version

import curvestudio


def test_version_is_exposed_and_matches_the_distribution() -> None:
    """``curvestudio.__version__`` must track pyproject.toml, so bundles and the web app
    report the same version as the installed distribution."""
    assert curvestudio.__version__ == version("curvestudio-core")
