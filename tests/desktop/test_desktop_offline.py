"""The desktop app is offline by design, with exactly one exception: its install record.

The owner's rule: "Desktop app will be completely free, without login, it only sends the
install data, nothing else." That one call is made by ``curvestudio_desktop.install_record``
over ``PySide6.QtNetwork`` and named nowhere else. Nothing under ``curvestudio_desktop``
may import any other network module, and that module may name no second destination: the
first test keeps the package free of Python's own network stacks (and the allowlist stays
empty), the second pins the single Qt import to the single URL.

Both checks parse the package's source with :mod:`ast`; they never import the modules they
inspect, so they also catch an import that only runs on a code path the tests never take.
"""

import ast
import re
from pathlib import Path

import curvestudio_desktop

NETWORK_MODULES = frozenset(
    {
        "urllib",
        "requests",
        "http",
        "httpx",
        "aiohttp",
        "socket",
        "ssl",
        "websocket",
        "websockets",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
    }
)
"""Top-level modules that can open a connection. Checked as written in the source (``import X``
and ``from X import …``), by top-level name, so ``http.client`` and ``urllib.request`` count."""

ALLOWED_NETWORK_IMPORTS: frozenset[tuple[str, str]] = frozenset()
"""Empty, and it stays empty: the install record goes over QtNetwork, which is not a member of
``NETWORK_MODULES``. The install record's own pin is ``QT_NETWORK_FILES`` below."""

QT_NETWORK_FILES = frozenset({"install_record.py"})
"""The one module of the package allowed to import ``PySide6.QtNetwork``."""

URL = re.compile(r"^https?://")
"""A string constant that starts like this names a network destination."""


def network_imports(package_dir: Path) -> list[str]:
    """Every network import under ``package_dir``, each as ``file:line: imports X``."""
    offenders: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(package_dir).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            else:
                continue
            for name in imported:
                top = name.split(".")[0]
                if top in NETWORK_MODULES and (relative, top) not in ALLOWED_NETWORK_IMPORTS:
                    offenders.append(f"{relative}:{node.lineno}: imports {name}")
    return offenders


def test_desktop_imports_no_network_module() -> None:
    package_dir = Path(curvestudio_desktop.__file__).resolve().parent
    offenders = network_imports(package_dir)
    assert not offenders, "network imports found:\n" + "\n".join(offenders)


def test_the_offline_guard_detects_a_planted_network_import(tmp_path: Path) -> None:
    """The guard passes on today's code, so prove it can fail: planted imports are named."""
    planted = tmp_path / "pkg"
    planted.mkdir()
    (planted / "evil.py").write_text(
        "import json\nimport socket\nfrom http.client import HTTPConnection\nimport ssl as tls\n",
        encoding="utf-8",
    )
    (planted / "clean.py").write_text("from pathlib import Path\n", encoding="utf-8")
    assert network_imports(planted) == [
        "evil.py:2: imports socket",
        "evil.py:3: imports http.client",
        "evil.py:4: imports ssl",
    ]


def qt_network_imports(package_dir: Path) -> dict[str, list[str]]:
    """``{file: [what was imported, …]}`` for every QtNetwork import under ``package_dir``."""
    found: dict[str, list[str]] = {}
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(package_dir).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "PySide6.QtNetwork":
                    names = [node.module]
                else:
                    names = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            hits = [name for name in names if name == "PySide6.QtNetwork"]
            if hits:
                found.setdefault(relative, []).extend(hits)
    return found


def url_literals(tree: ast.AST) -> list[str]:
    """Every string constant in ``tree`` that starts like a URL."""
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and URL.match(node.value)
    ]


def test_the_packages_only_network_call_is_the_install_record() -> None:
    package_dir = Path(curvestudio_desktop.__file__).resolve().parent

    # exactly one module imports QtNetwork, and it is the install record
    importers = qt_network_imports(package_dir)
    assert set(importers) == QT_NETWORK_FILES == {"install_record.py"}
    assert all(what == ["PySide6.QtNetwork"] for what in importers.values())

    # and it names exactly one network destination: the install-record URL constant
    record = ast.parse((package_dir / "install_record.py").read_text(encoding="utf-8"))
    assert url_literals(record) == []  # the URL itself lives in the package constant
    names = {node.id for node in ast.walk(record) if isinstance(node, ast.Name)}
    assert {name for name in names if name.endswith("URL")} == {"INSTALL_RECORD_URL"}
    package = ast.parse((package_dir / "__init__.py").read_text(encoding="utf-8"))
    assert url_literals(package) == [curvestudio_desktop.INSTALL_RECORD_URL]
    assert url_literals(package) == ["https://curvestudio.concreteinfo.cloud/api/v1/installs"]

    # the guard's own list is unchanged, nothing is allowlisted, and Python's network
    # stacks stay forbidden
    assert (
        frozenset(
            {
                "urllib",
                "requests",
                "http",
                "httpx",
                "aiohttp",
                "socket",
                "ssl",
                "websocket",
                "websockets",
                "ftplib",
                "smtplib",
                "telnetlib",
                "xmlrpc",
            }
        )
        == NETWORK_MODULES
    )
    assert frozenset() == ALLOWED_NETWORK_IMPORTS
    assert network_imports(package_dir) == []


def test_the_offline_guard_still_forbids_http_urllib_and_socket(tmp_path: Path) -> None:
    planted = tmp_path / "pkg"
    planted.mkdir()
    (planted / "again.py").write_text(
        "import http\nimport urllib.request\nimport socket\n", encoding="utf-8"
    )
    assert network_imports(planted) == [
        "again.py:1: imports http",
        "again.py:2: imports urllib.request",
        "again.py:3: imports socket",
    ]
