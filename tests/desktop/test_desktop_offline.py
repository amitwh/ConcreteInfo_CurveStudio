"""The desktop app is offline by design: nothing under ``curvestudio_desktop`` opens a network
connection.

The owner's rule: "Desktop app will be completely free, without login, it only sends the
install data, nothing else." The install record is a later project and is not written yet.
When it lands it will add exactly one explicitly allowlisted import (one module, in one
file) to this test — the guard itself is not to be loosened, and no other network import
is ever allowed. Until then the allowlist is empty, so any import of a module in
``NETWORK_MODULES`` fails this test with the file and line.

The check parses the package's source with :mod:`ast`; it never imports the modules it
inspects, so it also catches an import that only runs on a code path the tests never take.
"""

import ast
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
"""Empty on purpose. The install-record project adds one ``(file, module)`` entry here, and
nothing else ever does."""


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
    assert not offenders, "the desktop app is offline; network imports found:\n" + "\n".join(
        offenders
    )


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
