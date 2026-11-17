"""Architecture guard: curvestudio-core must never import GUI or web frameworks."""

import ast
from pathlib import Path

FORBIDDEN_MODULES = frozenset({"PySide6", "fastapi", "sqlalchemy", "starlette", "uvicorn"})

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"


FORBIDDEN_SUBMODULES = frozenset({"scipy.odr"})
"""Dotted modules that are forbidden although their top-level package is allowed."""


def _imported_names(node: ast.AST) -> list[str]:
    """Absolute module names a single import statement brings in (``from a import b`` -> a.b)."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
    return []


def find_forbidden_imports(source: str) -> list[str]:
    """Return forbidden modules imported by ``source`` (in import order).

    Top-level forbidden packages are reported by their top-level name; forbidden
    submodules (``scipy.odr``) are reported by their dotted name, whether imported as
    ``import scipy.odr``, ``from scipy.odr import X`` or ``from scipy import odr``.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        hits: list[str] = []
        for name in _imported_names(node):
            top = name.split(".")[0]
            if top in FORBIDDEN_MODULES:
                hits.append(top)
            hits.extend(
                sub for sub in FORBIDDEN_SUBMODULES if name == sub or name.startswith(sub + ".")
            )
        found.extend(dict.fromkeys(hits))  # one report per name per statement
    return found


def test_core_has_no_gui_or_web_imports() -> None:
    files = sorted(CORE_SRC.rglob("*.py"))
    assert files, f"no python files found under {CORE_SRC}; the guard would be vacuous"
    offenders: list[str] = []
    for path in files:
        for module in find_forbidden_imports(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(CORE_SRC)}: {module}")
    assert not offenders, f"forbidden imports in core: {offenders}"


def test_checker_flags_forbidden_imports() -> None:
    assert find_forbidden_imports("import PySide6") == ["PySide6"]
    assert find_forbidden_imports("import PySide6.QtCore as qt") == ["PySide6"]
    assert find_forbidden_imports("from fastapi import FastAPI") == ["fastapi"]
    assert find_forbidden_imports("from sqlalchemy.orm import Session") == ["sqlalchemy"]
    assert find_forbidden_imports("import os, starlette.routing, uvicorn") == [
        "starlette",
        "uvicorn",
    ]


def test_checker_ignores_allowed_and_relative_imports() -> None:
    snippet = "import numpy as np\nfrom scipy import optimize\nfrom . import sibling\n"
    assert find_forbidden_imports(snippet) == []


def test_forbidden_odr_import() -> None:
    """``scipy.odr`` is deprecated; no module under curvestudio may import it, in any spelling."""
    assert find_forbidden_imports("import scipy.odr") == ["scipy.odr"]
    assert find_forbidden_imports("import scipy.odr as odr") == ["scipy.odr"]
    assert find_forbidden_imports("from scipy.odr import ODR") == ["scipy.odr"]
    assert find_forbidden_imports("from scipy import odr") == ["scipy.odr"]
    assert find_forbidden_imports("from scipy import optimize, odr as o") == ["scipy.odr"]
    assert find_forbidden_imports("import scipy.optimize\nfrom scipy import stats") == []

    offenders: list[str] = []
    for path in sorted(CORE_SRC.rglob("*.py")):
        if "scipy.odr" in find_forbidden_imports(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(CORE_SRC)))
    assert not offenders, f"scipy.odr imported in core: {offenders}"
