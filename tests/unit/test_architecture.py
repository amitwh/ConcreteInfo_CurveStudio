"""Architecture guard: curvestudio-core must never import GUI or web frameworks."""

import ast
from pathlib import Path

FORBIDDEN_MODULES = frozenset({"PySide6", "fastapi", "sqlalchemy", "starlette", "uvicorn"})

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"


def find_forbidden_imports(source: str) -> list[str]:
    """Return forbidden top-level modules imported by ``source`` (in import order)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in FORBIDDEN_MODULES:
                found.append(top)
    return found


def test_core_has_no_gui_or_web_imports() -> None:
    offenders: list[str] = []
    for path in sorted(CORE_SRC.rglob("*.py")):
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
