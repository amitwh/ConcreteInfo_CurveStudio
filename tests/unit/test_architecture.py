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


SOURCE_ROOTS = tuple(sorted((Path(__file__).resolve().parents[2] / "packages").glob("*/src")))
"""Every workspace package's ``src`` directory (``packages/*/src``)."""

DYNAMIC_CALLS = frozenset({"eval", "exec", "parse_expr", "sympify"})
"""Names that run text as code; ``compile`` is checked separately (``re.compile`` is fine)."""

DYNAMIC_MODULES = frozenset({"pickle", "cPickle", "dill", "marshal"})


def find_dynamic_execution(source: str) -> list[str]:
    """Return uses of dynamic execution in ``source``: eval/exec/parse_expr/sympify/compile/pickle.

    ``eval``/``exec``/``parse_expr``/``sympify`` are flagged as a bare name or as an attribute
    (``sympy.sympify``). The builtin ``compile`` is flagged only as a bare call, so
    ``re.compile`` is allowed. Importing or naming ``pickle`` is flagged. ``lambdify`` is
    allowed: it is fed source printed from a SymPy tree with grammar-checked identifiers.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and (node.id in DYNAMIC_CALLS or node.id in DYNAMIC_MODULES):
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in DYNAMIC_CALLS:
            found.append(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "compile":
                found.append("compile")
        elif isinstance(node, ast.Import | ast.ImportFrom):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                found.extend(a.name for a in node.names if a.name in DYNAMIC_CALLS)
                if node.module and node.level == 0:
                    modules.append(node.module)
            found.extend(m for m in modules if m.split(".")[0] in DYNAMIC_MODULES)
    return found


def test_no_dynamic_execution_in_source() -> None:
    files = sorted(path for root in SOURCE_ROOTS for path in root.rglob("*.py"))
    assert files, "no python files found under packages/*/src; the guard would be vacuous"
    offenders: list[str] = []
    for path in files:
        for name in find_dynamic_execution(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {name}")
    assert not offenders, f"dynamic execution in source: {offenders}"


def test_dynamic_execution_checker_flags_each_spelling() -> None:
    assert find_dynamic_execution("eval('1')") == ["eval"]
    assert find_dynamic_execution("exec('1')") == ["exec"]
    assert find_dynamic_execution("import builtins\nbuiltins.eval('1')") == ["eval"]
    assert find_dynamic_execution("from sympy import sympify") == ["sympify"]
    assert find_dynamic_execution("sympy.sympify('x')") == ["sympify"]
    assert find_dynamic_execution("compile('1', '', 'eval')") == ["compile"]
    assert find_dynamic_execution("import pickle") == ["pickle"]
    assert find_dynamic_execution("from pickle import loads") == ["pickle"]


def test_dynamic_execution_checker_allows_safe_spellings() -> None:
    assert find_dynamic_execution("import re\nre.compile('a')") == []
    assert find_dynamic_execution("import ast\nast.literal_eval('1')") == []
    assert find_dynamic_execution("import sympy\nsympy.lambdify([], 1)") == []
