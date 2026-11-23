"""The allow-list for the tools layer, which import-linter 2.16 cannot express.

The layer may import the standard library, NumPy, SciPy, pydantic and the core layers below
it, and nothing else. ``import-linter`` here has no ``allowed`` contract type, so its
forbidden contract in ``pyproject.toml`` names the likely trespassers and this test holds
the full list.
"""

import ast
import sys
from pathlib import Path

CORE_SRC = Path(__file__).resolve().parents[3] / "packages" / "core" / "src"
TOOLS_SRC = CORE_SRC / "curvestudio" / "tools"

ALLOWED_TOP_LEVEL = frozenset({"curvestudio", "numpy", "scipy", "pydantic"})
ALLOWED_CORE_LAYERS = frozenset(
    {
        "curvestudio.tools",
        "curvestudio.data",
        "curvestudio.fitting",
        "curvestudio.models",
        "curvestudio.statistics",
        "curvestudio.diagnostics",
        "curvestudio.discovery",
        "curvestudio.workflow",
    }
)


def outside_allowlist(source: str) -> list[str]:
    """Absolute imports in ``source`` the tools layer may not make, in source order.

    A relative import is always fine. ``import x.y`` is judged by ``x`` and, when ``x`` is
    ``curvestudio``, by ``x.y``. ``from x import a, b`` is judged by ``x``; when ``x`` is
    ``curvestudio`` itself, each of ``curvestudio.a`` and ``curvestudio.b`` is judged too.
    """
    offenders: list[str] = []

    def judge(name: str) -> None:
        parts = name.split(".")
        top, layer = parts[0], ".".join(parts[:2])
        outside = top not in sys.stdlib_module_names and top not in ALLOWED_TOP_LEVEL
        wrong_layer = top == "curvestudio" and len(parts) > 1 and layer not in ALLOWED_CORE_LAYERS
        if outside or wrong_layer:
            offenders.append(name)

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                judge(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module:
                judge(module)
            if module == "curvestudio":
                for alias in node.names:
                    judge(f"curvestudio.{alias.name}")
    return offenders


def test_tools_import_only_the_standard_library_numpy_scipy_pydantic_and_the_core() -> None:
    files = sorted(TOOLS_SRC.rglob("*.py"))
    assert files, f"no python files under {TOOLS_SRC}; the allow-list would be vacuous"
    offenders = [
        f"{path.relative_to(TOOLS_SRC)}: {name}"
        for path in files
        for name in outside_allowlist(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"imports outside the tools layer's allow-list: {offenders}"


def test_checker_flags_each_trespass_and_allows_the_rest() -> None:
    assert outside_allowlist("import sympy") == ["sympy"]
    assert outside_allowlist("import statsmodels.api") == ["statsmodels.api"]
    assert outside_allowlist("from curvestudio.cli import main") == ["curvestudio.cli"]
    assert outside_allowlist("from curvestudio.expressions import grammar") == [
        "curvestudio.expressions"
    ]
    assert outside_allowlist("import curvestudio.project.archive") == [
        "curvestudio.project.archive"
    ]
    assert outside_allowlist("from curvestudio import workflow, project, testing") == [
        "curvestudio.project",
        "curvestudio.testing",
    ]
    assert outside_allowlist("from curvestudio.workflow import FitBundle\nimport pandas") == [
        "pandas"
    ]

    allowed = (
        "import math\n"
        "from collections.abc import Sequence\n"
        "from dataclasses import dataclass\n"
        "import numpy as np\n"
        "from numpy.typing import NDArray\n"
        "from scipy import stats\n"
        "from pydantic import BaseModel\n"
        "import curvestudio.workflow\n"
        "from curvestudio import workflow\n"
        "from curvestudio.workflow import FitBundle\n"
        "from curvestudio.tools._errors import ToolError\n"
        "from . import audit\n"
        "from ..tools import _common\n"
    )
    assert outside_allowlist(allowed) == []
