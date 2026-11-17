"""A model plugin must load whichever package is imported first, and may import from the core.

``curvestudio.models`` registers its starter set at the end of its own import. That must not
trigger plugin loading (a plugin doing ``from curvestudio.models import ModelSpec`` would fail
on the half-built package), so plugins load on the first public ``list_*`` / ``get_model`` call.
The fresh processes below install a fake entry point, since the test process hides real ones.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PLUGIN = """
from curvestudio.models import ModelExample, ModelSpec, Parameter, register


def register_models():
    register(
        ModelSpec(
            id="fake_plugin_model",
            name="Fake",
            category="power",
            expression="a*x",
            parameters=(Parameter("a"),),
            initial_guess=lambda x, y: {"a": 1.0},
            example=ModelExample({"a": 2.0}, (1.0, 2.0)),
        )
    )
"""

_DRIVER = """
import importlib.metadata as md
import sys

sys.path.insert(0, {plugin_dir!r})
real = md.entry_points


class Point:
    name = "fake"

    def load(self):
        import fake_plugin_pkg

        return fake_plugin_pkg.register_models


md.entry_points = lambda **kw: [Point()] if kw.get("group") == "curvestudio.models" else real(**kw)
{first_import}
from curvestudio.models import list_model_ids, plugin_failures

ids = list_model_ids()
assert plugin_failures() == (), plugin_failures()
assert "fake_plugin_model" in ids and "power" in ids, ids
"""


@pytest.mark.parametrize(
    "first_import",
    [
        "import curvestudio.models",
        "from curvestudio.models import ModelSpec",
        "import curvestudio.models.starter",
        "import curvestudio.fitting",
        "import curvestudio.workflow",
    ],
)
def test_plugin_importing_from_the_core_loads_in_any_import_order(
    tmp_path: Path, first_import: str
) -> None:
    (tmp_path / "fake_plugin_pkg.py").write_text(_PLUGIN, encoding="utf-8")
    code = textwrap.dedent(_DRIVER).format(plugin_dir=str(tmp_path), first_import=first_import)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "failed to load" not in done.stderr
