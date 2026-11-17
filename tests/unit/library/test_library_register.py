"""Registration of the library through the entry-point plugin mechanism."""

import subprocess
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

from curvestudio.models import (
    ModelExample,
    ModelSpec,
    Parameter,
    get_model,
    list_model_ids,
    load_plugins,
    plugin_failures,
)
from curvestudio.models.registry import _clear_registry
from curvestudio.models.starter import register_starter_models
from curvestudio_models import catalogue, register

pytestmark = pytest.mark.real_plugins


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    _clear_registry()
    yield
    _clear_registry()
    register_starter_models()


def _model(model_id: str, expression: str = "a+b/x") -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name="Probe",
        category="rational",
        expression=expression,
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.0, "b": 1.0},
        example=ModelExample({"a": 1.0, "b": 1.0}, (1.0, 2.0)),
    )


def _module(name: str, models: object = None, pairs: object = None) -> ModuleType:
    module = ModuleType(f"curvestudio_models.families.{name}")
    if models is not None:
        module.MODELS = models  # type: ignore[attr-defined]
    if pairs is not None:
        module.EQUIVALENT_PAIRS = pairs  # type: ignore[attr-defined]
    return module


def _fake_families(monkeypatch: pytest.MonkeyPatch, modules: dict[str, ModuleType]) -> None:
    """Replace the family modules; ``load_families`` reads them fresh (no cache)."""
    monkeypatch.setattr(register, "_family_modules", lambda: list(modules))
    monkeypatch.setattr(register, "_import_family", modules.__getitem__)
    monkeypatch.setattr(register, "_loaded", register.load_families)


def test_entry_point_registers_the_whole_catalogue() -> None:
    register_starter_models()
    assert "library" in load_plugins()
    assert plugin_failures() == ()
    ids = set(list_model_ids())
    assert {m.id for m in catalogue()} <= ids
    assert get_model("rational_1_1").category == "rational"


def test_register_all_is_idempotent() -> None:
    register.register_all()
    before = list_model_ids()
    register.register_all()
    assert list_model_ids() == before


def test_catalogue_is_a_stable_tuple() -> None:
    assert catalogue() is catalogue()
    assert isinstance(catalogue(), tuple)


def test_duplicate_id_between_modules_skips_the_later_module_naming_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_families(
        monkeypatch,
        {
            "one": _module("one", (_model("dup_model"),)),
            "two": _module("two", (_model("dup_model"), _model("other_model"))),
        },
    )
    loaded, failures = register.load_families()
    assert [f.name for f in loaded] == ["one"]
    assert set(failures) == {"two"}
    assert "'dup_model'" in str(failures["two"])
    assert "'one'" in str(failures["two"]) and "'two'" in str(failures["two"])


def test_a_broken_family_is_skipped_with_a_warning_and_the_others_still_register(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(name: str) -> ModuleType:
        if name == "crashes":
            raise ImportError("no module named nonsense")
        return modules[name]

    modules = {
        "fine_one": _module("fine_one", (_model("fine_a"),)),
        "no_models": _module("no_models"),
        "bad_pairs": _module("bad_pairs", (_model("bad_pair_model"),), pairs=(("x",),)),
        "crashes": _module("crashes"),
        "fine_two": _module("fine_two", (_model("fine_b"),), pairs=(("fine_b", "fine_a"),)),
    }
    monkeypatch.setattr(register, "_family_modules", lambda: list(modules))
    monkeypatch.setattr(register, "_import_family", explode)
    monkeypatch.setattr(register, "_loaded", register.load_families)
    caplog.set_level("WARNING", logger=register.__name__)

    loaded, failures = register.load_families()

    assert [f.name for f in loaded] == ["fine_one", "fine_two"]
    assert set(failures) == {"no_models", "bad_pairs", "crashes"}
    warned = " ".join(r.getMessage() for r in caplog.records)
    for name in ("no_models", "bad_pairs", "crashes"):
        assert f"'{name}'" in warned
    assert "no module named nonsense" in warned
    assert loaded[1].pairs == (("fine_b", "fine_a"),)


def test_register_all_skips_a_family_that_clashes_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from curvestudio.models import register as core_register

    core_register(_model("taken"))  # a different object under the id the family wants
    modules = {
        "clashing": _module("clashing", (_model("taken"), _model("never_registered"))),
        "healthy": _module("healthy", (_model("healthy_model"),)),
    }
    _fake_families(monkeypatch, modules)
    caplog.set_level("WARNING", logger=register.__name__)
    register.register_all()
    ids = set(list_model_ids())
    assert "healthy_model" in ids
    assert "never_registered" not in ids
    assert any("'clashing'" in r.getMessage() for r in caplog.records)


def test_a_real_family_failure_is_empty() -> None:
    assert register.family_failures() == {}


_CHECK = (
    "from curvestudio.models import list_model_ids, plugin_failures\n"
    "ids = list_model_ids()\n"
    "assert plugin_failures() == (), plugin_failures()\n"
    "assert len(ids) >= 17, ids\n"
    "assert 'rational_1_1' in ids and 'michaelis_menten' in ids, ids\n"
)


@pytest.mark.parametrize(
    "first_import",
    [
        "import curvestudio_models\n",
        "from curvestudio.models import ModelSpec\n",
        "import curvestudio_models.register\nimport curvestudio.models\n",
        "import curvestudio.models.starter\n",
    ],
    ids=["library_first", "core_first", "library_register_first", "starter_first"],
)
def test_library_loads_in_a_fresh_process_whichever_package_is_imported_first(
    first_import: str,
) -> None:
    done = subprocess.run(
        [sys.executable, "-c", first_import + _CHECK], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "failed to load" not in done.stderr
    assert "was skipped" not in done.stderr
