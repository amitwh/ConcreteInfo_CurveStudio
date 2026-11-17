"""Collect the engineering family modules and register them with the core registry.

Every module under ``curvestudio_engineering.families`` (private ``_names`` excluded) exposes
``MODELS: tuple[ModelSpec, ...]`` and may expose ``EQUIVALENT_PAIRS: tuple[tuple[str, str], ...]``
naming models that are reparametrisations of each other (a rate against a characteristic
time) or of a model in the general catalogue. A pair naming a model that does not exist is
kept as a declaration: it is reported by :func:`equivalent_pairs` and simply cannot be
checked until the partner exists.

A family module that cannot be imported, is malformed, or repeats an id of an earlier module
is **skipped with a warning naming it**: the other families still load. The skipped modules
are reported by :func:`family_failures`, and the engineering tests fail while any exists, so
a broken family cannot go unnoticed in development.
"""

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from functools import cache
from types import ModuleType

from curvestudio.models import ModelSpec, list_models, register

_log = logging.getLogger(__name__)

_FAMILIES_PACKAGE = "curvestudio_engineering.families"


@dataclass(frozen=True)
class Family:
    """One loaded family module: its short name, models and declared equivalent pairs."""

    name: str
    models: tuple[ModelSpec, ...]
    pairs: tuple[tuple[str, str], ...]


def _family_modules() -> list[str]:
    """Short names of the family modules, sorted so registration order is stable."""
    package = importlib.import_module(_FAMILIES_PACKAGE)
    return sorted(
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_")
    )


def _import_family(name: str) -> ModuleType:
    return importlib.import_module(f"{_FAMILIES_PACKAGE}.{name}")


def _read_family(name: str, module: ModuleType) -> Family:
    models = getattr(module, "MODELS", None)
    if not isinstance(models, tuple) or not all(isinstance(m, ModelSpec) for m in models):
        raise ValueError(f"family module {name!r} must define MODELS as a tuple of ModelSpec")
    pairs = getattr(module, "EQUIVALENT_PAIRS", ())
    if not isinstance(pairs, tuple) or not all(
        isinstance(p, tuple) and len(p) == 2 and p[0] != p[1] and all(isinstance(i, str) for i in p)
        for p in pairs
    ):
        raise ValueError(
            f"family module {name!r}: EQUIVALENT_PAIRS must be a tuple of (id, id) pairs of "
            "two different model ids"
        )
    return Family(name, models, pairs)


def load_families() -> tuple[tuple[Family, ...], dict[str, Exception]]:
    """Import every family module; return the loaded ones and the failures by module name.

    A duplicate id counts as a failure of the *later* module (the message names both).
    """
    owner: dict[str, str] = {}
    loaded: list[Family] = []
    failures: dict[str, Exception] = {}
    for name in _family_modules():
        try:
            family = _read_family(name, _import_family(name))
            for spec in family.models:
                if spec.id in owner:
                    raise ValueError(
                        f"model id {spec.id!r} is defined by both {owner[spec.id]!r} and {name!r}"
                    )
                owner[spec.id] = name
            ids = [m.id for m in family.models]
            if len(set(ids)) != len(ids):
                raise ValueError(f"family module {name!r} defines a model id twice")
        except Exception as exc:  # any import-time failure of one family must not drop the rest
            for model_id in [i for i, o in owner.items() if o == name]:
                del owner[model_id]
            failures[name] = exc
            _log.warning("engineering family %r was skipped: %s: %s", name, type(exc).__name__, exc)
        else:
            loaded.append(family)
    return tuple(loaded), failures


@cache
def _loaded() -> tuple[tuple[Family, ...], dict[str, Exception]]:
    return load_families()


@cache
def catalogue() -> tuple[ModelSpec, ...]:
    """All engineering models of the loaded families (the general catalogue is not included)."""
    return tuple(spec for family in _loaded()[0] for spec in family.models)


def families() -> tuple[Family, ...]:
    """The family modules that loaded, in registration order."""
    return _loaded()[0]


def family_failures() -> dict[str, Exception]:
    """Family modules that were skipped (import error, malformed, duplicate id), by name."""
    return dict(_loaded()[1])


def equivalent_pairs() -> tuple[tuple[str, str], ...]:
    """Every ``EQUIVALENT_PAIRS`` declaration of the loaded families, in registration order."""
    return tuple(pair for family in families() for pair in family.pairs)


def register_all() -> None:
    """Register every engineering model with the core registry, one family at a time.

    This is the ``curvestudio.models`` entry point and is idempotent. A family whose ids
    clash with models that are already registered is skipped with a warning naming it (and
    recorded in :func:`family_failures`); the other families are still registered.
    """
    existing = {m.id: m for m in list_models()}
    for family in families():
        clashes = [m.id for m in family.models if existing.get(m.id, m) is not m]
        if clashes:
            _skip(family.name, ValueError(f"model ids already registered by others: {clashes}"))
            continue
        try:
            for spec in family.models:
                if existing.get(spec.id) is not spec:
                    register(spec)
                    existing[spec.id] = spec
        except ValueError as exc:
            _skip(family.name, exc)


def _skip(name: str, exc: Exception) -> None:
    _loaded()[1][name] = exc
    _log.warning("engineering family %r was not registered: %s: %s", name, type(exc).__name__, exc)
