"""Process-wide registry of available models, keyed by ``ModelSpec.id``.

Models reach the registry in three ways: explicit :func:`register` calls, lazily built
*families* (:func:`register_family`; large generated catalogues addressed by id prefix, so
nothing is built until asked for), and *plugins* (:func:`load_plugins`; packages that
declare an entry point in the group ``curvestudio.models`` so the core never imports them).
"""

import difflib
import importlib.metadata
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from curvestudio.models.spec import ModelSpec

_log = logging.getLogger(__name__)

PLUGIN_GROUP = "curvestudio.models"
"""Entry-point group whose values are ``module:function`` callables taking no arguments."""

_REGISTRY: dict[str, ModelSpec] = {}


class _Family:
    def __init__(
        self,
        prefix: str,
        ids: Callable[[bool], Sequence[str]],
        build: Callable[[str], ModelSpec],
    ) -> None:
        self.prefix = prefix
        self.ids = ids
        self.build = build
        self.built: dict[str, ModelSpec] = {}


_FAMILIES: dict[str, _Family] = {}
_PLUGINS_DONE = False
_LOADED_PLUGINS: list[str] = []
_PLUGIN_FAILURES: list[PluginFailure] = []
_FAMILY_ERRORS: dict[str, BaseException] = {}
_LOAD_LOCK = threading.RLock()
_LOCAL = threading.local()


@dataclass(frozen=True)
class PluginFailure:
    """A plugin that failed to load: its entry-point name and the exception it raised."""

    name: str
    cause: BaseException


class PluginError(Exception):
    """A plugin failed to load or to register its models.

    Raised only by ``load_plugins(strict=True)``. ``name`` and ``cause`` describe the first
    failure; ``failures`` lists every failing plugin (all other plugins were still loaded).
    """

    def __init__(self, failures: Sequence[PluginFailure]) -> None:
        if not failures:
            raise ValueError("PluginError needs at least one failure")
        self.failures = tuple(failures)
        name, cause = self.failures[0].name, self.failures[0].cause
        self.name = name
        self.cause = cause
        others = [f.name for f in self.failures if f.name != name]
        extra = f" (also failed: {', '.join(others)})" if others else ""
        super().__init__(f"model plugin {name!r} failed: {type(cause).__name__}: {cause}{extra}")


def register(spec: ModelSpec) -> None:
    """Add ``spec``; a second model with the same id is a programming error."""
    if spec.id in _REGISTRY:
        raise ValueError(f"model id {spec.id!r} is already registered")
    for prefix in _FAMILIES:
        if spec.id.startswith(prefix):
            raise ValueError(
                f"model id {spec.id!r} starts with the prefix {prefix!r} of a registered "
                "family; it would silently shadow a family member"
            )
    _REGISTRY[spec.id] = spec


def register_family(
    prefix: str,
    ids: Callable[[bool], Sequence[str]],
    build: Callable[[str], ModelSpec],
) -> None:
    """Register an id-addressable family of models that are built only when requested.

    Args:
        prefix: every id of the family starts with this (non-empty, so it is unambiguous).
        ids: ``ids(exhaustive)`` lists the family's ids; ``exhaustive=True`` includes the
            long tail that is hidden from default listings.
        build: ``build(model_id)`` returns the spec for one id of the family (its ``id``
            must equal ``model_id``) and raises ``KeyError`` for an id it does not know.
    """
    if not prefix:
        raise ValueError("a model family needs a non-empty id prefix")
    if prefix in _FAMILIES:
        raise ValueError(f"model family {prefix!r} is already registered")
    clashing = sorted(i for i in _REGISTRY if i.startswith(prefix))
    if clashing:
        raise ValueError(
            f"model family prefix {prefix!r} collides with registered model ids {clashing[:3]}"
        )
    _FAMILIES[prefix] = _Family(prefix, ids, build)


def get_model(model_id: str) -> ModelSpec:
    """Return the model, or raise ``KeyError`` suggesting close ids.

    Looks at registered specs, then at families by id prefix (longest prefix first).
    """
    load_plugins()
    spec = _REGISTRY.get(model_id)
    if spec is not None:
        return spec
    family = _family_for(model_id)
    if family is not None:
        cached = family.built.get(model_id)
        if cached is not None:
            return cached
        built = family.build(model_id)
        if built.id != model_id:
            raise ValueError(
                f"family {family.prefix!r} built a model with id {built.id!r} for {model_id!r}"
            )
        family.built[model_id] = built
        return built
    close = difflib.get_close_matches(model_id, list_model_ids(), n=3, cutoff=0.5)
    hint = f"; did you mean {', '.join(repr(m) for m in close)}?" if close else ""
    if _PLUGIN_FAILURES:
        failed = ", ".join(repr(f.name) for f in _PLUGIN_FAILURES)
        hint += f" (model plugins that failed to load: {failed}; see plugin_failures())"
    raise KeyError(f"unknown model id {model_id!r}{hint}")


def list_models(category: str | None = None) -> list[ModelSpec]:
    """Registered models in registration order, optionally limited to one category.

    ``category`` matches exactly or as the first dotted segment (``engineering`` matches
    ``engineering.strength``). Family members are not built here; see :func:`list_model_ids`.
    """
    load_plugins()
    return [m for m in _REGISTRY.values() if _in_category(m.category, category)]


def list_model_ids(*, exhaustive: bool = False, category: str | None = None) -> list[str]:
    """Ids of registered models followed by each family's ids, without duplicates.

    ``exhaustive=True`` asks families for their full list. With ``category``, family
    members are built (but not compiled) to read their category; a member whose build
    raises is skipped and recorded in :func:`family_errors` instead of breaking the listing.
    """
    load_plugins()
    ids = list(_REGISTRY)
    seen = set(ids)
    for family in _FAMILIES.values():
        for model_id in family.ids(exhaustive):
            if model_id not in seen:
                seen.add(model_id)
                ids.append(model_id)
    if category is None:
        return ids
    return [i for i in ids if _member_in_category(i, category)]


def _member_in_category(model_id: str, category: str) -> bool:
    try:
        return _in_category(get_model(model_id).category, category)
    except Exception as exc:
        if model_id not in _FAMILY_ERRORS:
            _log.warning("model %r could not be built: %s: %s", model_id, type(exc).__name__, exc)
        _FAMILY_ERRORS[model_id] = exc
        return False


def family_errors() -> dict[str, BaseException]:
    """Family model ids whose build raised while listing, with the exception (a copy)."""
    return dict(_FAMILY_ERRORS)


def plugin_failures() -> tuple[PluginFailure, ...]:
    """Plugins that failed to load, in load order (empty until plugins have been loaded)."""
    return tuple(_PLUGIN_FAILURES)


def load_plugins(*, strict: bool = False) -> list[str]:
    """Load every plugin of the entry-point group ``curvestudio.models`` once.

    Each entry-point value is ``module:function``; the function is called with no
    arguments and registers models (and families) through this module. Loading is thread
    safe and happens once per process (until :func:`_clear_registry`): later calls, and
    concurrent first calls, return the same result once loading has finished. A plugin
    that looks a model up from inside its own function does not recurse.

    A plugin that fails (any exception, including ``SystemExit``) is recorded in
    :func:`plugin_failures`, logged as a warning once, and its partial registrations are
    rolled back so it is either fully loaded or not at all; the other plugins still load.
    ``KeyboardInterrupt`` aborts loading, undoes everything this call registered and
    propagates.

    Args:
        strict: raise :class:`PluginError` (after all plugins were tried) if any plugin
            failed, now or in an earlier call. The default returns normally.

    Returns:
        Names of the plugins that loaded successfully.
    """
    if not _PLUGINS_DONE and not getattr(_LOCAL, "loading", False):
        with _LOAD_LOCK:
            if not _PLUGINS_DONE:
                _LOCAL.loading = True
                try:
                    _load_all()
                finally:
                    _LOCAL.loading = False
    if strict and _PLUGIN_FAILURES:
        raise PluginError(_PLUGIN_FAILURES) from _PLUGIN_FAILURES[0].cause
    return list(_LOADED_PLUGINS)


def _load_all() -> None:
    global _PLUGINS_DONE
    models_before, families_before = set(_REGISTRY), set(_FAMILIES)
    try:
        for entry in importlib.metadata.entry_points(group=PLUGIN_GROUP):
            _load_one(entry)
    except KeyboardInterrupt:
        _rollback(models_before, families_before)
        _LOADED_PLUGINS.clear()
        _PLUGIN_FAILURES.clear()
        raise
    _PLUGINS_DONE = True


def _load_one(entry: importlib.metadata.EntryPoint) -> None:
    models_before, families_before = set(_REGISTRY), set(_FAMILIES)
    try:
        entry.load()()
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        _rollback(models_before, families_before)
        _PLUGIN_FAILURES.append(PluginFailure(entry.name, exc))
        _log.warning(
            "model plugin %r failed to load and was skipped: %s: %s",
            entry.name,
            type(exc).__name__,
            exc,
        )
    else:
        _LOADED_PLUGINS.append(entry.name)


def _rollback(models_before: set[str], families_before: set[str]) -> None:
    for model_id in set(_REGISTRY) - models_before:
        del _REGISTRY[model_id]
    for prefix in set(_FAMILIES) - families_before:
        del _FAMILIES[prefix]


def _family_for(model_id: str) -> _Family | None:
    matches = [f for p, f in _FAMILIES.items() if model_id.startswith(p)]
    return max(matches, key=lambda f: len(f.prefix), default=None)


def _in_category(actual: str, wanted: str | None) -> bool:
    return wanted is None or actual == wanted or actual.startswith(f"{wanted}.")


def _clear_registry() -> None:
    """Empty the registry, its families and the plugin-loaded flag. For tests only."""
    global _PLUGINS_DONE
    with _LOAD_LOCK:
        _REGISTRY.clear()
        _FAMILIES.clear()
        _LOADED_PLUGINS.clear()
        _PLUGIN_FAILURES.clear()
        _FAMILY_ERRORS.clear()
        _PLUGINS_DONE = False
