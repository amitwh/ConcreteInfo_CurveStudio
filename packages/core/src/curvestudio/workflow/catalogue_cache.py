"""A per-user disk cache of the built model catalogue.

Building the catalogue from scratch costs about two seconds: registering the model
plugins and rendering every model's LaTeX forces a SymPy compile per model, and the
result depends on nothing but the installed packages and their entry points. So the
built list (:func:`curvestudio.workflow.catalog.list_models`) is cached per user and
reloaded on the next start-up, which is what brings the desktop's model-library load
inside its two-second target.

A record is served only under an exact fingerprint of everything that decides its
content: the cache-format version, the versions of the core distribution and of every
distribution contributing a ``curvestudio.models`` entry point, the sorted set of
entry-point names, and a digest of the contributing packages' Python sources — an
editable install keeps its version while its content changes, so the sources are
covered too, while a frozen bundle (whose modules live in the archive, not as files)
contributes a stable empty digest there. The record also names the registry states it
may be served under — before and after the plugins register — so a state built up
in-process (tests, future callers registering models by hand) is never served stale.

On any mismatch, corruption or missing file the catalogue is built the ordinary way and
the cache is rewritten. Only a clean scan writes it: plugins not loaded before it, the
registry at its import-time starter baseline when it started, no plugin failure — so a
failing plugin is retried on every start-up instead of being frozen into the cache, and
in-process registrations are never persisted as if they were the environment's.

Registration is skipped on a hit: the plugins do not load until something asks for a
fitted model (:func:`curvestudio.models.get_model`), which is always a user action well
after start-up.

The cache never writes inside the repository or an installation: it lives under the
per-user cache directory (``$XDG_CACHE_HOME`` or ``~/.cache`` on Linux,
``~/Library/Caches`` on macOS, the local application-data directory on Windows,
under ``ConcreteInfo/CurveStudio``), overridable with :data:`ENV_CACHE_DIR` for tests
and packaging checks. Writes are atomic and best effort: a cache that cannot be written
only costs the rebuild it would have saved.
"""

import contextlib
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from curvestudio.models import registry as _registry
from curvestudio.models.starter import STARTER_MODELS

_log = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1
"""Bumped when the record's shape or meaning changes; older records then rebuild."""

ENV_CACHE_DIR = "CURVESTUDIO_CATALOGUE_CACHE_DIR"
"""Set to a directory to keep the catalogue cache there instead of the user's."""

CORE_DISTRIBUTION = "curvestudio-core"
"""The distribution whose version the fingerprint always carries (the starter set, the
expression grammar and the catalogue shape live in it)."""

_FILE_NAME = "model-catalogue.json"

_LOCK = threading.Lock()
_MEMORY: tuple[tuple[tuple[str, ...], str], list[Any]] | None = None
"""The last list served or built in this process, keyed by the registry's id state and
the content fingerprint. The state alone cannot tell two environments apart — an
in-process switch of the visible entry points (tests) leaves the registry as it was."""


def catalogue(build: Callable[[], list[Any]]) -> list[Any]:
    """The built catalogue: from cache when valid, via ``build`` when it is not.

    ``build`` is :func:`curvestudio.workflow.catalog._build_catalogue` — the ordinary
    scan, which loads and registers the plugins. It runs only on a miss; a hit serves
    the cached list and the registration stays deferred until a model is asked for.
    """
    global _MEMORY
    state = _registry_state()
    key = (state, fingerprint())
    served = _from_memory(key)
    if served is None:
        served = _from_disk(state)
    if served is not None:
        with _LOCK:
            _MEMORY = (key, served)
        return list(served)
    plugins_were_loaded = _registry.plugins_loaded()
    models = build()
    with _LOCK:
        _MEMORY = ((_registry_state(), fingerprint()), models)
    _store(models, miss_state=state, plugins_were_loaded=plugins_were_loaded)
    return list(models)


# -- what decides the content ------------------------------------------------------------


def fingerprint() -> str:
    """A hex sha256 of everything the catalogue's content depends on.

    The cache-format version, the core distribution's version, every contributing
    distribution's name and version, the sorted set of entry-point groups and names,
    and the contributing packages' Python sources.
    """
    parts = [
        CACHE_FORMAT_VERSION,
        sorted(_contributing_distributions()),
        sorted((entry.group, entry.name) for entry in _entry_points()),
        _source_digest(),
    ]
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entry_points() -> list[importlib.metadata.EntryPoint]:
    return list(importlib.metadata.entry_points(group=_registry.PLUGIN_GROUP))


def _contributing_distributions() -> set[tuple[str, str]]:
    """(name, version) of the core distribution and of every distribution that brings a
    plugin entry point. Where a distribution's metadata is absent, its entry point's own
    value stands in for the version — a frozen bundle that did not collect that
    dist-info still fingerprints stably, just less precisely."""
    found = {(CORE_DISTRIBUTION, _core_version())}
    for entry in _entry_points():
        dist = entry.dist
        if dist is not None:
            found.add((dist.name, dist.version))
        else:
            found.add((entry.name, entry.value))
    return found


def _core_version() -> str:
    try:
        return importlib.metadata.version(CORE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@functools.cache
def _source_digest() -> str:
    """A hex sha256 over the contributing packages' Python sources (paths and bytes).

    Every ``.py`` file under the core package and under each entry point's top-level
    package is hashed, so editing a catalogue source invalidates the cache even though
    the distribution's version did not move. A package whose files are not on disk (a
    frozen bundle's modules live in the archive) contributes nothing, which is stable.
    Memoized per process: the sources cannot change under a running interpreter.
    """
    hasher = hashlib.sha256()
    for root in _package_roots():
        for path in sorted(root.rglob("*.py")):
            hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
            try:
                hasher.update(path.read_bytes())
            except OSError:  # a source that cannot be read now cannot be cached for
                hasher.update(b"<unreadable>")  # later either; treat it as its own content
    return hasher.hexdigest()


def _package_roots() -> list[Path]:
    """Directories of the core package and of every entry point's top-level package."""
    roots: list[Path] = []
    modules = ["curvestudio", *(entry.value.partition(":")[0] for entry in _entry_points())]
    for module_name in modules:
        top_level = module_name.partition(".")[0]
        try:
            spec = importlib.util.find_spec(top_level)
        except ImportError, ValueError:
            continue
        locations = getattr(spec, "submodule_search_locations", None) if spec else None
        for location in locations or ():
            path = Path(location)
            if path.is_dir():
                roots.append(path)
    return roots


# -- the registry states a record may be served under ------------------------------------


def _registry_state() -> tuple[str, ...]:
    """The registry's ids as they stand right now, without loading plugins."""
    return tuple(sorted(_registry.peek_registered_ids()))


def _clean_state() -> tuple[str, ...]:
    """The import-time state every process starts in: the starter models alone."""
    return tuple(sorted(spec.id for spec in STARTER_MODELS))


# -- reading -----------------------------------------------------------------------------


def _from_memory(key: tuple[tuple[str, ...], str]) -> list[Any] | None:
    with _LOCK:
        if _MEMORY is not None and _MEMORY[0] == key:
            return list(_MEMORY[1])
    return None


def _from_disk(state: tuple[str, ...]) -> list[Any] | None:
    record = _read_record()
    if record is None:
        return None
    if record.get("format") != CACHE_FORMAT_VERSION:
        return None
    if record.get("fingerprint") != fingerprint():
        return None
    served_states = (record.get("baseline_ids"), record.get("registered_ids"))
    if not all(isinstance(ids, list) for ids in served_states):
        return None
    if list(state) not in served_states:  # a state built up in-process is never served stale
        return None
    models = record.get("models")
    if not isinstance(models, list) or not models:
        return None
    from curvestudio.workflow.catalog import ModelInfo  # deferred: catalog imports this module

    try:
        built = [ModelInfo.model_validate(model) for model in models]
    except Exception:  # a record that does not validate is corrupt: rebuild, never break
        _log.debug("model catalogue cache did not validate; rebuilding", exc_info=True)
        return None
    return built


def _read_record() -> dict[str, Any] | None:
    try:
        raw = record_path().read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


# -- writing -----------------------------------------------------------------------------


def _store(models: list[Any], *, miss_state: tuple[str, ...], plugins_were_loaded: bool) -> None:
    """Write the record a future process may be served, when this scan was the clean one.

    A scan that loaded the plugins itself, started from the starter baseline and finished
    without a plugin failure describes the environment exactly; anything else (a scan
    over in-process registrations, a failing plugin) must not be persisted.
    """
    if _registry.plugin_failures():
        _log.debug(
            "model catalogue not cached: %d plugin(s) failed to load",
            len(_registry.plugin_failures()),
        )
        return
    if plugins_were_loaded or miss_state != _clean_state():
        return
    record = {
        "format": CACHE_FORMAT_VERSION,
        "fingerprint": fingerprint(),
        "baseline_ids": list(miss_state),
        "registered_ids": list(_registry_state()),
        "models": [model.model_dump(mode="json") for model in models],
    }
    _write_record(record)


def _write_record(record: dict[str, Any]) -> None:
    path = record_path()
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a reader never sees a half-written record
    except OSError:
        _log.debug("model catalogue cache could not be written to %s", path, exc_info=True)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def cache_directory() -> Path:
    """The directory the cache lives in: :data:`ENV_CACHE_DIR` when set, else the
    per-user cache root's ``ConcreteInfo/CurveStudio`` (never inside a repository or an
    installation)."""
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return Path(override)
    return _user_cache_dir() / "ConcreteInfo" / "CurveStudio"


def record_path() -> Path:
    """Where the cache record file lives."""
    return cache_directory() / _FILE_NAME


def _user_cache_dir() -> Path:
    """The per-user cache root, by each platform's own convention: the local application
    data directory on Windows, ``~/Library/Caches`` on macOS, ``$XDG_CACHE_HOME`` or
    ``~/.cache`` elsewhere."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local)
        return Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def _reset_for_tests() -> None:
    """Forget everything memoized. For tests only: it restores a new process's view."""
    global _MEMORY
    _source_digest.cache_clear()
    with _LOCK:
        _MEMORY = None
