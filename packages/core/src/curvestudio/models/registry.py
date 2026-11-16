"""Process-wide registry of available models, keyed by ``ModelSpec.id``."""

import difflib

from curvestudio.models.spec import ModelSpec

_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    """Add ``spec``; a second model with the same id is a programming error."""
    if spec.id in _REGISTRY:
        raise ValueError(f"model id {spec.id!r} is already registered")
    _REGISTRY[spec.id] = spec


def get_model(model_id: str) -> ModelSpec:
    """Return the registered model, or raise ``KeyError`` suggesting close ids."""
    try:
        return _REGISTRY[model_id]
    except KeyError:
        close = difflib.get_close_matches(model_id, _REGISTRY, n=3, cutoff=0.5)
        hint = f"; did you mean {', '.join(repr(m) for m in close)}?" if close else ""
        raise KeyError(f"unknown model id {model_id!r}{hint}") from None


def list_models(category: str | None = None) -> list[ModelSpec]:
    """Registered models in registration order, optionally limited to one category."""
    return [m for m in _REGISTRY.values() if category is None or m.category == category]


def _clear_registry() -> None:
    """Empty the registry. For tests only."""
    _REGISTRY.clear()
