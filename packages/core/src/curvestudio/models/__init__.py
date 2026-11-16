"""Model layer: declarative model specifications and their registry."""

from curvestudio.models.registry import get_model, list_models, register
from curvestudio.models.spec import ModelSpec, Parameter

__all__ = ["ModelSpec", "Parameter", "get_model", "list_models", "register"]
