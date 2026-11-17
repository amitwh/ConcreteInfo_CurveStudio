"""Model layer: declarative model specifications and their registry."""

from curvestudio.models import starter as _starter  # noqa: F401  (registers the starter set)
from curvestudio.models.custom import (
    CustomModelDefinition,
    ParameterDefinition,
    build_custom_model,
)
from curvestudio.models.guess import GuessRule, guess_from_rules, rules_guess
from curvestudio.models.registry import (
    PluginError,
    PluginFailure,
    family_errors,
    get_model,
    list_model_ids,
    list_models,
    load_plugins,
    plugin_failures,
    register,
    register_family,
)
from curvestudio.models.spec import ModelExample, ModelSpec, Parameter

__all__ = [
    "CustomModelDefinition",
    "GuessRule",
    "ModelExample",
    "ModelSpec",
    "Parameter",
    "ParameterDefinition",
    "PluginError",
    "PluginFailure",
    "build_custom_model",
    "family_errors",
    "get_model",
    "guess_from_rules",
    "list_model_ids",
    "list_models",
    "load_plugins",
    "plugin_failures",
    "register",
    "register_family",
    "rules_guess",
]
