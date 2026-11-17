"""The CurveStudio hand-written model catalogue.

``catalogue()`` returns every library model (the core starter set is not included).
Registration with the core registry happens through the ``curvestudio.models`` entry
point, see :func:`curvestudio_models.register.register_all`.
"""

from curvestudio_models.register import catalogue, family_failures, register_all

__all__ = ["catalogue", "family_failures", "register_all"]
