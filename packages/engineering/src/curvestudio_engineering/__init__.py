"""The CurveStudio engineering (concrete and civil) model catalogue.

This is an optional package: ``curvestudio-core`` never imports it. It registers through the
``curvestudio.models`` entry point
(``engineering = curvestudio_engineering.register:register_all``), so a plain core install
has no engineering models and a user who does not need them pays nothing for them.

Every model carries :data:`ENGINEERING_NOTICE`: it is an empirical or theoretical
relationship with stated assumptions, not an acceptance equation of any code. Build one with
:func:`engineering_model`, which enforces the category, references and example and appends
the notice. ``catalogue()`` returns every engineering model without touching the core
registry.
"""

from curvestudio_engineering.notice import ENGINEERING_NOTICE, engineering_model
from curvestudio_engineering.register import (
    catalogue,
    equivalent_pairs,
    family_failures,
    register_all,
)

__all__ = [
    "ENGINEERING_NOTICE",
    "catalogue",
    "engineering_model",
    "equivalent_pairs",
    "family_failures",
    "register_all",
]
