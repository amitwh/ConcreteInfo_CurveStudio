"""``fit_to_json``: the complete bundle, wrapped with a report header."""

import json
from typing import Any

from curvestudio.workflow import FitBundle
from curvestudio_reporting.meta import (
    TOOL_NAME,
    assumptions_text,
    convention_of,
    generated_at,
    tool_versions,
)


def report_header(bundle: FitBundle) -> dict[str, Any]:
    """Who wrote the report, when, and under which statistical conventions."""
    return {
        "tool": TOOL_NAME,
        "versions": tool_versions(),
        "generated_at": generated_at(),
        "convention": convention_of(bundle),
        "assumptions": assumptions_text(),
    }


def fit_to_json(bundle: FitBundle) -> str:
    """Serialise ``bundle`` with a ``report`` header, as standard JSON.

    The ``fit`` value is the bundle's own dump; ``allow_nan=False`` enforces the
    project rule that undefined means ``null``, never NaN or Infinity (the field
    types reject non-finite floats already, so this is a guard, not a workaround).

    Raises:
        ValueError: a non-finite number is present (only possible through mutation
            of a list field after the fit).
    """
    return json.dumps(
        {"report": report_header(bundle), "fit": bundle.model_dump(mode="json")},
        allow_nan=False,
        indent=2,
    )
