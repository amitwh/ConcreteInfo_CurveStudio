"""Tools that read a finished fit and say what its evidence shows. Pure functions, no I/O."""

from curvestudio.statistics import CrossValidation
from curvestudio.tools._errors import ToolError
from curvestudio.tools.audit import (
    AuditCheck,
    ExcludedRow,
    FitAudit,
    SimplerModel,
    Verdict,
    audit_fit,
    expected_cv_ratio,
)
from curvestudio.tools.cross_validation import cross_validate
from curvestudio.tools.methods import MethodsText, Style, all_methods_texts, methods_text

__all__ = [
    "AuditCheck",
    "CrossValidation",
    "ExcludedRow",
    "FitAudit",
    "MethodsText",
    "SimplerModel",
    "Style",
    "ToolError",
    "Verdict",
    "all_methods_texts",
    "audit_fit",
    "cross_validate",
    "expected_cv_ratio",
    "methods_text",
]
