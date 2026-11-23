"""Tools that read a finished fit and say what its evidence shows. Pure functions, no I/O."""

from curvestudio.tools._errors import ToolError
from curvestudio.tools.audit import (
    AuditCheck,
    ExcludedRow,
    FitAudit,
    SimplerModel,
    Verdict,
    audit_fit,
)

__all__ = [
    "AuditCheck",
    "ExcludedRow",
    "FitAudit",
    "SimplerModel",
    "ToolError",
    "Verdict",
    "audit_fit",
]
