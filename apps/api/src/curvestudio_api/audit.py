"""Append-only audit trail. Rows are added in the caller's transaction."""

import uuid
from typing import Any

from sqlalchemy.orm import Session

from curvestudio_api.logging_setup import request_id_var
from curvestudio_api.orm import AuditLog


def record(
    db: Session,
    action: str,
    entity: str,
    *,
    user_id: uuid.UUID | None,
    entity_id: uuid.UUID | str | None = None,
    detail: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> None:
    """Add one audit row. ``detail`` must never carry passwords, tokens or personal data
    beyond what the action needs (an email is recorded for failed logins only as a hash)."""
    db.add(
        AuditLog(
            user_id=user_id,
            action=action,
            entity=entity,
            entity_id=None if entity_id is None else str(entity_id),
            detail=detail or {},
            request_id=request_id or request_id_var.get(),
        )
    )
