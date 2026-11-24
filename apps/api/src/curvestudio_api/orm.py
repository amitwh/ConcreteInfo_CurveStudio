"""The data model. Tables are snake_case plurals; ``created_at``/``updated_at`` everywhere
except ``audit_logs``, which is append-only: its only permitted change is erasing ``user_id``."""

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from curvestudio_api.db import UtcDateTime
from curvestudio_api.timeutil import utcnow

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base; its metadata is what the Alembic migration is checked against."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


DEFAULT_PREFERENCES: dict[str, bool] = {"error_reports": True, "report_byline": False}
"""The preferences an account starts with; the JSON column has no server default."""


def _default_preferences() -> dict[str, Any]:
    """A fresh copy per row — a shared mutable default would alias between users."""
    return dict(DEFAULT_PREFERENCES)


class Timestamps:
    """The ``created_at``/``updated_at`` pair, set by the ORM on insert and on update."""

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


# The name the tables in this module were written against.
_Timestamps = Timestamps


class User(_Timestamps, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(100), default="")
    is_active: Mapped[bool] = mapped_column(default=True)
    email_verified_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    # "member" or "admin"; the only values code writes. Read from the row on every request.
    role: Mapped[str] = mapped_column(String(20), default="member")
    last_login_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    # The IST calendar day of the latest authenticated request.
    last_active_on: Mapped[date | None] = mapped_column(default=None)
    # The column is nullable so the migration can add it; existing rows are backfilled and
    # new rows get the Python default below, so handlers always see an object.
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=True, default=_default_preferences
    )
    # A plan name (set by an admin or the entitlement connector); the resolver in ``quotas``
    # falls back to ``free`` when the name matches no row.
    plan_override: Mapped[str | None] = mapped_column(String(50), default=None)


Index("uq_users_email_lower", func.lower(User.email), unique=True)
"""Addresses are unique ignoring case, whatever path inserted them."""


class EmailToken(_Timestamps, Base):
    """A single-use e-mail verification or password-reset token; only its digest is stored."""

    __tablename__ = "email_tokens"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    purpose: Mapped[str] = mapped_column(String(16))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime)
    used_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)


class RefreshToken(_Timestamps, Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    replaced_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, default=None)


class OAuthCode(_Timestamps, Base):
    """A single-use code handed to the browser after a provider callback; only its digest is
    stored. It is exchanged out-of-band for an access/refresh pair by ``POST /auth/oauth/exchange``.
    """

    __tablename__ = "oauth_codes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    code_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime)
    used_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)


class UserDailyActivity(Base):
    """One row per account and IST day with authenticated requests. The ``(user_id, day)``
    pair is the primary key, which is also the uniqueness the schema needs."""

    __tablename__ = "user_daily_activity"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    day: Mapped[date] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class Dataset(_Timestamps, Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    n_rows: Mapped[int] = mapped_column(Integer)
    columns: Mapped[list[str]] = mapped_column(JSON)
    x_column: Mapped[str] = mapped_column(String(255))
    y_column: Mapped[str] = mapped_column(String(255))
    weight_column: Mapped[str | None] = mapped_column(String(255), default=None)
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    fingerprint: Mapped[str] = mapped_column(String(64))
    payload: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    # The table's other columns and the list of every column's kind: NULL on a dataset that
    # was stored before them, which is what tells the two forms apart.
    table: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True, default=None)
    table_columns: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=None)

    fits: Mapped[list[Fit]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )


class Fit(_Timestamps, Base):
    __tablename__ = "fits"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    bundle: Mapped[dict[str, Any]] = mapped_column(JSON, deferred=True)
    # The fit's cross-validation (the core's ``CrossValidation`` as JSON), stored by the
    # ``audit`` job. NULL until one has run. Deferred like the bundle: leave-one-out keeps one
    # held-out error per point, up to two thousand.
    cv: Mapped[dict[str, Any] | None] = mapped_column(JSON, deferred=True, default=None)
    status: Mapped[str] = mapped_column(String(20))
    autofit_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("autofits.id", ondelete="SET NULL"), index=True, default=None
    )
    # Both are set on a fit opened from an Auto-fit row; there is one such fit per row.
    candidate_id: Mapped[str | None] = mapped_column(String(100), default=None)

    dataset: Mapped[Dataset] = relationship(back_populates="fits")

    __table_args__ = (UniqueConstraint("autofit_id", "candidate_id"),)


class Autofit(_Timestamps, Base):
    """One finished Auto-fit run. ``result`` is ``AutoFitResult`` JSON without per-point vectors."""

    __tablename__ = "autofits"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, deferred=True)
    n_candidates: Mapped[int] = mapped_column(Integer)
    best_model_id: Mapped[str | None] = mapped_column(String(100), default=None)


class Job(_Timestamps, Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), index=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    result_ref: Mapped[uuid.UUID | None] = mapped_column(Uuid, default=None)
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    error_code: Mapped[str | None] = mapped_column(String(50), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)


class AuditLog(Base):
    """Append-only: triggers reject DELETE and every UPDATE except setting ``user_id`` to NULL
    (account erasure)."""

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), default=None, index=True
    )
    action: Mapped[str] = mapped_column(String(50))
    entity: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[str | None] = mapped_column(String(64), default=None)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class Plan(_Timestamps, Base):
    __tablename__ = "plans"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    limits: Mapped[dict[str, Any]] = mapped_column(JSON)
