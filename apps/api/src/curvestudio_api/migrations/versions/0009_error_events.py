"""Error capture: ``error_events``, the reported failures.

A row keeps the failure and the request that carried it. It never identifies an account, and
the table has no column for one: none may be added.

``status`` starts at ``new`` and ``occurrence_count`` at 1 from the ORM; neither column carries
a server default.

Revision ID: 0009
Revises: 0008
"""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "error_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source", sa.String(8), nullable=False),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("severity", sa.String(8), nullable=False),
        sa.Column("error_type", sa.String(200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("stack", sa.Text(), nullable=True),
        sa.Column("route", sa.String(200), nullable=True),
        sa.Column("method", sa.String(8), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("user_agent", sa.String(500), nullable=True),
        sa.Column("app_version", sa.String(40), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("context_json", sa.JSON(), nullable=True),
        sa.Column("console_log", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_error_events"),
    )
    op.create_index("ix_error_events_fingerprint", "error_events", ["fingerprint"])
    op.create_index("ix_error_events_status", "error_events", ["status"])
    op.create_index("ix_error_events_last_seen_at", "error_events", ["last_seen_at"])


def downgrade() -> None:
    op.drop_table("error_events")
