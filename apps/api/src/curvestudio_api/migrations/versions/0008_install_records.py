"""Desktop install records: ``install_records`` and ``install_events``.

An installation is known by its ``install_key`` and described by release, operating system and
architecture. ``install_events`` holds what an installation reported, and goes when its record
goes.

Neither table has a column for an address or an account, and none may be added. The desktop
app is used without signing in and sends nothing but this record; an installation is not a
person, and nothing here must be able to tie one back to a person.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "install_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("install_key", sa.String(64), nullable=False),
        sa.Column("app_version", sa.String(40), nullable=False),
        sa.Column("os_name", sa.String(16), nullable=False),
        sa.Column("os_version", sa.String(40), nullable=False),
        sa.Column("arch", sa.String(16), nullable=False),
        sa.Column("package", sa.String(20), nullable=False),
        sa.Column("session", sa.String(16), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_install_records"),
        sa.UniqueConstraint("install_key", name="uq_install_records_install_key"),
    )
    op.create_table(
        "install_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("install_record_id", sa.Uuid(), nullable=False),
        sa.Column("event", sa.String(16), nullable=False),
        sa.Column("from_version", sa.String(40), nullable=True),
        sa.Column("to_version", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_install_events"),
        sa.ForeignKeyConstraint(
            ["install_record_id"],
            ["install_records.id"],
            name="fk_install_events_install_record_id_install_records",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_install_events_install_record_id", "install_events", ["install_record_id"])


def downgrade() -> None:
    op.drop_table("install_events")
    op.drop_table("install_records")
