"""E-mail verification and password reset: ``users.email_verified_at`` and ``email_tokens``.

Existing accounts are backfilled as verified (their address was never checked before, so
refusing them now would lock everybody out).

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.execute("UPDATE users SET email_verified_at = created_at")
    op.create_table(
        "email_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("purpose", sa.String(16), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_email_tokens"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_email_tokens_user_id_users", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("token_hash", name="uq_email_tokens_token_hash"),
    )
    op.create_index("ix_email_tokens_user_id", "email_tokens", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_email_tokens_user_id", table_name="email_tokens")
    op.drop_table("email_tokens")
    # SQLite needs a table rebuild to drop a column; drop and restore the functional
    # unique index so the rebuild does not lose it.
    op.drop_index("uq_users_email_lower", table_name="users")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("email_verified_at")
    op.create_index("uq_users_email_lower", "users", [sa.text("lower(email)")], unique=True)
