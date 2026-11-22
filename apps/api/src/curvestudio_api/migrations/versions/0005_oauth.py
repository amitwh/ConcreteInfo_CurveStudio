"""OAuth sign-in: the single-use ``oauth_codes`` exchange table.

A provider callback stores the SHA-256 digest of a short-lived code; the web app claims it
once with ``POST /auth/oauth/exchange`` in return for an access/refresh pair.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oauth_codes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("code_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_oauth_codes"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_oauth_codes_user_id_users", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("code_hash", name="uq_oauth_codes_code_hash"),
    )
    op.create_index("ix_oauth_codes_user_id", "oauth_codes", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_oauth_codes_user_id", table_name="oauth_codes")
    op.drop_table("oauth_codes")
