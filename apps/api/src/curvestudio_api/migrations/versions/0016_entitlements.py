"""The licence connector: the ``entitlements`` table.

One row per account that activated a licence bought in the AppLicense portal: the licence's id
and plan code, the entitlement status (``active``, ``expired`` or ``released``), the licence's
own expiry and grace deadline, when it was last re-checked, and the licence JWT itself (needed to
re-check and to release the seat). ``user_id`` is unique — an account holds one licence at a
time — and cascades, so deleting an account takes its entitlement with it. ``expires_at`` is
nullable because a lifetime licence carries no expiry.

Revision ID: 0016
Revises: 0015
"""

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "entitlements",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("licence_id", sa.String(100), nullable=False),
        sa.Column("plan_code", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("grace_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_entitlements"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_entitlements_user_id_users",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("user_id", name="uq_entitlements_user_id"),
    )


def downgrade() -> None:
    op.drop_table("entitlements")
