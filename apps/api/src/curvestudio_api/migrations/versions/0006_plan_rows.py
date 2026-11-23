"""Plans: the seeded ``free`` row and the ``users.plan_override`` column.

``free`` starts at today's default limits so nothing tightens until the owner sets them (P5).
``plan_override`` names the plan row an account resolves to when billing is on; only an admin
or the entitlement connector writes it, and every change is audited.

Revision ID: 0006
Revises: 0005
"""

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

FREE_PLAN_ID = uuid.UUID("5e1f0057-0000-4000-8000-000000000002")
FREE_LIMITS = {
    "max_datasets": 200,
    "max_dataset_bytes": 500_000_000,
    "max_fits": 2000,
    "max_autofits_per_day": 100,
}
_plans = sa.table(
    "plans",
    sa.column("id", sa.Uuid),
    sa.column("name", sa.String),
    sa.column("limits", sa.JSON),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)


def upgrade() -> None:
    op.add_column("users", sa.Column("plan_override", sa.String(50), nullable=True))
    now = datetime.now(UTC)
    op.execute(
        _plans.insert().values(
            id=FREE_PLAN_ID, name="free", limits=FREE_LIMITS, created_at=now, updated_at=now
        )
    )


def downgrade() -> None:
    op.execute(_plans.delete().where(_plans.c.name == "free"))
    # SQLite needs a table rebuild to drop a column; drop and restore the functional
    # unique index so the rebuild does not lose it.
    op.drop_index("uq_users_email_lower", table_name="users")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("plan_override")
    op.create_index("uq_users_email_lower", "users", [sa.text("lower(email)")], unique=True)
