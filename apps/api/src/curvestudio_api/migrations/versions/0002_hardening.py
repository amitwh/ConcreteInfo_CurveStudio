"""Case-insensitive unique email, finite self-hosted quotas, audit erasure of ``user_id``.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

PLAN_LIMITS = {"max_datasets": 200, "max_dataset_bytes": 500_000_000, "max_fits": 2000}
MESSAGE = "audit_logs is append-only"
_COLUMNS = ("id", "action", "entity", "entity_id", "request_id", "created_at")


def _unchanged(comparison: str) -> str:
    """SQL saying every column except ``user_id`` and ``detail`` is unchanged."""
    return " AND ".join(f"NEW.{c} {comparison} OLD.{c}" for c in _COLUMNS)


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    op.create_index("uq_users_email_lower", "users", [sa.text("lower(email)")], unique=True)
    plans = sa.table("plans", sa.column("name", sa.String), sa.column("limits", sa.JSON))
    op.execute(plans.update().where(plans.c.name == "self-hosted").values(limits=PLAN_LIMITS))
    if _is_postgres():
        unchanged = _unchanged("IS NOT DISTINCT FROM")
        op.execute(
            "CREATE OR REPLACE FUNCTION audit_logs_append_only() RETURNS trigger AS $$ BEGIN "
            "IF TG_OP = 'UPDATE' AND OLD.user_id IS NOT NULL AND NEW.user_id IS NULL "
            f"AND {unchanged} AND NEW.detail::text IS NOT DISTINCT FROM OLD.detail::text "
            f"THEN RETURN NEW; END IF; RAISE EXCEPTION '{MESSAGE}'; END; $$ LANGUAGE plpgsql"
        )
    else:
        op.execute("DROP TRIGGER audit_logs_no_update")
        unchanged = _unchanged("IS")
        op.execute(
            "CREATE TRIGGER audit_logs_no_update BEFORE UPDATE ON audit_logs "
            "WHEN NOT (OLD.user_id IS NOT NULL AND NEW.user_id IS NULL "
            f"AND {unchanged} AND NEW.detail IS OLD.detail) "
            f"BEGIN SELECT RAISE(ABORT, '{MESSAGE}'); END"
        )


def downgrade() -> None:
    plans = sa.table("plans", sa.column("name", sa.String), sa.column("limits", sa.JSON))
    op.execute(
        plans.update().where(plans.c.name == "self-hosted").values(limits={"unlimited": True})
    )
    op.drop_index("uq_users_email_lower", table_name="users")
    if _is_postgres():
        op.execute(
            "CREATE OR REPLACE FUNCTION audit_logs_append_only() RETURNS trigger AS $$ "
            f"BEGIN RAISE EXCEPTION '{MESSAGE}'; END; $$ LANGUAGE plpgsql"
        )
    else:
        op.execute("DROP TRIGGER audit_logs_no_update")
        op.execute(
            "CREATE TRIGGER audit_logs_no_update BEFORE UPDATE ON audit_logs "
            f"BEGIN SELECT RAISE(ABORT, '{MESSAGE}'); END"
        )
