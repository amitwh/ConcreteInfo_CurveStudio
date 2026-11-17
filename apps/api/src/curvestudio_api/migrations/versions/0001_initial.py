"""Initial schema: users, refresh_tokens, datasets, fits, jobs, audit_logs, plans.

Revision ID: 0001
Revises:
"""

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SELF_HOSTED_PLAN_ID = uuid.UUID("5e1f0057-0000-4000-8000-000000000001")


def _timestamps() -> list[sa.Column[object]]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by", sa.Uuid(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_refresh_tokens"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_refresh_tokens_user_id_users", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("n_rows", sa.Integer(), nullable=False),
        sa.Column("columns", sa.JSON(), nullable=False),
        sa.Column("x_column", sa.String(255), nullable=False),
        sa.Column("y_column", sa.String(255), nullable=False),
        sa.Column("weight_column", sa.String(255), nullable=True),
        sa.Column("issues", sa.JSON(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_datasets"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_datasets_user_id_users", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_datasets_user_id", "datasets", ["user_id"])
    op.create_table(
        "fits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("bundle", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_fits"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_fits_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"], name="fk_fits_dataset_id_datasets", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_fits_user_id", "fits", ["user_id"])
    op.create_index("ix_fits_dataset_id", "fits", ["dataset_id"])
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("result_ref", sa.Uuid(), nullable=True),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_jobs_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"], name="fk_jobs_dataset_id_datasets", ondelete="CASCADE"
        ),
    )
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_dataset_id", "jobs", ["dataset_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("entity", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(64), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_audit_logs_user_id_users"),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    _make_audit_logs_append_only()
    plans = op.create_table(
        "plans",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("limits", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_plans"),
        sa.UniqueConstraint("name", name="uq_plans_name"),
    )
    now = datetime.now(UTC)
    op.bulk_insert(
        plans,
        [
            {
                "id": SELF_HOSTED_PLAN_ID,
                "name": "self-hosted",
                "limits": {"unlimited": True},
                "created_at": now,
                "updated_at": now,
            }
        ],
    )


def _make_audit_logs_append_only() -> None:
    message = "audit_logs is append-only"
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE FUNCTION audit_logs_append_only() RETURNS trigger AS $$ "
            f"BEGIN RAISE EXCEPTION '{message}'; END; $$ LANGUAGE plpgsql"
        )
        op.execute(
            "CREATE TRIGGER audit_logs_append_only BEFORE UPDATE OR DELETE ON audit_logs "
            "FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only()"
        )
    else:
        for event in ("UPDATE", "DELETE"):
            op.execute(
                f"CREATE TRIGGER audit_logs_no_{event.lower()} BEFORE {event} ON audit_logs "
                f"BEGIN SELECT RAISE(ABORT, '{message}'); END"
            )


def downgrade() -> None:
    op.drop_table("plans")
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP TRIGGER audit_logs_append_only ON audit_logs")
        op.execute("DROP FUNCTION audit_logs_append_only()")
    op.drop_table("audit_logs")
    op.drop_table("jobs")
    op.drop_table("fits")
    op.drop_table("datasets")
    op.drop_table("refresh_tokens")
    op.drop_table("users")
