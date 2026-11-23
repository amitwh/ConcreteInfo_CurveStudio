"""Account role and activity: ``users.role``, the activity stamps, ``users.preferences``
and ``user_daily_activity``.

``role`` is ``member`` or ``admin``; existing accounts become ``member`` via the server
default, new rows get it from the ORM. ``preferences`` is the account's JSON preference
object: the column carries no server default — existing rows are backfilled with today's
defaults and new rows get the ORM default — so handlers always see an object.

``user_daily_activity`` holds at most one row per account per (IST) day; the ``(user_id,
day)`` pair is its primary key, which is also the uniqueness the schema needs. Rows cascade
away with the account.

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

DEFAULT_PREFERENCES = {"error_reports": True, "report_byline": False}

_users = sa.table("users", sa.column("preferences", sa.JSON))


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("role", sa.String(20), nullable=False, server_default="member")
    )
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("last_active_on", sa.Date(), nullable=True))
    op.add_column("users", sa.Column("preferences", sa.JSON(), nullable=True))
    op.execute(_users.update().values(preferences=DEFAULT_PREFERENCES))
    op.create_table(
        "user_daily_activity",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "day", name="pk_user_daily_activity"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_daily_activity_user_id_users",
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    op.drop_table("user_daily_activity")
    # SQLite needs a table rebuild to drop a column; drop and restore the functional
    # unique index so the rebuild does not lose it.
    op.drop_index("uq_users_email_lower", table_name="users")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("preferences")
        batch.drop_column("last_active_on")
        batch.drop_column("last_login_at")
        batch.drop_column("role")
    op.create_index("uq_users_email_lower", "users", [sa.text("lower(email)")], unique=True)
