"""Request statistics for the admin console: ``request_stats_daily``.

One row per route and day (``(day, route_template)`` is unique) with the count, the error
count, a response-time histogram and the percentiles.

Revision ID: 0010
Revises: 0009
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "request_stats_daily",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("route_template", sa.String(200), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("errors", sa.Integer(), nullable=False),
        sa.Column("histogram", sa.JSON(), nullable=False),
        sa.Column("p50_ms", sa.Integer(), nullable=False),
        sa.Column("p95_ms", sa.Integer(), nullable=False),
        sa.Column("max_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_request_stats_daily"),
        sa.UniqueConstraint("day", "route_template", name="uq_request_stats_daily_day"),
    )


def downgrade() -> None:
    op.drop_table("request_stats_daily")
