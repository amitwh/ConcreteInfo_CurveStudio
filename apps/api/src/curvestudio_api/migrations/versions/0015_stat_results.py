"""Stored statistical results: the ``stat_results`` table.

One row per run of a statistical tool on the web: the choices a person sent, the result the
core returned, and the result's size for the bytes quota. ``dataset_id`` is nullable because a
tool that needs no data (a sample-size calculation) can be stored without a dataset; both
foreign keys cascade, so deleting a dataset or an account takes its results with it.

Revision ID: 0015
Revises: 0014
"""

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stat_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=True),
        sa.Column("tool", sa.String(40), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_stat_results"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_stat_results_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_stat_results_dataset_id_datasets",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_stat_results_user_id", "stat_results", ["user_id"])
    op.create_index("ix_stat_results_dataset_id", "stat_results", ["dataset_id"])


def downgrade() -> None:
    op.drop_index("ix_stat_results_dataset_id", table_name="stat_results")
    op.drop_index("ix_stat_results_user_id", table_name="stat_results")
    op.drop_table("stat_results")
