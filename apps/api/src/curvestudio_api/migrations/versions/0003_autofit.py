"""Auto-fit: the ``autofits`` table, job progress, and the link from a fit to the row it opened.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autofits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("n_candidates", sa.Integer(), nullable=False),
        sa.Column("best_model_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_autofits"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_autofits_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_autofits_dataset_id_datasets",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_autofits_user_id", "autofits", ["user_id"])
    op.create_index("ix_autofits_dataset_id", "autofits", ["dataset_id"])
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("progress", sa.JSON(), nullable=True))
    with op.batch_alter_table("fits") as batch:
        batch.add_column(sa.Column("autofit_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("candidate_id", sa.String(100), nullable=True))
        batch.create_foreign_key(
            "fk_fits_autofit_id_autofits", "autofits", ["autofit_id"], ["id"], ondelete="SET NULL"
        )
        batch.create_index("ix_fits_autofit_id", ["autofit_id"])
        batch.create_unique_constraint("uq_fits_autofit_id", ["autofit_id", "candidate_id"])


def downgrade() -> None:
    with op.batch_alter_table("fits") as batch:
        batch.drop_constraint("uq_fits_autofit_id", type_="unique")
        batch.drop_index("ix_fits_autofit_id")
        batch.drop_constraint("fk_fits_autofit_id_autofits", type_="foreignkey")
        batch.drop_column("candidate_id")
        batch.drop_column("autofit_id")
    with op.batch_alter_table("jobs") as batch:
        batch.drop_column("progress")
    op.drop_index("ix_autofits_dataset_id", table_name="autofits")
    op.drop_index("ix_autofits_user_id", table_name="autofits")
    op.drop_table("autofits")
