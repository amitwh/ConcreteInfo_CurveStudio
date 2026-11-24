"""The other columns of a dataset: the ``datasets.table`` column.

A dataset used to keep the x, y and weight columns of its upload and nothing else. It now
also keeps the table's other columns as CSV bytes here, so that the statistical tools can
read a group label, a second measurement or a batch code; the bytes count against the
dataset-bytes quota. The column is nullable and has no default: a dataset stored before it
exists keeps NULL, which says so. Such a dataset is never migrated.

Revision ID: 0013
Revises: 0012
"""

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(sa.Column("table", sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("datasets") as batch:
        batch.drop_column("table")
