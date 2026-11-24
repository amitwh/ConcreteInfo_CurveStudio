"""The columns of a stored table: the ``datasets.table_columns`` column.

The list of every column of a dataset's table with its kind and blank cells, computed once
when the upload is parsed and kept as JSON here, so that no route has to read and parse a
stored table that may be tens of megabytes to say what it holds. The column is nullable and
has no default: a dataset stored before it exists keeps NULL, which says that it was not
stored in this form. Such a dataset is never migrated.

Revision ID: 0014
Revises: 0013
"""

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(sa.Column("table_columns", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("datasets") as batch:
        batch.drop_column("table_columns")
