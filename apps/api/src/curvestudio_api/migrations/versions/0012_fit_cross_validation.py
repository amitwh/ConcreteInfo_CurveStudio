"""The cross-validation of a fit: the ``fits.cv`` column.

The ``audit`` job cross-validates a stored fit on its dataset's table and stores the core's
``CrossValidation`` as JSON here, where ``GET /fits/{id}/audit`` reads it. The column is
nullable and has no default: a fit that has not been cross-validated keeps NULL, which says so.

Revision ID: 0012
Revises: 0011
"""

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("fits") as batch:
        batch.add_column(sa.Column("cv", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("fits") as batch:
        batch.drop_column("cv")
