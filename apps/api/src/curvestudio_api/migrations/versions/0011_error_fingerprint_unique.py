"""One row per error: ``error_events.fingerprint`` becomes unique.

The roll-up looks a fingerprint up and inserts when it finds nothing, so two callers reporting
the same new error at once both inserted, and the console showed one error as several rows.
The plain lookup index gives way to the unique constraint the model declares (``unique=True``
on the column, named by the convention); as in 0003, batch mode adds it to the existing table.

Any rows a database at 0010 already holds for one fingerprint are collapsed first (the lowest
id stays), so the constraint can be created whatever the table holds.

Revision ID: 0011
Revises: 0010
"""

import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A self-join, not MIN(id): PostgreSQL has no minimum for UUIDs (SQLite, where the
    # tests run, accepts it on any column), and both engines compare UUIDs directly.
    op.execute(
        sa.text(
            "DELETE FROM error_events WHERE id IN ("
            "SELECT duplicate.id FROM error_events AS duplicate "
            "JOIN error_events AS kept"
            " ON duplicate.fingerprint = kept.fingerprint AND duplicate.id > kept.id)"
        )
    )
    op.drop_index("ix_error_events_fingerprint", table_name="error_events")
    with op.batch_alter_table("error_events") as batch:
        batch.create_unique_constraint("uq_error_events_fingerprint", ["fingerprint"])


def downgrade() -> None:
    with op.batch_alter_table("error_events") as batch:
        batch.drop_constraint("uq_error_events_fingerprint", type_="unique")
    op.create_index("ix_error_events_fingerprint", "error_events", ["fingerprint"])
