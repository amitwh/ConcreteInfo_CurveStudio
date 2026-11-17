"""Alembic environment: runs against the engine handed over by ``upgrade_to_head``."""

from alembic import context

from curvestudio_api.orm import Base

config = context.config
engine = config.attributes["engine"]

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()
