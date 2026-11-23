"""Alembic environment: runs against the engine handed over by ``upgrade_to_head``."""

from alembic import context

from curvestudio_api.modules import register_models
from curvestudio_api.orm import Base

config = context.config
engine = config.attributes["engine"]

# Every domain's table joins ``Base.metadata`` before it is compared with the database.
register_models()

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()
