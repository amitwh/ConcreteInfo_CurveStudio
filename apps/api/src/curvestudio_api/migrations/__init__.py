"""Alembic migrations, applied programmatically at start-up (``upgrade_to_head``)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine

SCRIPT_LOCATION = Path(__file__).parent


def _config(engine: Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", str(SCRIPT_LOCATION))
    config.attributes["engine"] = engine
    return config


def upgrade_to_head(engine: Engine) -> None:
    """Bring the database to the newest schema (a no-op when it already is)."""
    command.upgrade(_config(engine), "head")


def current_revision_is_head(engine: Engine) -> bool:
    """Whether the database is at the newest migration (used by the readiness probe)."""
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(_config(engine)).get_current_head()
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision() == head
