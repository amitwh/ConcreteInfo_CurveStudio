"""The blueprint-shaped domains. Each is a package with ``models.py`` (its tables),
``schemas.py`` (request and response shapes), ``service.py`` (the rules) and ``router.py``
(the HTTP surface).

Importing this package imports no domain: :func:`register_models` brings their tables in.
"""

from importlib import import_module


def register_models() -> None:
    """Import the ``models`` module of every domain.

    Importing a domain's models puts its tables on ``Base.metadata``, which Alembic and the
    model-versus-migration test compare with the database. Every place that makes that
    comparison must call this first.
    """
    for domain in ("admin", "errors", "installs"):
        import_module(f"curvestudio_api.modules.{domain}.models")
