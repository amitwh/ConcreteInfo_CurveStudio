"""CurveStudio API: a FastAPI backend over the ``curvestudio.workflow`` layer."""

__version__ = "1.13.0"

API_PREFIX = "/api/v1"
"""Where every API route lives. It is defined in this leaf module so that the application and
the middleware share one constant without importing each other."""

__all__ = ["API_PREFIX", "__version__"]
