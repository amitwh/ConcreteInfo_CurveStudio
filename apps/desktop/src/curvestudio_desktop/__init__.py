"""ConcreteInfo CurveStudio desktop application (PySide6).

The package is organised model-view-controller: ``controllers`` hold the logic and
stay free of Qt widgets (usable headless), ``views`` are thin widgets, and every
engine call goes through :mod:`curvestudio.workflow`, :mod:`curvestudio.project`
or :mod:`curvestudio.data` types — never the maths layers directly.
"""

__version__ = "0.11.0"

APP_NAME = "ConcreteInfo CurveStudio"
ORG_NAME = "ConcreteInfo"
NO_NETWORK_NOTE = "No network access"
