"""ConcreteInfo CurveStudio desktop application (PySide6).

The package is organised model-view-controller: ``controllers`` hold the logic and
stay free of Qt widgets (usable headless), ``views`` are thin widgets, and every
engine call goes through :mod:`curvestudio.workflow`, :mod:`curvestudio.project`
or :mod:`curvestudio.data` types — never the maths layers directly.
"""

__version__ = "1.12.0"

APP_NAME = "ConcreteInfo CurveStudio"
ORG_NAME = "ConcreteInfo"

INSTALL_RECORD_URL = "https://curvestudio.concreteinfo.cloud/api/v1/installs"
"""The one address the desktop app ever connects to: its anonymous install record."""

NO_NETWORK_NOTE = "Offline-ready · no account"
"""What the status bar says: the app works fully offline and has no account. The one thing it
sends is the install record (:mod:`curvestudio_desktop.install_record`)."""
