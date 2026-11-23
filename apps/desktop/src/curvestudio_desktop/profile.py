"""The report identity: who an exported report says it was prepared by.

The desktop app is offline and has no accounts, so the identity on a report is two
settings the user typed in Edit → Preferences: kept in QSettings under
``profile/report_name`` and ``profile/report_organisation`` and read by every export.
The exports pass the identity text on (``"Alice"``, ``"Lab"`` or ``"Alice, Lab"``); each
report adds the "Prepared by" words itself, as the web report does. The settings exist
only on this computer; nothing in the app sends them anywhere.

Clearing a field removes its key, so a user who never filled the dialog gets exactly
the reports the app has always written.
"""

from dataclasses import dataclass

from PySide6.QtCore import QSettings

NAME_KEY = "profile/report_name"
ORGANISATION_KEY = "profile/report_organisation"
MAX_BYLINE_FIELD = 100
"""Storage length of each field; the Preferences dialog caps its input at the same value."""


@dataclass(frozen=True)
class ReportIdentity:
    """The two optional parts of a "Prepared by" line."""

    name: str = ""
    organisation: str = ""

    def byline(self) -> str | None:
        """The identity text a report's byline names: the parts that are set, joined with ", ".

        ``None`` when neither part is set: the caller then writes the report as it
        always did, with no byline at all. The words "Prepared by" are not part of it:
        every renderer adds them itself, the web report included.
        """
        parts = [part for part in (self.name, self.organisation) if part]
        if not parts:
            return None
        return ", ".join(parts)


def _store(settings: QSettings | None) -> QSettings:
    return settings if settings is not None else QSettings()


def _field(value: object) -> str:
    """One field as stored and as printed: surrounding whitespace off, cut to the maximum."""
    return ("" if value is None else str(value).strip())[:MAX_BYLINE_FIELD]


def report_identity(settings: QSettings | None = None) -> ReportIdentity:
    """The stored identity (both parts empty when it was never set)."""
    store = _store(settings)
    return ReportIdentity(
        name=_field(store.value(NAME_KEY, "")),
        organisation=_field(store.value(ORGANISATION_KEY, "")),
    )


def set_report_identity(identity: ReportIdentity, settings: QSettings | None = None) -> None:
    """Store the identity; an empty part removes its key, so clearing restores the unset output."""
    store = _store(settings)
    for key, value in ((NAME_KEY, identity.name), (ORGANISATION_KEY, identity.organisation)):
        field = _field(value)
        if field:
            store.setValue(key, field)
        else:
            store.remove(key)
    store.sync()


__all__ = [
    "MAX_BYLINE_FIELD",
    "NAME_KEY",
    "ORGANISATION_KEY",
    "ReportIdentity",
    "report_identity",
    "set_report_identity",
]
