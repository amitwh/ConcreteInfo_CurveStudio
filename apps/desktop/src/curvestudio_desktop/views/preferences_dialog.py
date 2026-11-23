"""PreferencesDialog: the "Prepared by" identity on reports, the theme, and the one switch.

Edit → Preferences… (Ctrl+,). The two report fields are printed as a "Prepared by …" line
under the title of an exported PDF report, in the print header of an exported workbook and
along the top of an exported plot. They are kept in QSettings on this computer and read by
every export; the app has no account, so they never leave it. The Privacy group holds the
switch for the anonymous install record, the only thing the app ever sends. OK writes all
three groups; Cancel and Escape write nothing.
"""

from typing import cast

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from curvestudio_desktop.install_record import install_record_enabled, set_install_record_enabled
from curvestudio_desktop.profile import (
    MAX_BYLINE_FIELD,
    ReportIdentity,
    report_identity,
    set_report_identity,
)
from curvestudio_desktop.theme import (
    THEME_CHOICES,
    Preference,
    set_theme_preference,
    theme_preference,
)

TITLE = "Preferences"
HINT_TEXT = (
    'Printed as "Prepared by …" on PDF reports, exported plots and printed spreadsheets. '
    "It never leaves this computer."
)
PRIVACY_TEXT = (
    "On first launch, and again after an upgrade, CurveStudio sends one anonymous record "
    "(version, operating system, package type) so we know how many installations exist. "
    "It holds no account, no name and no project. Nothing else ever leaves this computer."
)


def _one_line(field: QLineEdit) -> str:
    """The field's text on one line: a paste can bring line breaks into a single-line field."""
    return " ".join(field.text().split())


class PreferencesDialog(QDialog):
    """Edit → Preferences…: who reports are prepared by, the theme, and the install record."""

    def __init__(self, parent: QWidget | None = None, *, settings: QSettings | None = None) -> None:
        super().__init__(parent)
        self._settings = settings if settings is not None else QSettings()
        self.setObjectName("preferencesDialog")
        self.setWindowTitle(TITLE)
        # Wide enough for the two fields; the hints wrap inside it rather than widening it.
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(8)
        layout.addWidget(self._build_reports())
        layout.addWidget(self._build_appearance())
        layout.addWidget(self._build_privacy())
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setDefault(True)
        layout.addWidget(self.buttons)

    # -- construction ------------------------------------------------------------------------

    def _build_reports(self) -> QGroupBox:
        identity = report_identity(self._settings)
        reports = QGroupBox("Reports")
        layout = QVBoxLayout(reports)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.name_edit = QLineEdit(identity.name)
        self.organisation_edit = QLineEdit(identity.organisation)
        for field, object_name in (
            (self.name_edit, "preferencesName"),
            (self.organisation_edit, "preferencesOrganisation"),
        ):
            field.setMaxLength(MAX_BYLINE_FIELD)
            field.setObjectName(object_name)
        form.addRow("Name shown on reports", self.name_edit)
        form.addRow("Organisation", self.organisation_edit)
        layout.addLayout(form)
        self.hint_label = QLabel(HINT_TEXT)
        self.hint_label.setWordWrap(True)
        self.hint_label.setObjectName("preferencesHint")
        layout.addWidget(self.hint_label)
        return reports

    def _build_appearance(self) -> QGroupBox:
        appearance = QGroupBox("Appearance")
        form = QFormLayout(appearance)
        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("preferencesTheme")
        for preference, label in THEME_CHOICES:  # the View → Theme choices, in that order
            self.theme_combo.addItem(label.replace("&", ""), preference)  # '&' is a menu mnemonic
        self.theme_combo.setCurrentIndex(
            self.theme_combo.findData(theme_preference(self._settings))
        )
        form.addRow("Theme", self.theme_combo)
        return appearance

    def _build_privacy(self) -> QGroupBox:
        privacy = QGroupBox("Privacy")
        layout = QVBoxLayout(privacy)
        self.install_record_check = QCheckBox("Send an anonymous install record")
        self.install_record_check.setObjectName("preferencesInstallRecord")
        self.install_record_check.setChecked(install_record_enabled(self._settings))
        layout.addWidget(self.install_record_check)
        self.privacy_label = QLabel(PRIVACY_TEXT)
        self.privacy_label.setWordWrap(True)
        self.privacy_label.setObjectName("preferencesPrivacyHint")
        layout.addWidget(self.privacy_label)
        return privacy

    # -- buttons -----------------------------------------------------------------------------

    def _accept(self) -> None:
        """OK: write all three groups, then close as accepted."""
        set_report_identity(
            ReportIdentity(_one_line(self.name_edit), _one_line(self.organisation_edit)),
            self._settings,
        )
        set_theme_preference(cast("Preference", self.theme_combo.currentData()), self._settings)
        set_install_record_enabled(self.install_record_check.isChecked(), self._settings)
        self.accept()
