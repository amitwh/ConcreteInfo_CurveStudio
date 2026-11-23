"""Edit → Preferences: the "Prepared by" identity on reports, the theme, and the one switch.

The two report fields are the name and organisation printed on an exported PDF report,
in an exported workbook's print header and along the top of an exported plot. They are
QSettings on this computer and nothing else: the desktop app has no account, so these
tests only ever read them back from the settings file. The third group is Privacy: the
switch for the anonymous install record, the only thing the app ever sends.
"""

import pytest
from desktop_data import GOOD_CSV
from openpyxl import load_workbook
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QMenu,
)

from curvestudio_desktop.controllers import exports
from curvestudio_desktop.install_record import INSTALL_RECORD_KEY, install_record_enabled
from curvestudio_desktop.profile import (
    MAX_BYLINE_FIELD,
    NAME_KEY,
    ORGANISATION_KEY,
    ReportIdentity,
    report_identity,
    set_report_identity,
)
from curvestudio_desktop.theme import THEME_CHOICES, THEME_SETTINGS_KEY, theme_preference
from curvestudio_desktop.views import main_window as main_window_module
from curvestudio_desktop.views.help_dialog import help_markdown
from curvestudio_desktop.views.preferences_dialog import HINT_TEXT, PRIVACY_TEXT, PreferencesDialog

IDENTITY = "Amit Haridas, ConcreteInfo"
"""What the settings hold and the exports pass on: the identity text, never the prefix."""
PREPARED = f"Prepared by {IDENTITY}"
"""What every renderer prints for it (the words are each renderer's own)."""


def ok_button(dialog: PreferencesDialog) -> QAbstractButton:
    button = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert button is not None
    return button


def test_preferences_dialog_saves_both_fields_to_settings(qtbot, settings: QSettings) -> None:
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Preferences"
    dialog.name_edit.setText("  Amit Haridas  ")
    dialog.organisation_edit.setText("ConcreteInfo")
    ok_button(dialog).click()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert settings.value(NAME_KEY) == "Amit Haridas"
    assert settings.value(ORGANISATION_KEY) == "ConcreteInfo"
    assert report_identity(settings) == ReportIdentity("Amit Haridas", "ConcreteInfo")


def test_preferences_dialog_reopens_with_saved_values(qtbot, settings: QSettings) -> None:
    first = PreferencesDialog(settings=settings)
    qtbot.addWidget(first)
    first.name_edit.setText("Amit Haridas")
    first.organisation_edit.setText("ConcreteInfo")
    first.theme_combo.setCurrentIndex(first.theme_combo.findData("dark"))
    ok_button(first).click()

    # a fresh settings object on the same file is what the next launch reads
    settings.sync()
    reopened = QSettings(settings.fileName(), QSettings.Format.IniFormat)
    second = PreferencesDialog(settings=reopened)
    qtbot.addWidget(second)
    assert second.name_edit.text() == "Amit Haridas"
    assert second.organisation_edit.text() == "ConcreteInfo"
    assert second.theme_combo.currentData() == "dark"


def test_preferences_dialog_clear_removes_the_byline(qtbot, settings: QSettings) -> None:
    set_report_identity(ReportIdentity("Amit Haridas", "ConcreteInfo"), settings)
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.name_edit.text() == "Amit Haridas"
    dialog.name_edit.clear()
    dialog.organisation_edit.clear()
    ok_button(dialog).click()

    assert settings.contains(NAME_KEY) is False
    assert settings.contains(ORGANISATION_KEY) is False
    assert report_identity(settings).byline() is None


def test_preferences_dialog_caps_input_at_100_characters(qtbot, settings: QSettings) -> None:
    assert MAX_BYLINE_FIELD == 100
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.name_edit.maxLength() == MAX_BYLINE_FIELD
    assert dialog.organisation_edit.maxLength() == MAX_BYLINE_FIELD
    dialog.name_edit.setText("n" * (MAX_BYLINE_FIELD + 50))
    ok_button(dialog).click()
    assert report_identity(settings).name == "n" * MAX_BYLINE_FIELD

    # the settings layer caps too, for a value that did not come through the field
    set_report_identity(ReportIdentity("o" * 150, "c" * 150), settings)
    assert report_identity(settings) == ReportIdentity("o" * 100, "c" * 100)


def test_preferences_dialog_keeps_a_pasted_block_on_one_line(qtbot, settings: QSettings) -> None:
    """A paste can leave line breaks in a single-line field; the printed byline is one line."""
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    dialog.name_edit.insert("Amy\nBo\tC\r\nD")
    assert "\n" in dialog.name_edit.text()  # the field itself keeps what was pasted
    ok_button(dialog).click()
    assert report_identity(settings).name == "Amy Bo C D"


def test_preferences_dialog_cancel_writes_nothing(qtbot, settings: QSettings) -> None:
    before = sorted(settings.allKeys())
    assert before, "the autouse mode seed gives the comparison something to hold"
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    dialog.name_edit.setText("Amit Haridas")
    dialog.organisation_edit.setText("ConcreteInfo")
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("dark"))
    cancel = dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel)
    assert cancel is not None
    cancel.click()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert sorted(settings.allKeys()) == before
    assert theme_preference(settings) == "system"

    escaped = PreferencesDialog(settings=settings)
    qtbot.addWidget(escaped)
    escaped.name_edit.setText("Amit Haridas")
    qtbot.keyClick(escaped, Qt.Key.Key_Escape)
    assert escaped.result() == QDialog.DialogCode.Rejected
    assert sorted(settings.allKeys()) == before


def test_preferences_dialog_theme_is_the_view_menu_setting(qtbot, settings: QSettings) -> None:
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    combo = dialog.theme_combo
    # the View → Theme choices, in its order; its labels carry '&' mnemonics a combo cannot use
    assert [combo.itemData(index) for index in range(combo.count())] == [
        preference for preference, _label in THEME_CHOICES
    ]
    assert [combo.itemText(index) for index in range(combo.count())] == [
        label.replace("&", "") for _preference, label in THEME_CHOICES
    ]
    assert combo.currentData() == "system"

    combo.setCurrentIndex(combo.findData("dark"))
    ok_button(dialog).click()
    assert settings.value(THEME_SETTINGS_KEY) == "dark"
    assert theme_preference(settings) == "dark"


def test_theme_choices_live_in_theme_and_feed_menu_and_dialog(
    make_window, qtbot, settings: QSettings
) -> None:
    """One list for both surfaces, so the dialog cannot drift from the View → Theme menu."""
    assert THEME_CHOICES == (("system", "&System"), ("light", "&Light"), ("dark", "&Dark"))
    window = make_window()
    qtbot.addWidget(window)
    theme_menu = next(m for m in window.menuBar().findChildren(QMenu) if m.title() == "&Theme")
    assert [action.text() for action in theme_menu.actions()] == [
        label for _preference, label in THEME_CHOICES
    ]
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    combo = dialog.theme_combo
    assert [combo.itemData(index) for index in range(combo.count())] == [
        preference for preference, _label in THEME_CHOICES
    ]
    # the window imports its dialog the way it imports the others: no local import to dodge a cycle
    assert main_window_module.PreferencesDialog is PreferencesDialog


def test_preferences_dialog_hint_names_every_surface_and_never_leaves(
    qtbot, settings: QSettings
) -> None:
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.hint_label.text() == HINT_TEXT
    for words in ("Prepared by", "PDF reports", "exported plots", "printed spreadsheets"):
        assert words in HINT_TEXT
    assert HINT_TEXT.endswith("It never leaves this computer.")


@pytest.mark.parametrize(
    ("identity", "text"),
    [
        (ReportIdentity("Amit Haridas", "ConcreteInfo"), "Amit Haridas, ConcreteInfo"),
        (ReportIdentity("Amit Haridas"), "Amit Haridas"),
        (ReportIdentity(organisation="ConcreteInfo"), "ConcreteInfo"),
        (ReportIdentity(), None),
    ],
    ids=["both", "name-only", "organisation-only", "neither"],
)
def test_report_identity_composes_the_byline(
    qtbot, settings: QSettings, identity: ReportIdentity, text: str | None
) -> None:
    """The identity text only: "Prepared by" is added by each renderer, as the web report does."""
    assert identity.byline() == text
    set_report_identity(identity, settings)  # and it round-trips through the settings file
    assert report_identity(settings) == identity
    assert report_identity(settings).byline() == text


def test_edit_menu_has_preferences_with_ctrl_comma(
    make_window, qtbot, settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    action = window.action_preferences
    assert action.text() == "Preferences…"
    assert action.shortcut().toString() == "Ctrl+,"
    assert action.statusTip() == "Report byline and theme"
    edit_menu = next(m for m in window.menuBar().findChildren(QMenu) if m.title() == "&Edit")
    entries = edit_menu.actions()
    assert entries[0] is window.action_undo
    assert entries[1].isSeparator()
    assert entries[2] is action and len(entries) == 3
    # and the help's shortcut table lists it, in the row style of its neighbours
    assert "| Preferences… (report byline and theme) | Ctrl+, |" in help_markdown()

    # the action opens the modal on the window's own settings; a theme chosen there is
    # applied at once, and the View → Theme check marks follow
    applied: list[str | None] = []
    real_apply = main_window_module.apply_theme

    def recording_apply(app, scheme=None, *, preference=None):
        applied.append(preference)
        return real_apply(app, scheme, preference=preference)

    monkeypatch.setattr(main_window_module, "apply_theme", recording_apply)
    opened: list[PreferencesDialog] = []

    def press_ok(self: PreferencesDialog) -> int:
        opened.append(self)
        self.name_edit.setText("Amit Haridas")
        self.theme_combo.setCurrentIndex(self.theme_combo.findData("dark"))
        ok_button(self).click()
        return int(self.result())

    monkeypatch.setattr(PreferencesDialog, "exec", press_ok)
    window.action_preferences.trigger()
    assert len(opened) == 1 and opened[0].parent() is window
    assert opened[0].result() == QDialog.DialogCode.Accepted
    assert report_identity(settings) == ReportIdentity("Amit Haridas")
    assert theme_preference(settings) == "dark"
    assert applied == ["dark"]
    assert window._theme_actions["dark"].isChecked()
    assert not window._theme_actions["system"].isChecked()

    def escape(self: PreferencesDialog) -> int:  # dismissed: nothing is applied again
        opened.append(self)
        return int(QDialog.DialogCode.Rejected)

    monkeypatch.setattr(PreferencesDialog, "exec", escape)
    window.action_preferences.trigger()
    assert len(opened) == 2
    assert applied == ["dark"]


def test_exports_carry_the_byline_when_set(
    make_window, qtbot, settings: QSettings, write_file, tmp_path, monkeypatch
) -> None:
    set_report_identity(ReportIdentity("Amit Haridas", "ConcreteInfo"), settings)
    window = make_window()
    qtbot.addWidget(window)
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("good.csv", GOOD_CSV))
    shown: list = []
    window.analysis.fit_finished.connect(lambda bundle, spec, meta: shown.append(bundle))
    window.analysis_panel.manual_view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)

    figures: list = []
    real_save = exports.save_figure
    monkeypatch.setattr(
        exports,
        "save_figure",
        lambda figure, target, fmt: figures.append(figure) or real_save(figure, target, fmt),
    )
    for name in ("fit.pdf", "fit.xlsx"):
        window.controller.paths.export_paths.append(tmp_path / name)
        window.export_results()
        qtbot.waitUntil(lambda name=name: (tmp_path / name).is_file(), timeout=30000)
        qtbot.waitUntil(lambda: not window.jobs.is_busy(), timeout=30000)
    window.controller.paths.export_paths.append(tmp_path / "fit.png")
    window.export_publication_plot()
    qtbot.waitUntil(lambda: len(figures) == 1, timeout=30000)
    qtbot.waitUntil(lambda: not window.jobs.is_busy(), timeout=30000)

    assert f"/Author ({IDENTITY})".encode() in (tmp_path / "fit.pdf").read_bytes()
    workbook = load_workbook(tmp_path / "fit.xlsx")
    assert [sheet.oddHeader.center.text for sheet in workbook.worksheets] == [PREPARED] * 5
    assert [text.get_text() for text in figures[0].texts if text.get_text() == PREPARED] == [
        PREPARED
    ]
    assert (tmp_path / "fit.png").stat().st_size > 0


def test_preferences_dialog_hint_wraps_instead_of_widening_the_dialog(
    qtbot, settings: QSettings
) -> None:
    """The hint is a long sentence and the dialog holds two short fields: the sentence wraps."""
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    hint = dialog.hint_label

    assert hint.wordWrap()
    assert dialog.minimumWidth() == 520
    assert dialog.sizeHint().width() < hint.fontMetrics().horizontalAdvance(HINT_TEXT)

    dialog.show()
    assert dialog.width() <= 600


# -- Privacy: the install-record switch ---------------------------------------------------------


def test_the_privacy_group_holds_the_install_record_switch_on_by_default(
    qtbot, settings: QSettings
) -> None:
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)

    titles = [group.title() for group in dialog.findChildren(QGroupBox)]
    assert titles == ["Reports", "Appearance", "Privacy"]
    check = dialog.findChild(QCheckBox, "preferencesInstallRecord")
    assert check is dialog.install_record_check
    assert check.text() == "Send an anonymous install record"
    assert check.isChecked() is True  # on unless the user turned it off
    assert settings.contains(INSTALL_RECORD_KEY) is False  # the default is an absent key
    assert install_record_enabled(settings) is True


def test_the_privacy_text_says_what_is_sent_and_that_nothing_else_is(
    qtbot, settings: QSettings
) -> None:
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)

    note = dialog.findChild(QLabel, "preferencesPrivacyHint")
    assert note is dialog.privacy_label
    assert note.text() == PRIVACY_TEXT
    assert note.wordWrap()
    for words in (
        "first launch",
        "after an upgrade",
        "one anonymous record",
        "version, operating system, package type",
        "no account, no name and no project",
        "Nothing else ever leaves this computer",
    ):
        assert words in PRIVACY_TEXT


def test_ok_writes_the_install_record_switch_and_cancel_writes_nothing(
    qtbot, settings: QSettings
) -> None:
    dialog = PreferencesDialog(settings=settings)
    qtbot.addWidget(dialog)
    dialog.install_record_check.setChecked(False)
    assert settings.contains(INSTALL_RECORD_KEY) is False  # nothing is written before OK
    ok_button(dialog).click()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert settings.value(INSTALL_RECORD_KEY) is False
    assert install_record_enabled(settings) is False

    # a fresh settings object on the same file is what the next launch reads
    settings.sync()
    reopened = PreferencesDialog(
        settings=QSettings(settings.fileName(), QSettings.Format.IniFormat)
    )
    qtbot.addWidget(reopened)
    assert reopened.install_record_check.isChecked() is False

    reopened.install_record_check.setChecked(True)
    before = sorted(settings.allKeys())
    cancel = reopened.buttons.button(QDialogButtonBox.StandardButton.Cancel)
    assert cancel is not None
    cancel.click()
    assert sorted(settings.allKeys()) == before
    assert install_record_enabled(settings) is False
