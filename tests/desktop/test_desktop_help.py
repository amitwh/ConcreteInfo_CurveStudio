"""Help, About and the vendored icon set: the user-facing documentation surfaces.

The canonical help text lives in ``docs/help-content.md``; it is copied verbatim into
the desktop package so the frozen application can render it. These tests pin that copy
to the source, the eleven sections in order, the Markdown rendering (tables, lists,
bold, inline code), search, the menu/shortcut wiring, the About contract (logo, links,
runtime versions, credits) and the Lucide icon mapping.
"""

from importlib import resources
from pathlib import Path

import numpy
import PySide6
import pytest
import scipy
from PySide6.QtCore import Qt, qVersion
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from curvestudio_desktop import __version__
from curvestudio_desktop.views import icons

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_HELP = REPO_ROOT / "docs" / "help-content.md"

#: Every internal name the views ask :func:`icons.icon` for. A missing entry must fail
#: here, not render blank in the toolbar.
INTERNAL_ICON_NAMES = {
    "new",
    "open",
    "save",
    "import",
    "clipboard",
    "undo",
    "log-x",
    "log-y",
    "reset",
    "export",
    "details",
    "help",
    "about",
    "collapse",
    "expand",
    "float",
    "close",
    "panel-project",
    "panel-analysis",
    "panel-table",
}


def canonical_section_titles() -> tuple[str, ...]:
    """The ``##`` headings of the canonical document, in file order."""
    return tuple(
        line[3:].strip()
        for line in CANONICAL_HELP.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    )


def lucide_dir():
    return resources.files("curvestudio_desktop") / "resources" / "icons" / "lucide"


# -- icons ------------------------------------------------------------------------------------


def test_every_internal_icon_name_maps_to_an_existing_vendored_file() -> None:
    assert set(icons._ICONS) == INTERNAL_ICON_NAMES
    for name, filename in icons._ICONS.items():
        asset = lucide_dir() / filename
        assert asset.is_file(), f"{name} maps to a missing file {filename}"
        source = asset.read_text(encoding="utf-8")
        assert "currentColor" in source, f"{filename} is not a Lucide stroke icon"


def test_every_icon_renders_a_non_null_pixmap_at_16_and_24_px(qtbot) -> None:
    for name in sorted(INTERNAL_ICON_NAMES):
        rendered = icons.icon(name, "#335577")
        for size in (16, 24):
            pixmap = rendered.pixmap(size, size)
            assert not pixmap.isNull(), f"{name} rendered a null {size}px pixmap"
            assert pixmap.width() >= size and pixmap.height() >= size


def test_an_unknown_icon_name_raises_instead_of_returning_blank(qtbot) -> None:
    with pytest.raises(KeyError):
        icons.icon("not-a-real-icon")


def test_icons_use_one_stroke_weight_and_the_theme_colour() -> None:
    for name in INTERNAL_ICON_NAMES:
        source = icons._svg_text(name, "#123456")
        assert "currentColor" not in source, f"{name} kept the unresolved colour"
        assert "#123456" in source, f"{name} did not take the theme colour"
        assert source.count('stroke-width="1.8"') == 1, (
            f"{name} does not use the toolbar's single stroke weight"
        )
        assert 'stroke-width="2"' not in source


def test_the_lucide_licence_is_packaged() -> None:
    licence = lucide_dir() / "LUCIDE-LICENSE.txt"
    assert licence.is_file()
    assert "ISC License" in licence.read_text(encoding="utf-8")


# -- help content and rendering ---------------------------------------------------------------


def test_help_content_ships_the_canonical_document_verbatim() -> None:
    from curvestudio_desktop.views.help_dialog import help_markdown

    assert help_markdown() == CANONICAL_HELP.read_text(encoding="utf-8")


def test_help_dialog_lists_the_eleven_canonical_sections_in_order(qtbot, settings) -> None:
    from curvestudio_desktop.views.help_dialog import HelpDialog

    dialog = HelpDialog(settings=settings)
    qtbot.addWidget(dialog)
    titles = dialog.sections()
    assert len(titles) == 11
    assert titles == canonical_section_titles()
    assert [dialog.section_list.item(i).text() for i in range(11)] == list(titles)
    assert titles[0] == "1. Getting started"
    assert titles[-1] == "11. About, licence, projects and policy"


def test_markdown_renders_tables_lists_bold_links_and_inline_code() -> None:
    from curvestudio_desktop.views.help_dialog import (
        help_markdown,
        markdown_to_html,
        parse_sections,
    )

    sections = {section.title: section for section in parse_sections(help_markdown())}

    options = markdown_to_html(sections["3. Auto-fit"].body)
    assert "<table" in options and "<th>Option</th>" in options
    assert "<code>auto</code>" in options
    assert "<strong>Options</strong>" in options

    grammar = markdown_to_html(sections["4. Manual fit and custom equations"].body)
    assert "<ul>" in grammar and "<li>" in grammar
    assert "<strong>lower &lt; upper</strong>" in grammar
    assert "<code>log10(x)</code>" in grammar

    steps = markdown_to_html(sections["1. Getting started"].body)
    assert "<ol>" in steps and steps.count("<li>") >= 5

    about = markdown_to_html(sections["11. About, licence, projects and policy"].body)
    assert '<a href="https://www.concreteinfo.in">' in about
    assert "<em>" in about or "<strong>" in about


def test_selecting_a_section_renders_its_text_in_the_browser(qtbot, settings) -> None:
    from curvestudio_desktop.views.help_dialog import HelpDialog

    dialog = HelpDialog(settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.select_section("3. Auto-fit") is True
    assert dialog.current_section() == "3. Auto-fit"
    html = dialog.browser.toHtml()
    assert "3. Auto-fit" in html
    assert "<table" in html  # the options and ranking-column tables survive rendering
    text = dialog.browser.toPlainText()
    assert "multi-starts" in text
    assert "adj. R²" in text


def test_search_filters_the_section_list_and_jumps_to_the_first_body_match(qtbot, settings) -> None:
    from curvestudio_desktop.views.help_dialog import HelpDialog

    dialog = HelpDialog(settings=settings)
    qtbot.addWidget(dialog)
    dialog.search.setText("LOWER < UPPER")  # body text, case-insensitively
    QApplication.processEvents()
    visible = [
        dialog.section_list.item(i)
        for i in range(dialog.section_list.count())
        if not dialog.section_list.item(i).isHidden()
    ]
    assert [item.text() for item in visible] == ["4. Manual fit and custom equations"]
    assert dialog.current_section() == "4. Manual fit and custom equations"

    dialog.search.clear()
    QApplication.processEvents()
    assert all(
        not dialog.section_list.item(i).isHidden() for i in range(dialog.section_list.count())
    )
    assert len(dialog.sections()) == 11


def test_an_empty_search_result_says_so(qtbot, settings) -> None:
    from curvestudio_desktop.views.help_dialog import HelpDialog

    dialog = HelpDialog(settings=settings)
    qtbot.addWidget(dialog)
    dialog.search.setText("no section contains this phrase")
    QApplication.processEvents()
    assert all(dialog.section_list.item(i).isHidden() for i in range(dialog.section_list.count()))
    assert "No sections match" in dialog.browser.toPlainText()
    assert dialog.current_section() is None


def test_help_dialog_is_non_modal_and_remembers_its_geometry(qtbot, settings) -> None:
    from curvestudio_desktop.views.help_dialog import GEOMETRY_KEY, HelpDialog

    dialog = HelpDialog(settings=settings)
    qtbot.addWidget(dialog)
    assert dialog.isModal() is False
    assert dialog.windowFlags() & Qt.WindowType.Window
    dialog.resize(777, 523)
    dialog.save_geometry()
    assert settings.contains(GEOMETRY_KEY)

    fresh = HelpDialog(settings=settings)
    qtbot.addWidget(fresh)
    assert fresh.size() == dialog.size()


# -- main-window wiring -----------------------------------------------------------------------


def test_help_menu_and_f1_open_the_browser(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    actions = {action.text(): action for action in window.help_menu.actions()}
    assert "&Help contents" in actions
    assert "&Keyboard shortcuts" in actions
    assert "&About" in actions
    assert "About &Qt" in actions
    assert window.action_help_contents.shortcut().toString() == "F1"

    window.action_help_contents.trigger()
    dialog = window.help_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert dialog.isModal() is False
    assert dialog.current_section() is not None


def test_keyboard_shortcuts_action_opens_that_section(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.action_shortcuts.trigger()
    assert window.help_dialog is not None
    assert "Keyboard shortcuts" in (window.help_dialog.current_section() or "")


def test_about_qt_calls_the_application_dialog(make_window, qtbot, monkeypatch) -> None:
    window = make_window()
    qtbot.addWidget(window)
    called: list[object] = []
    monkeypatch.setattr(QApplication, "aboutQt", lambda self: called.append(self))
    window.action_about_qt.trigger()
    assert called == [QApplication.instance()]


# -- About dialog -----------------------------------------------------------------------------


def test_about_dialog_shows_logo_version_description_and_links(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "About CurveStudio"

    asset = resources.files("curvestudio_desktop") / "resources" / "brand" / "concreteinfo-logo.png"
    expected = QPixmap()
    assert expected.loadFromData(asset.read_bytes(), "PNG")
    shown = dialog.logo_label.pixmap()
    assert shown is not None and not shown.isNull()
    # never inverted or recoloured: exactly the staged brand asset
    assert shown.toImage() == expected.toImage()
    assert "#ffffff" in dialog.logo_label.styleSheet().lower()

    assert __version__ in dialog.version_label.text()
    assert "fit" in dialog.description_label.text().lower()
    assert "https://www.concreteinfo.in" in dialog.website_label.text()
    assert dialog.website_label.openExternalLinks() is True
    assert "AGPL-3.0-or-later" in dialog.licence_label.text()
    assert "https://www.gnu.org/licenses/agpl-3.0.html" in dialog.licence_label.text()
    assert dialog.website_label.openExternalLinks() is True


def test_about_dialog_reads_the_runtime_stack_versions(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    text = dialog.version_text()
    for value in (
        qVersion(),
        PySide6.__version__,
        numpy.__version__,
        scipy.__version__,
    ):
        assert value in text
    import platform

    assert platform.python_version() in text


def test_about_dialog_lists_every_credit(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    credits = dialog.credits_text()
    for token in ("LGPL-3.0", "BSD-3-Clause", "MIT", "OFL-1.1", "ISC"):
        assert token in credits
    assert "NumPy" in credits and "SciPy" in credits and "SymPy" in credits
    assert "pyqtgraph" in credits
    assert "Inter" in credits and "JetBrains Mono" in credits
    assert "Lucide" in credits


def test_full_text_keeps_the_existing_licence_and_version_contract(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    text = dialog.full_text()
    assert __version__ in text
    assert "AGPL" in text
    assert "PySide6" in text and qVersion()[:3] in text


# -- screenshots in the help ------------------------------------------------------------------


def test_markdown_renders_a_screenshot_with_its_caption() -> None:
    from curvestudio_desktop.views.help_dialog import markdown_to_html

    html = markdown_to_html("![The workspace](images/workspace.png)\n\nAfter the image.")
    assert '<img src="images/workspace.png" width="700"' in html
    assert 'alt="The workspace"' in html
    assert "The workspace</span>" in html  # the caption is visible, not only the alt text
    assert "<p>After the image.</p>" in html


def test_every_screenshot_the_help_references_is_packaged() -> None:
    import re

    from curvestudio_desktop.views.help_dialog import help_markdown

    references = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", help_markdown())
    assert references, "the help references no screenshots at all"
    help_root = resources.files("curvestudio_desktop") / "resources" / "help"
    for reference in references:
        asset = help_root / reference
        assert asset.is_file(), f"{reference} is referenced but not packaged"
        assert asset.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{reference} is not a PNG"


def test_the_help_browser_resolves_relative_image_sources(qtbot, settings) -> None:
    from curvestudio_desktop.views.help_dialog import HelpDialog

    dialog = HelpDialog(settings=settings)
    qtbot.addWidget(dialog)
    search_paths = dialog.browser.searchPaths()
    assert search_paths
    assert Path(search_paths[0]).name == "help"
    assert (Path(search_paths[0]) / "images").is_dir()


def test_about_lists_the_other_projects_and_contact(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import CONTACT_EMAIL, OTHER_PROJECTS, AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    html = dialog.sections.toHtml()
    for name, url, _description in OTHER_PROJECTS:
        assert name in html, name
        assert url in html, url
    assert CONTACT_EMAIL in html
    assert "https://www.concreteinfo.in" in html
    assert "GitHub Desktop for Linux" in OTHER_PROJECTS[0][0]
    # the same facts are in the copyable text, not only the widget
    text = dialog.full_text()
    assert "PC Power Calculator" in text and CONTACT_EMAIL in text


def test_about_explains_the_licence_and_free_use(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    html = dialog.sections.toHtml()
    assert "AGPL-3.0-or-later" in html
    assert "desktop app stays free" in html
    assert "hosted service" in html


def test_about_carries_the_disclaimer_and_the_policy(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    html = dialog.sections.toHtml()
    # the disclaimer's three load-bearing claims, and the policy's two halves
    assert "does not decide what the data mean" in html
    assert "not evidence that the model is physically right" in html
    assert "provided “as is”, without warranty" in html or "provided &quot;as is&quot;" in html
    assert "Privacy and fair-use policy" in html
    assert "not sold, and it is not shared for advertising" in html
    assert "Argon2" in html
    assert "no telemetry" in html
