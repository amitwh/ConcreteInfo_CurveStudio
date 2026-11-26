"""Help, About and the vendored icon set: the user-facing documentation surfaces.

The canonical help text lives in ``docs/help-content.md``; it is copied verbatim into
the desktop package so the frozen application can render it. These tests pin that copy
to the source, the twenty-five sections in order, the Markdown rendering (tables, lists,
bold, inline code), search, the menu/shortcut wiring, the About contract (logo, links,
runtime versions, credits), the shared list of other ConcreteInfo projects and
the Lucide icon mapping.
"""

import json
import re
import shutil
import subprocess
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
CANONICAL_PROJECTS = REPO_ROOT / "docs" / "other-projects.json"
WEB_APP = REPO_ROOT / "apps" / "web"

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
    "chevron-down",
    "close",
    "panel-project",
    "panel-analysis",
    "panel-table",
    # the status-bar footer's links (globe is Lucide's; linkedin/github are
    # Simple Icons' CC0 brand marks — Lucide dropped brand icons)
    "globe",
    "linkedin",
    "github",
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
    # the two brand marks (linkedin, github) are Simple Icons' CC0 *filled* glyphs —
    # they carry no stroke at all, only the themed fill
    filled = {"linkedin", "github"}
    for name in INTERNAL_ICON_NAMES:
        source = icons._svg_text(name, "#123456")
        assert "currentColor" not in source, f"{name} kept the unresolved colour"
        assert "#123456" in source, f"{name} did not take the theme colour"
        if name in filled:
            continue
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


def test_the_web_renderer_agrees_on_the_section_structure() -> None:
    """Design item 10: the same Markdown through both apps' renderers.

    The desktop splits and renders ``docs/help-content.md`` with its own
    ``parse_sections``/``markdown_to_html``; the web renders the same file with
    react-markdown + remark-gfm — the stack ``/help`` uses — through
    ``apps/web/scripts/help-structure.mjs``. The heading/section structure must be
    identical, level for level and word for word.
    """
    from curvestudio_desktop.views.help_dialog import markdown_to_html, parse_sections

    node = shutil.which("node")
    if node is None or not (WEB_APP / "node_modules" / "react-markdown").is_dir():
        pytest.skip("the web renderer needs node and apps/web's dependencies (npm ci in apps/web)")

    rendered = subprocess.run(
        [node, "scripts/help-structure.mjs"],
        cwd=WEB_APP,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rendered.returncode == 0, rendered.stderr
    web_headings = [(heading["level"], heading["text"]) for heading in json.loads(rendered.stdout)]

    canonical = CANONICAL_HELP.read_text(encoding="utf-8")
    desktop_headings: list[tuple[int, str]] = []
    for section in parse_sections(canonical):
        desktop_headings.append((2, section.title))
        desktop_headings.extend(
            (3, match.group(1))
            for match in re.finditer(r"<h3>(.*?)</h3>", markdown_to_html(section.body))
        )
    assert web_headings == desktop_headings


def test_help_dialog_lists_the_twenty_five_canonical_sections_in_order(qtbot, settings) -> None:
    from curvestudio_desktop.views.help_dialog import HelpDialog

    dialog = HelpDialog(settings=settings)
    qtbot.addWidget(dialog)
    titles = dialog.sections()
    assert len(titles) == 25
    assert titles == canonical_section_titles()
    assert [dialog.section_list.item(i).text() for i in range(25)] == list(titles)
    assert titles[0] == "1. Getting started"
    assert titles[10] == "11. Fit audit and methods text"
    assert titles[11] == "12. Tools that start from a fit"
    assert titles[12] == "13. Statistics tools"
    assert titles[20] == "21. Sample size and power"
    assert titles[21] == "22. Advanced models"
    assert titles[22] == "23. Evidence synthesis"
    assert titles[23] == "24. Bayesian tests"
    assert titles[-1] == "25. About, licence, projects and policy"


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

    about = markdown_to_html(sections["25. About, licence, projects and policy"].body)
    assert '<a href="https://www.concreteinfo.in">' in about
    assert "<em>" in about or "<strong>" in about


def test_markdown_renders_the_fit_audit_section_table_bold_and_inline_code() -> None:
    from curvestudio_desktop.views.help_dialog import (
        help_markdown,
        markdown_to_html,
        parse_sections,
    )

    sections = {section.title: section for section in parse_sections(help_markdown())}
    rendered = markdown_to_html(sections["11. Fit audit and methods text"].body)

    # the table of checks: its two headers, then one row for each of the eleven checks
    assert rendered.count("<table>") == 1
    assert "<th>Check</th><th>If it is flagged</th>" in rendered
    assert rendered.count("<tr>") == 12
    assert "<td>The fit converged</td>" in rendered
    assert "<td>There are at least 10 points per parameter</td>" in rendered
    assert "<td>No simpler model fits as well (from an Auto-fit ranking)</td>" in rendered

    assert "<strong>fit audit</strong>" in rendered
    assert "<strong>Reading a verdict.</strong>" in rendered
    assert "<strong>Not applicable</strong>" in rendered
    assert "<code>curvestudio audit</code>" in rendered
    assert "<code>curvestudio methods --style markdown</code>" in rendered
    assert "<code>booktabs</code>" in rendered
    # nothing of the section leaked out as raw Markdown
    assert "**" not in rendered and "`" not in rendered


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
    assert len(dialog.sections()) == 25


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


# -- shared project list ----------------------------------------------------------------------


def test_the_packaged_other_projects_json_is_byte_identical_to_the_canonical_copy() -> None:
    packaged = resources.files("curvestudio_desktop") / "resources" / "help" / "other-projects.json"
    assert packaged.is_file(), "other-projects.json is not packaged"
    assert packaged.read_bytes() == CANONICAL_PROJECTS.read_bytes()


def test_the_shared_projects_list_holds_five_short_entries_with_valid_kinds() -> None:
    packaged = resources.files("curvestudio_desktop") / "resources" / "help" / "other-projects.json"
    entries = json.loads(packaged.read_bytes())["entries"]
    assert len(entries) == 5
    for entry in entries:
        assert len(entry["summary"]) <= 90
        assert entry["kind"] in {"open-source", "commercial"}


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


def _packaged_project_entries() -> list[dict[str, str]]:
    """The shared five-entry list exactly as packaged (name, url, summary, kind)."""
    packaged = resources.files("curvestudio_desktop") / "resources" / "help" / "other-projects.json"
    entries: list[dict[str, str]] = json.loads(packaged.read_bytes())["entries"]
    return entries


def test_about_lists_the_shared_projects_and_contact(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import CONTACT_EMAIL, AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    html = dialog.sections.toHtml()
    assert "Other tools we build and maintain" in html
    entries = _packaged_project_entries()
    assert len(entries) == 5
    # every shared entry appears with its link, open source first, commercial second
    positions = []
    for entry in entries:
        assert entry["name"] in html, entry["name"]
        assert entry["url"] in html, entry["url"]
        positions.append(html.index(entry["name"]))
    assert positions == sorted(positions)
    assert CONTACT_EMAIL in html
    assert "https://www.concreteinfo.in" in html
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
    # the desktop half: no account, and the one thing it does send is named, with its switch
    assert "no account" in html
    assert "one anonymous install record" in html
    assert "Edit → Preferences → Privacy" in html
    assert "Nothing else ever leaves your computer" in html
    assert "nothing is uploaded" not in html and "no telemetry" not in html


# -- example datasets --------------------------------------------------------------------------


def test_help_section_25_names_all_three_example_datasets() -> None:
    body = CANONICAL_HELP.read_text(encoding="utf-8").split(
        "## 25. About, licence, projects and policy", 1
    )[1]
    for filename in ("example.csv", "example-decay.csv", "example-kinetics.csv"):
        assert f"`{filename}`" in body, filename


@pytest.mark.parametrize(
    ("filename", "rows", "columns"),
    [
        ("example.csv", None, ("age_days", "strength_mpa")),
        ("example-decay.csv", 30, ("x", "y")),
        ("example-kinetics.csv", 24, ("x", "y")),
    ],
)
def test_every_bundled_example_parses_as_a_dataset(
    filename: str, rows: int | None, columns: tuple[str, ...]
) -> None:
    from curvestudio.data import read_dataset

    with resources.as_file(resources.files("curvestudio_desktop") / "resources" / filename) as path:
        dataset = read_dataset(path)
    frame = dataset.original
    assert tuple(frame.columns) == columns
    if rows is not None:
        assert len(frame) == rows


def test_the_package_ships_exactly_three_example_datasets() -> None:
    shipped = sorted(
        entry.name
        for entry in (resources.files("curvestudio_desktop") / "resources").iterdir()
        if entry.is_file() and entry.name.startswith("example") and entry.suffix == ".csv"
    )
    assert shipped == ["example-decay.csv", "example-kinetics.csv", "example.csv"]
