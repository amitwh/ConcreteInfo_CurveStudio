"""The desktop app's promise says the same true thing wherever it is written down.

The app works fully offline with no account, and sends exactly one anonymous install
record per version (version, operating system, package type). Every surface that used
to say "nothing is uploaded" or "no network calls, ever" now says that instead.

The web half of the same policy says, wherever it is written down, that the service keeps
a dataset as the whole table that was uploaded, not only the columns used for fitting.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from curvestudio.tools.stats import FAMILIES
from curvestudio_desktop import NO_NETWORK_NOTE
from curvestudio_desktop.views.dialogs import POLICY_HTML

REPO = Path(__file__).resolve().parents[2]
CANONICAL_HELP = REPO / "docs" / "help-content.md"
PACKAGED_HELP = REPO / "apps/desktop/src/curvestudio_desktop/resources/help/help-content.md"

SURFACES: dict[str, Path] = {
    "root README": REPO / "README.md",
    "desktop README": REPO / "apps" / "desktop" / "README.md",
    "desktop help": CANONICAL_HELP,
    # The web help needs no surface of its own since 6eaa2a8: the web renders this same
    # canonical document (the old hand-kept help.ts copy is gone; the web tests pin the
    # sync and the rendering).
    "web About page": REPO / "apps" / "web" / "src" / "pages" / "AboutPage.tsx",
}

KEY_PHRASES = (
    "one anonymous install record",
    "version, operating system, package type",
    "Edit → Preferences → Privacy",
    "Nothing else ever leaves your computer",
)

OLD_PROMISES = (
    "nothing is uploaded anywhere",
    "no network calls, ever",
    "there is no telemetry",
    "No network access",
)

WHAT_THE_WEB_KEEPS = "the whole table you uploaded"
THE_SHORTER_WORDING_IT_REPLACES = "your datasets and fits are stored in your account"

WEB_NOTICE_COPIES: dict[str, Callable[[], str]] = {
    "the canonical help": lambda: CANONICAL_HELP.read_text(encoding="utf-8"),
    "the packaged help": lambda: PACKAGED_HELP.read_text(encoding="utf-8"),
    "the web About page": lambda: SURFACES["web About page"].read_text(encoding="utf-8"),
    "the desktop About policy": lambda: POLICY_HTML,
}


def squeezed(text: str) -> str:
    """The text with every run of whitespace folded: a Markdown line break is still one sentence."""
    return " ".join(text.split())


def assert_states_the_promise(surface: str, text: str) -> None:
    for phrase in KEY_PHRASES:
        assert phrase in text, f"{surface} does not say {phrase!r}"
    for promise in OLD_PROMISES:
        assert promise not in text, f"{surface} still promises {promise!r}"


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_every_surface_states_the_promise(surface: str) -> None:
    assert_states_the_promise(surface, squeezed(SURFACES[surface].read_text(encoding="utf-8")))


def test_the_about_policy_states_the_promise() -> None:
    assert_states_the_promise("the desktop About policy", squeezed(POLICY_HTML))


def test_the_web_about_page_names_what_the_account_keeps() -> None:
    text = squeezed(SURFACES["web About page"].read_text(encoding="utf-8"))
    for kept in ("e-mail address", "name if you give one", "last day", "error-report choice"):
        assert kept in text, kept
    assert "anonymous error report" in text
    assert "Profile → Privacy" in text


@pytest.mark.parametrize("surface", sorted(WEB_NOTICE_COPIES))
def test_every_copy_saying_what_the_web_keeps_names_the_whole_table(surface: str) -> None:
    assert WHAT_THE_WEB_KEEPS in squeezed(WEB_NOTICE_COPIES[surface]()), surface


def test_the_canonical_help_no_longer_carries_the_shorter_wording() -> None:
    text = squeezed(CANONICAL_HELP.read_text(encoding="utf-8"))
    assert THE_SHORTER_WORDING_IT_REPLACES not in text


def test_the_packaged_help_is_the_canonical_help_byte_for_byte() -> None:
    assert PACKAGED_HELP.read_bytes() == CANONICAL_HELP.read_bytes()


HELP_SURFACES: dict[str, Path] = {
    "the canonical help": CANONICAL_HELP,
    "the packaged help": PACKAGED_HELP,
    # "the web help" is the canonical help too since 6eaa2a8 (the web renders it; the
    # old hand-kept copy is gone).
}


@pytest.mark.parametrize("surface", sorted(HELP_SURFACES))
def test_every_help_surface_names_statistics_and_each_family(surface: str) -> None:
    text = squeezed(HELP_SURFACES[surface].read_text(encoding="utf-8"))
    assert "Statistics" in text, f"{surface} does not name the Statistics tools"
    for _, title in FAMILIES:
        assert title in text, f"{surface} does not name the {title!r} family"


def test_the_status_bar_note_is_the_new_one() -> None:
    assert NO_NETWORK_NOTE == "Offline-ready · no account"


# -- one name per thing (design item 17) --------------------------------------------------
#
# The glossary: "Auto-fit" (hyphenated, capital A), "dataset" as one word, "project" on
# the desktop and "workspace" on the web, "ranking" for the Auto-fit table and "results"
# for one fit's numbers. The banned spellings below are the three the design names. The
# code's and the API's lowercase "autofit" (routes, job kinds, object names, CSS classes)
# is not user-visible and stays — which is why the "Autofit" pattern is case-sensitive.

BANNED_SPELLINGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Autofit", re.compile(r"\bAutofit\b")),
    ("auto fit", re.compile(r"\bauto fit\b", re.IGNORECASE)),
    ("data set", re.compile(r"\bdata set\b", re.IGNORECASE)),
)

_GENERATED_SCHEMA = Path("apps/web/src/api/schema.d.ts")
"""The OpenAPI schema's "Autofit" spellings are the API's own route and schema names."""


def terminology_surfaces() -> list[tuple[str, str]]:
    """Every file that carries user-visible copy, as ``(relative name, text)``.

    The packaged desktop help is byte-for-byte the canonical help (pinned above), so the
    canonical document covers both; the web's synced copy under ``src/content`` is the
    same file again. The web's generated API schema is excluded on purpose.
    """
    files = [
        REPO / "README.md",
        REPO / "apps" / "desktop" / "README.md",
        REPO / "apps" / "web" / "README.md",
        REPO / "apps" / "web" / "index.html",
        CANONICAL_HELP,
        *sorted((REPO / "apps" / "desktop" / "src").rglob("*.py")),
        *sorted((REPO / "apps" / "web" / "src").rglob("*.ts*")),
    ]
    return [
        (str(path.relative_to(REPO)), path.read_text(encoding="utf-8"))
        for path in files
        if path.is_file() and path.relative_to(REPO) != _GENERATED_SCHEMA
    ]


@pytest.mark.parametrize(
    "surface_name, text", terminology_surfaces(), ids=[name for name, _ in terminology_surfaces()]
)
def test_the_copy_never_uses_the_banned_spellings(surface_name: str, text: str) -> None:
    for banned, pattern in BANNED_SPELLINGS:
        assert not pattern.search(text), f"{surface_name} says {banned!r}"


def test_the_help_defines_both_apps_containers() -> None:
    """The desktop's container is a project, the web's a workspace; the help says both."""
    text = squeezed(CANONICAL_HELP.read_text(encoding="utf-8"))
    assert "empty project" in text, "the help no longer defines the desktop's project"
    assert "your workspace" in text, "the help no longer defines the web's workspace"
