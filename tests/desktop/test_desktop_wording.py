"""The desktop app's promise says the same true thing wherever it is written down.

The app works fully offline with no account, and sends exactly one anonymous install
record per version (version, operating system, package type). Every surface that used
to say "nothing is uploaded" or "no network calls, ever" now says that instead.
"""

from pathlib import Path

import pytest

from curvestudio_desktop import NO_NETWORK_NOTE
from curvestudio_desktop.views.dialogs import POLICY_HTML

REPO = Path(__file__).resolve().parents[2]
CANONICAL_HELP = REPO / "docs" / "help-content.md"
PACKAGED_HELP = REPO / "apps/desktop/src/curvestudio_desktop/resources/help/help-content.md"

SURFACES: dict[str, Path] = {
    "root README": REPO / "README.md",
    "desktop README": REPO / "apps" / "desktop" / "README.md",
    "desktop help": CANONICAL_HELP,
    "web help": REPO / "apps" / "web" / "src" / "content" / "help.ts",
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


def test_the_packaged_help_is_the_canonical_help_byte_for_byte() -> None:
    assert PACKAGED_HELP.read_bytes() == CANONICAL_HELP.read_bytes()


def test_the_status_bar_note_is_the_new_one() -> None:
    assert NO_NETWORK_NOTE == "Offline-ready · no account"
