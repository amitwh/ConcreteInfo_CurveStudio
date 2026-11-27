"""Write the GitHub release body for a CurveStudio tag.

    uv run python apps/desktop/packaging/release_notes.py [release-notes.md]

The body is the install lines for the platform's artifacts plus the changelog
section for the app's own version, so a release never advertises a different
version from the one it ships. The release workflow calls this from both the
Linux and the Windows job; keeping it a script (not an inline heredoc in YAML)
makes the exact text testable.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

PACKAGING = Path(__file__).resolve().parent
REPO = PACKAGING.parents[2]
CHANGELOG = REPO / "CHANGELOG.md"
DEFAULT_OUTPUT = "release-notes.md"

INSTALL_LINES = """\
**Linux (x86_64)**
- `.deb`: `sudo apt install ./curvestudio-desktop_{version}_amd64.deb`
- `AppImage`: `chmod +x curvestudio-desktop-{version}-linux-x86_64.AppImage` then run it
- `.tar.gz`: unpack and run `curvestudio-desktop/curvestudio-desktop`

**Windows (x86_64)**
- `curvestudio-desktop-{version}-windows-x86_64.zip`: unpack and run `curvestudio-desktop.exe`
- `curvestudio-desktop-{version}-windows-x86_64-setup.exe` when the installer
  step succeeded
"""


def changelog_section(changelog: str, version: str) -> str:
    """The ``## [version]`` section of the changelog, without its heading."""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match is not None else ""


def build_notes(version: str, changelog: str) -> str:
    """The release body for ``version``: what it is, how to install it, what changed."""
    section = changelog_section(changelog, version) or "See CHANGELOG.md."
    return (
        f"ConcreteInfo CurveStudio {version} — curve fitting for paired x/y measurements.\n\n"
        f"{INSTALL_LINES.format(version=version)}\n"
        "Every artifact's SHA-256 is in the `SHA256SUMS-*` file next to it.\n\n"
        "## Changes\n\n"
        f"{section}\n"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    version = metadata.version("curvestudio-desktop")
    output = Path(arguments[0]) if arguments else Path(DEFAULT_OUTPUT)
    output.write_text(build_notes(version, CHANGELOG.read_text(encoding="utf-8")), "utf-8")
    print(f"release notes for {version}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
