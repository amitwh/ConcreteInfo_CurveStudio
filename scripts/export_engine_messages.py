"""Write the core's code-to-sentence mapping where the web app can import it.

The web cannot import Python, so ``curvestudio.messages`` — the one authored source —
is exported to ``apps/web/src/content/engine-messages.json``, committed so the web
build needs no Python. ``tests/unit/test_messages.py`` fails while the two disagree.

Usage: ``uv run scripts/export_engine_messages.py`` (from the repository root).
"""

import json
from pathlib import Path

from curvestudio.messages import SENTENCES

TARGET = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "web"
    / "src"
    / "content"
    / "engine-messages.json"
)


def main() -> int:
    """Write the mapping with stable formatting; print what was written."""
    body = json.dumps(dict(sorted(SENTENCES.items())), indent=2, ensure_ascii=False)
    TARGET.write_text(f"{body}\n", encoding="utf-8")
    print(f"wrote {len(SENTENCES)} sentences to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
