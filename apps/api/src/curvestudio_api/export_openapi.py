"""Write ``apps/api/openapi.json``: ``python -m curvestudio_api.export_openapi [path]``."""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from curvestudio_api.app import create_app
from curvestudio_api.settings import Settings

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "openapi.json"


def generate() -> dict[str, Any]:
    """The OpenAPI document of the application (no database or server is started)."""
    with tempfile.TemporaryDirectory(prefix="curvestudio-openapi-") as scratch:
        settings = Settings(
            env="test",
            database_url=f"sqlite:///{scratch}/openapi.db",
            data_dir=Path(scratch),
            cors_origins=[],
        )
        schema: dict[str, Any] = create_app(settings).openapi()
    return schema


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    target.write_text(render(generate()), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
