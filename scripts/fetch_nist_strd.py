"""Download the NIST StRD reference datasets used by the numerical benchmarks.

Why a script and not a test fixture: the data are vendored (committed unchanged) so tests never
touch the network. This script exists only to (re)populate ``tests/data/nist_strd`` once, and to
make the provenance of every file reproducible. Existing files are skipped so that committed
files are never silently overwritten; pass ``--force`` to replace them deliberately.

Only the standard library is used. Any non-200 response, empty body or network error aborts with
a non-zero exit status and a message naming the URL; nothing is ever synthesised.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

NLS_URL = "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/{name}.dat"
LLS_URL = "https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/{name}.dat"

NONLINEAR = (
    "Misra1a",
    "Misra1b",
    "Misra1c",
    "Misra1d",
    "DanWood",
    "Chwirut2",
    "Lanczos3",
    "Gauss1",
    "BoxBOD",
    "Rat42",
    "MGH09",
    "Eckerle4",
)
LINEAR = ("Norris", "Pontius", "Wampler1", "Wampler4", "Filip")

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "tests" / "data" / "nist_strd"
TIMEOUT_SECONDS = 30.0
# NIST rejects urllib's default "Python-urllib" agent with HTTP 403, so identify ourselves.
USER_AGENT = "CurveStudio-fetch-nist-strd/1.0 (test-data vendoring)"


def fetch(url: str, *, timeout: float = TIMEOUT_SECONDS) -> bytes:
    """Return the body at ``url``; exit loudly on any failure (never return partial data)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body: bytes = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"error: could not download {url}: {exc}") from exc
    if status != 200:
        raise SystemExit(f"error: {url} returned HTTP {status}, expected 200")
    if not body.strip():
        raise SystemExit(f"error: {url} returned an empty body")
    return body


def download_all(dest: Path, *, force: bool = False) -> list[Path]:
    """Download every dataset under ``dest``; return the paths actually written."""
    written: list[Path] = []
    for subdir, template, names in (
        ("nonlinear", NLS_URL, NONLINEAR),
        ("linear", LLS_URL, LINEAR),
    ):
        target_dir = dest / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = target_dir / f"{name}.dat"
            if target.exists() and not force:
                print(f"skip   {target} (exists; use --force to replace)")
                continue
            target.write_bytes(fetch(template.format(name=name)))
            print(f"wrote  {target}")
            written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="output directory")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)
    download_all(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
