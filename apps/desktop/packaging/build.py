"""Cross-platform build driver for the CurveStudio desktop app.

    uv run python apps/desktop/packaging/build.py           # icons + bundle + selftest + archive
    uv run python apps/desktop/packaging/build.py selftest  # run the frozen --selftest only
    uv run python apps/desktop/packaging/build.py icons     # regenerate build/icons only
    uv run python apps/desktop/packaging/build.py sums      # rewrite dist/release checksums

Produces, relative to the repository root:

    build/icons/                    generated icon PNG/ICO files (never committed)
    build/pyinstaller/              PyInstaller work files
    dist/curvestudio-desktop/       the one-folder bundle
    dist/release/curvestudio-desktop-<version>-<os>-<arch>.tar.gz (Linux)
                                     ... .zip (Windows) — plus SHA256SUMS-<os>-<arch>

The frozen executable's ``--selftest`` runs one fit and one Auto-fit through the
spawn process pool, so a green build proves ``freeze_support()`` and the
model-library entry points survived freezing. The Linux archive additionally
carries the ``.desktop`` file and the icon for menu integration.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import tarfile
import zipfile
from importlib import metadata
from pathlib import Path

PACKAGING = Path(__file__).resolve().parent
REPO = PACKAGING.parents[2]
APP = "curvestudio-desktop"
BUNDLE = "curvestudio-desktop"  # COLLECT() name in the spec; same on every OS
DIST = REPO / "dist"
RELEASE = DIST / "release"
WORK = REPO / "build"
SELFTEST_TIMEOUT = 900  # a cold-start Auto-fit pool in a frozen app can be slow
SELFTEST_OK = "SELFTEST OK"

ARCHITECTURES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}


def app_version() -> str:
    return metadata.version("curvestudio-desktop")


def os_name() -> str:
    return "windows" if sys.platform == "win32" else "linux"


def cpu_arch() -> str:
    machine = platform.machine().lower()
    try:
        return ARCHITECTURES[machine]
    except KeyError:
        raise SystemExit(f"unknown CPU architecture for packaging: {machine!r}") from None


def is_windows() -> bool:
    return sys.platform == "win32"


def make_icons() -> Path:
    """Generate the icon files (deterministic; see make_icon.py)."""
    sys.path.insert(0, str(PACKAGING))  # make_icon sits next to this driver
    try:
        import make_icon
    finally:
        sys.path.remove(str(PACKAGING))
    icons = WORK / "icons"
    for path in make_icon.write_icons(icons):
        print(f"icon: {path}")
    return icons


def run_pyinstaller() -> None:
    """Build the one-folder bundle with the spec in this directory."""
    env = dict(os.environ)
    if is_windows():
        env.setdefault("CURVESTUDIO_ICON", str(WORK / "icons" / f"{APP}.ico"))
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(DIST),
        "--workpath",
        str(WORK / "pyinstaller"),
        str(PACKAGING / "curvestudio-desktop.spec"),
    ]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env, cwd=REPO)


def frozen_executable() -> Path:
    exe = DIST / BUNDLE / (APP + (".exe" if is_windows() else ""))
    if not exe.is_file():
        raise SystemExit(f"frozen executable not found: {exe} — did the build run?")
    return exe


def run_frozen_selftest() -> None:
    """Run the frozen ``--selftest``; fail unless it prints the OK line."""
    exe = frozen_executable()
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"  # Qt Widgets renders offscreen on Windows too
    print(f"+ {exe} --selftest", flush=True)
    proc = subprocess.run(
        [str(exe), "--selftest"],
        capture_output=True,
        text=True,
        env=env,
        cwd=DIST,
        timeout=SELFTEST_TIMEOUT,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0 or SELFTEST_OK not in proc.stdout:
        raise SystemExit(f"frozen selftest failed (exit code {proc.returncode})")


def check_bundle() -> None:
    """Fail loudly when a data ship the selftest cannot reach went missing.

    The selftest covers modules through use (Qt, numpy/scipy/sympy, pyqtgraph,
    the registry entry points, the spawn pool). Files that only matter later —
    report fonts and the SVG plugin — are checked by presence here.
    """
    internal = DIST / BUNDLE / "_internal"
    checks: list[tuple[bool, str]] = [
        (
            any(internal.glob("matplotlib/mpl-data/fonts/ttf/*.ttf")),
            "matplotlib fonts (mpl-data) for the PDF reports",
        ),
        (
            any(internal.glob("reportlab/fonts/*.pfb"))
            or any(internal.glob("reportlab/fonts/*.ttf")),
            "reportlab fonts for the PDF reports",
        ),
        (
            any(
                internal.glob(
                    # PySide6 wheel layout: libqsvg.so / qsvg.dll
                    "PySide6/Qt/plugins/imageformats/*svg*"
                )
            ),
            "the Qt SVG image format plugin",
        ),
        (
            any(internal.glob("PySide6/Qt/plugins/iconengines/*")),
            "the Qt icon engine plugins",
        ),
        (
            any(internal.glob("curvestudio_models-*.dist-info/entry_points.txt")),
            "curvestudio-models dist-info (registry entry points)",
        ),
        (
            any(internal.glob("curvestudio_engineering-*.dist-info/entry_points.txt")),
            "curvestudio-engineering dist-info (registry entry points)",
        ),
        (
            (internal / "curvestudio_desktop" / "resources" / "example.csv").is_file(),
            "the bundled example dataset",
        ),
    ]
    missing = [what for ok, what in checks if not ok]
    if missing:
        raise SystemExit("frozen bundle is missing: " + "; ".join(missing))


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def make_archive(icons: Path) -> Path:
    """Pack the bundle (Linux: with .desktop file and icon) into dist/release."""
    RELEASE.mkdir(parents=True, exist_ok=True)
    stem = f"{APP}-{app_version()}-{os_name()}-{cpu_arch()}"
    if is_windows():
        archive = RELEASE / f"{stem}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted((DIST / BUNDLE).rglob("*")):
                if file.is_file():
                    # as_posix: zip entries must always use forward slashes
                    zf.write(file, file.relative_to(DIST).as_posix())
    else:
        archive = RELEASE / f"{stem}.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(DIST / BUNDLE, arcname=BUNDLE)
            tf.add(PACKAGING / "linux" / f"{APP}.desktop", arcname=f"{APP}.desktop")
            tf.add(icons / f"{APP}.png", arcname=f"{APP}.png")
    return archive


def write_sha256sums() -> Path:
    """(Re)write the checksum file for everything currently in dist/release."""
    sums = RELEASE / f"SHA256SUMS-{os_name()}-{cpu_arch()}"
    lines = []
    for path in sorted(RELEASE.iterdir()):
        if path == sums or not path.is_file():
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums


def build() -> None:
    icons = make_icons()
    run_pyinstaller()
    run_frozen_selftest()
    check_bundle()
    archive = make_archive(icons)
    sums = write_sha256sums()
    print(f"bundle: {DIST / BUNDLE} ({dir_size(DIST / BUNDLE) / 1e6:.1f} MB)")
    print(f"archive: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    print(f"checksums: {sums}")


def main(argv: list[str] | None = None) -> int:
    assert __doc__ is not None  # this module always has a docstring
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step",
        nargs="?",
        choices=("build", "selftest", "icons", "sums"),
        default="build",
        help="build: everything (default); selftest: frozen smoke only; "
        "icons: regenerate the icon files; sums: rewrite the release checksums",
    )
    args = parser.parse_args(argv)
    if args.step == "build":
        build()
    elif args.step == "selftest":
        run_frozen_selftest()
    elif args.step == "icons":
        make_icons()
    else:
        print(write_sha256sums())
    return 0


if __name__ == "__main__":
    sys.exit(main())
