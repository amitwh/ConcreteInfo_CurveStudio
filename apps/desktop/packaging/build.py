"""Cross-platform build driver for the CurveStudio desktop app.

    uv run python apps/desktop/packaging/build.py           # icons + bundle + selftest + archive
    uv run python apps/desktop/packaging/build.py selftest  # run the frozen --selftest only
    uv run python apps/desktop/packaging/build.py icons     # regenerate build/icons only
    uv run python apps/desktop/packaging/build.py sums      # rewrite dist/release checksums
    uv run python apps/desktop/packaging/build.py deb       # Debian package from the bundle
    uv run python apps/desktop/packaging/build.py appimage  # AppImage from the bundle
    uv run python apps/desktop/packaging/build.py release   # build + deb + appimage + checksums

Produces, relative to the repository root:

    build/icons/                    generated icon PNG/ICO files (never committed)
    build/pyinstaller/              PyInstaller work files
    dist/curvestudio-desktop/       the one-folder bundle
    dist/release/curvestudio-desktop-<version>-<os>-<arch>.tar.gz (Linux)
                                     ... .zip (Windows)
    dist/release/curvestudio-desktop_<version>_<debarch>.deb (Linux, ``deb``)
    dist/release/curvestudio-desktop-<version>-linux-<arch>.AppImage (Linux, ``appimage``)
                                     plus SHA256SUMS-<os>-<arch>

The frozen executable's ``--selftest`` runs one fit and one Auto-fit through the
spawn process pool, so a green build proves ``freeze_support()`` and the
model-library entry points survived freezing. The Linux archive additionally
carries the ``.desktop`` file and the icon for menu integration.

Every artifact also carries its ``package`` word in ``package.txt`` beside the
executable (``deb``, ``appimage``, ``tarball``, ``windows_zip``; the Inno Setup
installer brings its own, ``windows_setup``): the app's anonymous install record
reports which kind of download it runs from. The shared bundle itself stays
neutral, so no artifact inherits another's word.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import platform
import shutil
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

PACKAGE_FILE = "package.txt"
"""Beside the frozen executable: which artifact the app came from, for its install record."""
ARCHIVE_PACKAGES = {"windows": "windows_zip", "linux": "tarball"}
"""The word each OS's plain archive carries (the portable zip, the tarball)."""

ARCHITECTURES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}
DEB_ARCHITECTURES = {"x86_64": "amd64", "arm64": "arm64"}
APPIMAGE_ARCHITECTURES = {"x86_64": "x86_64", "arm64": "aarch64"}

DISPLAY_NAME = "ConcreteInfo CurveStudio"
MAINTAINER = "Amit Haridas <amit.wh@gmail.com>"
HOMEPAGE = "https://github.com/amitwh/ConcreteInfo_CurveStudio"
APPIMAGE_TOOL_ENV = "APPIMAGETOOL"
DEB_DEPENDS = (
    "libc6, libgl1, libegl1, libxkbcommon-x11-0, libdbus-1-3, libxcb-cursor0, "
    "libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-randr0, "
    "libxcb-render-util0, libxcb-shape0, libxcb-xinerama0, libxcb-xkb1, "
    "libfontconfig1, libfreetype6, libexpat1"
)
APPIMAGE_APPRUN = '#!/bin/sh\nexec "$APPDIR/usr/lib/curvestudio-desktop/curvestudio-desktop" "$@"\n'


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


def write_package_file(directory: Path, package: str) -> Path:
    """Leave the artifact's ``package`` word in ``directory``, beside its executable."""
    path = directory / PACKAGE_FILE
    path.write_text(package + "\n", encoding="utf-8")
    return path


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
    """Run the frozen ``--selftest``; fail unless the OK line was reported.

    A windowed Windows build has no console to capture, so the app also writes its
    output to ``CURVESTUDIO_SELFTEST_OUT``; either channel is accepted.
    """
    exe = frozen_executable()
    log = WORK / "selftest.log"
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"  # Qt Widgets renders offscreen on Windows too
    env["CURVESTUDIO_SELFTEST_OUT"] = str(log)
    # A build is not an installation: the smoke test must never post an install record
    # (an empty destination means "never send" to the app).
    env["CURVESTUDIO_INSTALL_RECORD_URL"] = ""
    if log.exists():
        log.unlink()
    print(f"+ {exe} --selftest", flush=True)
    proc = subprocess.run(
        [str(exe), "--selftest"],
        capture_output=True,
        text=True,
        env=env,
        cwd=DIST,
        timeout=SELFTEST_TIMEOUT,
    )
    captured = proc.stdout + proc.stderr
    if log.exists():
        captured += log.read_text(encoding="utf-8")
    sys.stdout.write(captured)
    if proc.returncode != 0 or SELFTEST_OK not in captured:
        raise SystemExit(f"frozen selftest failed (exit code {proc.returncode})")


def _has_qt_plugin(internal: Path, kind: str, pattern: str) -> bool:
    """Is a Qt plugin of this kind in the bundle? The PySide6 wheel puts plugins under
    ``PySide6/Qt/plugins`` on Linux and under ``PySide6/plugins`` on Windows."""
    return any(
        any(internal.glob(f"PySide6/{layout}/{kind}/{pattern}"))
        for layout in ("Qt/plugins", "plugins")
    )


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
            _has_qt_plugin(internal, "imageformats", "*svg*"),
            "the Qt SVG image format plugin",
        ),
        (
            _has_qt_plugin(internal, "iconengines", "*"),
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
            all(
                (internal / "curvestudio_desktop" / "resources" / name).is_file()
                for name in ("example.csv", "example-decay.csv", "example-kinetics.csv")
            ),
            "the bundled example datasets",
        ),
    ]
    missing = [what for ok, what in checks if not ok]
    if missing:
        raise SystemExit("frozen bundle is missing: " + "; ".join(missing))


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def make_archive(icons: Path) -> Path:
    """Pack the bundle (Linux: with .desktop file and icon) into dist/release.

    The archive's own ``package`` word goes in as an entry, beside the executable in the
    archive but never as a file in the shared bundle, so the installer built afterwards
    from that same bundle cannot pick it up by accident.
    """
    RELEASE.mkdir(parents=True, exist_ok=True)
    stem = f"{APP}-{app_version()}-{os_name()}-{cpu_arch()}"
    word = (ARCHIVE_PACKAGES[os_name()] + "\n").encode("utf-8")
    if is_windows():
        archive = RELEASE / f"{stem}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted((DIST / BUNDLE).rglob("*")):
                if file.is_file():
                    # as_posix: zip entries must always use forward slashes
                    zf.write(file, file.relative_to(DIST).as_posix())
            zf.writestr(f"{BUNDLE}/{PACKAGE_FILE}", word)
    else:
        archive = RELEASE / f"{stem}.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(DIST / BUNDLE, arcname=BUNDLE)
            entry = tarfile.TarInfo(f"{BUNDLE}/{PACKAGE_FILE}")
            entry.size = len(word)
            entry.mode = 0o644
            tf.addfile(entry, io.BytesIO(word))
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


def _map_arch(mapping: dict[str, str], arch: str, package: str) -> str:
    try:
        return mapping[arch]
    except KeyError:
        raise SystemExit(f"no {package} architecture mapping for {arch!r}") from None


def deb_arch(arch: str) -> str:
    return _map_arch(DEB_ARCHITECTURES, arch, "Debian")


def appimage_arch(arch: str) -> str:
    return _map_arch(APPIMAGE_ARCHITECTURES, arch, "AppImage")


def hicolor_icon_sizes(icons: Path) -> list[int]:
    """The square icon sizes present in ``icons`` (``icon-<n>.png``)."""
    sizes = []
    for path in icons.glob("icon-*.png"):
        try:
            sizes.append(int(path.stem.removeprefix("icon-")))
        except ValueError:
            continue  # not an icon-<size>.png file
    return sorted(sizes)


def deb_control(version: str, arch: str, installed_kb: int) -> str:
    """The Debian ``DEBIAN/control`` text for the bundled app."""
    description = [
        f"Description: {DISPLAY_NAME} desktop",
        " Fit curves to concrete test data entirely offline: over 100 models,",
        " Auto-fit ranking, project files and PDF reports.",
        " .",
        " The package installs the bundled Qt application under /usr/lib and a",
        " launcher on PATH.",
    ]
    lines = [
        f"Package: {BUNDLE}",
        f"Version: {version}",
        f"Architecture: {arch}",
        f"Maintainer: {MAINTAINER}",
        f"Installed-Size: {installed_kb}",
        "Section: science",
        "Priority: optional",
        f"Homepage: {HOMEPAGE}",
        f"Depends: {DEB_DEPENDS}",
        *description,
    ]
    return "\n".join(lines) + "\n"


def deb_copyright() -> str:
    """A short machine-readable copyright file for the Debian package."""
    return (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
        "Upstream-Name: ConcreteInfo CurveStudio\n"
        f"Upstream-Contact: {MAINTAINER}\n"
        f"Source: {HOMEPAGE}\n"
        "\n"
        "Files: *\n"
        "Copyright: 2026 Amit Haridas\n"
        "License: AGPL-3.0-or-later\n"
        " This program is free software: you can redistribute it and/or modify it\n"
        " under the terms of the GNU Affero General Public License as published by\n"
        " the Free Software Foundation, either version 3 of the License, or (at\n"
        " your option) any later version. The full text of the license is at\n"
        " https://www.gnu.org/licenses/agpl-3.0.txt\n"
    )


def _copy_bundle(destination: Path) -> None:
    shutil.copytree(DIST / BUNDLE, destination, symlinks=True)


def _sum_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _set_tree_modes(root: Path) -> None:
    """Directories 0o755; executables 0o755; every other file 0o644 (symlinks untouched)."""
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir() or path.stat().st_mode & 0o111:
            path.chmod(0o755)
        else:
            path.chmod(0o644)


def make_deb_tree() -> Path:
    """Stage the Debian package payload under ``build/deb`` and return the tree root."""
    version = app_version()
    arch = deb_arch(cpu_arch())
    staging = WORK / "deb" / f"{BUNDLE}_{version}_{arch}"
    shutil.rmtree(staging, ignore_errors=True)

    library = staging / "usr" / "lib" / BUNDLE
    library.parent.mkdir(parents=True)
    _copy_bundle(library)
    write_package_file(library, "deb")

    launcher = staging / "usr" / "bin" / BUNDLE
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(f"/usr/lib/{BUNDLE}/{BUNDLE}")

    applications = staging / "usr" / "share" / "applications"
    applications.mkdir(parents=True)
    shutil.copy2(PACKAGING / "linux" / f"{BUNDLE}.desktop", applications / f"{BUNDLE}.desktop")

    icons = WORK / "icons"
    for size in hicolor_icon_sizes(icons):
        target = staging / "usr/share/icons/hicolor" / f"{size}x{size}" / "apps"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icons / f"icon-{size}.png", target / f"{BUNDLE}.png")

    document = staging / "usr" / "share" / "doc" / BUNDLE
    document.mkdir(parents=True)
    (document / "copyright").write_text(deb_copyright(), encoding="utf-8")

    _set_tree_modes(staging)
    installed_kb = -(-_sum_file_bytes(staging) // 1024)  # round up like dpkg-deb

    control = staging / "DEBIAN" / "control"
    control.parent.mkdir(parents=True)
    control.write_text(deb_control(version, arch, installed_kb), encoding="utf-8")
    control.chmod(0o644)
    return staging


def deb() -> Path:
    """Build the Debian package from a finished bundle (Linux host only)."""
    if is_windows():
        raise SystemExit("the deb step builds a Debian package; run it on Linux (dpkg-deb)")
    staging = make_deb_tree()
    RELEASE.mkdir(parents=True, exist_ok=True)
    package = RELEASE / f"{BUNDLE}_{app_version()}_{deb_arch(cpu_arch())}.deb"
    command = ["dpkg-deb", "--root-owner-group", "--build", str(staging), str(package)]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    if not package.is_file() or package.stat().st_size == 0:
        raise SystemExit(f"dpkg-deb produced no package at {package}")
    subprocess.run(["dpkg-deb", "--info", str(package)], check=True)
    return package


def make_appdir() -> Path:
    """Stage the AppImage AppDir under ``build/appimage`` and return its root."""
    appdir = WORK / "appimage" / "CurveStudio.AppDir"
    shutil.rmtree(appdir, ignore_errors=True)

    library = appdir / "usr" / "lib" / BUNDLE
    library.parent.mkdir(parents=True)
    _copy_bundle(library)
    # assembled like the .deb's tree, but the image is its own package kind
    write_package_file(library, "appimage")

    app_run = appdir / "AppRun"
    app_run.write_text(APPIMAGE_APPRUN, encoding="utf-8")
    app_run.chmod(0o755)

    icons = WORK / "icons"
    shutil.copy2(icons / f"{BUNDLE}.png", appdir / f"{BUNDLE}.png")
    shutil.copy2(PACKAGING / "linux" / f"{BUNDLE}.desktop", appdir / f"{BUNDLE}.desktop")
    (appdir / ".DirIcon").symlink_to(f"{BUNDLE}.png")

    applications = appdir / "usr" / "share" / "applications"
    applications.mkdir(parents=True)
    shutil.copy2(PACKAGING / "linux" / f"{BUNDLE}.desktop", applications / f"{BUNDLE}.desktop")
    hicolor = appdir / "usr/share/icons/hicolor" / "256x256" / "apps"
    hicolor.mkdir(parents=True)
    shutil.copy2(icons / f"{BUNDLE}.png", hicolor / f"{BUNDLE}.png")
    return appdir


def _find_appimagetool() -> str | None:
    return os.environ.get(APPIMAGE_TOOL_ENV) or shutil.which("appimagetool")


def appimage() -> Path:
    """Build the AppImage from a finished bundle (Linux host only)."""
    if is_windows():
        raise SystemExit("the appimage step builds a Linux AppImage; run it on Linux")
    tool = _find_appimagetool()
    if tool is None:
        raise SystemExit(
            "appimagetool not found — it is a host tool (not a Python dependency); "
            f"install it or point the {APPIMAGE_TOOL_ENV} environment variable at it"
        )
    appdir = make_appdir()
    RELEASE.mkdir(parents=True, exist_ok=True)
    image = RELEASE / f"{BUNDLE}-{app_version()}-linux-{cpu_arch()}.AppImage"
    env = dict(os.environ)
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"  # no FUSE needed on the build host
    env["ARCH"] = appimage_arch(cpu_arch())
    env["VERSION"] = app_version()
    command = [tool, str(appdir), str(image)]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env)
    if not image.is_file() or image.stat().st_size == 0:
        raise SystemExit(f"appimagetool produced no image at {image}")
    return image


def release() -> None:
    """Build the bundle, both Linux packages, and the release checksums."""
    build()
    package = deb()
    image = appimage()
    sums = write_sha256sums()
    print(f"deb: {package} ({package.stat().st_size / 1e6:.1f} MB)")
    print(f"appimage: {image} ({image.stat().st_size / 1e6:.1f} MB)")
    print(f"checksums: {sums}")


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
        choices=("build", "deb", "appimage", "release", "selftest", "icons", "sums"),
        default="build",
        help="build: everything except the Linux packages (default); deb: Debian package; "
        "appimage: AppImage; release: build + deb + appimage + checksums; "
        "selftest: frozen smoke only; icons: regenerate the icon files; "
        "sums: rewrite the release checksums",
    )
    args = parser.parse_args(argv)
    if args.step == "build":
        build()
    elif args.step == "deb":
        print(deb())
    elif args.step == "appimage":
        print(appimage())
    elif args.step == "release":
        release()
    elif args.step == "selftest":
        run_frozen_selftest()
    elif args.step == "icons":
        make_icons()
    else:
        print(write_sha256sums())
    return 0


if __name__ == "__main__":
    sys.exit(main())
