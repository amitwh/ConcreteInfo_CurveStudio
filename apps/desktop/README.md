# CurveStudio desktop app

PySide6 shell for ConcreteInfo CurveStudio. All engine calls go through
`curvestudio.workflow` (shared with the web API), `curvestudio.project`
(`.cstudio` project files, format version 2: datasets, fits, Auto-fit runs and
statistics results), `curvestudio.data` types and `curvestudio.discovery`
(Auto-fit runs, streamed through a worker) — never the maths layers directly;
exports use `curvestudio_plotting` / `curvestudio_reporting`.

Auto-fit is the primary workflow: importing or selecting a dataset starts a run
by itself (Manual fit mode never starts anything). Runs stream a live ranking,
are cancellable, and every result is stored in the project.

## Run

```bash
uv run curvestudio-desktop            # the app
uv run curvestudio-desktop --selftest # headless smoke: window + example + fit + Auto-fit
```

## Install (standalone build)

Standalone builds are attached to every [GitHub
Release](https://github.com/amitwh/ConcreteInfo_CurveStudio/releases) (the
*Desktop release* workflow also produces them on demand via *Run workflow*):

- **Linux** — `curvestudio-desktop-<version>-linux-x86_64.tar.gz`
- **Windows** — `curvestudio-desktop-<version>-windows-x86_64.zip` (portable;
  releases from CI also carry
  `curvestudio-desktop-<version>-windows-x86_64-setup.exe`, the Inno Setup
  installer)

Verify a download against the matching `SHA256SUMS-<os>-<arch>` file
(`sha256sum -c SHA256SUMS-linux-x86_64`), then:

- **Linux**: extract anywhere and run `./curvestudio-desktop/curvestudio-desktop`.
  The archive also carries `curvestudio-desktop.desktop` and a `curvestudio-desktop.png`
  icon; to register the app in your menu, copy the icon to
  `~/.local/share/icons/` and the `.desktop` file to `~/.local/share/applications/`,
  editing its `Exec=` line to point at the extracted executable.
- **Windows**: extract the zip and run `curvestudio-desktop\curvestudio-desktop.exe`
  (or run the installer, which adds Start-menu and optional desktop shortcuts).

**Works fully offline, no account needed.** On first launch and after an upgrade the app
sends one anonymous install record (version, operating system, package type) so we know
how many installations exist; turn it off under Edit → Preferences → Privacy. Nothing
else ever leaves your computer: there are no updates, no error reports and no account,
and the app ships everything it needs (model catalogue, fonts, Qt, the Python stack).
A crash is written to a local `crash.log` (the dialog's *Open log folder* button leads
there) and is never sent. First run needs nothing but the extracted files.

## Build from source

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/); PyInstaller arrives
with the `desktop` dependency group (never in the server image).

```bash
uv sync --group desktop
QT_QPA_PLATFORM=offscreen uv run python apps/desktop/packaging/build.py
```

The driver, in order: generates the icon files (`packaging/make_icon.py`,
deterministic, from the geometry in `packaging/icon.svg`), runs PyInstaller with
`packaging/curvestudio-desktop.spec`, runs the frozen executable's
`--selftest` (one fit and one Auto-fit through the spawn process pool — this is
what proves `freeze_support()` and the model-library entry points survived
freezing), checks that the report fonts and Qt SVG plugins landed in the bundle,
and packs `dist/release/curvestudio-desktop-<version>-<os>-<arch>.tar.gz|zip`
plus a `SHA256SUMS-<os>-<arch>` file. Every artifact carries its package word
(`tarball`, `windows_zip`, `deb`, `appimage`, `windows_setup`) in a `package.txt`
beside the executable, which the app's install record reports. `build.py` also takes
the steps `selftest`, `icons` and `sums` on their own.

On Windows the same command produces the portable zip; the optional installer
(`packaging/windows/curvestudio.iss`) is built by CI with Inno Setup — `iscc
/DMyAppVersion=<version> apps\desktop\packaging\windows\curvestudio.iss` after a
build. Linux CI installs the Qt offscreen libraries (`libegl1 libgl1
libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libfontconfig1`) so the frozen
selftest can run headless.

The frozen build is a windowed application on Windows — no console window sits next
to the GUI (`packaging/curvestudio-desktop.spec` builds it with `console=False`
there; the other platforms keep the console). The `--selftest` output stays
capturable everywhere: with no console to read, the app also writes it to
`CURVESTUDIO_SELFTEST_OUT`, and `build.py` accepts either channel.

### Linux packages

On a Linux host the same driver also builds native packages from a finished
bundle:

```bash
uv run python apps/desktop/packaging/build.py deb       # .deb
uv run python apps/desktop/packaging/build.py appimage  # AppImage
uv run python apps/desktop/packaging/build.py release   # build + deb + appimage + checksums
```

`deb` and `appimage` expect the bundle from a previous `build`; `release` runs
the whole chain (`build`, then both packages, then the checksums). They write to
`dist/release/` alongside the tarball:

- `curvestudio-desktop_<version>_amd64.deb` (or `_arm64.deb`)
- `curvestudio-desktop-<version>-linux-x86_64.AppImage` (or `...-arm64.AppImage`)
- `SHA256SUMS-linux-x86_64`, rewritten to include everything in the directory

Both need a host tool that `uv sync` does not install:

- **deb** — `dpkg-deb` (Debian/Ubuntu `dpkg` package).
- **appimage** — `appimagetool` from the AppImageKit continuous build. It may
  sit on `PATH` or be named by the `APPIMAGETOOL` environment variable. The
  driver runs it with `APPIMAGE_EXTRACT_AND_RUN=1`, so FUSE is not required.

## Tests

```bash
uv run pytest tests/desktop
```

Tests run offscreen (`QT_QPA_PLATFORM=offscreen`, set by `tests/desktop/conftest.py`)
and default to Manual fit mode so background Auto-fit never races them; the
Auto-fit tests opt in explicitly.
Licensed AGPL-3.0-or-later, like the rest of the repository.
