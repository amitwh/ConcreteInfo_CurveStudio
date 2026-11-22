"""Static checks on the desktop packaging files (spec, build driver, release workflow).

PyInstaller itself is not run here (it is a heavy build; ``apps/desktop/packaging/build.py``
does the real build and the frozen ``--selftest``). What we can assert cheaply is shape:
the files exist, the spec and the scripts are valid Python, the workflow is valid YAML
whose actions are pinned by full commit SHA, and the pieces the frozen app depends on
(entry-point metadata, the spawn-pool guard) are actually referenced.
"""

import ast
import importlib.util
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PACKAGING = REPO / "apps" / "desktop" / "packaging"
WORKFLOW = REPO / ".github" / "workflows" / "desktop-release.yml"

SPEC = PACKAGING / "curvestudio-desktop.spec"
BUILD = PACKAGING / "build.py"
ICON_SCRIPT = PACKAGING / "make_icon.py"
ICON_SOURCE = PACKAGING / "icon.svg"
ISS = PACKAGING / "windows" / "curvestudio.iss"
DESKTOP_ENTRY = PACKAGING / "linux" / "curvestudio-desktop.desktop"

SHA_PIN = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}(#.*)?$")


def test_packaging_files_exist() -> None:
    for path in (SPEC, BUILD, ICON_SCRIPT, ICON_SOURCE, ISS, DESKTOP_ENTRY):
        assert path.is_file(), f"missing packaging file: {path}"
        assert path.stat().st_size > 0, f"empty packaging file: {path}"


def test_spec_is_valid_python_and_collects_the_frozen_necessities() -> None:
    source = SPEC.read_text(encoding="utf-8")
    tree = ast.parse(source)  # a spec is Python; PyInstaller exec()s it
    assert isinstance(tree, ast.Expression) or isinstance(tree.body[0], ast.stmt)
    # entry points: importlib.metadata must find the plugin distributions in the bundle
    for dist in ("curvestudio-models", "curvestudio-engineering"):
        assert f'"{dist}"' in source, f"the spec must copy_metadata({dist!r})"
    # the model families are imported dynamically (entry points), so they must be
    # collected as hidden imports explicitly
    assert "collect_submodules" in source and "curvestudio_models" in source
    assert "curvestudio_engineering" in source
    # reporting/plotting data: matplotlib ships its fonts as data files
    assert "matplotlib" in source


def test_build_driver_is_valid_python_and_gates_on_selftest() -> None:
    source = BUILD.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert any("selftest" in name for name in functions), (
        "build.py must run the frozen --selftest (the point of the build driver)"
    )
    assert "PyInstaller" in source and "SHA256SUMS" in source


def test_icon_script_and_svg_are_present_and_script_parses() -> None:
    ast.parse(ICON_SCRIPT.read_text(encoding="utf-8"))
    svg = ICON_SOURCE.read_text(encoding="utf-8")
    assert svg.lstrip().startswith("<") and "</svg>" in svg


def test_release_workflow_is_valid_yaml_with_sha_pinned_actions() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    # dispatch and tag triggers, as the plan requires
    on: object = workflow.get(True, workflow.get("on"))  # YAML 1.1 parses bare `on:` as true
    assert isinstance(on, dict)
    assert "workflow_dispatch" in on
    assert on.get("push") == {"tags": ["v*"]}
    jobs = workflow["jobs"]
    runners = {job["runs-on"] for job in jobs.values()}
    assert {"ubuntu-latest", "windows-latest"} <= runners
    # every action use is pinned by a full 40-hex commit SHA (no floating tags);
    # a trailing "# vX.Y" comment (the ci.yml style) is allowed after the SHA
    text = WORKFLOW.read_text(encoding="utf-8")
    uses = [
        line.split("uses:", 1)[1].split("#", 1)[0].strip()
        for line in text.splitlines()
        if "uses:" in line
    ]
    assert uses, "the workflow uses no actions?"
    unpinned = [use for use in uses if not SHA_PIN.match(use)]
    assert not unpinned, f"actions not pinned by full commit SHA: {unpinned}"
    # the tag release step exists and uploads the release archives
    assert "dist/release" in text


def test_inno_script_targets_the_bundle_and_release_dir() -> None:
    text = ISS.read_text(encoding="utf-8")
    assert "curvestudio-desktop" in text and "[Files]" in text and "[Setup]" in text


def test_build_outputs_are_git_ignored() -> None:
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("dist/", "build/"):
        assert pattern in text, f".gitignore must ignore {pattern}"


def _load_build() -> ModuleType:
    """Import apps/desktop/packaging/build.py, which is not an installed package."""
    spec = importlib.util.spec_from_file_location("curvestudio_packaging_build", BUILD)
    assert spec is not None and spec.loader is not None, f"cannot load {BUILD}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()

APP = "curvestudio-desktop"


@dataclass
class FakeRepo:
    """A throwaway repository layout wired into the build driver's module globals."""

    work: Path
    dist: Path
    release: Path
    icons: Path


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRepo:
    work = tmp_path / "build"
    dist = tmp_path / "dist"
    release = dist / "release"
    icons = work / "icons"
    icons.mkdir(parents=True)

    # a minimal stand-in for the PyInstaller one-folder bundle
    bundle = dist / APP
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "_internal" / "data.txt").write_text("payload", encoding="utf-8")
    executable = bundle / APP
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    (icons / "icon-16.png").write_bytes(b"png16")
    (icons / "icon-32.png").write_bytes(b"png32")
    (icons / "icon-256.png").write_bytes(b"png256")
    (icons / f"{APP}.png").write_bytes(b"png256")

    monkeypatch.setattr(build, "WORK", work)
    monkeypatch.setattr(build, "DIST", dist)
    monkeypatch.setattr(build, "RELEASE", release)
    monkeypatch.setattr(build, "app_version", lambda: "1.2.3")
    monkeypatch.setattr(build, "cpu_arch", lambda: "x86_64")
    return FakeRepo(work=work, dist=dist, release=release, icons=icons)


def test_deb_control_text() -> None:
    text = build.deb_control("1.2.3", "amd64", 1234)
    lines = text.splitlines()
    assert "Package: curvestudio-desktop" in lines
    assert "Version: 1.2.3" in lines
    assert "Architecture: amd64" in lines
    assert "Maintainer: Amit Haridas <amit.wh@gmail.com>" in lines
    assert "Installed-Size: 1234" in lines
    assert "Section: science" in lines
    assert "Priority: optional" in lines
    assert "Homepage: https://github.com/amitwh/ConcreteInfo_CurveStudio" in lines
    depends = next(line for line in lines if line.startswith("Depends:"))
    for library in ("libc6", "libxcb-cursor0", "libxcb-xkb1", "libfontconfig1"):
        assert library in depends

    description = lines[lines.index("Description: ConcreteInfo CurveStudio desktop") :]
    assert len(description) >= 3, "the description must have a short line plus detail"
    assert all(line.startswith(" ") for line in description[1:]), (
        "every description continuation line starts with a space"
    )
    assert " ." in description, "the blank detail line is the Debian ' .' convention"


def test_deb_tree_layout(fake_repo: FakeRepo) -> None:
    staging = build.make_deb_tree()
    assert staging == fake_repo.work / "deb" / "curvestudio-desktop_1.2.3_amd64"

    library = staging / "usr" / "lib" / APP
    assert (library / "_internal" / "data.txt").read_text(encoding="utf-8") == "payload"
    assert stat.S_IMODE(library.stat().st_mode) == 0o755
    assert stat.S_IMODE((library / APP).stat().st_mode) & 0o111, "the main binary stays executable"

    launcher = staging / "usr" / "bin" / APP
    assert launcher.is_symlink()
    assert os.readlink(launcher) == f"/usr/lib/{APP}/{APP}"

    assert (staging / "usr" / "share" / "applications" / f"{APP}.desktop").is_file()
    assert (
        staging / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / f"{APP}.png"
    ).is_file()
    assert (staging / "usr" / "share" / "doc" / APP / "copyright").is_file()

    control = staging / "DEBIAN" / "control"
    assert stat.S_IMODE(control.stat().st_mode) == 0o644
    control_text = control.read_text(encoding="utf-8")
    assert "Package: curvestudio-desktop" in control_text
    assert "Version: 1.2.3" in control_text
    assert "Architecture: amd64" in control_text


def test_appdir_layout(fake_repo: FakeRepo) -> None:
    appdir = build.make_appdir()
    assert appdir == fake_repo.work / "appimage" / "CurveStudio.AppDir"

    app_run = appdir / "AppRun"
    assert app_run.read_text(encoding="utf-8") == (
        '#!/bin/sh\nexec "$APPDIR/usr/lib/curvestudio-desktop/curvestudio-desktop" "$@"\n'
    )
    assert stat.S_IMODE(app_run.stat().st_mode) == 0o755

    dir_icon = appdir / ".DirIcon"
    assert dir_icon.is_symlink()
    assert os.readlink(dir_icon) == f"{APP}.png"
    assert (appdir / f"{APP}.png").is_file()
    assert (appdir / f"{APP}.desktop").is_file()
    assert (appdir / "usr" / "lib" / APP / "_internal" / "data.txt").is_file()
    assert (appdir / "usr" / "share" / "applications" / f"{APP}.desktop").is_file()
    assert (
        appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / f"{APP}.png"
    ).is_file()


@pytest.mark.parametrize(("arch", "expected"), [("x86_64", "amd64"), ("arm64", "arm64")])
def test_deb_arch_mapping(arch: str, expected: str) -> None:
    assert build.deb_arch(arch) == expected


@pytest.mark.parametrize(("arch", "expected"), [("x86_64", "x86_64"), ("arm64", "aarch64")])
def test_appimage_arch_mapping(arch: str, expected: str) -> None:
    assert build.appimage_arch(arch) == expected


def test_appimage_requires_the_appimagetool_host_tool(
    fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build, "_find_appimagetool", lambda: None)
    with pytest.raises(SystemExit) as excinfo:
        build.appimage()
    message = str(excinfo.value)
    assert "appimagetool" in message
    assert "host tool" in message


def test_frozen_selftest_reads_the_log_file_when_there_is_no_console(
    fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A windowed Windows build prints nothing on stdout, so the driver reads the file."""
    import subprocess

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        Path(str(env["CURVESTUDIO_SELFTEST_OUT"])).write_text(
            "SELFTEST OK window, table (20 rows)\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    build.run_frozen_selftest()  # must not raise
    assert (fake_repo.work / "selftest.log").is_file()


def test_frozen_selftest_still_fails_without_the_ok_line(
    fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="nothing useful\n", stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="frozen selftest failed"):
        build.run_frozen_selftest()


def test_release_notes_carry_install_lines_and_the_version_s_changelog_section() -> None:
    """The release body is generated from the changelog, never hand-written in YAML."""
    spec = importlib.util.spec_from_file_location(
        "curvestudio_packaging_release_notes", PACKAGING / "release_notes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    changelog = (
        "# Changelog\n\n"
        "## [1.2.3] - 2026-01-02\n\n### Fixed\n- A thing that was broken.\n\n"
        "## [1.2.2] - 2026-01-01\n\n### Fixed\n- An older thing.\n"
    )
    notes = module.build_notes("1.2.3", changelog)
    assert "CurveStudio 1.2.3" in notes
    assert "curvestudio-desktop_1.2.3_amd64.deb" in notes
    assert "curvestudio-desktop-1.2.3-linux-x86_64.AppImage" in notes
    assert "curvestudio-desktop-1.2.3-windows-x86_64.zip" in notes
    assert "SHA256SUMS" in notes
    assert "A thing that was broken." in notes
    assert "An older thing." not in notes  # only this version's section
    assert "See CHANGELOG.md." in module.build_notes("9.9.9", changelog)


def test_the_release_workflow_publishes_on_tags_and_dry_runs_on_dispatch() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # YAML parses the bare key `on` as True; accept either spelling
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["push"]["tags"] == ["v*"]
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"]["contents"] == "write"

    jobs = workflow["jobs"]
    assert set(jobs) == {"linux", "windows"}
    for job in jobs.values():
        steps = job["steps"]
        publish = [step for step in steps if "action-gh-release" in str(step.get("uses", ""))]
        assert len(publish) == 1, "each platform job publishes its own artifacts once"
        assert publish[0]["if"] == "startsWith(github.ref, 'refs/tags/v')"
        # the tag must match the app version before anything is built or published
        assert any("does not match the app version" in str(step.get("run", "")) for step in steps)
    linux = str(jobs["linux"])
    assert "build.py release" in linux
    assert "sha256sum -c" in linux
    assert "release_notes.py" in linux
    windows = str(jobs["windows"])
    assert "build.py sums" in windows
    assert "release_notes.py" in windows
