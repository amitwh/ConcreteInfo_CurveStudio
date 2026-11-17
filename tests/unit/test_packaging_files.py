"""Static checks on the desktop packaging files (spec, build driver, release workflow).

PyInstaller itself is not run here (it is a heavy build; ``apps/desktop/packaging/build.py``
does the real build and the frozen ``--selftest``). What we can assert cheaply is shape:
the files exist, the spec and the scripts are valid Python, the workflow is valid YAML
whose actions are pinned by full commit SHA, and the pieces the frozen app depends on
(entry-point metadata, the spawn-pool guard) are actually referenced.
"""

import ast
import re
from pathlib import Path

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
    matrix = workflow["jobs"]["release"]["strategy"]["matrix"]["os"]
    assert {"ubuntu-latest", "windows-latest"} <= set(matrix)
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
