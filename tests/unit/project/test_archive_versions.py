"""The project file's schema versions: version 2 keeps a statistics result.

A version-2 file carries a tool result's plain JSON in a ``stat`` analysis. The project
layer checks the result's required fields on load (it sits below the tools layer and cannot
hold the ``ToolResult`` model itself); the desktop validates the full model when it shows
the result. A version-1 file (written before this format) still opens and keeps everything
it had; saving it writes version 2. Any other version is refused as
``UNSUPPORTED_VERSION``.
"""

import hashlib
import io
import json
import zipfile
from typing import Any

import pandas as pd
import pytest

from curvestudio.data import Dataset
from curvestudio.project import (
    Analysis,
    Project,
    ProjectDataset,
    ProjectError,
    load_project,
    save_project,
)
from curvestudio.tools.stats import Counts, ToolResult

CREATED_AT = "2026-11-24T09:00:00+05:30"
STAT_PATH = "analyses/stat-ds1-descriptives.json"


def make_dataset() -> Dataset:
    return Dataset("beam", pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}))


def make_result() -> ToolResult:
    """One real result model, carrying the fields the project format keeps."""
    return ToolResult(
        tool="descriptives",
        title="Descriptive statistics of y",
        level=0.95,
        interpretation="The mean of y is 4.",
        methods_fragment="The mean and standard deviation were taken.",
        counts=Counts(n_offered=3, n_used=3, n_excluded=0),
    )


def make_analysis() -> Analysis:
    return Analysis(
        id="stat-ds1-descriptives",
        dataset_id="ds1",
        kind="stat",
        config={"tool": "descriptives", "values": "y"},
        result=make_result().model_dump(mode="json"),
    )


def make_project() -> Project:
    return Project(
        name="Statistics demo",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="ds1", name="Beam tests", dataset=make_dataset()),),
        analyses=(make_analysis(),),
        plot_settings={},
    )


def save_bytes(project: Project) -> bytes:
    buffer = io.BytesIO()
    save_project(project, buffer)
    return buffer.getvalue()


def load_bytes(data: bytes) -> Project:
    return load_project(io.BytesIO(data))


def members_of(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def rezip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def rehash(members: dict[str, bytes], *paths: str) -> None:
    """Point the manifest's hashes at the rewritten payloads."""
    manifest = json.loads(members["manifest.json"])
    rewritten = set(paths)
    for entry in manifest["entries"]:
        if entry["path"] in rewritten:
            entry["sha256"] = hashlib.sha256(members[entry["path"]]).hexdigest()
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")


def downgrade_to_version_1(data: bytes) -> bytes:
    """The same archive as the previous format wrote it: schema_version 1 everywhere."""
    members = members_of(data)
    manifest = json.loads(members["manifest.json"])
    manifest["schema_version"] = 1
    rewritten: list[str] = []
    for name, payload in list(members.items()):
        if name == "manifest.json" or not name.endswith(".json"):
            continue
        document = json.loads(payload)
        if "schema_version" not in document:
            continue
        document["schema_version"] = 1
        members[name] = json.dumps(document).encode("utf-8")
        rewritten.append(name)
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    rehash(members, *rewritten)
    return rezip(members)


def patch_manifest_version(data: bytes, version: int) -> bytes:
    members = members_of(data)
    manifest = json.loads(members["manifest.json"])
    manifest["schema_version"] = version
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    return rezip(members)


def patch_payload_version(data: bytes, path: str, version: int) -> bytes:
    members = members_of(data)
    document = json.loads(members[path])
    document["schema_version"] = version
    members[path] = json.dumps(document).encode("utf-8")
    rehash(members, path)
    return rezip(members)


def patch_stat_result(data: bytes, result: Any) -> bytes:
    members = members_of(data)
    document = json.loads(members[STAT_PATH])
    document["result"] = result
    members[STAT_PATH] = json.dumps(document).encode("utf-8")
    rehash(members, STAT_PATH)
    return rezip(members)


def expect_error(data: bytes, code: str, fragment: str = "") -> ProjectError:
    with pytest.raises(ProjectError) as caught:
        load_bytes(data)
    assert caught.value.code == code
    assert fragment in caught.value.message
    return caught.value


def test_version_2_round_trips() -> None:
    project = make_project()
    assert load_bytes(save_bytes(project)) == project


def test_written_file_is_version_2_in_the_manifest_and_every_payload() -> None:
    members = members_of(save_bytes(make_project()))
    assert json.loads(members["manifest.json"])["schema_version"] == 2
    assert json.loads(members["datasets/ds1/operations.json"])["schema_version"] == 2
    assert json.loads(members[STAT_PATH])["schema_version"] == 2


def test_version_1_archive_opens_and_is_saved_as_version_2() -> None:
    project = make_project()
    downgraded = downgrade_to_version_1(save_bytes(project))

    loaded = load_bytes(downgraded)
    assert loaded == project  # every value it had survives, nothing is added

    resaved = members_of(save_bytes(loaded))
    assert json.loads(resaved["manifest.json"])["schema_version"] == 2
    assert json.loads(resaved["datasets/ds1/operations.json"])["schema_version"] == 2
    assert json.loads(resaved[STAT_PATH])["schema_version"] == 2


def test_manifest_version_3_is_unsupported() -> None:
    expect_error(patch_manifest_version(save_bytes(make_project()), 3), "UNSUPPORTED_VERSION", "3")


def test_payload_version_3_is_unsupported() -> None:
    data = patch_payload_version(save_bytes(make_project()), STAT_PATH, 3)
    expect_error(data, "UNSUPPORTED_VERSION", "3")


def test_stat_analysis_round_trips_with_its_result_and_choices() -> None:
    loaded = load_bytes(save_bytes(make_project()))
    analysis = loaded.analyses[0]
    assert analysis.kind == "stat"
    assert isinstance(analysis.result, dict)
    assert analysis.result == make_result().model_dump(mode="json")
    assert analysis.config == {"tool": "descriptives", "values": "y"}


def test_stat_result_that_is_not_a_result_model_is_corrupt() -> None:
    data = patch_stat_result(save_bytes(make_project()), {"not": "a result"})
    expect_error(data, "CORRUPT", "statistics result")


def test_stat_result_that_is_not_an_object_is_corrupt() -> None:
    data = patch_stat_result(save_bytes(make_project()), [1, 2, 3])
    expect_error(data, "CORRUPT")
