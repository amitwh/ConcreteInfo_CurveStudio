"""Hardening of ``.cstudio`` loading: crafted archives must fail as ``ProjectError``.

The archives here are built with :mod:`zipfile` (or by patching sizes in a real
archive's central directory) so every limit is exercised before extraction.
"""

import hashlib
import io
import json
import random
import stat
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from curvestudio.data import Dataset
from curvestudio.project import (
    Analysis,
    Project,
    ProjectDataset,
    ProjectError,
    load_project,
    save_project,
)

CREATED_AT = "2026-11-17T08:24:00+05:30"
MEGABYTE = 1024 * 1024


def zipped(entries: Sequence[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


def saved_project() -> bytes:
    dataset = Dataset("t", pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}))
    project = Project(
        name="p",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="ds1", name="t", dataset=dataset),),
        analyses=(
            Analysis(
                id="a1",
                dataset_id="ds1",
                kind="autofit",
                config={},
                result={"rows": []},
            ),
        ),
    )
    buffer = io.BytesIO()
    save_project(project, buffer)
    return buffer.getvalue()


def members_of(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def pack(members: dict[str, bytes], manifest: dict[str, object]) -> bytes:
    return zipped([("manifest.json", json.dumps(manifest).encode()), *members.items()])


def rewrite(data: bytes, mutate: Callable[[dict[str, bytes]], None]) -> bytes:
    """Rebuild ``data`` with mutated payloads and a manifest re-hashed to match."""
    members = members_of(data)
    manifest = json.loads(members.pop("manifest.json"))
    mutate(members)
    manifest["entries"] = [
        {"path": name, "sha256": hashlib.sha256(body).hexdigest()} for name, body in members.items()
    ]
    return pack(members, manifest)


def central_directory(blob: bytes | bytearray) -> int:
    end = blob.rfind(b"PK\x05\x06")
    assert end >= 0, "no end-of-central-directory record"
    return int.from_bytes(blob[end + 16 : end + 20], "little")


def patch_declared_sizes(
    data: bytes,
    needles: Sequence[str],
    *,
    uncompressed: int | None = None,
    compressed: int | None = None,
) -> bytes:
    """Rewrite sizes the central directory declares, leaving the payload untouched."""
    blob = bytearray(data)
    offset = central_directory(blob)
    patched = 0
    while blob[offset : offset + 4] == b"PK\x01\x02":
        name_len = int.from_bytes(blob[offset + 28 : offset + 30], "little")
        extra_len = int.from_bytes(blob[offset + 30 : offset + 32], "little")
        comment_len = int.from_bytes(blob[offset + 32 : offset + 34], "little")
        name = bytes(blob[offset + 46 : offset + 46 + name_len]).decode()
        if any(needle in name for needle in needles):
            if compressed is not None:
                blob[offset + 20 : offset + 24] = compressed.to_bytes(4, "little")
            if uncompressed is not None:
                blob[offset + 24 : offset + 28] = uncompressed.to_bytes(4, "little")
            patched += 1
        offset += 46 + name_len + extra_len + comment_len
    assert patched, "no matching central-directory entry"
    return bytes(blob)


def patch_extract_version(data: bytes, version: int) -> bytes:
    """Rewrite the first central-directory entry's "version needed to extract"."""
    blob = bytearray(data)
    offset = central_directory(blob)
    assert blob[offset : offset + 4] == b"PK\x01\x02", "no central-directory entry"
    blob[offset + 6 : offset + 8] = version.to_bytes(2, "little")
    return bytes(blob)


def expect_error(data: bytes, code: str, fragment: str = "") -> ProjectError:
    with pytest.raises(ProjectError) as caught:
        load_project(io.BytesIO(data))
    assert caught.value.code == code
    assert fragment in caught.value.message
    return caught.value


# --- path and entry safety -------------------------------------------------------------


def test_parent_directory_entry_is_refused() -> None:
    expect_error(
        zipped([("manifest.json", b"{}"), ("../evil.csv", b"x")]),
        "UNSAFE_ARCHIVE",
        "parent directory",
    )


def test_absolute_path_entry_is_refused() -> None:
    expect_error(
        zipped([("manifest.json", b"{}"), ("/etc/passwd", b"x")]), "UNSAFE_ARCHIVE", "absolute"
    )


def test_backslash_entry_is_refused() -> None:
    expect_error(
        zipped([("manifest.json", b"{}"), ("datasets\\ds1\\original.csv", b"x")]),
        "UNSAFE_ARCHIVE",
        "backslash",
    )


def test_symlink_entry_is_refused() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", b"{}")
        info = zipfile.ZipInfo("datasets/ds1/original.csv")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"/etc/passwd")
    expect_error(buffer.getvalue(), "UNSAFE_ARCHIVE", "symlink")


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_duplicate_entry_names_are_refused() -> None:
    expect_error(
        zipped(
            [("manifest.json", b"{}"), ("plot_settings.json", b"{}"), ("plot_settings.json", b"{}")]
        ),
        "UNSAFE_ARCHIVE",
        "duplicate",
    )


def test_directory_entry_is_refused() -> None:
    expect_error(
        zipped([("manifest.json", b"{}"), ("datasets/", b"")]), "UNSAFE_ARCHIVE", "directory"
    )


def test_payload_names_with_a_trailing_newline_are_refused() -> None:
    members = members_of(saved_project())
    del members["manifest.json"]
    for name, body in members.items():
        expect_error(
            zipped([("manifest.json", b"{}"), (name + "\n", body)]),
            "UNSAFE_ARCHIVE",
            "control",
        )


@pytest.mark.parametrize("control", ["\x01", "\x1f", "\x7f"])
def test_control_characters_in_entry_names_are_refused(control: str) -> None:
    expect_error(
        zipped([("manifest.json", b"{}"), (f"datasets/ds1/original.csv{control}", b"x")]),
        "UNSAFE_ARCHIVE",
        "control",
    )


def test_manifest_paths_with_a_trailing_newline_are_refused() -> None:
    members = members_of(saved_project())
    manifest = json.loads(members.pop("manifest.json"))
    manifest["entries"][0]["path"] += "\n"
    expect_error(pack(members, manifest), "UNSAFE_ARCHIVE", "control")


# --- size and compression limits -------------------------------------------------------


def test_entry_count_is_capped() -> None:
    entries = [("manifest.json", b"{}")] + [(f"f{index}.txt", b"") for index in range(2000)]
    expect_error(zipped(entries), "UNSAFE_ARCHIVE", "entries")


def test_compression_ratio_bomb_is_refused() -> None:
    members = members_of(saved_project())
    manifest = json.loads(members.pop("manifest.json"))
    members["datasets/ds1/bomb.bin"] = b"\0" * (17 * MEGABYTE)
    expect_error(pack(members, manifest), "UNSAFE_ARCHIVE", "ratio")


def test_whitespace_json_bomb_is_refused_by_its_declared_ratio() -> None:
    members = members_of(saved_project())
    manifest = json.loads(members.pop("manifest.json"))
    members["datasets/ds1/bomb.json"] = b" " * 1024
    data = pack(members, manifest)
    expect_error(
        patch_declared_sizes(data, ["bomb.json"], uncompressed=190 * MEGABYTE),
        "UNSAFE_ARCHIVE",
        "ratio",
    )


def test_manifest_listing_too_many_entries_is_refused() -> None:
    manifest = {
        "schema_version": 1,
        "entries": [
            {"path": f"e{index}", "sha256": hashlib.sha256(str(index).encode()).hexdigest()}
            for index in range(2001)
        ],
    }
    expect_error(
        zipped([("manifest.json", json.dumps(manifest).encode())]), "UNSAFE_ARCHIVE", "lists"
    )


def test_huge_manifest_is_refused_before_it_is_read() -> None:
    manifest = {
        "schema_version": 1,
        "entries": [
            {"path": f"e{index}", "sha256": hashlib.sha256(str(index).encode()).hexdigest()}
            for index in range(100_000)
        ],
    }
    data = zipped([("manifest.json", json.dumps(manifest).encode())])
    start = time.perf_counter()
    error = expect_error(data, "UNSAFE_ARCHIVE", "manifest.json")
    assert time.perf_counter() - start < 1.0, "the oversized manifest must be refused unread"
    assert error.code == "UNSAFE_ARCHIVE"


def test_declared_entry_size_is_capped() -> None:
    data = patch_declared_sizes(saved_project(), ["original.csv"], uncompressed=201 * MEGABYTE)
    expect_error(data, "UNSAFE_ARCHIVE", "expands to")


def test_declared_total_size_is_capped() -> None:
    with zipfile.ZipFile(io.BytesIO(saved_project())) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()]
    data = zipped([*members, ("extra.bin", b"x")])
    every_entry = [
        "manifest.json",
        "plot_settings.json",
        "original.csv",
        "cleaned.csv",
        "operations.json",
        "extra.bin",
    ]
    patched = patch_declared_sizes(
        data, every_entry, uncompressed=200 * MEGABYTE, compressed=1 * MEGABYTE
    )
    expect_error(patched, "UNSAFE_ARCHIVE", "expands to more than")


# --- integrity of the payload ----------------------------------------------------------


def test_tampered_payload_fails_its_checksum() -> None:
    data = saved_project()
    members = members_of(data)
    members["datasets/ds1/original.csv"] = b"x,y\n9,9\n"
    manifest = json.loads(members.pop("manifest.json"))
    expect_error(pack(members, manifest), "CHECKSUM_MISMATCH", "original.csv")


def test_cleaned_csv_must_match_the_replayed_log() -> None:
    def tamper(members: dict[str, bytes]) -> None:
        members["datasets/ds1/cleaned.csv"] = b"x,y\n1.0,99.0\n2.0,4.0\n"

    expect_error(rewrite(saved_project(), tamper), "CORRUPT", "cleaned.csv")


def test_tampered_original_csv_fails_the_fingerprint() -> None:
    def tamper(members: dict[str, bytes]) -> None:
        members["datasets/ds1/original.csv"] = b"x,y\n1.0,3.0\n2.0,5.0\n"

    expect_error(rewrite(saved_project(), tamper), "CORRUPT", "fingerprint")


def test_oversized_integer_cell_says_number_too_large() -> None:
    dataset = Dataset("ints", pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}))
    project = Project(
        name="p",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="ints", dataset=dataset),),
    )
    buffer = io.BytesIO()
    save_project(project, buffer)

    def tamper(members: dict[str, bytes]) -> None:
        members["datasets/d/original.csv"] = b"x,y\n" + b"9" * 5000 + b",3.0\n" + b"2,4.0\n"

    expect_error(rewrite(buffer.getvalue(), tamper), "CORRUPT", "number too large")


def test_oversized_integer_in_json_says_number_too_large() -> None:
    def tamper(members: dict[str, bytes]) -> None:
        members["plot_settings.json"] = b'{"n": ' + b"9" * 5000 + b"}"

    expect_error(rewrite(saved_project(), tamper), "CORRUPT", "number too large")


def test_entry_not_listed_in_the_manifest_is_refused() -> None:
    members = members_of(saved_project())
    manifest = json.loads(members.pop("manifest.json"))
    members["datasets/ds1/extra.csv"] = b"x\n1\n"
    expect_error(pack(members, manifest), "UNSAFE_ARCHIVE", "not listed")


def test_manifest_path_traversal_is_refused() -> None:
    members = members_of(saved_project())
    manifest = json.loads(members.pop("manifest.json"))
    manifest["entries"].append({"path": "../evil.csv", "sha256": "0" * 64})
    expect_error(pack(members, manifest), "UNSAFE_ARCHIVE", "parent directory")


# --- format and version ----------------------------------------------------------------


def test_unknown_schema_version_is_named() -> None:
    data = zipped([("manifest.json", json.dumps({"schema_version": 7}).encode())])
    expect_error(data, "UNSUPPORTED_VERSION", "7")


def test_non_zip_bytes_are_not_a_project() -> None:
    expect_error(b"these are not the bytes of a zip archive", "NOT_A_PROJECT")


def test_empty_file_is_not_a_project() -> None:
    expect_error(b"", "NOT_A_PROJECT")


def test_truncated_archive_is_not_a_project() -> None:
    data = saved_project()
    expect_error(data[: len(data) // 2], "NOT_A_PROJECT")


def test_missing_path_says_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(ProjectError) as caught:
        load_project(tmp_path / "missing.cstudio")
    assert caught.value.code == "NOT_A_PROJECT"
    assert "file not found" in caught.value.message


def test_unsupported_zip_extract_version_is_a_project_error() -> None:
    expect_error(patch_extract_version(saved_project(), 67), "CORRUPT", "unsupported ZIP")


def test_random_byte_flips_never_leak_another_exception() -> None:
    data = saved_project()
    rng = random.Random(20261117)
    for _ in range(500):
        mutated = bytearray(data)
        mutated[rng.randrange(len(mutated))] ^= 1 << rng.randrange(8)
        try:
            load_project(io.BytesIO(bytes(mutated)))
        except ProjectError:
            continue


def test_missing_manifest_is_not_a_project() -> None:
    expect_error(zipped([("plot_settings.json", b"{}")]), "NOT_A_PROJECT", "manifest")


def test_manifest_that_is_not_json_is_corrupt() -> None:
    expect_error(zipped([("manifest.json", b"{not json")]), "CORRUPT", "JSON")


def test_manifest_with_the_wrong_shape_is_corrupt() -> None:
    expect_error(zipped([("manifest.json", b'{"hello": "world"}')]), "CORRUPT", "schema_version")
    expect_error(
        zipped([("manifest.json", b'{"schema_version": 1, "project_name": 3}')]), "CORRUPT"
    )


def test_manifest_missing_a_dependency_version_is_corrupt() -> None:
    members = members_of(saved_project())
    manifest = json.loads(members.pop("manifest.json"))
    del manifest["versions"]["sympy"]
    expect_error(pack(members, manifest), "CORRUPT", "sympy")


# --- model-level guards ----------------------------------------------------------------


def test_dataset_ids_cannot_escape_the_archive() -> None:
    dataset = Dataset("t", pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValidationError):
        ProjectDataset(id="../evil", name="t", dataset=dataset)
    with pytest.raises(ValidationError):
        ProjectDataset(id="a/b", name="t", dataset=dataset)


def test_project_rejects_duplicate_ids_and_dangling_references() -> None:
    dataset = Dataset("t", pd.DataFrame({"x": [1.0]}))
    entry = ProjectDataset(id="d", name="t", dataset=dataset)
    with pytest.raises(ValidationError):
        Project(name="p", created_at=CREATED_AT, datasets=(entry, entry))
    dangling = Analysis(id="a", dataset_id="missing", kind="autofit", config={}, result={})
    with pytest.raises(ValidationError):
        Project(name="p", created_at=CREATED_AT, analyses=(dangling,))


def test_case_insensitively_colliding_dataset_ids_are_refused_on_save() -> None:
    dataset = Dataset("t", pd.DataFrame({"x": [1.0]}))
    project = Project(
        name="p",
        created_at=CREATED_AT,
        datasets=(
            ProjectDataset(id="ds1", name="a", dataset=dataset),
            ProjectDataset(id="DS1", name="b", dataset=dataset),
        ),
    )
    with pytest.raises(ProjectError) as caught:
        save_project(project, io.BytesIO())
    assert caught.value.code == "UNSUPPORTED_DATA"
    assert "case-insensitively" in caught.value.message


def test_case_insensitively_colliding_analysis_ids_are_refused_on_save() -> None:
    dataset = Dataset("t", pd.DataFrame({"x": [1.0]}))
    project = Project(
        name="p",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="ds1", name="t", dataset=dataset),),
        analyses=(
            Analysis(id="a1", dataset_id="ds1", kind="autofit", config={}, result={}),
            Analysis(id="A1", dataset_id="ds1", kind="autofit", config={}, result={}),
        ),
    )
    with pytest.raises(ProjectError) as caught:
        save_project(project, io.BytesIO())
    assert caught.value.code == "UNSUPPORTED_DATA"
    assert "case-insensitively" in caught.value.message


@pytest.mark.parametrize(
    ("member", "twin"),
    [
        ("datasets/ds1/original.csv", "datasets/DS1/original.csv"),
        ("analyses/a1.json", "analyses/A1.json"),
    ],
)
def test_case_insensitively_colliding_entry_names_are_refused_on_load(
    member: str, twin: str
) -> None:
    members = members_of(saved_project())
    manifest = json.loads(members.pop("manifest.json"))
    members[twin] = members[member]
    expect_error(pack(members, manifest), "UNSAFE_ARCHIVE", "case-insensitively")


def test_project_requires_an_ist_timestamp() -> None:
    with pytest.raises(ValidationError):
        Project(name="p", created_at="2026-11-17T08:24:00+00:00")


def test_load_never_leaks_another_exception() -> None:
    data = saved_project()
    blobs = [
        b"",
        b"not a zip",
        data[: len(data) // 3],
        zipped([("manifest.json", b"\xff\xfe")]),
        zipped([("manifest.json", b"[]")]),
        zipped([("manifest.json", b'{"schema_version": true}')]),
    ]
    for blob in blobs:
        with pytest.raises(ProjectError):
            load_project(io.BytesIO(blob))
