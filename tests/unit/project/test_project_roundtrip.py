"""Round-trip behaviour of the ``.cstudio`` project format."""

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from curvestudio.data import DataError, Dataset, ExcludeRows, RenameColumn, TransformColumn
from curvestudio.fitting import FitResult, fit_polynomial
from curvestudio.project import (
    Analysis,
    Project,
    ProjectDataset,
    ProjectError,
    load_project,
    save_project,
)

CREATED_AT = "2026-11-17T08:24:00+05:30"


def make_dataset() -> Dataset:
    return Dataset(
        "beam",
        pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5], "y": [1.0, 2.1, 2.9, 4.2, 4.8]}),
    )


def make_project() -> Project:
    dataset = make_dataset()
    result = fit_polynomial(dataset.xy("x", "y"), 2)
    return Project(
        name="Demo project",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="ds1", name="Beam tests", dataset=dataset),),
        analyses=(
            Analysis(
                id="fit1",
                dataset_id="ds1",
                kind="fit",
                config={"model": "polynomial", "degree": 2},
                result=result,
                note="quadratic trend",
            ),
        ),
        plot_settings={"x_label": "Age (days)", "log_y": False, "series": [1, 2.5]},
    )


def save_to_bytes(project: Project) -> bytes:
    buffer = io.BytesIO()
    save_project(project, buffer)
    return buffer.getvalue()


def load_from_bytes(data: bytes) -> Project:
    return load_project(io.BytesIO(data))


def archive_members(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_round_trip_equals_the_original_project() -> None:
    project = make_project()
    assert load_from_bytes(save_to_bytes(project)) == project


def test_dataset_fingerprint_is_preserved() -> None:
    project = make_project()
    loaded = load_from_bytes(save_to_bytes(project))
    assert loaded.datasets[0].dataset.fingerprint() == project.datasets[0].dataset.fingerprint()
    assert loaded.datasets[0].dataset.name == "beam"
    assert loaded.datasets[0].name == "Beam tests"


def test_polynomial_fit_numbers_are_bit_identical() -> None:
    project = make_project()
    loaded = load_from_bytes(save_to_bytes(project))
    before = project.analyses[0].result
    after = loaded.analyses[0].result
    assert isinstance(before, FitResult)
    assert isinstance(after, FitResult)
    assert after.to_json() == before.to_json()
    for old, new in zip(before.parameters, after.parameters, strict=True):
        assert new.value.hex() == old.value.hex()
        assert new.std_err is not None and old.std_err is not None
        assert new.std_err.hex() == old.std_err.hex()


def test_path_target_writes_and_reads(tmp_path: Path) -> None:
    target = tmp_path / "demo.cstudio"
    project = make_project()
    save_project(project, target)
    assert target.is_file()
    assert load_project(target) == project


def test_binary_file_object_target_and_source() -> None:
    buffer = io.BytesIO()
    project = make_project()
    save_project(project, buffer)
    assert load_project(buffer) == project


def test_archive_layout_and_manifest() -> None:
    data = save_to_bytes(make_project())
    members = archive_members(data)
    assert set(members) == {
        "manifest.json",
        "datasets/ds1/original.csv",
        "datasets/ds1/cleaned.csv",
        "datasets/ds1/operations.json",
        "analyses/fit1.json",
        "plot_settings.json",
    }
    manifest = json.loads(members["manifest.json"])
    assert manifest["schema_version"] == 1
    assert manifest["project_name"] == "Demo project"
    assert manifest["created_at"] == CREATED_AT
    assert set(manifest["versions"]) == {"python", "numpy", "scipy", "sympy", "curvestudio"}
    listed = {entry["path"]: entry["sha256"] for entry in manifest["entries"]}
    assert set(listed) == set(members) - {"manifest.json"}
    for path, digest in listed.items():
        assert hashlib.sha256(members[path]).hexdigest() == digest


def test_floats_are_written_with_shortest_repr() -> None:
    dataset = Dataset(
        "awkward",
        pd.DataFrame({"x": [0.1, 1e-300, -0.0], "y": [5e-324, 1.7976931348623157e308, 1 / 3]}),
    )
    project = Project(
        name="floats",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="awkward", dataset=dataset),),
    )
    text = archive_members(save_to_bytes(project))["datasets/d/original.csv"].decode()
    assert "0.1" in text
    assert "1e-300" in text
    assert "-0.0" in text
    assert "5e-324" in text


def test_awkward_floats_round_trip_bit_exactly() -> None:
    dataset = Dataset(
        "awkward",
        pd.DataFrame({"x": [0.1, 1e-300, -0.0], "y": [5e-324, 1.7976931348623157e308, 1 / 3]}),
    )
    project = Project(
        name="floats",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="awkward", dataset=dataset),),
    )
    loaded = load_from_bytes(save_to_bytes(project)).datasets[0].dataset
    assert loaded.fingerprint() == dataset.fingerprint()
    original = loaded.original
    assert original["x"].to_numpy()[0] == 0.1
    assert np.signbit(original["x"].to_numpy()[2]), "negative zero must keep its sign"
    assert original["y"].to_numpy()[1] == 1.7976931348623157e308


def test_non_finite_original_round_trips_and_xy_still_refuses() -> None:
    dataset = Dataset("edges", pd.DataFrame({"x": [1.0, 2.0], "y": [math.inf, 3.0]}))
    project = Project(
        name="edges",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="edges", dataset=dataset),),
    )
    loaded = load_from_bytes(save_to_bytes(project)).datasets[0].dataset
    assert loaded.fingerprint() == dataset.fingerprint()
    assert math.isinf(loaded.original["y"].to_numpy()[0])
    with pytest.raises(DataError) as caught:
        loaded.xy("x", "y")
    assert caught.value.code == "NON_FINITE"


def test_missing_original_round_trips_and_xy_still_refuses() -> None:
    dataset = Dataset("edges", pd.DataFrame({"x": [1.0, 2.0], "y": [math.nan, 3.0]}))
    project = Project(
        name="edges",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="edges", dataset=dataset),),
    )
    loaded = load_from_bytes(save_to_bytes(project)).datasets[0].dataset
    assert loaded.fingerprint() == dataset.fingerprint()
    assert math.isnan(loaded.original["y"].to_numpy()[0])
    with pytest.raises(DataError) as caught:
        loaded.xy("x", "y")
    assert caught.value.code == "MISSING_VALUE"


def test_operation_log_round_trips_and_replays() -> None:
    base = Dataset("ops", pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]}))
    dataset = (
        base.with_op(TransformColumn("x", "lx", "log"))
        .with_op(ExcludeRows((1,)))
        .with_op(RenameColumn("y", "strength"))
    )
    project = Project(
        name="ops",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="ops", dataset=dataset),),
    )
    loaded = load_from_bytes(save_to_bytes(project)).datasets[0].dataset
    assert loaded.ops == dataset.ops
    assert loaded.fingerprint() == dataset.fingerprint()
    assert loaded.table().equals(dataset.table())
    assert loaded.active_mask().tolist() == dataset.active_mask().tolist()


def test_cleaned_csv_is_readable_with_pandas() -> None:
    base = Dataset("ops", pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]}))
    dataset = base.with_op(TransformColumn("x", "lx", "log")).with_op(ExcludeRows((1,)))
    project = Project(
        name="ops",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="ops", dataset=dataset),),
    )
    members = archive_members(save_to_bytes(project))
    cleaned = pd.read_csv(io.BytesIO(members["datasets/d/cleaned.csv"]))
    assert list(cleaned.columns) == ["x", "y", "lx"]
    assert cleaned["x"].tolist() == [1.0, 2.0, 3.0]
    assert cleaned["lx"].to_numpy() == pytest.approx(np.log([1.0, 2.0, 3.0]))
    original = pd.read_csv(io.BytesIO(members["datasets/d/original.csv"]))
    assert list(original.columns) == ["x", "y"]


def test_mixed_dtypes_round_trip() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "count": [1, 2, 3],
            "flag": [True, False, True],
            "label": ["alpha", None, "gamma"],
            "when": pd.to_datetime(
                ["2026-01-01", "2026-02-03T04:05:06", "2026-03-04"], format="mixed"
            ),
        }
    )
    dataset = Dataset("mixed", frame)
    project = Project(
        name="mixed",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="mixed", dataset=dataset),),
    )
    loaded = load_from_bytes(save_to_bytes(project)).datasets[0].dataset
    assert loaded.fingerprint() == dataset.fingerprint()
    assert [str(dtype) for dtype in loaded.original.dtypes] == [
        str(dtype) for dtype in frame.dtypes
    ]
    assert loaded.original.equals(frame)


def test_autofit_analysis_keeps_opaque_json() -> None:
    dataset = make_dataset()
    ranking: dict[str, Any] = {
        "rank_by": "AICc",
        "rows": [{"rank": 1, "model": "linear", "r2": 0.99, "params": {"a": 0.1, "b": 2.0}}],
        "failed": [],
    }
    project = Project(
        name="autofit",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="beam", dataset=dataset),),
        analyses=(
            Analysis(
                id="auto1",
                dataset_id="d",
                kind="autofit",
                config={"rank_by": "AICc"},
                result=ranking,
                note="",
            ),
        ),
    )
    loaded = load_from_bytes(save_to_bytes(project))
    assert loaded.analyses[0].result == ranking
    assert loaded == project


def test_empty_project_round_trips() -> None:
    project = Project(name="empty", created_at=CREATED_AT)
    loaded = load_from_bytes(save_to_bytes(project))
    assert loaded == project
    assert loaded.datasets == ()
    assert loaded.plot_settings == {}


def test_saving_twice_produces_identical_bytes() -> None:
    project = make_project()
    assert save_to_bytes(project) == save_to_bytes(project)


def test_highly_compressible_constant_columns_round_trip() -> None:
    frame = pd.DataFrame({"x": [1.0] * 100_000, "y": [2.0] * 100_000})
    dataset = Dataset("constants", frame)
    project = Project(
        name="constants",
        created_at=CREATED_AT,
        datasets=(ProjectDataset(id="d", name="constants", dataset=dataset),),
    )
    loaded = load_from_bytes(save_to_bytes(project)).datasets[0].dataset
    assert loaded.fingerprint() == dataset.fingerprint()
    assert loaded.table().equals(dataset.table())


def test_failed_save_leaves_the_existing_file_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "demo.cstudio"
    project = make_project()
    save_project(project, target)
    before = target.read_bytes()

    original_writestr = zipfile.ZipFile.writestr
    writes = 0

    def failing_writestr(
        self: zipfile.ZipFile, info_or_name: Any, data: Any, *args: Any, **kwargs: Any
    ) -> None:
        nonlocal writes
        writes += 1
        if writes > 1:
            raise OSError("disk full")
        original_writestr(self, info_or_name, data, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", failing_writestr)
    with pytest.raises(ProjectError) as caught:
        save_project(project, target)
    assert caught.value.code == "SAVE_FAILED"
    assert target.read_bytes() == before
    assert load_project(target) == project
    assert [path.name for path in tmp_path.iterdir()] == [target.name]


def test_save_into_a_missing_directory_is_save_failed(tmp_path: Path) -> None:
    with pytest.raises(ProjectError) as caught:
        save_project(make_project(), tmp_path / "missing" / "demo.cstudio")
    assert caught.value.code == "SAVE_FAILED"
