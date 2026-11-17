"""Pydantic models describing a ``.cstudio`` project.

A project holds datasets (the untouched original table plus the reversible operation
log) and analyses over them. Every stored value must survive a JSON round trip
unchanged: "not defined" is ``None``, and NaN or infinity in free-form JSON is refused
so a saved file is always standard JSON.
"""

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from curvestudio.data import Dataset
from curvestudio.fitting import FitResult
from curvestudio.fitting.result import IST

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
"""Dataset and analysis ids: one safe path segment inside the archive."""

Id = Annotated[str, Field(pattern=ID_PATTERN, max_length=200)]

VERSION_KEYS = ("python", "numpy", "scipy", "sympy", "curvestudio")
"""Dependency versions every manifest records."""


def _require_plain_json(value: Any, field: str) -> None:
    """Reject values JSON cannot carry exactly (NaN, infinity, tuples, sets, ...)."""
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be plain JSON data: {exc}") from exc
    if json.loads(text) != value:
        raise ValueError(f"{field} does not survive a JSON round trip unchanged")


class ProjectDataset(BaseModel):
    """One dataset as stored in a project: its id, display name and original table."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    id: Id
    name: str = Field(min_length=1)
    dataset: Dataset


class Analysis(BaseModel):
    """One recorded analysis: what was run and what it produced.

    ``kind="fit"`` holds a complete :class:`FitResult`; ``kind="autofit"`` holds the
    Auto-fit table as opaque JSON (validated to be plain JSON data) until the Auto-fit
    result type lands.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Id
    dataset_id: Id
    kind: Literal["fit", "autofit"]
    config: dict[str, Any]
    result: FitResult | dict[str, Any]
    note: str = ""

    @field_validator("config")
    @classmethod
    def _config_is_plain_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_plain_json(value, "config")
        return value

    @model_validator(mode="after")
    def _result_matches_kind(self) -> Analysis:
        if self.kind == "fit" and not isinstance(self.result, FitResult):
            raise ValueError('a "fit" analysis needs a FitResult result')
        if self.kind == "autofit":
            if isinstance(self.result, FitResult):
                raise ValueError('an "autofit" analysis stores opaque JSON, not a FitResult')
            _require_plain_json(self.result, "result")
        return self


class Project(BaseModel):
    """Everything a ``.cstudio`` file carries. Frozen: edits create a new project."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    created_at: str
    datasets: tuple[ProjectDataset, ...] = ()
    analyses: tuple[Analysis, ...] = ()
    plot_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _require_ist_timestamp(cls, value: str) -> str:
        try:
            stamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"created_at is not ISO 8601: {value!r}") from exc
        if stamp.utcoffset() != IST.utcoffset(None):
            raise ValueError(f"created_at must carry the +05:30 (IST) offset: {value!r}")
        return value

    @field_validator("plot_settings")
    @classmethod
    def _settings_are_plain_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_plain_json(value, "plot_settings")
        return value

    @model_validator(mode="after")
    def _ids_are_unique_and_resolve(self) -> Project:
        dataset_ids = [entry.id for entry in self.datasets]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("dataset ids must be unique")
        analysis_ids = [analysis.id for analysis in self.analyses]
        if len(set(analysis_ids)) != len(analysis_ids):
            raise ValueError("analysis ids must be unique")
        unknown = sorted({analysis.dataset_id for analysis in self.analyses} - set(dataset_ids))
        if unknown:
            raise ValueError(f"analyses refer to unknown dataset ids: {', '.join(unknown)}")
        return self


class ManifestEntry(BaseModel):
    """One payload file listed in ``manifest.json`` with its SHA-256."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Manifest(BaseModel):
    """``manifest.json``: the archive's table of contents and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    app: str = "curvestudio"
    app_version: str
    versions: dict[str, str]
    project_name: str
    created_at: str
    entries: tuple[ManifestEntry, ...] = ()

    @field_validator("versions")
    @classmethod
    def _versions_are_complete(cls, value: dict[str, str]) -> dict[str, str]:
        missing = [key for key in VERSION_KEYS if key not in value]
        if missing:
            raise ValueError(f"versions must record {', '.join(missing)}")
        return value
