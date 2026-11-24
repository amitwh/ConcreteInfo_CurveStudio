"""Dataset upload, listing, detail and delete."""

import re
import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, Query, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import defer

from curvestudio.workflow import WorkflowError
from curvestudio_api import audit, quotas
from curvestudio_api.deps import Ctx, CurrentUser, Db, own_dataset
from curvestudio_api.errors import ApiError
from curvestudio_api.importer import precheck
from curvestudio_api.orm import Dataset
from curvestudio_api.schemas import DatasetDetailOut, DatasetOut, IssueOut, PointsOut
from curvestudio_api.storage import decode_payload
from curvestudio_api.workflow_errors import api_error

router = APIRouter(prefix="/datasets", tags=["datasets"])

MAX_NAME_LENGTH = 100
DEFAULT_NAME = "dataset"
DEFAULT_POINTS = 2000
MAX_POINTS = 20_000
_NAME_UNSAFE = re.compile(r"[^\w .()+-]+")


def sanitise_name(raw: str | None) -> str:
    """A display name safe to store and show: no path separators or control characters."""
    if raw is None:
        return ""
    cleaned = _NAME_UNSAFE.sub("", raw.replace("\\", "/").rsplit("/", 1)[-1])
    return " ".join(cleaned.split())[:MAX_NAME_LENGTH].strip(" .")


def _blank_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _fields(dataset: Dataset) -> dict[str, object]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "n_rows": dataset.n_rows,
        "columns": list(dataset.columns),
        "x_column": dataset.x_column,
        "y_column": dataset.y_column,
        "weight_column": dataset.weight_column,
        "issues": [IssueOut.model_validate(i) for i in dataset.issues],
        "created_at": dataset.created_at,
    }


def dataset_out(dataset: Dataset) -> DatasetOut:
    return DatasetOut.model_validate(_fields(dataset))


@router.post("", status_code=201, response_model=DatasetOut)
def create_dataset(
    user: CurrentUser,
    ctx: Ctx,
    db: Db,
    file: Annotated[UploadFile, File(description="csv, tsv, txt, json or xlsx table")],
    name: Annotated[str | None, Form(max_length=300)] = None,
    x_column: Annotated[str | None, Form(max_length=255)] = None,
    y_column: Annotated[str | None, Form(max_length=255)] = None,
    weight_column: Annotated[str | None, Form(max_length=255)] = None,
) -> DatasetOut:
    """Upload a table. The size and row caps are enforced before the file is parsed."""
    limit = ctx.settings.max_upload_bytes
    raw = file.file.read(limit + 1)
    if len(raw) > limit:
        raise ApiError(
            413,
            "TOO_LARGE",
            f"the file is larger than the limit of {limit:,} bytes",
            {"code": "TOO_LARGE"},
        )
    filename = file.filename or ""
    quotas.check(db, ctx.settings, user.id, "datasets")
    precheck(raw, filename, ctx.settings.max_rows)  # cheap refusals, no process needed
    run = ctx.imports.run(
        user.id,
        {
            "raw": raw,
            "filename": filename,
            "x_column": _blank_to_none(x_column),
            "y_column": _blank_to_none(y_column),
            "weight_column": _blank_to_none(weight_column),
            "max_bytes": limit,
            "max_rows": ctx.settings.max_rows,
        },
    )
    meta = run.meta
    if meta["status"] == "error":
        raise api_error(WorkflowError(str(meta["code"]), str(meta["message"])))
    if meta["status"] == "unusable":
        issues = meta["issues"]
        errors = [i for i in issues if i["severity"] == "error"]
        first = errors[0] if errors else None
        raise ApiError(
            422,
            "VALIDATION_ERROR",
            first["message"] if first else "the data cannot be used for a fit",
            {
                "code": first["code"] if first else "UNUSABLE_DATA",
                "columns": meta["columns"],
                "issues": issues,
            },
        )
    quotas.check(db, ctx.settings, user.id, "dataset_bytes", adding_bytes=len(run.payload))
    dataset = Dataset(
        id=uuid.uuid4(),
        user_id=user.id,
        name=sanitise_name(name) or sanitise_name(str(meta["name"])) or DEFAULT_NAME,
        n_rows=int(meta["n_rows"]),
        columns=list(meta["columns"]),
        x_column=str(meta["x_column"]),
        y_column=str(meta["y_column"]),
        weight_column=meta["weight_column"],
        issues=list(meta["issues"]),
        fingerprint=str(meta["fingerprint"]),
        payload=run.payload,
    )
    db.add(dataset)
    audit.record(
        db,
        "dataset.create",
        "dataset",
        user_id=user.id,
        entity_id=dataset.id,
        detail={"n_rows": dataset.n_rows, "bytes": len(raw)},
    )
    db.commit()
    return dataset_out(dataset)


@router.get("", response_model=list[DatasetOut])
def list_datasets(user: CurrentUser, db: Db) -> list[DatasetOut]:
    rows = db.scalars(
        select(Dataset)
        .where(Dataset.user_id == user.id)
        .options(defer(Dataset.payload))
        .order_by(Dataset.created_at.desc(), Dataset.id)
    )
    return [dataset_out(d) for d in rows]


def _sample(values: np.ndarray, keep: np.ndarray) -> list[float]:
    return [float(v) for v in values[keep]]


@router.get("/{dataset_id}", response_model=DatasetDetailOut)
def get_dataset(
    dataset_id: uuid.UUID,
    user: CurrentUser,
    db: Db,
    points: Annotated[
        int,
        Query(
            ge=0,
            le=MAX_POINTS,
            description=(
                f"at most this many stride-sampled points (default {DEFAULT_POINTS}); 0 for none"
            ),
        ),
    ] = DEFAULT_POINTS,
) -> DatasetDetailOut:
    dataset = own_dataset(db, user, dataset_id)
    sampled: PointsOut | None = None
    if points > 0:
        xy = decode_payload(
            dataset.payload,
            x_name=dataset.x_column,
            y_name=dataset.y_column,
            fingerprint=dataset.fingerprint,
        )
        n = int(xy.x.size)
        keep = (
            np.arange(n)
            if n <= points
            else np.unique(np.linspace(0, n - 1, points).round().astype(np.intp))
        )
        sampled = PointsOut(
            n_total=n,
            x=_sample(xy.x, keep),
            y=_sample(xy.y, keep),
            w=None if xy.weights is None else _sample(xy.weights, keep),
        )
    return DatasetDetailOut.model_validate({**_fields(dataset), "points": sampled})


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: uuid.UUID, user: CurrentUser, ctx: Ctx, db: Db) -> Response:
    """Delete a dataset together with its fits and jobs (running jobs are stopped first)."""
    dataset = own_dataset(db, user, dataset_id)
    ctx.jobs.cancel_for_dataset(dataset.id)
    # No name in the row: a dataset's name (often a file name) can identify a person, and the
    # audit trail outlives the account.
    audit.record(db, "dataset.delete", "dataset", user_id=user.id, entity_id=dataset.id)
    db.delete(dataset)
    db.commit()
    return Response(status_code=204)
