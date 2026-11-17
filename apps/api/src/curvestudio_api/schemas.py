"""Request and response bodies. Timestamps serialise as ISO 8601 with the +05:30 offset."""

import re
import uuid
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr

from curvestudio_api.timeutil import IstDatetime

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s.]+")


def _normalise_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 254 or not _EMAIL.fullmatch(email):
        raise ValueError("not a valid email address")
    return email


Email = Annotated[str, AfterValidator(_normalise_email)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterIn(_Body):
    email: Email
    password: SecretStr = Field(min_length=10, max_length=128)
    name: str = Field(default="", max_length=100)


class LoginIn(_Body):
    email: Email
    password: SecretStr = Field(min_length=1, max_length=128)


class RefreshIn(_Body):
    refresh_token: str = Field(min_length=1, max_length=512)


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    created_at: IstDatetime


class HealthOut(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    version: str


class ReadyOut(BaseModel):
    status: Literal["ready"]


class IssueOut(BaseModel):
    code: str
    severity: Literal["error", "warning"]
    message: str
    rows: list[int] = []


class DatasetOut(BaseModel):
    id: uuid.UUID
    name: str
    n_rows: int
    columns: list[str]
    x_column: str
    y_column: str
    weight_column: str | None
    issues: list[IssueOut]
    created_at: IstDatetime


class PointsOut(BaseModel):
    """Stride-sampled points; ``w`` is ``None`` when the dataset has no weights."""

    n_total: int
    x: list[float]
    y: list[float]
    w: list[float] | None


class DatasetDetailOut(DatasetOut):
    points: PointsOut | None


class JobOut(BaseModel):
    id: uuid.UUID
    kind: Literal["fit", "autofit"]
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    dataset_id: uuid.UUID
    spec: dict[str, Any]
    result_ref: uuid.UUID | None
    error_code: str | None
    error_message: str | None
    created_at: IstDatetime
    started_at: IstDatetime | None
    finished_at: IstDatetime | None


class JobCreatedOut(BaseModel):
    job_id: uuid.UUID


class FitSummaryOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    status: str
    spec: dict[str, Any]
    created_at: IstDatetime


class PredictIn(_Body):
    x: list[float] = Field(max_length=100_000)
    band: Literal["confidence", "prediction"] | None = None
