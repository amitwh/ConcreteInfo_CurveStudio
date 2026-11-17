"""Request and response bodies. Timestamps serialise as ISO 8601 with the +05:30 offset."""

import re
import uuid
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from curvestudio_api.passwords import weakness
from curvestudio_api.timeutil import IstDatetime

_LOGIN_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s.]+")
_REGISTER_EMAIL = re.compile(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]{2,}")


def _require_utf8(value: str) -> str:
    """Reject text that cannot be encoded (a lone surrogate from a JSON ``\\ud800`` escape)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("contains characters that are not valid text") from exc
    return value


def _normalise(value: str, pattern: re.Pattern[str]) -> str:
    email = _require_utf8(value).strip().lower()
    if len(email) > 254 or not pattern.fullmatch(email):
        raise ValueError("not a valid email address")
    return email


Email = Annotated[str, AfterValidator(lambda v: _normalise(v, _LOGIN_EMAIL))]
RegisterEmail = Annotated[str, AfterValidator(lambda v: _normalise(v, _REGISTER_EMAIL))]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _secret_utf8(value: SecretStr) -> SecretStr:
    _require_utf8(value.get_secret_value())
    return value


class RegisterIn(_Body):
    email: RegisterEmail
    password: SecretStr = Field(min_length=10, max_length=128)
    name: Annotated[str, AfterValidator(_require_utf8)] = Field(default="", max_length=100)

    @field_validator("password")
    @classmethod
    def _password_text(cls, value: SecretStr) -> SecretStr:
        return _secret_utf8(value)

    @model_validator(mode="after")
    def _password_policy(self) -> RegisterIn:
        problem = weakness(self.password.get_secret_value(), self.email)
        if problem is not None:
            raise ValueError(problem)
        return self


class LoginIn(_Body):
    email: Email
    password: SecretStr = Field(min_length=1, max_length=128)

    @field_validator("password")
    @classmethod
    def _password_text(cls, value: SecretStr) -> SecretStr:
        return _secret_utf8(value)


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


class PlanOut(BaseModel):
    name: str
    max_datasets: int
    max_dataset_bytes: int
    max_fits: int


class UsageOut(BaseModel):
    datasets: int
    dataset_bytes: int
    fits: int


class MeOut(UserOut):
    plan: PlanOut
    usage: UsageOut


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
