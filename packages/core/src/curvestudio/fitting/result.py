"""Serializable fit result, status and provenance.

A :class:`FitResult` is the complete, self-describing outcome of one fit: the estimates,
how they were obtained and how trustworthy they are. It must survive a JSON round trip
unchanged, so every float field rejects NaN/inf at construction. Callers express "not
defined" with ``None`` (singular covariance, unavailable standard error), which keeps the
JSON standard-compliant and makes missing information explicit instead of silent.
"""

import json
import platform
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from importlib import metadata
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator

from curvestudio.fitting.metrics import FiniteFloat, FitStatistics

IST = timezone(timedelta(hours=5, minutes=30))
"""Fixed +05:30 offset. Built by hand because Windows CI ships no tz database."""

_TRACKED_DISTRIBUTIONS = {
    "curvestudio": "curvestudio-core",
    "numpy": "numpy",
    "scipy": "scipy",
    "lmfit": "lmfit",
}


class FitStatus(StrEnum):
    """Outcome category. A failed fit is reported as such, never as a plausible result."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"


class ParameterEstimate(BaseModel):
    """One fitted parameter. ``None`` uncertainty means it could not be determined."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: FiniteFloat
    std_err: FiniteFloat | None = None
    ci_low: FiniteFloat | None = None
    ci_high: FiniteFloat | None = None


class Provenance(BaseModel):
    """What produced a result: software versions, input data identity, time and seed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    software_versions: dict[str, str]
    dataset_fingerprint: str
    created_at: str
    seed: int

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


def _version_of(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def collect_provenance(dataset_fingerprint: str, seed: int) -> Provenance:
    """Snapshot the environment now. A package that is not installed reads ``"unknown"``."""
    versions = {key: _version_of(dist) for key, dist in _TRACKED_DISTRIBUTIONS.items()}
    versions["python"] = platform.python_version()
    return Provenance(
        software_versions=versions,
        dataset_fingerprint=dataset_fingerprint,
        created_at=datetime.now(IST).isoformat(timespec="seconds"),
        seed=seed,
    )


class FitResult(BaseModel):
    """Complete, immutable record of one fit.

    The model is frozen, but list/dict fields are not deep-frozen: do not mutate them.
    :meth:`to_json` re-checks finiteness, so a mutation that sneaks in NaN/inf fails
    loudly rather than producing a file that cannot be read back.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    equation: str
    equation_fitted: str | None
    parameters: tuple[ParameterEstimate, ...]
    covariance: list[list[FiniteFloat]] | None
    correlation: list[list[FiniteFloat]] | None
    stats: FitStatistics | None
    fitted: list[FiniteFloat]
    residuals: list[FiniteFloat]
    status: FitStatus
    termination_reason: str
    method: str
    loss: str
    initial_params: dict[str, FiniteFloat]
    n_starts: int
    warnings: tuple[str, ...]
    provenance: Provenance

    def to_json(self) -> str:
        """Serialise to standard JSON (floats keep full ``repr`` precision).

        Raises:
            ValueError: a non-finite number is present (only possible through mutation
                of a list field).
        """
        return json.dumps(self.model_dump(mode="python"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse :meth:`to_json` output; invalid JSON or non-finite values raise ``ValueError``."""

        def reject_constant(token: str) -> Any:
            raise ValueError(f"non-finite JSON constant {token!r} is not allowed")

        return cls.model_validate(json.loads(text, parse_constant=reject_constant))
