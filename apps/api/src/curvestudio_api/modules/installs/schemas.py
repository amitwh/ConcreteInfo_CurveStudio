"""The bodies of the install-record domain. A report names no person and no address: an
installation is known by a hash of ``install_id`` and by nothing else."""

import uuid
from datetime import date

from pydantic import BaseModel, Field, field_validator

from curvestudio_api.schemas import _Body

_MAX = 40


class InstallRecordIn(_Body):
    """One report from a desktop installation."""

    install_id: uuid.UUID
    app_version: str = Field(min_length=1, max_length=_MAX)
    os_version: str = Field(max_length=_MAX)
    os_name: str = Field(min_length=1, max_length=_MAX)
    arch: str = Field(min_length=1, max_length=_MAX)
    package: str = Field(min_length=1, max_length=_MAX)
    session: str = Field(min_length=1, max_length=_MAX)

    @field_validator("install_id")
    @classmethod
    def _version_four(cls, value: uuid.UUID) -> uuid.UUID:
        """Only a version 4 UUID. The other versions are derived from something: version 1
        from the clock and the network card's address, which is just what is not stored here."""
        if value.version != 4:
            raise ValueError("install_id must be a version 4 UUID")
        return value


class NameCount(BaseModel):
    """How many installations hold ``name``."""

    name: str
    count: int


class WeekCount(BaseModel):
    """How many installations were first seen in the week that starts on ``week_start``."""

    week_start: date
    count: int


class InstallSummaryOut(BaseModel):
    """The figures the admin console shows."""

    total: int
    new_30d: int
    latest_release: str
    latest_share: float
    by_version: list[NameCount]
    by_os: list[NameCount]
    by_arch: list[NameCount]
    by_package: list[NameCount]
    by_session: list[NameCount]
    new_per_week: list[WeekCount]
