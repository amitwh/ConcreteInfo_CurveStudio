"""The licence connector: settings, activation, release, the daily re-check and the migration.

The portal is a fake: a real generated RSA key pair signs the licence tokens, and a fake HTTP
function answers the connector's four calls (activate, check, deactivate, public key).
"""

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import pytest
from alembic import command
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from curvestudio_api import licences
from curvestudio_api.app import create_app
from curvestudio_api.licences import LicenceError, PortalUnreachable
from curvestudio_api.migrations import _config, upgrade_to_head
from curvestudio_api.orm import AuditLog, Entitlement, Plan, User
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import utcnow

KEY = "CURVESTUDIO-TEST-KEY-0001"
LICENCE_URL = "https://licence.example.test"
APP_SLUG = "curvestudio"


def _key_pair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    return private_pem, public_pem


@dataclass
class Call:
    """One request the connector made to the fake portal."""

    url: str
    payload: dict[str, Any] | None
    token: str | None


@dataclass
class Portal:
    """A fake AppLicense service over a real database with one account and one paid plan."""

    app: Any
    settings: Settings
    private_pem: str
    public_pem: str
    user_id: uuid.UUID
    licence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    calls: list[Call] = field(default_factory=list)
    down: bool = False
    check_valid: bool = True
    renew: bool = True
    activate_message: str = "Key not found or invalid"
    check_message: str = "License has been revoked"
    features: list[str] = field(default_factory=lambda: ["pro", "export"])
    app_slug_in_token: str = APP_SLUG
    device: str | None = None
    expires_at: datetime | None = None
    token_exp: datetime | None = None

    def __post_init__(self) -> None:
        self.expires_at = utcnow().replace(microsecond=0) + timedelta(days=30)
        self.token_exp = utcnow().replace(microsecond=0) + timedelta(hours=1)

    @property
    def sessions(self) -> Callable[[], Session]:
        return self.app.state.ctx.session_factory  # type: ignore[no-any-return]

    def token(self, **overrides: Any) -> str:
        claims: dict[str, Any] = {
            "license_id": self.licence_id,
            "app_slug": self.app_slug_in_token,
            "client_id": "client-1",
            "device_fingerprint": self.device or f"web:{self.user_id}",
            "license_type": "per_device",
            "features": list(self.features),
            "expires_at": None if self.expires_at is None else self.expires_at.isoformat(),
            "exp": int(self.token_exp.timestamp()) if self.token_exp else 0,
            "iat": int(utcnow().timestamp()),
            "jti": uuid.uuid4().hex,
        }
        claims.update(overrides)
        return jwt.encode(claims, self.private_pem, algorithm="RS256")

    def handle(
        self,
        url: str,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(Call(url, payload, token))
        if self.down:
            raise PortalUnreachable("could not reach the licence service")
        if url.endswith("/api/verify/public-key"):
            return {"public_key": self.public_pem}
        if url.endswith("/api/verify/activate"):
            if payload is None or payload.get("key") != KEY:
                return {
                    "valid": False,
                    "license_jwt": None,
                    "message": self.activate_message,
                    "features": [],
                }
            return {
                "valid": True,
                "license_jwt": self.token(),
                "message": None,
                "features": list(self.features),
            }
        if url.endswith("/api/verify/check"):
            if not self.check_valid:
                return {
                    "valid": False,
                    "license_jwt": None,
                    "message": self.check_message,
                    "features": [],
                }
            return {
                "valid": True,
                "license_jwt": self.token() if self.renew else None,
                "message": None,
                "features": list(self.features),
            }
        if url.endswith("/api/verify/deactivate"):
            return {
                "valid": True,
                "license_jwt": None,
                "message": "Device deactivated successfully",
                "features": [],
            }
        raise AssertionError(f"unexpected URL {url}")


def licence_settings(directory: Path, public_pem: str, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "env": "test",
        "database_url": f"sqlite:///{directory}/api.db",
        "data_dir": directory,
        "licence_url": LICENCE_URL,
        "licence_app_id": APP_SLUG,
        "licence_plan_map": '{"pro": "paid"}',
        "licence_public_key_pem": public_pem,
        "licence_portal_url": "https://licence.example.test/pricing",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def portal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Portal]:
    private_pem, public_pem = _key_pair()
    settings = licence_settings(tmp_path, public_pem)
    app = create_app(settings)
    upgrade_to_head(app.state.ctx.engine)
    with app.state.ctx.session_factory() as db:
        db.add(Plan(name="paid", limits={"max_datasets": 10, "max_dataset_bytes": 1000}))
        user = User(email=f"licence-{uuid.uuid4().hex[:8]}@example.org", password_hash="x")
        db.add(user)
        db.commit()
        user_id = user.id
    licences._public_keys.clear()
    fake = Portal(app, settings, private_pem, public_pem, user_id)
    monkeypatch.setattr(licences, "_http_json", fake.handle)
    yield fake
    licences._public_keys.clear()


def user_of(portal: Portal, db: Session) -> User:
    user = db.get(User, portal.user_id)
    assert user is not None
    return user


def stored(portal: Portal) -> Entitlement:
    """The account's entitlement row, read now (the deferred token included) for a detached look."""
    with portal.sessions() as db:
        row = db.scalar(select(Entitlement).where(Entitlement.user_id == portal.user_id))
        assert row is not None
        _ = row.token
        return row


def audits(db: Session, action: str) -> list[AuditLog]:
    return list(db.scalars(select(AuditLog).where(AuditLog.action == action)))


def audit_blob(db: Session) -> str:
    return json.dumps(
        [[a.action, a.entity, a.entity_id, a.detail] for a in db.scalars(select(AuditLog)).all()]
    )


# -- settings -------------------------------------------------------------------------------------


def test_a_configured_connector_turns_billing_on(tmp_path: Path) -> None:
    settings = licence_settings(tmp_path, "PEM")
    assert settings.billing_enabled is True
    assert settings.licence_plans == {"pro": "paid"}


def test_an_empty_plan_map_is_no_mapping(tmp_path: Path) -> None:
    settings = licence_settings(tmp_path, "PEM", licence_plan_map="")
    assert settings.licence_plans == {}


def test_a_malformed_plan_map_names_its_variable(tmp_path: Path) -> None:
    settings = licence_settings(tmp_path, "PEM", licence_plan_map="{not json")
    with pytest.raises(ValueError, match="CURVESTUDIO_LICENCE_PLAN_MAP"):
        _ = settings.licence_plans
    wrong = licence_settings(tmp_path, "PEM", licence_plan_map='["pro"]')
    with pytest.raises(ValueError, match="CURVESTUDIO_LICENCE_PLAN_MAP"):
        _ = wrong.licence_plans


# -- activation -----------------------------------------------------------------------------------


def row_grace(portal: Portal) -> datetime:
    """The grace an activation stores: the token's expiry plus seven days."""
    assert portal.token_exp is not None
    return portal.token_exp + timedelta(days=7)


def test_activation_stores_the_entitlement_and_moves_the_plan(portal: Portal) -> None:
    with portal.sessions() as db:
        row = licences.activate(db, portal.settings, user_of(portal, db), KEY)
        assert row.status == "active"
        assert row.licence_id == portal.licence_id
        assert row.plan_code == "pro"
        assert row.grace_until == row_grace(portal)
        assert row.last_checked_at is not None

    request = [call for call in portal.calls if call.url.endswith("/api/verify/activate")]
    assert request[0].payload == {
        "key": KEY,
        "device_fingerprint": f"web:{portal.user_id}",
        "app_slug": APP_SLUG,
    }
    row = stored(portal)
    assert row.token and row.expires_at == portal.expires_at
    with portal.sessions() as db:
        assert user_of(portal, db).plan_override == "paid"
        activated = audits(db, "user.licence_activate")
        assert len(activated) == 1
        assert activated[0].detail["licence_id"] == portal.licence_id
        assert activated[0].detail["plan_code"] == "pro"
        assert activated[0].entity_id == str(portal.user_id)
        changes = audits(db, "user.plan_change")
        assert [c.detail for c in changes] == [{"from": None, "to": "paid"}]


def test_the_key_and_the_token_never_reach_the_audit_trail(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    token = stored(portal).token
    with portal.sessions() as db:
        licences.release(db, portal.settings, user_of(portal, db))
        licences.recheck(db, portal.settings, now=utcnow() + timedelta(days=2))
        blob = audit_blob(db)
    assert KEY not in blob
    assert token not in blob
    assert "license_jwt" not in blob


def test_an_invalid_key_is_refused_with_the_services_message(portal: Portal) -> None:
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), "WRONG-KEY")
    assert failure.value.code == "LICENCE_INVALID"
    assert failure.value.message == portal.activate_message
    with portal.sessions() as db:
        assert db.scalar(select(Entitlement)) is None
        assert user_of(portal, db).plan_override is None


def test_a_plan_code_no_mapping_names_is_refused(portal: Portal) -> None:
    portal.features = ["gold", "export"]
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_UNKNOWN_PLAN"
    assert "gold" in failure.value.message
    with portal.sessions() as db:
        assert db.scalar(select(Entitlement)) is None
        assert user_of(portal, db).plan_override is None


def test_a_plan_mapped_to_an_unknown_plan_is_refused(portal: Portal) -> None:
    portal.settings = portal.settings.model_copy(update={"licence_plan_map": '{"pro": "platinum"}'})
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_UNKNOWN_PLAN"
    assert "platinum" in failure.value.message


def test_a_token_for_another_application_is_refused(portal: Portal) -> None:
    portal.app_slug_in_token = "some-other-app"
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_INVALID"


def test_a_token_for_another_device_is_refused(portal: Portal) -> None:
    portal.device = f"web:{uuid.uuid4()}"
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_DEVICE"


def test_an_expired_token_is_refused(portal: Portal) -> None:
    portal.token_exp = utcnow().replace(microsecond=0) - timedelta(minutes=1)
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_EXPIRED"


def test_an_expired_licence_is_refused(portal: Portal) -> None:
    portal.expires_at = utcnow().replace(microsecond=0) - timedelta(days=1)
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_EXPIRED"


def test_a_token_signed_by_another_key_is_refused(portal: Portal) -> None:
    other_private, _ = _key_pair()
    portal.private_pem = other_private
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_INVALID"


def test_an_unreachable_service_on_activation_is_typed(portal: Portal) -> None:
    portal.down = True
    with portal.sessions() as db, pytest.raises(LicenceError) as failure:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    assert failure.value.code == "LICENCE_UNREACHABLE"


def test_the_public_key_is_fetched_once_when_none_is_configured(portal: Portal) -> None:
    unconfigured = portal.settings.model_copy(update={"licence_public_key_pem": None})
    device = f"web:{portal.user_id}"
    assert licences.verify_token(portal.token(), unconfigured, device=device).licence_id == (
        portal.licence_id
    )
    licences.verify_token(portal.token(), unconfigured, device=device)
    fetches = [call for call in portal.calls if call.url.endswith("/api/verify/public-key")]
    assert len(fetches) == 1


# -- release --------------------------------------------------------------------------------------


def test_release_returns_the_account_to_free(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    with portal.sessions() as db:
        licences.release(db, portal.settings, user_of(portal, db))
    row = stored(portal)
    assert row.status == "released"
    with portal.sessions() as db:
        assert user_of(portal, db).plan_override is None
        assert len(audits(db, "user.licence_release")) == 1
        changes = [c.detail for c in audits(db, "user.plan_change")]
        assert changes[-1] == {"from": "paid", "to": None}
    deactivations = [call for call in portal.calls if call.url.endswith("/api/verify/deactivate")]
    assert deactivations[0].payload == {
        "device_fingerprint": f"web:{portal.user_id}",
        "app_slug": APP_SLUG,
    }
    assert deactivations[0].token == row.token


def test_release_still_happens_when_the_service_is_down(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    portal.down = True
    with portal.sessions() as db:
        licences.release(db, portal.settings, user_of(portal, db))
    assert stored(portal).status == "released"
    with portal.sessions() as db:
        assert user_of(portal, db).plan_override is None
        assert len(audits(db, "user.licence_release")) == 1


# -- the daily re-check ---------------------------------------------------------------------------


def test_recheck_waits_a_day_and_renews_the_token(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    first_token = stored(portal).token
    with portal.sessions() as db:
        assert licences.recheck(db, portal.settings, now=utcnow()) == 0  # checked at activation

    portal.token_exp = utcnow().replace(microsecond=0) + timedelta(hours=3)
    later = utcnow() + timedelta(hours=25)
    with portal.sessions() as db:
        assert licences.recheck(db, portal.settings, now=later) == 1
    row = stored(portal)
    assert row.token != first_token
    assert row.grace_until == row_grace(portal)
    assert row.last_checked_at == later
    assert row.status == "active"
    checks = [call for call in portal.calls if call.url.endswith("/api/verify/check")]
    assert checks[0].payload == {"app_slug": APP_SLUG}
    assert checks[0].token == first_token


def test_a_refusal_inside_the_grace_keeps_the_plan(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    portal.check_valid = False
    inside = utcnow() + timedelta(days=1)
    with portal.sessions() as db:
        assert licences.recheck(db, portal.settings, now=inside) == 1
    row = stored(portal)
    assert row.status == "active"
    assert row.last_checked_at == inside  # the service gave a real answer
    with portal.sessions() as db:
        assert user_of(portal, db).plan_override == "paid"


def test_an_unreachable_service_inside_the_grace_keeps_the_plan(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    checked_at = stored(portal).last_checked_at
    portal.down = True
    inside = utcnow() + timedelta(days=1)
    with portal.sessions() as db:
        assert licences.recheck(db, portal.settings, now=inside) == 1
    row = stored(portal)
    assert row.status == "active"
    assert row.last_checked_at == checked_at  # no answer, so no answer is recorded
    with portal.sessions() as db:
        assert user_of(portal, db).plan_override == "paid"


def test_a_lapsed_licence_returns_the_account_to_free(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    portal.check_valid = False
    after = stored(portal).grace_until + timedelta(seconds=1)
    assert after is not None
    with portal.sessions() as db:
        assert licences.recheck(db, portal.settings, now=after) == 1
    row = stored(portal)
    assert row.status == "expired"
    assert row.last_checked_at == after
    with portal.sessions() as db:
        assert user_of(portal, db).plan_override is None
        assert [c.detail for c in audits(db, "user.plan_change")][-1] == {
            "from": "paid",
            "to": None,
        }
        expired = audits(db, "user.licence_expired")
        assert len(expired) == 1
        assert expired[0].detail["licence_id"] == portal.licence_id


def test_an_unreachable_service_after_the_grace_expires_the_licence(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    portal.down = True
    after = stored(portal).grace_until + timedelta(seconds=1)
    assert after is not None
    with portal.sessions() as db:
        assert licences.recheck(db, portal.settings, now=after) == 1
    assert stored(portal).status == "expired"
    with portal.sessions() as db:
        assert user_of(portal, db).plan_override is None


def test_recheck_without_a_renewed_token_keeps_the_stored_dates(portal: Portal) -> None:
    with portal.sessions() as db:
        licences.activate(db, portal.settings, user_of(portal, db), KEY)
    before = stored(portal)
    portal.renew = False
    later = utcnow() + timedelta(hours=25)
    with portal.sessions() as db:
        assert licences.recheck(db, portal.settings, now=later) == 1
    row = stored(portal)
    assert row.token == before.token
    assert row.grace_until == before.grace_until
    assert row.last_checked_at == later


# -- billing off ----------------------------------------------------------------------------------


def test_every_licence_function_refuses_while_billing_is_off(portal: Portal) -> None:
    off = portal.settings.model_copy(
        update={"licence_url": "", "licence_app_id": "", "billing_enabled": False}
    )
    with portal.sessions() as db:
        user = user_of(portal, db)
        for call in (
            lambda: licences.activate(db, off, user, KEY),
            lambda: licences.release(db, off, user),
            lambda: licences.status(db, user, off),
            lambda: licences.recheck(db, off),
        ):
            with pytest.raises(LicenceError) as failure:
                call()
            assert failure.value.code == "BILLING_OFF"
        assert db.scalar(select(Entitlement)) is None


def test_a_licence_status_read_needs_billing_on(portal: Portal) -> None:
    with portal.sessions() as db:
        user = user_of(portal, db)
        assert licences.status(db, user, portal.settings) is None
        licences.activate(db, portal.settings, user, KEY)
        assert licences.status(db, user, portal.settings) is not None


# -- the table ------------------------------------------------------------------------------------


def test_migration_0016_upgrades_and_downgrades(tmp_path: Path) -> None:
    settings = Settings(env="test", database_url=f"sqlite:///{tmp_path}/m.db", data_dir=tmp_path)
    app = create_app(settings)
    engine = app.state.ctx.engine
    upgrade_to_head(engine)
    with engine.connect() as connection:
        inspector = inspect(connection)
        assert "entitlements" in inspector.get_table_names()
        columns = {column["name"]: column for column in inspector.get_columns("entitlements")}
        assert columns["token"]["nullable"] is False
        assert columns["expires_at"]["nullable"] is True
        unique = inspector.get_unique_constraints("entitlements")
        foreign = inspector.get_foreign_keys("entitlements")
    assert [rule["name"] for rule in unique] == ["uq_entitlements_user_id"]
    assert foreign[0]["referred_table"] == "users"
    assert foreign[0]["options"]["ondelete"] == "CASCADE"

    command.downgrade(_config(engine), "0015")
    with engine.connect() as connection:
        assert "entitlements" not in inspect(connection).get_table_names()
    upgrade_to_head(engine)
    with engine.connect() as connection:
        assert "entitlements" in inspect(connection).get_table_names()


def _entitlement(user_id: uuid.UUID, licence_id: str) -> Entitlement:
    return Entitlement(
        user_id=user_id, licence_id=licence_id, plan_code="pro", status="active", token="t"
    )


def test_an_account_holds_one_entitlement_and_deleting_it_takes_the_row(portal: Portal) -> None:
    with portal.sessions() as db:
        user = user_of(portal, db)
        db.add(_entitlement(user.id, "a"))
        db.add(_entitlement(user.id, "b"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    with portal.sessions() as db:
        db.add(_entitlement(portal.user_id, "a"))
        db.commit()
    with portal.sessions() as db:
        db.delete(user_of(portal, db))
        db.commit()
    with portal.sessions() as db:
        assert db.scalar(select(Entitlement)) is None
