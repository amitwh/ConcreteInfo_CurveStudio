"""The licence routes under /me, the housekeeping re-check and the admin console's view of it.

The portal is a fake over a real generated RSA key pair, as in ``test_api_licences.py``; these
tests go through HTTP: the routes' error envelopes, the plan an activation lands the account
on, the audit rows a replacement activation writes, and what an administrator sees of the
entitlement.
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import make_settings, new_email, signup
from curvestudio_api import licences
from curvestudio_api.app import create_app
from curvestudio_api.db import make_engine, make_session_factory
from curvestudio_api.export_openapi import DEFAULT_PATH, generate, render
from curvestudio_api.housekeeping import Housekeeper
from curvestudio_api.migrations import upgrade_to_head
from curvestudio_api.modules.admin.requeststats import RequestStats
from curvestudio_api.orm import AuditLog, Plan
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import utcnow

API = "/api/v1"
LICENCE = f"{API}/me/licence"
KEY = "CURVESTUDIO-ROUTES-KEY-1"
LICENCE_URL = "https://licence.example.test"
APP_SLUG = "curvestudio"
PORTAL_URL = "https://licence.example.test/pricing"


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


def _licence_settings(directory: Path, public_pem: str, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "licence_url": LICENCE_URL,
        "licence_app_id": APP_SLUG,
        "licence_plan_map": '{"pro": "paid"}',
        "licence_public_key_pem": public_pem,
        "licence_portal_url": PORTAL_URL,
    }
    values.update(overrides)
    return make_settings(directory, **values)


@dataclass
class Portal:
    """The licence service's answers: an activation, a release and nothing else."""

    private_pem: str
    licence_id: str
    valid: bool = True
    message: str = "Key not found or invalid"
    features: list[str] = field(default_factory=lambda: ["pro", "export"])
    down: bool = False
    device: str | None = None
    token_exp: datetime = field(default_factory=lambda: utcnow() + timedelta(hours=1))
    expires_at: datetime = field(default_factory=lambda: utcnow() + timedelta(days=30))
    deactivations: list[tuple[dict[str, Any] | None, str | None]] = field(default_factory=list)

    def token(self) -> str:
        assert self.device is not None  # set by the activation call
        claims: dict[str, Any] = {
            "license_id": self.licence_id,
            "app_slug": APP_SLUG,
            "device_fingerprint": self.device,
            "features": list(self.features),
            "expires_at": self.expires_at.isoformat(),
            "exp": int(self.token_exp.timestamp()),
            "iat": int(utcnow().timestamp()),
        }
        return jwt.encode(claims, self.private_pem, algorithm="RS256")

    def __call__(
        self,
        url: str,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        if self.down:
            raise licences.PortalUnreachable("could not reach the licence service")
        if url.endswith("/api/verify/activate"):
            if payload is None:
                raise AssertionError("activation without a payload")
            self.device = payload.get("device_fingerprint")
            if not self.valid or payload.get("key") != KEY:
                return {
                    "valid": False,
                    "license_jwt": None,
                    "message": self.message,
                    "features": [],
                }
            return {
                "valid": True,
                "license_jwt": self.token(),
                "message": None,
                "features": list(self.features),
            }
        if url.endswith("/api/verify/deactivate"):
            self.deactivations.append((payload, token))
            return {
                "valid": True,
                "license_jwt": None,
                "message": "Device deactivated",
                "features": [],
            }
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def licence_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, Portal]]:
    private_pem, public_pem = _key_pair()
    app = create_app(_licence_settings(tmp_path, public_pem))
    upgrade_to_head(app.state.ctx.engine)
    with app.state.ctx.session_factory() as db:
        db.add(Plan(name="paid", limits={"max_datasets": 10, "max_dataset_bytes": 1000}))
        db.commit()
    portal = Portal(private_pem=private_pem, licence_id=str(uuid.uuid4()))
    monkeypatch.setattr(licences, "_http_json", portal)
    licences._public_keys.clear()
    with TestClient(app) as client:
        yield client, portal
    licences._public_keys.clear()


def audits(client: TestClient, action: str) -> list[AuditLog]:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        return list(db.scalars(select(AuditLog).where(AuditLog.action == action)))


# -- the routes ------------------------------------------------------------------------------------


def test_the_three_routes_need_a_signed_in_account(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, _portal = licence_client
    assert client.get(LICENCE).status_code == 401
    assert client.post(LICENCE, json={"key": KEY}).status_code == 401
    assert client.delete(LICENCE).status_code == 401


def test_billing_off_answers_404_billing_off_and_me_reports_no_licence(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory())  # the connector is not configured
    headers = signup(client)
    for response in (
        client.get(LICENCE, headers=headers),
        client.post(LICENCE, headers=headers, json={"key": KEY}),
        client.delete(LICENCE, headers=headers),
    ):
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "NOT_FOUND"
        assert body["detail"]["code"] == "BILLING_OFF"
    me = client.get(f"{API}/me", headers=headers).json()
    assert me["licence"] is None
    assert me["plan"]["name"] == "self-hosted"


def test_activation_answers_201_and_me_shows_the_licence_and_the_mapped_plan(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, portal = licence_client
    headers = signup(client)
    response = client.post(LICENCE, headers=headers, json={"key": KEY})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["plan_code"] == "pro"
    assert body["plan"] == "paid"  # the mapped CurveStudio plan, not the portal's code
    assert body["status"] == "active"
    assert body["portal_url"] == PORTAL_URL
    assert body["message"] == ""
    assert body["expires_at"] is not None and body["expires_at"].endswith("+05:30")
    assert body["grace_until"] is not None and body["grace_until"].endswith("+05:30")

    me = client.get(f"{API}/me", headers=headers).json()
    assert me["plan"]["name"] == "paid"
    assert me["licence"]["id"] == body["id"]
    assert me["licence"]["plan"] == "paid"

    read = client.get(LICENCE, headers=headers)
    assert read.status_code == 200
    assert read.json()["id"] == body["id"]
    assert len(audits(client, "user.licence_activate")) == 1
    assert portal.deactivations == []


def test_reading_without_a_licence_answers_404_licence_none(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, _portal = licence_client
    headers = signup(client)
    response = client.get(LICENCE, headers=headers)
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "NOT_FOUND"
    assert body["detail"]["code"] == "LICENCE_NONE"


def test_a_second_activation_releases_the_first_and_says_so(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, portal = licence_client
    headers = signup(client)
    first = client.post(LICENCE, headers=headers, json={"key": KEY}).json()
    portal.licence_id = str(uuid.uuid4())  # the second key is another licence

    second = client.post(LICENCE, headers=headers, json={"key": KEY})
    assert second.status_code == 201, second.text
    body = second.json()
    assert body["id"] == first["id"]  # one entitlement row, reused
    assert "released" in body["message"]

    assert len(portal.deactivations) == 1  # the seat went back to the portal
    payload, token = portal.deactivations[0]
    assert payload is not None and payload["device_fingerprint"].startswith("web:")
    assert token is not None
    assert len(audits(client, "user.licence_activate")) == 2
    assert len(audits(client, "user.licence_release")) == 1
    me = client.get(f"{API}/me", headers=headers).json()
    assert me["plan"]["name"] == "paid"  # the new licence put it back


def test_delete_answers_204_and_returns_the_account_to_free(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, _portal = licence_client
    headers = signup(client)
    assert client.post(LICENCE, headers=headers, json={"key": KEY}).status_code == 201

    response = client.delete(LICENCE, headers=headers)
    assert response.status_code == 204

    me = client.get(f"{API}/me", headers=headers).json()
    assert me["plan"]["name"] == "free"
    assert me["licence"] is not None and me["licence"]["status"] == "released"
    read = client.get(LICENCE, headers=headers)
    assert read.status_code == 200 and read.json()["status"] == "released"
    assert len(audits(client, "user.licence_release")) == 1


def test_an_invalid_key_is_a_422_with_the_services_message(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, portal = licence_client
    headers = signup(client)
    portal.message = "that key has been revoked"
    response = client.post(LICENCE, headers=headers, json={"key": "WRONG-KEY"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["detail"]["code"] == "LICENCE_INVALID"
    assert body["message"] == "that key has been revoked"
    me = client.get(f"{API}/me", headers=headers).json()
    assert me["plan"]["name"] == "free" and me["licence"] is None


def test_an_unknown_plan_code_is_a_422_naming_the_code(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, portal = licence_client
    headers = signup(client)
    portal.features = ["gold", "export"]
    response = client.post(LICENCE, headers=headers, json={"key": KEY})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["detail"]["code"] == "LICENCE_UNKNOWN_PLAN"
    assert "gold" in body["message"]


def test_an_unreachable_service_is_a_503(
    licence_client: tuple[TestClient, Portal],
) -> None:
    client, portal = licence_client
    headers = signup(client)
    portal.down = True
    response = client.post(LICENCE, headers=headers, json={"key": KEY})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "SERVICE_UNAVAILABLE"
    assert body["detail"]["code"] == "LICENCE_UNREACHABLE"


# -- the admin console ----------------------------------------------------------------------------


def test_the_admin_payload_carries_the_entitlement(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_pem, public_pem = _key_pair()
    owner = new_email()
    client = client_factory(
        settings_factory(
            admin_emails=[owner],
            licence_url=LICENCE_URL,
            licence_app_id=APP_SLUG,
            licence_plan_map='{"pro": "paid"}',
            licence_public_key_pem=public_pem,
            licence_portal_url=PORTAL_URL,
        )
    )
    admin = signup(client, owner)
    member = signup(client)
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(Plan(name="paid", limits={"max_datasets": 10}))
        db.commit()
    monkeypatch.setattr(licences, "_http_json", Portal(private_pem, str(uuid.uuid4())))
    licences._public_keys.clear()
    assert client.post(LICENCE, headers=member, json={"key": KEY}).status_code == 201
    licences._public_keys.clear()

    listed = client.get(f"{API}/admin/users", headers=admin).json()["items"]
    licenced = next(item for item in listed if item["plan"] == "paid")
    assert licenced["plan_override"] == "paid"
    assert licenced["licence"] is not None
    assert licenced["licence"]["status"] == "active"
    assert licenced["licence"]["plan_code"] == "pro"
    assert licenced["licence"]["expires_at"] is not None
    assert licenced["licence"]["last_checked_at"] is not None
    detail = client.get(f"{API}/admin/users/{licenced['id']}", headers=admin).json()
    assert detail["user"]["licence"] == licenced["licence"]
    plain = next(item for item in listed if item["plan"] != "paid")
    assert plain["licence"] is None


# -- the housekeeper ------------------------------------------------------------------------------


def _housekeeper(settings: Settings, recheck: Any) -> Housekeeper:
    engine = make_engine(settings)
    upgrade_to_head(engine)
    return Housekeeper(
        make_session_factory(engine),
        1.0,
        settings=settings,
        stats=RequestStats(),
        recheck=recheck,
    )


def test_the_housekeeper_calls_recheck_when_due(tmp_path: Path) -> None:
    calls: list[tuple[Session, Settings]] = []

    def spy(db: Session, settings: Settings, now: Any = None) -> int:
        calls.append((db, settings))
        return 0

    settings = _licence_settings(tmp_path, "PEM")  # billing is on
    keeper = _housekeeper(settings, spy)
    keeper.run_once()
    assert len(calls) == 1
    assert calls[0][1] is settings  # the same settings the app was built with

    keeper.run_once()
    assert len(calls) == 1  # the hour has not passed

    keeper._last_recheck = time.monotonic() - 3601.0  # type: ignore[attr-defined]
    keeper.run_once()
    assert len(calls) == 2


def test_the_housekeeper_does_not_recheck_while_billing_is_off(tmp_path: Path) -> None:
    calls: list[object] = []

    def spy(db: Session, settings: Settings, now: Any = None) -> int:
        calls.append(db)
        return 0

    keeper = _housekeeper(make_settings(tmp_path), spy)
    keeper.run_once()
    keeper._last_recheck = None  # type: ignore[attr-defined]
    keeper.run_once()
    assert calls == []


def test_the_housekeeper_survives_a_recheck_that_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(db: Session, settings: Settings, now: Any = None) -> int:
        raise RuntimeError("the portal fell over")

    keeper = _housekeeper(_licence_settings(tmp_path, "PEM"), boom)
    with caplog.at_level(logging.ERROR, logger="curvestudio_api"):
        result = keeper.run_once()  # must not raise
    assert result.refresh_tokens == 0
    assert any("licence re-check failed" in row.getMessage() for row in caplog.records)
    keeper.run_once()  # and it keeps running
    keeper._last_recheck = time.monotonic() - 3601.0  # type: ignore[attr-defined]
    keeper.run_once()


# -- the OpenAPI document -------------------------------------------------------------------------


def test_the_committed_openapi_covers_the_licence_routes_and_is_fresh() -> None:
    schema = generate()
    assert set(schema["paths"][f"{API}/me/licence"]) == {"get", "post", "delete"}
    assert DEFAULT_PATH.read_text(encoding="utf-8") == render(schema), (
        "apps/api/openapi.json is stale: run `uv run python -m curvestudio_api.export_openapi`"
    )
