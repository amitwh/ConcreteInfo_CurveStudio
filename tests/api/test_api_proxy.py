"""Client addresses behind a reverse proxy: only what the trusted proxies appended counts."""

from collections.abc import Callable
from typing import Any

import pytest
from api_test_helpers import PASSWORD, new_email
from fastapi.testclient import TestClient

from curvestudio_api.proxy import client_address

API = "/api/v1"


@pytest.mark.parametrize(
    ("forwarded", "peer", "hops", "expected"),
    [
        ([], "10.0.0.1", 1, "10.0.0.1"),  # no header: the peer
        (["1.2.3.4"], "10.0.0.1", 1, "1.2.3.4"),
        (["evil, 1.2.3.4"], "10.0.0.1", 1, "1.2.3.4"),  # leftmost is spoofable, ignored
        (["a, b, 1.2.3.4"], "10.0.0.1", 1, "1.2.3.4"),
        (["a, 5.6.7.8, 1.2.3.4"], "10.0.0.1", 2, "5.6.7.8"),
        (["5.6.7.8", "1.2.3.4"], "10.0.0.1", 1, "1.2.3.4"),  # several header lines
        (["1.2.3.4"], "10.0.0.1", 0, "10.0.0.1"),  # no proxy: header ignored
        (["1.2.3.4"], "10.0.0.1", 2, "10.0.0.1"),  # fewer entries than proxies
        (["not-an-ip"], "10.0.0.1", 1, "10.0.0.1"),  # garbage never becomes the key
        (["[2001:db8::1]"], "10.0.0.1", 1, "2001:db8::1"),
        (["1.2.3.4"], None, 0, "unknown"),
    ],
)
def test_client_address(forwarded: list[str], peer: str | None, hops: int, expected: str) -> None:
    assert client_address(forwarded, peer, hops) == expected


def attempts(
    client: TestClient, email: str, forwarded: Callable[[int], str], count: int
) -> list[int]:
    return [
        client.post(
            f"{API}/auth/login",
            json={"email": email, "password": "wrong password!!"},
            headers={"X-Forwarded-For": forwarded(n)},
        ).status_code
        for n in range(count)
    ]


def test_rotating_a_spoofed_leftmost_address_does_not_evade_the_limit(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    statuses = attempts(client, new_email(), lambda n: f"203.0.113.{n}, 198.51.100.7", 14)
    assert statuses[:10] == [401] * 10
    assert statuses[10:] == [429] * 4


def test_two_real_clients_behind_the_proxy_are_limited_independently(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    email = new_email()
    first = attempts(client, email, lambda n: "x, 198.51.100.1", 12)
    second = attempts(client, email, lambda n: "x, 198.51.100.2", 3)
    assert 429 in first
    assert second == [401, 401, 401]


def test_without_a_proxy_the_header_is_ignored(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(trusted_proxy_hops=0))
    statuses = attempts(client, new_email(), lambda n: f"203.0.113.{n}", 12)
    assert 429 in statuses  # every request has the same socket peer


def test_two_proxies_use_the_second_address_from_the_right(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(trusted_proxy_hops=2))
    email = new_email()
    first = attempts(client, email, lambda n: f"{n}.0.0.1, 198.51.100.1, 10.0.0.9", 12)
    second = attempts(client, email, lambda n: "x, 198.51.100.2, 10.0.0.9", 2)
    assert 429 in first and second == [401, 401]


def test_refresh_is_limited_per_token_whatever_the_address(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    email = new_email()
    client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD})
    token = client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD}).json()[
        "refresh_token"
    ]
    statuses = [
        client.post(
            f"{API}/auth/refresh",
            json={"refresh_token": token},
            headers={"X-Forwarded-For": f"192.0.2.{n}"},
        ).status_code
        for n in range(13)
    ]
    assert 429 in statuses
    other = client.post(
        f"{API}/auth/refresh",
        json={"refresh_token": "another-token"},
        headers={"X-Forwarded-For": "192.0.2.99"},
    )
    assert other.status_code == 401  # a different token has its own allowance
