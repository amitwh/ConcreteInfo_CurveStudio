"""The real client address behind reverse proxies, without trusting what a client can forge.

Each trusted proxy appends the address it received the request from to ``X-Forwarded-For``.
With ``hops`` trusted proxies in front, the client is therefore the ``hops``-th entry counted
from the right; everything left of it is supplied by the client and ignored.
"""

import ipaddress
from collections.abc import Iterable


def _address(text: str) -> str | None:
    candidate = text.strip().strip("[]")
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def client_address(forwarded_for: Iterable[str], peer: str | None, hops: int) -> str:
    """The client address to rate-limit on.

    ``forwarded_for`` holds the values of every ``X-Forwarded-For`` header, ``peer`` is the
    socket peer. With ``hops == 0`` (no proxy) only the peer counts. Otherwise the ``hops``-th
    address from the right is used; a header that is missing, too short or not an IP address
    falls back to the peer, so a malformed header never lets a client choose its own key.
    """
    fallback = peer or "unknown"
    if hops <= 0:
        return fallback
    entries = [part for value in forwarded_for for part in value.split(",") if part.strip()]
    if len(entries) < hops:
        return fallback
    return _address(entries[-hops]) or fallback
