"""Password hashing (Argon2id), RS256 access tokens and opaque refresh tokens."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from curvestudio_api.errors import ApiError
from curvestudio_api.timeutil import utcnow

ALGORITHM = "RS256"
ACCESS_TYPE = "access"

_hasher = PasswordHasher()  # argon2-cffi defaults: Argon2id with the RFC 9106 low-memory profile
_DUMMY_HASH = _hasher.hash("curvestudio-timing-equaliser")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored_hash: str | None, password: str) -> bool:
    """Whether ``password`` matches. With no stored hash a dummy one is checked, so an unknown
    account takes as long as a wrong password."""
    try:
        return _hasher.verify(stored_hash or _DUMMY_HASH, password) and stored_hash is not None
    except VerificationError, InvalidHashError:
        return False


def password_needs_rehash(stored_hash: str) -> bool:
    return _hasher.check_needs_rehash(stored_hash)


def hash_token(token: str) -> str:
    """The SHA-256 hex digest under which a refresh token is stored (never the token itself)."""
    return hashlib.sha256(token.encode()).hexdigest()


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


@dataclass(frozen=True)
class JwtKeys:
    private_pem: str
    public_pem: str


def issue_access_token(keys: JwtKeys, user_id: uuid.UUID, ttl: timedelta) -> str:
    now = utcnow()
    claims = {
        "sub": str(user_id),
        "typ": ACCESS_TYPE,
        "iat": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, keys.private_pem, algorithm=ALGORITHM)


def unauthorized(message: str = "authentication required") -> ApiError:
    return ApiError(401, "UNAUTHORIZED", message, headers={"WWW-Authenticate": "Bearer"})


def verify_access_token(keys: JwtKeys, token: str) -> uuid.UUID:
    """The user id in a valid access token.

    Raises:
        ApiError: 401 for a bad signature, wrong algorithm, expiry, wrong ``typ`` or ``sub``.
    """
    try:
        claims = jwt.decode(
            token,
            keys.public_pem,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "iat", "sub", "typ"]},
        )
        if claims["typ"] != ACCESS_TYPE:
            raise jwt.InvalidTokenError("not an access token")
        return uuid.UUID(str(claims["sub"]))
    except (jwt.PyJWTError, ValueError) as exc:
        raise unauthorized("invalid or expired access token") from exc
