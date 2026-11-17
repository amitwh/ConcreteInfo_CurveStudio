"""Settings read from the environment (prefix ``CURVESTUDIO_``); secrets never have defaults."""

from pathlib import Path
from typing import Annotated, Literal, Self

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PRIVATE_KEY_FILE = "jwt_private.pem"
PUBLIC_KEY_FILE = "jwt_public.pem"


class Settings(BaseSettings):
    """Every runtime knob of the API. See ``.env.example`` for the full list."""

    model_config = SettingsConfigDict(
        env_prefix="CURVESTUDIO_", env_file=None, extra="ignore", frozen=True
    )

    database_url: str = "sqlite:///./data/curvestudio.db"
    data_dir: Path = Path("./data")
    env: Literal["dev", "test", "prod"] = "dev"
    jwt_private_key_pem: SecretStr | None = None
    jwt_public_key_pem: SecretStr | None = None
    access_ttl_min: int = Field(default=15, ge=1)
    refresh_ttl_days: int = Field(default=14, ge=1)
    max_upload_bytes: int = Field(default=20_000_000, ge=1)
    max_rows: int = Field(default=1_000_000, ge=1)
    job_timeout_s: float = Field(default=120.0, gt=0)
    job_workers: int = Field(default=2, ge=1)
    job_grace_s: float = Field(default=10.0, ge=0)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    registration_enabled: bool = True
    billing_enabled: bool = False
    web_dist: Path | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("cors_origins")
    @classmethod
    def _explicit_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if "*" in origin:
                raise ValueError(
                    "CORS origins must be listed explicitly; wildcards are not allowed"
                )
        return value

    @field_validator("database_url")
    @classmethod
    def _driver(cls, value: str) -> str:
        """``postgresql://`` means the psycopg 3 driver this package ships."""
        for prefix in ("postgresql://", "postgres://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix) :]
        return value

    @field_validator("web_dist", mode="before")
    @classmethod
    def _empty_dist_is_off(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _keys(self) -> Self:
        given = (self.jwt_private_key_pem is not None, self.jwt_public_key_pem is not None)
        if self.env == "prod" and not all(given):
            raise ValueError(
                "CURVESTUDIO_JWT_PRIVATE_KEY_PEM and CURVESTUDIO_JWT_PUBLIC_KEY_PEM are required "
                "when CURVESTUDIO_ENV=prod"
            )
        if any(given) and not all(given):
            raise ValueError(
                "set both CURVESTUDIO_JWT_PRIVATE_KEY_PEM and CURVESTUDIO_JWT_PUBLIC_KEY_PEM, "
                "or neither"
            )
        return self


def _normalise_pem(pem: str) -> str:
    """Accept PEMs pasted into an env var with literal ``\\n`` sequences."""
    return pem.replace("\\n", "\n").strip() + "\n"


def _check_pair(private_pem: str, public_pem: str) -> None:
    try:
        private = serialization.load_pem_private_key(private_pem.encode(), password=None)
        public = serialization.load_pem_public_key(public_pem.encode())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"the JWT key PEMs are not valid RSA keys: {exc}") from exc
    if not isinstance(private, rsa.RSAPrivateKey) or not isinstance(public, rsa.RSAPublicKey):
        raise ValueError("the JWT keys must be RSA keys (RS256)")
    if private.public_key().public_numbers() != public.public_numbers():
        raise ValueError("the JWT public key does not belong to the JWT private key")


def load_jwt_keys(settings: Settings) -> tuple[str, str]:
    """The RS256 ``(private_pem, public_pem)`` pair.

    From the environment when set. Otherwise (dev and test only; production settings cannot
    be built without them) a pair is generated into ``data_dir`` once and reused.
    """
    if settings.jwt_private_key_pem is not None and settings.jwt_public_key_pem is not None:
        private_pem = _normalise_pem(settings.jwt_private_key_pem.get_secret_value())
        public_pem = _normalise_pem(settings.jwt_public_key_pem.get_secret_value())
        _check_pair(private_pem, public_pem)
        return private_pem, public_pem
    private_path = settings.data_dir / PRIVATE_KEY_FILE
    public_path = settings.data_dir / PUBLIC_KEY_FILE
    if private_path.exists() and public_path.exists():
        private_pem = private_path.read_text(encoding="ascii")
        public_pem = public_path.read_text(encoding="ascii")
        _check_pair(private_pem, public_pem)
        return private_pem, public_pem
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
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    private_path.touch(mode=0o600, exist_ok=True)
    private_path.write_text(private_pem, encoding="ascii")
    public_path.write_text(public_pem, encoding="ascii")
    return private_pem, public_pem
