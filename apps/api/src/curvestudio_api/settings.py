"""Settings read from the environment (prefix ``CURVESTUDIO_``); secrets never have defaults."""

from pathlib import Path
from typing import Annotated, Literal, Self

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PRIVATE_KEY_FILE = "jwt_private.pem"
PUBLIC_KEY_FILE = "jwt_public.pem"
MIN_RSA_BITS = 2048


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
    job_queue_max: int = Field(default=500, ge=1)
    job_max_result_bytes: int = Field(default=250_000_000, ge=1_000)
    autofit_timeout_s: float = Field(default=180.0, gt=0)
    autofit_budget_s: float = Field(default=120.0, gt=0)
    autofit_workers: int = Field(default=3, ge=1)
    import_timeout_s: float = Field(default=20.0, gt=0)
    import_concurrency: int = Field(default=2, ge=1)
    import_memory_mb: int = Field(default=1024, ge=64)
    trusted_proxy_hops: int = Field(default=1, ge=0)
    quota_max_datasets: int | None = Field(default=None, ge=1)
    quota_max_dataset_bytes: int | None = Field(default=None, ge=1)
    quota_max_fits: int | None = Field(default=None, ge=1)
    quota_max_autofits_per_day: int | None = Field(default=None, ge=1)
    housekeeping_interval_s: float = Field(default=3600.0, gt=0)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Comma-separated addresses whose accounts are promoted to administrator at their next
    # successful sign-in. Empty by default; the list never demotes.
    admin_emails: Annotated[list[str], NoDecode] = Field(default_factory=list)
    registration_enabled: bool = True
    billing_enabled: bool = False
    web_dist: Path | None = None
    # Public address of the web app, used in the links of transactional email.
    app_url: str = "http://localhost:5173"
    require_email_verification: bool = True
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = ""
    smtp_password: SecretStr | None = None
    smtp_from: str = ""
    email_from_name: str = "CurveStudio"
    verification_ttl_hours: int = Field(default=24, ge=1)
    reset_ttl_minutes: int = Field(default=60, ge=5)
    # Social sign-in. A provider is enabled only when both its id and secret are set.
    oauth_google_client_id: str = ""
    oauth_google_client_secret: SecretStr | None = None
    oauth_microsoft_client_id: str = ""
    oauth_microsoft_client_secret: SecretStr | None = None
    oauth_microsoft_tenant: str = "common"
    oauth_linkedin_client_id: str = ""
    oauth_linkedin_client_secret: SecretStr | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("admin_emails", mode="before")
    @classmethod
    def _split_admin_emails(cls, value: object) -> object:
        """``CURVESTUDIO_ADMIN_EMAILS=a@b.co, c@d.co``."""
        return value.split(",") if isinstance(value, str) else value

    @field_validator("admin_emails")
    @classmethod
    def _fold_admin_emails(cls, value: list[str]) -> list[str]:
        """Entries are addresses, matched in lower case however the list was built."""
        return [part.strip().lower() for part in value if part.strip()]

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

    @field_validator("jwt_private_key_pem", "jwt_public_key_pem", mode="before")
    @classmethod
    def _empty_key_is_unset(cls, value: object) -> object:
        """``CURVESTUDIO_JWT_..._PEM=`` (as in ``.env.example``) means not set."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "quota_max_datasets",
        "quota_max_dataset_bytes",
        "quota_max_fits",
        "quota_max_autofits_per_day",
        mode="before",
    )
    @classmethod
    def _empty_quota_is_plan_default(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("web_dist", mode="before")
    @classmethod
    def _empty_dist_is_off(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("smtp_password", mode="before")
    @classmethod
    def _empty_smtp_password_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator(
        "oauth_google_client_secret",
        "oauth_microsoft_client_secret",
        "oauth_linkedin_client_secret",
        mode="before",
    )
    @classmethod
    def _empty_oauth_secret_is_unset(cls, value: object) -> object:
        """``CURVESTUDIO_OAUTH_..._CLIENT_SECRET=`` (as in ``.env.example``) means not set."""
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _autofit_budget_within_timeout(self) -> Self:
        if self.autofit_budget_s >= self.autofit_timeout_s:
            raise ValueError(
                "CURVESTUDIO_AUTOFIT_BUDGET_S (when a run is stopped and its partial result "
                "kept) must be below CURVESTUDIO_AUTOFIT_TIMEOUT_S (when the job is killed)"
            )
        return self

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
    if private.key_size < MIN_RSA_BITS or public.key_size < MIN_RSA_BITS:
        raise ValueError(f"the JWT keys must be RSA keys of at least {MIN_RSA_BITS} bits")
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
