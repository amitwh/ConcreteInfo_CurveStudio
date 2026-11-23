"""Static checks on the self-hosting files in ``deploy/``.

These files change rarely, but a typo in an environment variable name or a broken compose
file would otherwise only surface at deploy time, on someone else's server. The full
behavioural check is ``deploy/smoke.sh`` against a running container (CI runs it against a
freshly built image); what we can assert cheaply here is shape.
"""

import sys
from pathlib import Path

import pytest
import yaml

from curvestudio_api.settings import Settings

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"

# Variables consumed by docker-compose.yml itself (not by the API's Settings).
COMPOSE_ONLY_VARS = frozenset(
    {"CURVESTUDIO_DOMAIN", "CURVESTUDIO_DB_USER", "CURVESTUDIO_DB_PASSWORD", "CURVESTUDIO_DB_NAME"}
)

# Process-level variables read straight from the environment by curvestudio_api.main.run,
# not through Settings.
PROCESS_ONLY_VARS = frozenset({"CURVESTUDIO_HOST", "CURVESTUDIO_PORT"})

SCRIPTS = ("gen-keys.sh", "backup.sh", "restore.sh", "smoke.sh")


def test_env_example_covers_real_settings_only() -> None:
    """Every CURVESTUDIO_* name in deploy/.env.example is a real setting (or compose-only)."""
    text = (DEPLOY / ".env.example").read_text(encoding="utf-8")
    names = [line.split("=", 1)[0] for line in text.splitlines() if line.startswith("CURVESTUDIO_")]
    assert names, "deploy/.env.example defines no CURVESTUDIO_* variables"
    unknown = [
        name
        for name in names
        if name not in COMPOSE_ONLY_VARS
        and name not in PROCESS_ONLY_VARS
        and name.removeprefix("CURVESTUDIO_").lower() not in Settings.model_fields
    ]
    assert unknown == [], f"deploy/.env.example names unknown settings: {unknown}"
    # every API setting is documented somewhere: deploy/.env.example or apps/api/.env.example
    api_example = (REPO / "apps/api/.env.example").read_text(encoding="utf-8")
    api_names = [
        line.split("=", 1)[0]
        for line in api_example.splitlines()
        if line.startswith("CURVESTUDIO_")
    ]
    documented = set(names) | set(api_names)
    missing = {f"CURVESTUDIO_{field.upper()}" for field in Settings.model_fields} - documented
    assert not missing, f"settings absent from both .env.example files: {sorted(missing)}"


def test_env_example_has_no_secrets() -> None:
    text = (DEPLOY / ".env.example").read_text(encoding="utf-8")
    assert "BEGIN PRIVATE KEY" not in text and "BEGIN PUBLIC KEY" not in text
    for line in text.splitlines():
        if line.startswith(("CURVESTUDIO_DB_PASSWORD=", "CURVESTUDIO_DATABASE_URL=")):
            assert "CHANGE" in line or "example" in line, f"not a placeholder: {line}"


def test_compose_stack_shape() -> None:
    doc = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text(encoding="utf-8"))
    services = doc["services"]
    assert set(services) == {"app", "db", "caddy"}
    app = services["app"]
    assert app["build"]["context"] == ".."
    assert app["build"]["dockerfile"] == "deploy/Dockerfile"
    assert app["restart"] == "unless-stopped"
    assert "curvestudio-data:/data" in app["volumes"]
    assert app["env_file"] == ".env"
    assert "healthcheck" in app, "the app service needs a healthcheck for dependent services"
    assert app["depends_on"]["db"]["condition"] == "service_healthy"

    db = services["db"]
    assert db["image"] == "postgres:17-alpine"
    assert "POSTGRES_PASSWORD" in db["environment"]
    assert "healthcheck" in db
    assert any("postgres" in v for v in db["volumes"])

    caddy = services["caddy"]
    assert caddy["image"] == "caddy:2"
    assert "80:80" in caddy["ports"] and "443:443" in caddy["ports"]
    assert "./Caddyfile:/etc/caddy/Caddyfile:ro" in caddy["volumes"]


def test_dockerfile_shape() -> None:
    text = (DEPLOY / "Dockerfile").read_text(encoding="utf-8")
    stages = [line.split()[1] for line in text.splitlines() if line.startswith("FROM ")]
    assert stages[0].startswith("node:24"), f"web builder must pin a node 24 tag: {stages}"
    assert stages[-1].startswith("python:3.14"), f"runtime must pin python 3.14: {stages}"
    for needle in (
        "npm ci",
        "npm run build",
        "uv sync --frozen --no-dev",
        "packages/model_library/pyproject.toml",  # the entry-point catalogues must be synced
        "packages/engineering/pyproject.toml",
        "TZ=Asia/Kolkata",
        "PYTHONUNBUFFERED=1",
        "10001",  # non-root uid
        "EXPOSE 8000",
        "/api/v1/health",
        "--start-period=40s",
        'CMD ["/app/.venv/bin/curvestudio-api"]',
    ):
        assert needle in text, f"Dockerfile is missing: {needle}"
    assert "USER curvestudio" in text or "USER 10001" in text
    # `uv sync --package curvestudio-api` would install only the API's own dependencies and
    # silently drop the model catalogues from /api/v1/models.
    assert "--package" not in text, "the image must sync the whole workspace, not one package"


def test_dockerignore_keeps_the_context_clean_and_secret_free() -> None:
    text = (REPO / ".dockerignore").read_text(encoding="utf-8")
    for needle in (
        ".git",
        ".venv",
        "node_modules",
        "tests",
        "docs",
        "apps/web/dist",
        "deploy/.env",
    ):
        assert needle in text, f".dockerignore must exclude {needle}"


def test_caddyfile_proxy_shape() -> None:
    text = (DEPLOY / "Caddyfile").read_text(encoding="utf-8")
    assert "{$CURVESTUDIO_DOMAIN}" in text
    assert "reverse_proxy app:8000" in text
    assert "encode" in text
    for header in ("X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy"):
        assert header in text, f"Caddyfile should set {header}"


@pytest.mark.parametrize("name", SCRIPTS)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only: executable bits are not tracked on Windows checkouts",
)
def test_scripts_are_executable_and_strict(name: str) -> None:
    path = DEPLOY / name
    assert path.is_file(), f"deploy/{name} is missing"
    assert path.stat().st_mode & 0o111, f"deploy/{name} must be executable"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!"), f"deploy/{name} needs a shebang"
    assert "set -eu" in text, f"deploy/{name} must fail on errors"
    if name == "smoke.sh":
        assert "pipefail" in text


def test_readme_links_the_self_hosting_guide() -> None:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "deploy/README.md" in text, "the root README should link the self-hosting guide"


def test_caddyfile_redirects_both_old_hosts_preserving_uri() -> None:
    """The redirect service answers both retired addresses with a 301 that keeps the path."""
    caddyfile = (DEPLOY / "redirect" / "Caddyfile").read_text(encoding="utf-8")
    assert "curvestudio.concreteinfo.net" in caddyfile
    assert "curvestudio.concreteinfo.co.in" in caddyfile
    assert "redir https://curvestudio.concreteinfo.cloud{uri} permanent" in caddyfile


def test_dockerfile_serves_the_caddyfile() -> None:
    dockerfile = (DEPLOY / "redirect" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM caddy:2-alpine" in dockerfile
    assert "COPY Caddyfile /etc/caddy/Caddyfile" in dockerfile
    assert "EXPOSE 80 443" in dockerfile
