# CurveStudio web app image: one container, one uvicorn process.
#
# Build from the repository root (the whole repo is the context; see .dockerignore):
#   docker build -f deploy/Dockerfile -t curvestudio .
#
# Stage 1 builds the React SPA (apps/web) with a pinned Node LTS. Stage 2 installs the uv
# workspace (core + model catalogues + API) into a virtual environment with pinned uv, copies
# the SPA in, and runs the API as a non-root user. The API serves the SPA itself
# (CURVESTUDIO_WEB_DIST), so there is no nginx and no second process: jobs and the rate
# limiter live inside the single uvicorn process (see apps/api/README.md).

# --- stage 1: the SPA ---------------------------------------------------------------------------
FROM node:24.22.3-alpine AS web

WORKDIR /build
# Dependencies first so source edits do not invalidate the npm ci layer.
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

# --- stage 2: the API + SPA runtime --------------------------------------------------------------
FROM python:3.14-slim

ENV TZ=Asia/Kolkata \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CURVESTUDIO_HOST=0.0.0.0 \
    CURVESTUDIO_PORT=8000 \
    CURVESTUDIO_DATA_DIR=/data \
    CURVESTUDIO_WEB_DIST=/app/web

# tzdata so TZ=Asia/Kolkata resolves; curl because Coolify's health check shells out to curl/wget.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

WORKDIR /app

# Manifests first: this layer (all external wheels) only changes when the lockfile does.
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/README.md packages/core/
COPY packages/model_library/pyproject.toml packages/model_library/README.md packages/model_library/
COPY packages/engineering/pyproject.toml packages/engineering/README.md packages/engineering/
COPY apps/api/pyproject.toml apps/api/README.md apps/api/
RUN uv sync --frozen --no-dev --no-install-workspace

# Sources: rebuilds only the workspace members (curvestudio-core, -models, -engineering, -api)
# as wheels. The catalogue packages MUST be installed: the model library reaches the API
# through their curvestudio.models entry points, which is what makes /api/v1/models list it.
COPY packages/ packages/
COPY apps/api/ apps/api/
RUN uv sync --frozen --no-dev --no-editable \
 && uv cache clean \
 && rm -f /usr/local/bin/uv /usr/local/bin/uvx \
 && groupadd --gid 10001 curvestudio \
 && useradd --uid 10001 --gid 10001 --home-dir /app --shell /usr/sbin/nologin curvestudio \
 && mkdir /data && chown curvestudio:curvestudio /data

COPY --from=web /build/dist /app/web

USER curvestudio
VOLUME /data
EXPOSE 8000

# /api/v1/health really queries the database, so this is a readiness probe, not just liveness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["/app/.venv/bin/python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"]

# X-Forwarded-For is parsed by the app itself (CURVESTUDIO_TRUSTED_PROXY_HOPS); uvicorn's own
# proxy-header handling is switched off in curvestudio_api.main.run.
CMD ["/app/.venv/bin/curvestudio-api"]
