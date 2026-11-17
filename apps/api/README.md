# curvestudio-api

FastAPI backend for CurveStudio (AGPL-3.0-or-later). It contains no maths: every number comes
from `curvestudio.workflow` (`import_dataset`, `fit_dataset`, `predict`), so the web app and
the desktop app show identical results.

## Run

```
cp apps/api/.env.example .env      # edit; see the comments in the file
uv run curvestudio-api             # one uvicorn process on CURVESTUDIO_HOST:CURVESTUDIO_PORT
```

The schema is migrated (Alembic, `0001`) at start-up. With `CURVESTUDIO_WEB_DIST` pointing at
the built web app the same process serves it, with a fallback to `index.html` for non-`/api`
paths. The OpenAPI document is `apps/api/openapi.json`; regenerate it with
`uv run python -m curvestudio_api.export_openapi` (a test fails when it is stale).

## Deployment notes

* **One API process per database.** Jobs run in a `spawn` process pool owned by that process;
  the rate limiter is in memory; jobs left `queued`/`running` by a previous process are failed
  (`SERVER_RESTART`) at start-up. Scale up with `CURVESTUDIO_JOB_WORKERS`, not with more
  uvicorn workers.
* Behind a reverse proxy set `FORWARDED_ALLOW_IPS` to the proxy address so the auth rate limit
  sees real client addresses.
* `CURVESTUDIO_ENV=prod` refuses to start without `CURVESTUDIO_JWT_PRIVATE_KEY_PEM` and
  `CURVESTUDIO_JWT_PUBLIC_KEY_PEM` (a matching RSA pair) and adds HSTS.
* Postgres: `CURVESTUDIO_DATABASE_URL=postgresql://user:password@host/db` (psycopg 3).
* A stored fit holds per-row arrays, so its size grows with the dataset (about 130 MB of JSON
  at 500 000 rows). Lower `CURVESTUDIO_MAX_ROWS` for hosted deployments.
