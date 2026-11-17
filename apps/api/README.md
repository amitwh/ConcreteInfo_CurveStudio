# curvestudio-api

FastAPI backend for CurveStudio (AGPL-3.0-or-later). It contains no maths: every number comes
from `curvestudio.workflow` (`import_dataset`, `fit_dataset`, `predict`), so the web app and
the desktop app show identical results.

## Run

```
cp apps/api/.env.example .env      # edit; see the comments in the file
uv run curvestudio-api             # one uvicorn process on CURVESTUDIO_HOST:CURVESTUDIO_PORT
```

The schema is migrated (Alembic, up to `0003`) at start-up. With `CURVESTUDIO_WEB_DIST` pointing at
the built web app the same process serves it, with a fallback to `index.html` for non-`/api`
paths. The OpenAPI document is `apps/api/openapi.json`; regenerate it with
`uv run python -m curvestudio_api.export_openapi` (a test fails when it is stale).

## Deployment notes

* **One API process per database.** Jobs run in a `spawn` process pool owned by that process;
  the rate limiter is in memory; jobs left `queued`/`running` by a previous process are failed
  (`SERVER_RESTART`) at start-up. Scale up with `CURVESTUDIO_JOB_WORKERS`, not with more
  uvicorn workers.
* Behind a reverse proxy set `CURVESTUDIO_TRUSTED_PROXY_HOPS` to the number of proxies (default
  1). The client address used for rate limits is the Nth entry from the right of
  `X-Forwarded-For`, so a client cannot choose its own address by sending the header itself;
  with `0` the header is ignored. Do not expose the API directly with the default of 1.
* Every fit job runs in its own `spawn` process and every upload is parsed in its own short-lived
  process (time, memory and concurrency limits), so one user's cancel, timeout, crash or heavy
  file cannot disturb another's work. Jobs are scheduled round-robin across users.
* Quotas (datasets, dataset bytes, stored fits) come from the `self-hosted` row of `plans` and can
  be overridden with `CURVESTUDIO_QUOTA_*`; `GET /me` shows limits and usage. `DELETE /me` erases an
  account (its audit rows keep no user id).
* `CURVESTUDIO_ENV=prod` refuses to start without `CURVESTUDIO_JWT_PRIVATE_KEY_PEM` and
  `CURVESTUDIO_JWT_PUBLIC_KEY_PEM` (a matching RSA pair) and adds HSTS.
* Postgres: `CURVESTUDIO_DATABASE_URL=postgresql://user:password@host/db` (psycopg 3).
* A stored fit holds per-row arrays, so its size grows with the dataset (about 130 MB of JSON
  at 500 000 rows). Results above `CURVESTUDIO_JOB_MAX_RESULT_BYTES` fail the job
  (`RESULT_TOO_LARGE`); lower `CURVESTUDIO_MAX_ROWS` for hosted deployments.

## Auto-fit

Auto-fit is the primary workflow: it fits every applicable model to a dataset and ranks the
results (the engine is `curvestudio.discovery`; the API adds jobs, storage and limits).

| Endpoint | |
|---|---|
| `POST /datasets/{id}/autofit` | body `AutoFitRequest` (`rank_by`, `families`, `exhaustive`, `max_parameters`, `n_starts` 1-8, `cv_folds` 2-10, `seed`, `curve_top_n` 1-10, up to 10 `custom_models`) -> 202 `{job_id}` |
| `GET /jobs/{id}` | `progress` while it runs (`done`, `total`, `converged`, `failed`, `skipped`, `elapsed_s`, `top` = the best five rows so far, `partial`), `result_id` when it succeeded |
| `POST /jobs/{id}/cancel` | stops the run; nothing is stored |
| `GET /datasets/{id}/autofits`, `GET /autofits/{id}?rank_by=`, `DELETE /autofits/{id}` | list, read (re-ranked by another criterion without refitting) and delete stored runs |
| `GET /autofits/{id}/export?format=csv\|json` | one CSV table of candidates (ranked rows first, then the ones with no rank), or the JSON of `GET` |
| `POST /autofits/{id}/rows/{candidate_id}/open` | 202 `{job_id, fit_id}`: refits that row (same seed, so the same numbers) into an ordinary stored fit (`/fits/{id}`: curve, bands, diagnostics, predict, export). Opening a row twice gives the same fit (`fit_id` is set the second time) |

A request the engine refuses is a 422 whose `detail.code` is `TOO_FEW_POINTS` (fewer than four
points), `UNKNOWN_FAMILY`, `NO_CANDIDATES`, `INVALID_CUSTOM_MODEL` or `DUPLICATE_MODEL_ID`. The
check is cheap (it counts points and enumerates candidates; it does not fit) and runs in the
request thread pool, never on the event loop; the fitting happens in the job.

* **Time.** A run is stopped at `CURVESTUDIO_AUTOFIT_BUDGET_S` (default 120 s): the candidates
  fitted so far are ranked and stored with `partial: true` (job `succeeded`). The hard watchdog
  `CURVESTUDIO_AUTOFIT_TIMEOUT_S` (default 180 s, plus `CURVESTUDIO_JOB_GRACE_S`) is the backstop:
  a run that is still alive then is killed and fails with `TIMEOUT`, keeping no result. The budget
  must be below the timeout. One candidate may take at most 20 s (or the budget, if shorter).
* **Cores.** Each run starts its own pool of at most `CURVESTUDIO_AUTOFIT_WORKERS` (default 3)
  worker processes, so one tenant cannot take every core; a run of four candidates or fewer stays
  in its job process. The global `CURVESTUDIO_JOB_WORKERS` cap on concurrent jobs still applies.
* **Processes.** An Auto-fit job is not a daemon process (it has children) and leads its own
  process group. Cancel, timeout and shutdown kill the group with the pool; a job process whose API
  process is killed outright notices and takes its pool with it. (Windows has no process groups: there cancel and timeout kill the job process only, and its pool
  workers are not guaranteed to end. Run the API on Linux.)
* **Quota.** `CURVESTUDIO_QUOTA_MAX_AUTOFITS_PER_DAY` (else `max_autofits_per_day` in the plan
  row, else 100) runs per user per IST calendar day, counting cancelled and failed runs too;
  over it: 429 `QUOTA_EXCEEDED` with `Retry-After`. Auto-fit runs do not count against the stored
  fits quota; an opened row does.
* **Storage.** A stored run is `AutoFitResult` JSON without per-point vectors (rows, metrics,
  parameters, reasons and the curves of the top rows are kept), so it stays small; opening a row
  recomputes the full fit. Prediction from an opened row works for registered models and
  polynomials; a row opened from a custom equation cannot be used with `/fits/{id}/predict`
  (the fit specification cannot name a custom model by id).
