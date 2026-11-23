# curvestudio-api

FastAPI backend for CurveStudio (AGPL-3.0-or-later). It contains no maths: every number comes
from `curvestudio.workflow` (`import_dataset`, `fit_dataset`, `predict`), so the web app and
the desktop app show identical results.

## Run

```
cp apps/api/.env.example .env      # edit; see the comments in the file
uv run curvestudio-api             # one uvicorn process on CURVESTUDIO_HOST:CURVESTUDIO_PORT
```

The schema is migrated (Alembic, up to `0005`) at start-up. With `CURVESTUDIO_WEB_DIST` pointing at
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

## Email verification and password reset

Registration sends a branded verification e-mail and sign-in is refused until the address is
confirmed. Forgot-password sends a single-use reset link. Tokens are random
(`secrets.token_urlsafe(32)`); only their SHA-256 digest is stored, in `email_tokens`, with a
purpose (`verify` 24 h, `reset` 60 min) and a one-shot `used_at`. Issuing a new token of a
purpose invalidates the previous unused one. Links use `CURVESTUDIO_APP_URL`:
`/verify-email?token=…` and `/reset-password?token=…`.

| Endpoint | Behaviour |
|---|---|
| `POST /auth/register` | 201 with `email_verified: false`; queues the verification e-mail |
| `POST /auth/login` | correct password for an unverified address -> 403 `detail.code = EMAIL_NOT_VERIFIED`, no tokens |
| `POST /auth/verify-email` `{token}` | marks the address verified (single use) -> `{email, verified: true}` |
| `POST /auth/verify-email/resend` `{email}` | always 202; sends only when the account exists and is unverified |
| `POST /auth/password/forgot` `{email}` | always 202; sends a reset link when the account exists |
| `POST /auth/password/reset` `{token, new_password}` | re-hashes, verifies the address and revokes every refresh token |

Both "always 202" endpoints return the same generic message whatever the address, so they
never reveal whether an account exists. Reset enforces the same password policy as
registration. All six are rate-limited with the existing limiter.

Mail goes out from the background of the request (a failing send never delays or fails the
response; it is logged). With no `CURVESTUDIO_SMTP_HOST` the send is skipped and logged. In
`CURVESTUDIO_ENV=prod` the API refuses to start when verification is required and no SMTP host
is set. Configuration: `CURVESTUDIO_APP_URL`, `CURVESTUDIO_REQUIRE_EMAIL_VERIFICATION`,
`CURVESTUDIO_SMTP_HOST/PORT/USER/PASSWORD/FROM`, `CURVESTUDIO_EMAIL_FROM_NAME`,
`CURVESTUDIO_VERIFICATION_TTL_HOURS`, `CURVESTUDIO_RESET_TTL_MINUTES`.

## Social sign-in (Google, Microsoft, LinkedIn)

OAuth 2.0 authorization code + PKCE, implemented with the standard library (no dependency).
A provider is enabled only when both its client id and secret are set; the providers endpoint
lists the enabled ones and nothing is exposed otherwise. The flow keeps
the provider token server-side: the callback mints a single-use exchange code (120 s, only its
SHA-256 digest is stored in `oauth_codes`) and the web app redeems it once for an access/refresh
pair.

| Endpoint | Behaviour |
|---|---|
| `GET /auth/oauth/providers` | `{"providers": ["google", …]}` for configured providers only |
| `GET /auth/oauth/{provider}/start` | 404 unless enabled; 302 to the provider with `state` + S256 PKCE, sets the HttpOnly `oauth_flow` cookie (10 min) |
| `GET /auth/oauth/{provider}/callback` | validates state/provider, exchanges the code, reads the profile, verifies the address, signs the user in and 302s to `CURVESTUDIO_APP_URL/oauth/callback?code=…`; on failure 302s to `/login` with `oauth_error=state` or `oauth_error=provider` (never a 500) |
| `POST /auth/oauth/exchange` `{code}` | claims the code atomically (single use, unexpired) -> `TokenOut`; an invalid/used code is 400 `INVALID_TOKEN` |

A new account is created with a random, unusable password hash (so password sign-in is not a
way in) and `email_verified_at` set; an existing unverified account is marked verified instead
of duplicated. All endpoints are rate-limited. Provider accounts are matched by address
(case-insensitive) — there is no separate identity table.

Register the exact callback URI at each provider:

```
https://<app host>/api/v1/auth/oauth/{provider}/callback
```

Step-by-step console setup: [`OAUTH_SETUP.md`](OAUTH_SETUP.md).

Setup: create an OAuth client per provider and set `CURVESTUDIO_OAUTH_GOOGLE_CLIENT_ID/SECRET`,
`CURVESTUDIO_OAUTH_MICROSOFT_CLIENT_ID/SECRET` (+ `CURVESTUDIO_OAUTH_MICROSOFT_TENANT`, default
`common`), `CURVESTUDIO_OAUTH_LINKEDIN_CLIENT_ID/SECRET`. The scopes are fixed: Google
`openid email profile`, Microsoft `openid email profile`, LinkedIn `openid profile
email` (the app needs the "Sign In with LinkedIn using OpenID Connect" product). The redirect URI is derived
from `CURVESTUDIO_APP_URL`; the provider compares it byte for byte, so the web app and the API
must share an origin.

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
