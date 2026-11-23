# Self-hosting CurveStudio

CurveStudio's web app ships as **one container**: a single uvicorn process serving both the
API (`/api/v1/...`) and the built React SPA, with fit jobs running in an in-process worker
pool. The stack below adds PostgreSQL and a Caddy reverse proxy with automatic HTTPS.

Everything lives in `deploy/`:

| File | What it is |
|---|---|
| `Dockerfile` | multi-stage image: Node builds the SPA, Python 3.14 runs the API (non-root, uid 10001) |
| `docker-compose.yml` | app + PostgreSQL 17 + Caddy 2, healthchecks, named volumes |
| `Caddyfile` | automatic HTTPS for your domain, gzip/zstd, security headers |
| `.env.example` | every environment variable with safe placeholders |
| `gen-keys.sh` | generates the JWT signing pair and prints `.env` lines |
| `smoke.sh` | end-to-end check against any running deployment |
| `backup.sh` / `restore.sh` | timestamped `pg_dump` / restore through the compose stack |

## Requirements

- A Linux host with Docker Engine **and the compose plugin** (`docker compose version` works).
- Ports 80 and 443 free and reachable from the internet (Let's Encrypt needs port 80 for the
  first certificate).
- A DNS record for your domain (`fit.example.com`) pointing at that host.
- ~1 GB RAM for a small install (fits run in worker subprocesses).

## First run

```bash
git clone https://github.com/amitwh/ConcreteInfo_CurveStudio.git   # or your fork
cd ConcreteInfo_CurveStudio/deploy
cp .env.example .env
$EDITOR .env          # set CURVESTUDIO_DOMAIN and both CHANGE_ME passwords
./gen-keys.sh >> .env # JWT signing keys; then delete the two empty placeholder lines
docker compose up -d --build
docker compose ps     # wait for app to report "healthy" (first build takes a few minutes)
./smoke.sh https://YOUR_DOMAIN
```

`CURVESTUDIO_ENV=prod` (the `.env.example` default) means the API refuses to start without the
JWT keys and sends HSTS. The database schema is migrated automatically at start-up - there is
no separate migration step.

The data layout:

| Volume | Contents |
|---|---|
| `curvestudio-db` | everything: accounts, datasets, fits, jobs, audit log |
| `curvestudio-data` | nearly empty in this stack (SQLite/dev keys only when you switch off Postgres) |
| `caddy-caddydata` | TLS certificates |

`deploy/.env` (which holds the JWT private key) is **not** in any volume - back it up
yourself, and never commit it.

## Upgrading

```bash
cd ConcreteInfo_CurveStudio
git pull
cd deploy
./backup.sh                       # always take a backup first
docker compose up -d --build      # rebuilds the image and restarts the stack
./smoke.sh https://YOUR_DOMAIN
```

Schema migrations run at start-up. To pin a version, check out its tag before building.

## Backup and restore

```bash
./backup.sh                    # deploy/backups/curvestudio-YYYYMMDD-HHMMSS.sql.gz (IST stamp)
./backup.sh /var/backups       # or any existing directory
./restore.sh backups/curvestudio-20260101-053000.sql.gz   # interactive, destructive
```

Backups can run while the app is up. Restore drops and recreates the database, asks for
confirmation, then restarts the app. Schedule `backup.sh` from cron and copy the files off
the host; also keep a copy of `deploy/.env` (the JWT keys) somewhere safe - a database
restored without its original key pair logs everyone out.

## Coolify

The same image deploys on Coolify as **one application** (no compose needed):

1. New resource → application, Git provider, this repository.
2. Build pack **Dockerfile**, Dockerfile location `deploy/Dockerfile`, **base directory `/`**
   (the whole repo is the build context).
3. Port: the container listens on **8000** (`ports_exposes` 8000). Health check path
   `/api/v1/health` (it queries the database, so wait for "healthy" before routing traffic).
4. Domain: your own `curvestudio.<your-domain>` host behind the Coolify proxy.
5. Database: create a database in the **shared-postgres** standalone resource and connect the
   app to it (Coolify adds the `coolify` network and the env vars). The internal hostname is
   the **resource UUID**, i.e. `CURVESTUDIO_DATABASE_URL=postgresql://<user>:<password>@<resource-uuid>:5432/<dbname>`
   (on coolify-hs the shared-postgres UUID is `j2496obuv0owvzbge9dg4rfu`).
6. Environment: the variables from `.env.example` (domain and `CURVESTUDIO_DB_*` do not apply
   here). `CURVESTUDIO_ENV=prod` plus the `gen-keys.sh` output as minimum.
7. **Exactly one replica.** Jobs, the rate limiter and the queue are in-process state: run one
   container per database and scale with `CURVESTUDIO_JOB_WORKERS`, never with replicas.
8. Proxy hops: behind Cloudflare → Traefik → container set `CURVESTUDIO_TRUSTED_PROXY_HOPS=2`
   (both proxies append to `X-Forwarded-For`; `Cf-Connecting-Ip` is not used). Behind Traefik
   only, `1`. The client address for rate limits is taken from the Nth `X-Forwarded-For`
   entry counted from the right, so clients cannot forge it by sending the header themselves.

## Security checklist

- [ ] `CURVESTUDIO_ENV=prod` (startup fails without JWT keys; HSTS enabled)
- [ ] JWT keys generated fresh per install (`gen-keys.sh`), 3072-bit RSA, never committed
- [ ] strong `CURVESTUDIO_DB_PASSWORD`, database not published outside the compose network
- [ ] `CURVESTUDIO_REGISTRATION_ENABLED=false` if the instance should be invite-only
- [ ] `CURVESTUDIO_TRUSTED_PROXY_HOPS` equals the real number of proxies in front
- [ ] HTTPS terminates at Caddy/Coolify's proxy; port 8000 of the container is never published
- [ ] `deploy/.env` mode 600, excluded by `.gitignore` and `.dockerignore`
- [ ] backups scheduled **and** restore tested once; `.env` backed up alongside
- [ ] the container runs as uid 10001 with a read-only root filesystem (`--read-only --tmpfs /tmp` plus the `/data` volume works out of the box)

## Environment variables

Every variable is documented inline in [`deploy/.env.example`](.env.example) and in
[`apps/api/.env.example`](../apps/api/.env.example). The deploy-specific ones:

| Variable | Meaning |
|---|---|
| `CURVESTUDIO_DOMAIN` | public domain; Caddy gets its certificate for it |
| `CURVESTUDIO_DB_USER` / `CURVESTUDIO_DB_PASSWORD` / `CURVESTUDIO_DB_NAME` | the Postgres container's credentials (must match `CURVESTUDIO_DATABASE_URL`) |
| `CURVESTUDIO_DATABASE_URL` | what the API connects to (`postgresql://...`, psycopg 3) |
| `CURVESTUDIO_WEB_DIST` | the built SPA inside the image - keep `/app/web` |
| `CURVESTUDIO_TRUSTED_PROXY_HOPS` | proxies in front (see above) |

## Troubleshooting

- **`app` never becomes healthy** - `docker compose logs app`; the usual causes are a
  misformed JWT key pair (`CURVESTUDIO_ENV=prod` checks it at start-up) or a database the API
  cannot reach.
- **502 from Caddy** - the app is still starting (first boot runs migrations; the health
  start-period is 40 s) or it crashed; check `docker compose ps` and `docker compose logs app`.
- **Certificate errors** - port 80 must be reachable for the first Let's Encrypt issuance;
  confirm DNS points at this host.

## Redirecting the old domains

The app used to live at `curvestudio.concreteinfo.net` and `curvestudio.concreteinfo.co.in`.
`deploy/redirect/` is a small Caddy image for those retired addresses: it answers both old
domains with a permanent (301) redirect to `https://curvestudio.concreteinfo.cloud`,
preserving the path and query, so old links keep working after the move. It deploys as its
**own Coolify application on the old host** with both old domains attached - Dockerfile build
pack pointing at `deploy/redirect/Dockerfile`, base directory `/` - put up once the old
application on that host has stopped.
