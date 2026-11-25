# CurveStudio performance baselines — P6 Task 0

Date: 25/11/2026 (IST). Commit: `fd101c2`. Machine: the development workstation — AMD
Ryzen 5 5600G (12 threads), 61 GB RAM, Linux 7.0.0, otherwise idle while measuring.

Every figure below was produced by a script in `/home/amith/tmp/cs-verify/perf/` (not part
of the repository) whose exact command is recorded with its table, so any number can be
reproduced. Every clock is the script's own (`performance.now()`, `time.monotonic()`,
`time.time()`, curl's `%{time_total}`); nothing is estimated. `Not measured` means exactly
that: the target exists, no baseline was taken, and the reason is given. These are the
before-figures every later P6 task is judged against; each task records its after-figure
here, on the same setup.

## The targets this file feeds

From `docs/superpowers/specs/2026-11-23-performance-design.md`:

| Surface | Target | Baseline (25/11/2026) |
| --- | --- | --- |
| Public page, 10 Mbit/s / 40 ms | LCP ≤ 2.0 s | **1.27 s** median (local, uncompressed serving — caveat below) |
| Public page | JS ≤ 100 KB gzip | **184.3 KB** gzip (the single entry chunk; raw 611.4 KB) |
| Workspace, same profile | first chart ≤ 3 s of sign-in | **2.17 s** median |
| Workspace | 100 000-row dataset plotted ≤ 2 s | **0.16 s** median (of the 20 000-point sample — see the caveat) |
| Server | Auto-fit 100 points: median ≤ 3 s, p95 ≤ 6 s | **15.9 s / 16.4 s** (2-CPU replica) |
| Server | 100 000-row CSV import ≤ 2 s | **2.19 s** median (2-CPU replica) |
| Desktop | window visible ≤ 1.0 s | **4.71 s** median (offscreen source run) |
| Desktop | model list ready ≤ 2.0 s | **4.88 s** median |
| Desktop | first Auto-fit overhead ≤ 0.3 s | **0.56 s** per run on a warm pool (1.05 s pool creation on the first) |

## Web: the built bundles

Command: `node /home/amith/tmp/cs-verify/perf/bundles.mjs <worktree>/apps/web/dist`, run
after `npm ci && npm run build` in `apps/web` (vite 8). Raw sizes are the filesystem's;
gzip is `zlib` level 6 over the same bytes. Full output: `bundles.out` next to the script.

| Asset | Raw | Gzip | Loaded |
| --- | ---: | ---: | --- |
| `assets/plotly.min-*.js` | 4,651,969 | 1,409,192 | lazily, on the first chart |
| `assets/react-pdf.browser-*.js` | 1,201,877 | 441,284 | lazily, on report export |
| `assets/index-*.js` (the entry) | 611,413 | 184,328 | every page, including the public page |
| `assets/AdminConsole-*.js` | 74,890 | 21,034 | lazily, the admin console |
| `assets/fitReport-*.js` | 8,417 | 2,544 | lazily, the report body |
| `assets/index-*.css` | 61,491 | 14,273 | every page |
| `curvestudio-logo.png` (header) | 117,710 | — (PNG) | every page |
| `assets/inter-latin-*.woff2` | 48,256 | — (already Brotli) | first screen |
| all JS together | 6,552,165 | 2,060,247 | |

There is no route split: the entry chunk carries the public page and the whole signed-in
workspace (every page is a static import in `App.tsx`), which is what the 100 KB-gzip
target measures. The two lazy giants (Plotly, the PDF renderer) are the targets of
Tasks 2–3.

### The chart bundle after Task 2, 25/11/2026

The chart library is now built from the `plotly.js` source package's modules — `lib/core`
plus the traces the app registers (`bar`, `box`, `scattergl`; `scatter` is part of core)
— instead of the `plotly.js-dist-min` whole bundle; same library, same locked version
4.3.0 (dist-min ships exactly one file, no partial bundle, so the module route is the
library's supported way to register less). Command: `node
/home/amith/tmp/cs-verify/perf/bundles.mjs <dist>`, run on the build at `81eb1f7`
(before) and the Task 2 build (after); full outputs `bundles.task2-before.out` and
`bundles.task2-after.out` next to the script.

| Asset | Raw before | Gzip before | Raw after | Gzip after |
| --- | ---: | ---: | ---: | ---: |
| the chart chunk (`plotly.min-*.js` → `plotly-bundle-*.js`) | 4,651,969 | 1,409,192 | 1,495,001 | 505,951 |
| the chart chunk's CSS (was inlined in the JS; now a separate file loaded with it) | — | — | 82,962 | 10,471 |
| what a first chart downloads together | 4,651,969 | 1,409,192 | 1,577,963 | 516,422 |

−67.8 % raw, −63.3 % gzip on the first-chart payload. The Plotly `histogram` trace is
deliberately not registered: the statistics mode bins server-side and draws its
histogram as `bar` traces, so the module would be dead bytes. Every other asset is
unchanged (the entry chunk moved 611,413 → 611,476 raw with the import-specifier text);
all JS together 6,552,165 → 3,395,661 raw, 2,060,247 → 1,158,042 gzip.

Screenshot comparison, same date: `node
/home/amith/tmp/cs-verify/perf/screenshots_task2.mjs <dist> <out-dir>` drives the real
UI against a fresh API — the public page's model gallery, a workspace fit (data, fitted
curve, 95 % band), the statistics result's histogram and box plot, and the 100 000-row
dataset's `scattergl` chart. PNGs under
`/home/amith/tmp/cs-verify/perf/screenshots/task2/{before,after}`; all five pairs are
pixel-identical (`compare -metric AE` = 0 differing pixels) and the drawn trace types
are unchanged (`screenshots.json` in each directory).

## Web: public page LCP under throttle

Command: `node /home/amith/tmp/cs-verify/perf/web_timings.mjs` (section A of the output
JSON). Setup: the API (`uv run curvestudio-api`, SQLite, `CURVESTUDIO_ENV=test`) serving
the built SPA from the worktree on 127.0.0.1:8793; headless Chromium (the repository's
playwright) with CDP `Network.emulateNetworkConditions` — download 10 Mbit/s
(1 312 500 B/s), one-way latency 20 ms, i.e. the plan's "10 Mbit/s and 40 ms RTT". One
warm-up (discarded) then 7 runs, a fresh browser context each; LCP read from a buffered
`PerformanceObserver`.

| Figure | Value |
| --- | --- |
| LCP, median | 1,272 ms (runs 1,268–1,300) |
| LCP element | an `<h2>` heading (the observer reports the tag; the page's hero and section headings are `h2`, the logo `img` is smaller) |
| Transferred per load, median | 806,100 bytes over 10 requests |

Caveat, recorded honestly: the local API serves the SPA **uncompressed** — `spa.py`
returns plain `FileResponse`s and the app adds no gzip middleware — while production
compresses at the edge (Caddy `encode zstd gzip`). The local LCP therefore includes the
entry script raw (611 KB, ~0.5 s of the throttle budget alone). It is the correct
before-figure for Tasks 1–3 (same setup before and after) and a pessimistic stand-in for
the deployed page; the deployed bundle fetch below carries the edge-compressed number.

## Web: the first chart after sign-in, throttled

Same script and throttle (section B). The run signs in through the real form on the public
page (fill e-mail and password, click "Sign in"), then clicks the 100-point dataset
(`examples/decay.csv`) in the workspace. The clock runs from the submit click to the first
`.js-plotly-plot` root appearing — Plotly is a dynamic import, so this includes its
1.4 MB-gzip download under throttle. Fresh context per run; the Auto-fit each selection
starts is drained to completion before the next run.

| Figure | Value |
| --- | --- |
| Submit click → first chart, median | 2,172 ms (runs 2,141–2,211) |

## Web: a 100 000-row dataset's plot time

Same script (section C), **no throttle**. Signed in with a fresh refresh token per run
(they are single-use), the 100-point dataset is selected first so the chart bundle is
already loaded, then the generated 100 000-row dataset is clicked; the clock runs from the
row click to the chart's Data trace holding the API's stride sample.

| Figure | Value |
| --- | --- |
| Row click → chart shows the sample, median | 161 ms (runs 139–190) |

Two facts shape this number and belong to the baseline:

- the workspace asks the API for at most `CHART_POINTS = 20 000` stride-sampled points
  (`GET /datasets/{id}?points=20000`, `MAX_POINTS = 20_000` in `datasets.py`), so the full
  100 000 rows never reach the chart today — the target's spirit (draw a huge dataset
  fast) is only half-exercised, which is exactly what Task 4 changes and re-measures;
- above `SCATTER_MAX_SVG_POINTS = 5 000` with WebGL the trace is `scattergl`; this headless
  Chromium has SwiftShader WebGL, so all 20 000 sampled points draw.

## Server: the local constrained replica

Command: `bash /home/amith/tmp/cs-verify/perf/docker_replica.sh`. The production image
(`deploy/Dockerfile`, built from this worktree at `fd101c2`) run with the production
limits — `docker run --cpus=2 --memory=2g --init` — SQLite at `/data/perf.db`,
`CURVESTUDIO_ENV=test`, port published on 127.0.0.1:8794, the image's own health check
active (`GET /api/v1/health` every 30 s, exactly as production). This is the replica the
plan names (`--cpus=2 --memory=2g`); `systemd-run` was not needed. After the measurements
the script removes the container and the image; nothing is left running.

### Per-route latency percentiles

`node server_replica.mjs http://127.0.0.1:8794` — 60 sequential requests per route (10 ms
gap), keep-alive, each timed around `fetch` with `performance.now()`.

| Route | Median | p95 | Notes |
| --- | ---: | ---: | --- |
| `GET /` (the SPA's HTML) | 2 ms | 2 ms | |
| `GET /api/v1/health` | 2 ms | 2 ms | queries the database |
| `GET /api/v1/ready` | 6 ms | 8 ms | queries the database |
| `GET` entry JS (raw — the API does not gzip) | 3 ms | 4 ms | 611 KB over localhost, warm cache |
| `GET` entry CSS (raw) | 2 ms | 2 ms | |
| `GET /api/v1/me` | 5 ms | 6 ms | authenticated |
| `GET /api/v1/models` | 152 ms | 195 ms | authenticated; the first call after boot was 2,332 ms (entry-point scan), and every call re-enumerates the catalogue |
| `GET /api/v1/datasets` | 3 ms | 4 ms | authenticated |
| `GET /api/v1/stats/tools` | 3 ms | 4 ms | authenticated; first call 56 ms |

The `GET /models` cost is per-request, not just cold — worth knowing when Tasks 5–6 move
work off the request path.

### Auto-fit on 100 points

1 warm-up (discarded) + 15 recorded runs of `POST /datasets/{id}/autofit` on
`examples/decay.csv` (the default configuration: 174 candidates, 3 workers), polled to job
success.

| Figure | Value |
| --- | --- |
| Wall, POST → job succeeded, median | 15,899 ms |
| Wall, p95 | 16,354 ms (runs 15,674–16,398) |
| `created_at → started_at` per the API's own clocks | median 2 ms (the database transition only — the process-spawn cost hides inside the run; the probe below measures it directly) |

### A 100 000-row CSV import

3 runs of `POST /datasets` with the generated 100 000-row CSV (2.7 MB), timed to the 201.

| Figure | Value |
| --- | --- |
| Import wall, median | 2,189 ms (runs 2,184–2,190) |

### Worker start-up time

`worker_spawn_probe.py`, copied into the container and run with `/app/.venv/bin/python`:
it spawns one job process exactly the way `jobs.py` does (`spawn` context, target
`job_process_main`, an Auto-fit task's pickled arguments on a 100-point dataset) and times
`Process.start()` to the `STARTED` bytes arriving on the pipe — the moment the parent
begins the job's wall-clock budget. The process is killed right after the hello, so no fit
runs. 12 samples.

| Figure | Value |
| --- | --- |
| Spawn → `STARTED`, median | 1,503.7 ms (min 1,471.2, max 1,566.2) |

This is the per-job cost Task 5's warm pool removes: at ~1.5 s of pure process start-up,
an Auto-fit that needs 16 s wall carries nearly a tenth of it before any mathematics, and
a queue of small fits would pay it per job.

## Server: the deployed service, from outside

Command: `bash /home/amith/tmp/cs-verify/perf/deployed_probe.sh` — read-only GETs only,
no tokens, no sign-ups, no install records, run from this workstation over the public
internet to `https://curvestudio.concreteinfo.cloud` (Caddy edge with `encode zstd gzip`;
curl `--http2 --compressed`, so the numbers are what a negotiating browser sees). 60
samples per URL, 0.1 s apart, each timed by curl's `%{time_total}`. The home bundle is the
entry script the served `index.html` names (`/assets/index-BaS1SyQY.js` at the deployed
commit; the deployed build predates this worktree's).

| URL | Median | p95 | Bytes (compressed) |
| --- | ---: | ---: | ---: |
| `GET /` | 45.2 ms | 54.6 ms | 761 |
| `GET /api/v1/ready` | 56.4 ms | 64.9 ms | 18 |
| `GET` the home bundle | 70.2 ms | 121.9 ms | 180,408 |

Auth-gated per-route percentiles are **local-replica-only by policy**: measuring them
against production would need an account and a token there, and this task's rules allow
read-only, unauthenticated GETs and nothing else. The replica table above carries them.

## Desktop: cold start

Command: `uv run python /home/amith/tmp/cs-verify/perf/desktop_coldstart.py` (from the
worktree). It spawns `desktop_child.py` as a fresh process five times
(`QT_QPA_PLATFORM=offscreen`, throwaway `QSettings`), which prints one `MARK` per phase;
the parent clocks every phase from its own spawn moment. The window-visible moment
offscreen is approximated the way the render scripts do it: `show()` + event-loop pumping
+ one forced render (`grab()`); on a real compositor the window becomes visible within a
frame of that point. The Auto-fit phases use the app's own runner
(`analysis_jobs.run_autofit` with the shared warm pool), on the bundled `example.csv`
(20 rows, exponential family, 2 workers).

| Phase (median of 5, from process spawn) | Seconds |
| --- | ---: |
| `t0` (uv + interpreter up) | 0.025 |
| `app_module` (`curvestudio_desktop.app` imported) | 0.100 |
| `qapp` (QApplication, fonts, icon) | 1.749 |
| `window_built` (`build_main_window` returned — the heavy module-level imports) | 4.684 |
| **`window_shown`** | **4.713** (runs 4.697–4.757) |
| **`model_list`** (first `list_models()` returned; 200 models) | **4.882** |
| first Auto-fit (creates the pool) | 6.510 (its own duration 1.606 s) |
| second Auto-fit, warm pool | 7.045 (its own duration 0.559 s) |

So: the window waits ~4.7 s on imports before it can appear (2.9 s of it inside the
main-window module — the design's 2.6 s figure has grown), the model list is ready 0.17 s
after that, and the first Auto-fit pays ~1.05 s of pool creation (1.606 s vs 0.559 s),
while even the warm run costs 0.56 s — above the 0.3 s target.

## What is Not measured, and why

- **The `--startup-timing` flag's per-phase figures** — the flag does not exist yet; P6
  Task 6 builds it, and the packaged builds are measured with it then (this file's desktop
  table is the source-run stand-in).
- **Packaged (frozen) desktop start-up** — no packaged build is produced in Task 0; Task 6
  measures the Linux release machine (and Windows when available) and records which.
- **Auth-gated per-route percentiles on the deployed service** — read-only, no-token
  policy for production; the constrained local replica carries the authenticated routes.
- **A 1,000,000-row dataset's plot time** — the plan measures 100 000 rows at this stage;
  Task 4 measures both 100 000 and 1 000 000 when the WebGL/decimation threshold lands.
- **The health check's daily log lines** — the standing figure is ~2,900 lines/day,
  measured 23/11/2026 and recorded in the design doc; Task 7 re-measures it before and
  after the DEBUG change rather than duplicating it here.
- **Local LCP with edge-equivalent compression** — the local API serves raw bytes (no gzip
  middleware); reproducing Caddy's `encode` locally was out of scope for the baseline. The
  deployed bundle fetch above carries the compressed transfer time.
