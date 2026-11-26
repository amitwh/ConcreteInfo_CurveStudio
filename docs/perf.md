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

### Task 1, 25/11/2026 — after the route split

Command: the same `node /home/amith/tmp/cs-verify/perf/bundles.mjs <worktree>/apps/web/dist`
after `npm ci && npm run build`, on the Task 1 worktree (code state `81eb1f7` plus the
route-split change). Full output: `bundles-task1.out` next to the script.

| Asset | Raw | Gzip | Loaded |
| --- | ---: | ---: | --- |
| `assets/index-*.js` (the entry) | 353,989 | 110,474 | every page, including the public page |
| `assets/jsx-runtime-*.js` | 8,243 | 3,114 | modulepreloaded by the entry |
| `assets/rolldown-runtime-*.js` | 726 | 429 | modulepreloaded by the entry |
| **the public page's JS, total** | **362,958** | **114,017** | |
| `assets/WorkspacePage-*.js` | 125,737 | 34,110 | lazily, the workspace |
| `assets/HelpPage-*.js` | 65,508 | 22,073 | lazily, help (page or overlay) |
| `assets/AdminConsole-*.js` | 75,064 | 21,085 | lazily, the admin console |
| `assets/AuthenticatedApp-*.js` | 13,315 | 4,526 | lazily, the signed-in shell |
| `assets/AboutPage-*.js` | 8,988 | 3,406 | lazily, about (page or overlay) |
| `assets/ProfilePage-*.js` | 9,824 | 3,134 | lazily, the profile overlay |
| `assets/RegisterPage-*.js` | 3,343 | 1,286 | lazily, sign-up |
| `assets/ResetPasswordPage-*.js` | 2,531 | 1,041 | lazily, password reset |
| `assets/ForgotPasswordPage-*.js` | 1,819 | 887 | lazily, password recovery |
| `assets/VerifyEmailPage-*.js` | 1,780 | 715 | lazily, e-mail verification |
| `assets/OAuthCallbackPage-*.js` | 1,037 | 615 | lazily, the OAuth callback |
| all JS together | 6,559,739 | 2,071,275 | |

The split: every page but the public home is `React.lazy` behind a `LazyBoundary`
(`src/components/States.tsx`), one chunk per page, and the signed-in shell itself — the
navigation bar with its workspace tabs, the overlays' Modal wiring, the workspace-set
state — is its own chunk (`src/authenticated/AuthenticatedApp.tsx`), so the anonymous
home page downloads no workspace code at all. The entry fell from 611.4 KB raw /
184.3 KB gzip to 354.0 / 110.5, and the public page's total JS (entry plus its two
modulepreloads) from 184.3 to 114.0 KB gzip: **against the ≤ 100 KB target this is not
met, 14 KB over**. What remains is library floor, not page code — by the source-map
attribution of the entry's minified bytes, roughly 57% is react-dom (with react and
scheduler), ~11% react-router, ~10% the home page itself, ~6% TanStack Query's core
(mounted by `main.tsx`'s root provider, whose every consumer is behind auth — moving
that provider behind sign-in is the one remaining in-app lever, worth ~7 KB gzip), the
rest the API client, error reporter, shell and icons. No planned P6 task removes those
bytes; the LCP target below is the one this task moves.

### Task 3, 25/11/2026 — the header asset sized, the fonts preloaded

Command: the same `node /home/amith/tmp/cs-verify/perf/bundles.mjs <worktree>/apps/web/dist`
after `npm run build`, on the Task 3 worktree (code state `1efe2e8` plus the Task 3
change). Full output: `bundles.task3.out` next to the script.

| Asset | Before | After |
| --- | --- | --- |
| `curvestudio-logo.png` (every page's header; also embedded in the PDF report) | 661 × 176 PNG, 117,710 B | 330 × 88 PNG-8 (256 colours), **17,895 B** |

−84.8 %. The header renders the logo at 165 × 44 CSS px (the `width`/`height`
attributes on the three `<img>`s that use it: the public header, the workspace header,
the home footer), so 330 × 88 is its full 2 × retina size; the PDF report header prints
it at 26 pt height, where 88 px is ~2.5 × device pixels. Produced with the system's own
tools — ImageMagick Catrom resize, pngquant 256-colour quantisation, optipng — no new
dependency. Format comparison on the same 330 × 88 render: lossless WebP 24,976 B,
*larger* than the palette PNG, so the PNG-8 is the smallest format that renders
identically; lossy WebP q90 was 8,206 B but is not pixel-identical and react-pdf (the
report embeds this same file as a data URL) cannot use WebP. Visual verification:
`node /home/amith/tmp/cs-verify/perf/task3_probe.mjs` (same env switches as
`web_timings.mjs`; not committed) screenshots the header logo element at
deviceScaleFactor 2 with the old asset build and the new one — PNGs under
`/home/amith/tmp/cs-verify/perf/screenshots/task3/{before,after}`; the two renders are
visually identical (side-by-side inspected: same text, colours, proportions, tagline
legibility; the pixel diff, RMSE 0.14 public / 0.026 workspace header, is edge
anti-aliasing from the two resampling paths of the same art).

The two Latin font files the first screen renders are now preloaded in the served
HTML — `<link rel="preload" href="/assets/inter-latin-wght-normal-Dx4kXJAl.woff2"
as="font" type="font/woff2" crossorigin="anonymous">` and the same for
`jetbrains-mono-latin-wght-normal-B9CIFXIH.woff2` (48,256 B and 40,404 B). The hashes
exist only at build time, so `fontPreloadPlugin` (`apps/web/src/lib/fontPreload.ts`,
wired into `vite.config.ts`) finds each family's latin subset among the emitted assets
after the bundle is complete and has Vite inject the two links into `index.html`; its
`latinFontPreloads` unit tests pin the selection (latin only, never `latin-ext`, never
ambiguous) and the attributes. Which two to preload was verified, not assumed: the
probe above also lists every woff2 the public page actually fetches — exactly those two
latin files, plus the 19.0 KB `inter-greek` subset some greek-range glyph on the page
pulls in (not latin, deliberately not preloaded).

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

### Task 1, 25/11/2026 — LCP after the route split

Command (the harness grew environment switches for the parallel P6 tasks; same
measurement, same throttle):

```
CS_REPO=<task1 worktree> CS_RUN=/home/amith/tmp/cs-verify/perf/run-web-task1 \
CS_PORT=8795 CS_SECTIONS=A,B \
CS_SHOTS=/home/amith/tmp/cs-verify/perf/screenshots/task1 \
node /home/amith/tmp/cs-verify/perf/web_timings.mjs
```

Setup identical to the baseline (section A of the same script: the local API serving the
built SPA, CDP 10 Mbit/s / 40 ms RTT, one warm-up then 7 runs, fresh context each),
except port 8795 so a parallel task's run cannot collide. Full output:
`web_timings-task1.out` and `.err` next to the script; comparison screenshots in
`screenshots/task1/` (`public.png`, `workspace-fit.png`).

| Figure | Baseline (25/11/2026) | Task 1, 25/11/2026 |
| --- | ---: | ---: |
| LCP, median | 1,272 ms | **1,068 ms** (runs 1,064–1,100) — target ≤ 2.0 s, met |
| LCP element | an `<h2>` heading | an `<h2>` heading |
| Transferred per load, median | 806,100 bytes over 10 requests | 557,219 bytes over 11 requests |

Machine activity at measurement time: no parallel web build was running (checked by
process list; load average ≈ 3 on the 12-thread workstation); the P6 Task 2 implementer
may build concurrently at other moments, which is why the size table above — the
deterministic figure — is the one to compare on. The uncompressed-serving caveat above
still applies to both columns equally.

### Task 3, 25/11/2026 — LCP with the sized logo and the preloaded fonts

Command: section A of the same run as the first-chart table below (`CS_RUN=run-web-task3`,
port 8799, same throttle and script). Full output: `web_timings-task3.out` and `.err`
next to the script.

| Figure | Task 1, 25/11/2026 | Task 3, 25/11/2026 |
| --- | ---: | ---: |
| LCP, median | 1,068 ms | **1,112 ms** (runs 1,104–1,128) — target ≤ 2.0 s, met |
| LCP element | an `<h2>` heading | an `<h2>` heading |
| Transferred per load, median | 557,219 bytes over 11 requests | 547,356 bytes over 11 requests |

The logo's −99.8 KB shows in the transferred figure, but the two font preloads
(88.7 KB together) now share the throttle's bandwidth with the entry script from the
first bytes, and the LCP element — an `<h2>` painted fallback-then-swapped — lands
~44 ms later. That is the honest cost of the fonts arriving with the CSS instead of
after it: the first font-dependent paint no longer waits on a post-CSS fetch chain.
The run ranges do not overlap (1,064–1,100 vs 1,104–1,128), so the +44 ms is real,
small, and inside every target. Machine activity at measurement time (~21:50 IST):
load average ≈ 1.5–1.9 on the 12-thread workstation, no parallel builds (process list
checked); a background browser sync and an rclone transfer were active throughout.

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

Task 1, 25/11/2026 — observational re-measure on the same setup (section B of the same
run as the LCP table above): median **2,658 ms** (runs 2,644–2,946). The route split made
sign-in fetch its chunks in sequence — the signed-in shell, then the workspace page, then
Plotly — where the monolithic entry needed only Plotly; ~0.5 s of the difference is those
two extra chunk fetches under the 40 ms-RTT throttle. Task 3's idle prefetch of the
workspace and chart bundles when a visitor is signed in is the designed remedy; the ≤ 3 s
target still holds, but the figure is worse than the baseline and is recorded here so
Task 3 is judged against the honest number.

### Task 3, 25/11/2026 — the idle prefetch

Command: section B of the same run as the Task 3 LCP table (`web_timings.mjs`,
`CS_RUN=run-web-task3`, port 8799, same throttle, fresh context per run). Full output
`web_timings-task3.out` next to the script.

| Figure | Baseline (25/11/2026) | Task 1, 25/11/2026 | Task 3, 25/11/2026 |
| --- | ---: | ---: | ---: |
| Submit click → first chart, median | 2,172 ms | 2,658 ms | **1,802 ms** (runs 1,787–1,957) — target ≤ 3 s, met |

The prefetch is `useChartPrefetch` (`apps/web/src/lib/chartPrefetch.ts`, mounted in
`App.tsx`): once the auth status is `authed`, `requestIdleCallback` — with a 2 s timeout,
and a same-length timer where the API is missing — starts the same dynamic import
PlotChart's loader uses, so the browser's module map deduplicates and a chart mounted
mid-prefetch waits on the in-flight fetch instead of starting a second one. An anonymous
visitor schedules nothing. The mechanism was checked directly, not inferred:
`node /home/amith/tmp/cs-verify/perf/task3_prefetch_check.mjs` (same throttle, one
scripted run) shows both chart resources — the JS chunk and its CSS — requested at
t0+1,054 ms and complete by t0+1,094 ms, before the dataset list had even rendered
(t0+1,727 ms), and no further chart fetch after the row click: the first chart now
waits on its render, not its download. That recovers the whole Task 1 regression and
beats the pre-split baseline by 370 ms.

Judgement call, recorded: the shell (`AuthenticatedApp`) and workspace chunks were NOT
additionally prefetched. In the one state the plan allows prefetching in — a visitor
signed in — those two chunks are already being fetched by the render itself (the
`LazyBoundary`s fetch them the moment the authed tree mounts), so an idle prefetch for
them is a no-op by construction; the measured 1,802 ms, with the chart chunk provably
in memory before the click, confirms the chart bundle was the only chunk worth pulling
early. What remains on the path is the sign-in API call and the sequential
shell → workspace fetch chain, which no signed-in prefetch can overlap.

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

### Task 4, 25/11/2026 — large datasets: the constraint measured, then the design

**Where does a large plot's time actually go?** Measured first, before any design:
`node /home/amith/tmp/cs-verify/perf/task4_probe.mjs` (full outputs `task4_probe.out`,
`task4_probe2.out` next to the script) starts the worktree's API on 127.0.0.1:8797, seeds
the baseline generator's `decay` (100 rows), `gen-100k` and `gen-1m`, and times the three
candidate constraints. Server figures are curl `time_total`, median of 5; parse figures
are an in-page `fetch`+`response.json()`, median of 3; draw figures are `Plotly.newPlot`
`scattergl` markers through the app's own chart-bundle chunk on SwiftShader WebGL, median
of 3 after a warm-up (the second pass, whose GL actually painted — see the CSP note).

| Stage | 100k rows @20k pts | 100k full | 1M rows @20k pts | 1M @100k | 1M @200k | 1M @500k | 1M full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| API serve (median) | 21 ms | 35–39 ms | 166–174 ms | 193–205 ms | 229–278 ms | 373–405 ms | 383–402 ms |
| body size | 0.34 MiB | 1.70 MiB | 0.34 MiB | 1.69 MiB | 3.38 MiB | 8.47 MiB | 16.95 MiB |
| browser fetch+parse | 29 ms | 49–63 ms | 179–185 ms | 203–205 ms | 247–248 ms | 399–439 ms | 501–508 ms |
| SwiftShader GL draw, `newPlot` (marker count) | 53 ms (20k) | 80 ms (100k) | — | — | 107 ms (200k) | 268 ms (500k) | 587–1 171 ms (1M) |

**The design that landed:**

- **API:** `MAX_POINTS` 20 000 → 1 000 000 (`routers/datasets.py`). The client-chosen
  sample stays capped and validated — `points=1 000 001` is a 422 — and anything larger
  is stride-sampled down server-side. Never an unbounded fetch.
- **The chart's request** (`workspace/traces.ts`, `chartSamplePoints`): above 20 000 rows
  — exactly where the old fixed request began losing rows, since at or below it the API
  already returned every row — and only with WebGL, the chart asks for
  `min(n_rows, 1 000 000)`; every other case is byte-identical to the old request, so the
  SVG path and WebGL-less browsers fetch no more than before. An upload passes its own
  `n_rows` because the dataset list has not re-rendered when the selection starts.
- **Display decimation** (`GL_DISPLAY_MAX_POINTS = 200 000`, `decimateMinMax`): above it
  the data trace is min/max envelope decimated — the x-range splits into 100 000 buckets,
  each bucket contributes its least-y and greatest-y sample (once, when one sample wins
  both), and the range's least-x and greatest-x samples are always kept — so the drawn
  y-envelope per bucket is exact (a spike inside a bucket cannot vanish) at ≤ 200 002
  points, and the drawn x-span never shrinks. Non-finite samples draw no marker and are
  skipped; buckets are by x-value, so unsorted datasets decimate just as correctly. The
  budget sits just below the measured render knee (107 ms at 200k markers vs 268 ms at
  500k and ~0.6–1.2 s at 1M on SwiftShader, the slowest renderer the app targets) and
  keeps the 100 000-row target dataset drawing every row, undecimated.
- **The fitted curve and bands are never decimated** (the smoothness invariant, pinned by
  a test): they stay the engine's own 200-point and 2×200-point polygons, verbatim.
- `SCATTER_MAX_SVG_POINTS = 5 000` stays the SVG/`scattergl` trace-type threshold,
  unchanged — the mechanism the app already had.

**The bug the screenshots found — the CSP was silencing every WebGL chart.** The first
screenshot pass showed Plotly's "WebGL is not supported" banner where the `scattergl`
clouds should have been — on the before build and the after build alike, at 20 000 points
as much as at 100 000. Traced and verified with fresh-browser A/B runs (marker-pixel
counts on page screenshots, `task4_paint_truth.mjs`): a bare `getContext('webgl')`
succeeds in this Chromium, hardware GL (`--use-angle=gl`) or SwiftShader, and the full
`plotly.js` dist bundle behaves exactly like the app's module-built chunk — but Plotly's
regl pipeline generates code at runtime, and the API's Content-Security-Policy
(`script-src 'self' 'wasm-unsafe-eval'`, deliberate for the PDF renderer) blocked it
silently: regl's context creation failed and Plotly covered the chart with the banner.
Adding `'unsafe-eval'` to `script-src` (`middleware.py`, with the reasoning in place of
the old no-unsafe-eval note) makes the same page paint 4 592 marker samples where the
banner was — with the policy enforced, on both renderers. Which means **from the day that
header landed (22/11, `036d168`) every web chart above 5 000 points had been showing the
banner instead of data, in every browser, and every earlier "scattergl" screenshot
(Task 0's baseline note, Task 2's five pairs) had been photographing the banner on both
sides of its comparisons.** The relaxation is a real security-posture trade — the price
of shipping plotly.js-gl — and is flagged for the owner to ratify; the self-only source
allowlist, nosniff and no-framing are unchanged.

**End-to-end, before and after** (`node /home/amith/tmp/cs-verify/perf/task4_timings.mjs`,
outputs `task4_timings-{before,after}.out`; before = the saved pre-Task-4 build in
`task4-dist-before/`, after = this worktree's). Same shape as section C: no throttle, the
100-point dataset selected first so the chart bundle is loaded, clock from the dataset-row
click to the Data trace holding the target count. Two deliberate differences from the Task
0 baseline, recorded honestly: the workspace sits in **Manual fit** (a selection there
starts no Auto-fit, so no background job competes with the clock — Task 0 drained after
each run instead), and logins are paced under the route's 10/min limit (a run every ~10 s
trips it otherwise; observed, then fixed in the harness). The browser cache is cleared per
run and the detail fetch's bytes are asserted non-zero, so no run rides a cache hit.

| Dataset | Before (20 000-pt sample) | After (full fetch) | Drawn points | Target |
| --- | ---: | ---: | ---: | --- |
| gen-100k, median of 5 | 142 ms (runs 137–376) | **350 ms** (runs 325–521) | 20 000 → **100 000** | ≤ 2 s, met |
| gen-1m, median of 5 | 369 ms (runs 289–370) | **1 023 ms** (runs 1 011–1 212) | 20 000 → **200 001** (of 1 000 000 fetched) | none set; recorded |

Both columns are measured with the CSP fix in place and the GL raster actually painting —
the after-figure buys five times the drawn points for ~+210 ms at 100k, and at 1M the
whole dataset (16.96 MiB) is fetched, min/max-decimated and drawn in about a second.
Machine activity during the runs: load average ≈ 1.4–1.7 on the 12-thread workstation (no
parallel web build running; an rclone and a small PHP process were resident earlier in the
evening).

**Screenshots** (`node /home/amith/tmp/cs-verify/perf/task4_screenshots.mjs <out-dir>`,
real renders after the CSP fix): below the threshold (`decay`, 100 rows — the plain SVG
path) and above it (`gen-100k`, `gen-1m`), before and after, under
`/home/amith/tmp/cs-verify/perf/screenshots/task4/`. The below-threshold pair is
pixel-identical (`compare -metric AE` = 0) — the path below the threshold is exactly
today's. The above-threshold pairs both show the rendered decay cloud (verified by
marker-pixel counts: ~1 900 samples in the plot area, where the pre-fix banner runs
showed ~2); they differ by 3.5 % of pixels — the same cloud drawn from 5× and 50× the
points, marginally denser. `screenshots.json` in each directory records the trace types
and counts.

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

## Server: warm job workers (Task 5, 25/11/2026)

Each `CURVESTUDIO_JOB_WORKERS` slot is now a warm worker (`apps/api/.../worker_pool.py`):
a `spawn` process that imports the whole fitting stack once and then forks a throwaway
child per job. The worker never runs job code, so a job's process is as fresh as the
one-spawn-per-job it replaces — proven by test (`tests/api/test_api_jobs_warm_pool.py`): a
fit and a cross-validation through ONE worker are byte-identical to the same jobs in
fresh `spawn` processes (canonical JSON; the one blanked field is the bundle's
provenance wall-clock stamp), and a cancelled job's environment, working directory,
module globals and files do not reach the next job in that worker. The kill machinery
keeps its strength: the child leads its own process group, so a timeout or a cancel is
one `killpg` — an Auto-fit run's engine pool dies with the job — a timed-out job's worker
is replaced, a cancel in the fork's hello window is routed through the child's own
announced pid (tested with the scheduler's tick blocked), and a hard-killed API process
takes worker, child and grandchildren with it (`PR_SET_PDEATHSIG` plus the parent
watches; tested). A worker recycles after **50 jobs** or above **384 MB** of its own
resident size — `CURVESTUDIO_JOB_WORKER_MAX_JOBS` / `CURVESTUDIO_JOB_WORKER_MEMORY_MB`:
a job's memory never touches the worker (each job dies with its child), so the limits
bound the worker's slow drift, not job size; the measured post-import resident size is
**202.3 MB** (VmSize 1,192 MB), giving the default ≈1.9× headroom, and 50 jobs bounds
any leak to a spawn's cost every fifty jobs. The upload parser keeps its fresh,
self-capped child, exactly as design step 11 requires.

Command: `bash /home/amith/tmp/cs-verify/perf/task5_docker_replica.sh <before|after>` —
the same constrained replica as Task 0 (`docker run --cpus=2 --memory=2g --init`, the
production image, its own health check; torn down after). `before` builds exactly
`579ce27` (`git archive`, no working-tree changes); `after` builds the Task 5 tree. Both
passes ran back-to-back on the otherwise-idle workstation, minutes apart. The scripts:
`server_replica_task5.mjs` (the Task 0 Auto-fit/import measurements), the new
`task5_fit_wall.mjs` (1 warm-up + 15 small fits — `POST /datasets/{id}/fits`, polynomial
degree 1 on `decay.csv`, polled to job success; the surface the per-job spawn used to
dominate) and `warm_worker_probe.py` (a warm worker's spawn cost, its per-job dispatch
cost and its resident size, replacing Task 0's one-shot `worker_spawn_probe.py`).

| Figure (2-CPU/2-GB replica) | Before (`579ce27`) | After (Task 5) |
| --- | ---: | ---: |
| small fit, POST → job succeeded, median | 1,636 ms (runs 1,571–1,734) | **71 ms** (runs 70–73) |
| small fit, p95 | 1,678 ms | 73 ms |
| Auto-fit 100 points, median | 16,774 ms (runs 16,018–17,515) | **14,912 ms** (runs 14,495–15,296) |
| Auto-fit 100 points, p95 | 17,401 ms | 15,063 ms |
| 100 000-row CSV import, median | 2,294 ms (runs 2,245–2,320) | 2,239 ms (runs 2,212–2,317) |
| per-job process start (probe) | 1,581 ms spawn → hello | **5 ms** warm dispatch → hello (median; a worker's first spawn still 1,633 ms) |

The per-job spawn is gone where it was the whole story: a small fit drops 1,636 → 71 ms
(23×) and a warm worker hands a job to its forked child in 5 ms, against the 1.58 s a
fresh process needed. Against the plan's targets, honestly: **the Auto-fit 3 s / 6 s
target is not met** (14.9 s / 15.1 s) — the ~1.9 s this task removed was the job
process; what remains is the search itself (174 candidates on 2 CPUs) plus the engine
pool each run starts inside the job (`discovery/pool.py`, ~2 s of interpreter starts per
run, three workers) — design step 10, Task 6's warm Auto-fit pool, is that lever. The
**2 s import target is not met either** (2.24 s, within noise of before): the parser's
fresh, self-capped child is deliberately untouched (step 11 is Task 6's warm parent for
it). Both are recorded as not started on their next lever, not as failures of this one.

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

## Server: the health check's log volume (Task 7, 25/11/2026)

Who probes the app, verified rather than assumed. Production's only standing prober is the
image's own HEALTHCHECK (`deploy/Dockerfile`: `GET /api/v1/health` every 30 s, 5 s timeout,
3 retries, 40 s start period) — Coolify's own HTTP health check is switched off for the
application (`health_check_enabled=false` with `custom_healthcheck_found=true`, read from
the Coolify API 25/11), so Coolify reports "healthy" from the image's check and the cadence
is exactly that 30 s. The Sentinel is Coolify's server-metrics collector (server-side Docker
stats, 10 s refresh); it never calls the app's HTTP endpoints. `/api/v1/ready` has no
standing prober anywhere in the repository — the deployed-probe figures above were ad-hoc
curl runs — but it is a probe route and is quieted with `/health`.

Command: `bash /home/amith/tmp/cs-verify/perf/health_log_probe.sh <port> 10` — starts the
worktree's API (`uv run curvestudio-api`, SQLite, `CURVESTUDIO_ENV=test`), waits for
readiness, truncates the start-up lines, then issues exactly 10 GETs of `/api/v1/health` at
the production cadence (one per 30 s) and counts every line the process wrote to stderr from
the first probe to the last. Before = the worktree at `8daa069` (the Task 0 code); after =
the Task 7 change (the same record logged at DEBUG for the two probe paths).

| Figure | Before | After |
| --- | ---: | ---: |
| Probes, all HTTP 200 | 10 | 10 |
| stderr lines, first probe to last | 10 | 0 |
| of which middleware `request` records | 10 (all INFO) | 0 |

One line per probe before — the request middleware's `request` record at INFO; nothing else
wrote anything, including uvicorn's access log (`log_config=None` leaves `uvicorn.access`
unrendered at INFO), so the probe's line was the whole of its log cost. Zero lines after:
the same record now logs at DEBUG for `/api/v1/health` and `/api/v1/ready`, and the JSON
handler's INFO threshold drops it. Extrapolated at the HEALTHCHECK's cadence —
86,400 s/day ÷ 30 s = 2,880 probes/day — that is **2,880 request lines a day before, 0
after**, and the standing design-doc figure (~2,900 lines/day, measured on production
23/11/2026) is the same arithmetic plus start-period retries and restarts, which corroborates
it. Failures stay loud: a probe that fails still raises the health route's own
`database check failed` ERROR record; only the routine 200 line went quiet, and the
per-route request statistics still count every probe as before.

## The production container's capacity — a recommendation (Task 7, 25/11/2026)

The plan's Task 7 asks for the host's spare capacity against the container's limit, ending
in a recommendation. **The limit change is the owner's decision and nothing in any deployment
was modified.** Host figures: the Coolify server-metrics API (the Sentinel's own collection,
5-minute buckets) for a 24-hour window, read 25/11/2026 ~21:05 IST from coolify-vps, server
`localhost` — one host carrying 9 resources: this app (`curvestudio-vps`), `cams2` and its
PgBouncer, the shared Postgres, Redis, MariaDB and RabbitMQ, phpMyAdmin and `cmd_app`, plus
Coolify itself.

| Figure | Value |
| --- | --- |
| CPU, whole host, 24 h | typically 11–25 %, brief peaks ~34 % |
| load1 | mostly < 1.5, spikes to ~4.4 (job bursts) |
| Memory | 16.76 GB total, 5.25–5.61 GB used across the window (~32–33 %) |
| Disk | 207 GB total, 30.5 % used |
| The app's container limit | 2 CPUs / 2 GB (Coolify `limits_cpus=2`, `limits_memory=2 GiB`, read 25/11 — the plan's stated production limit) |

What the app asks of that limit, from this file's own measurements plus the application's
production environment (read from Coolify 25/11): production runs `CURVESTUDIO_JOB_WORKERS=1`
and `CURVESTUDIO_AUTOFIT_WORKERS=2` — the defaults are 2 and 3, so production already chose
the conservative end. One Auto-fit is therefore 1 job process + 2 fit workers, three
model-stack processes, plus the API process serving requests, all inside the 2-CPU cap; the
replica's measured cost of that shape of work is the Auto-fit table above (**15.9 s median,
16.4 s p95** at 2 CPUs with the 3-worker default; the design's target is 3 s) and the
**1.50 s** per-job process spawn. The upload parser child is separately capped at
`CURVESTUDIO_IMPORT_MEMORY_MB=1024` (the default; production does not override it) — one
child allowed half the container's whole memory limit.

The arithmetic:

- **CPU.** During a fit the cap, not the host, binds: the 24-h host CPU (app and all nine
  resources together) sat at 11–25 % with ~34 % peaks, so the majority of the host's CPU was
  idle while the app's three fit processes shared 2 CPUs — 2 ÷ 3 ≈ 0.67 CPU each. The
  load1 spikes to ~4.4 are the same bursts engaging roughly four cores' worth of host before
  the cap flattens them into a longer queue. Raising the limit to 4 CPUs gives each fit
  process about a core (4 ÷ 3 ≈ 1.33, 2× today's share). Auto-fit's 174 candidate fits are
  independent, so wall time should fall roughly with that share until the serial parts
  dominate: a rough expectation is 15.9 s → ~9 s, the 1.5 s spawn and the data setup not
  shrinking — an expectation, not a measurement; nothing was re-measured above 2 CPUs, and
  4 CPUs alone does not reach the 3 s target (Tasks 5–6's warm pools are the bigger lever
  on the serial part). The cost to the neighbours is bounded: Docker's `--cpus` is a ceiling,
  not a reservation, so the raise only removes the cap during bursts; the shared databases
  and RabbitMQ keep whatever CPU they claim first.
- **Memory.** The 2 GB limit has been enough (the production container has
  `restart_count=0` and the replica never OOMed), but the tight case is an import's
  1-GB-capped child landing while an Auto-fit's three model-stack processes are resident —
  per-process RSS on the server is **not measured**, so this is a configured ceiling
  argument, not an observed one. The host's spare is 16.76 − 5.61 ≈ 11.2 GB: raising the
  container to 3 GB takes the host's worst case to ~6.6 GB used (~39 %), which the host
  would not notice. (Disk is nowhere near binding at 30.5 %.)

**Recommendation — the owner's call, not made here:** if Auto-fit wall time on the server
matters, raise the container to **4 CPUs / 3 GB** and leave `CURVESTUDIO_JOB_WORKERS=1`;
with 4 CPUs, `CURVESTUDIO_AUTOFIT_WORKERS=3` (the default) becomes the natural second step —
re-measure the Auto-fit table here at the new limit before taking it, and watch memory, since
a fourth model-stack process is exactly the un-measured RSS question above. If the limit
stays 2 CPU / 2 GB, the current `AUTOFIT_WORKERS=2` is already the right setting — more fit
workers than CPUs adds contention and memory without throughput — and the Auto-fit gain must
come from Tasks 5–6's pools instead. Scaling is vertical only either way: the deployment
policy is exactly one replica, with `CURVESTUDIO_JOB_WORKERS` as the lever, never replicas
(`deploy/README.md`).

**Addendum (26/11/2026, after Task 5's warm workers landed):** the memory picture above
predates the warm pool. A warm worker's measured post-import resident size is **202.3 MB**
("Server: warm job workers"), and workers stay resident with no idle expiry — so the
container now holds an idle floor of about **405 MB at the default
`CURVESTUDIO_JOB_WORKERS=2`** before any job runs. The 2 GB limit absorbs that (the floor
plus one import's 1-GB-capped child plus one Auto-fit's engine pool still fits under the
argument above), but the 3 GB recommendation now carries this floor in its margin, and
dropping to `CURVESTUDIO_JOB_WORKERS=1` on a 2 GB box trades the idle floor (405 → 202 MB)
against the second concurrent job.

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
  measured 23/11/2026 and recorded in the design doc; Task 7 has since measured it before
  and after the DEBUG change — see "the health check's log volume" above.
- **Local LCP with edge-equivalent compression** — the local API serves raw bytes (no gzip
  middleware); reproducing Caddy's `encode` locally was out of scope for the baseline. The
  deployed bundle fetch above carries the compressed transfer time.


## 2026-11-26 — P6 Task 6: the window first, the library behind it

Commands (the Task 0 instruments): the desktop cold-start harness
`uv run python /home/amith/tmp/cs-verify/perf/desktop_coldstart.py` from the Task 6
worktree, plus one variant child (`desktop_child_default.py`, held beside the harness)
identical except its Auto-fit runs at the app's own worker count (`workers=None` → the
pool the app pre-warms) — the standing harness pins `workers=2` for run-to-run stability,
which builds a pool of its own that no pre-warm warms. Before = the worktree re-measured
before the change (Task 0's figures within noise): `window_shown` 4.955 s, `model_list`
5.127 s, first-Auto-fit pool overhead 1.095 s.

| Phase (median of 5, from process spawn) | Before | After | Target | Verdict |
| --- | ---: | ---: | ---: | --- |
| `qapp` | 1.756 | 0.142 | — | the window-icon import no longer pulls the window in |
| `window_built` | 4.923 | 0.565 | ≈ `qapp` + ε | met (0.42 s over it) |
| **`window_shown`** | **4.955** | **0.593** | ≤ 1.0 s | **met** |
| `model_list` | 5.127 | 4.233 | ≤ 2.0 s | **missed** — the load itself is 2.82 s; below |
| **first Auto-fit overhead, the app's pre-warmed pool** | **1.095** | **0.153** | ≤ 0.3 s | **met** |
| first Auto-fit overhead, the instrument's own 2-worker pool | 1.095 | 1.051 | — | unchanged by design: that pool is not the pre-warmed one |

The app-default rows behind the overhead figures: `pool_warm` 6.133 s, first Auto-fit
0.437 s of its own after it, second 0.284 s — 0.153 s of first-run overhead.

What moved. The main-window module and every view and controller it imports now resolve
their heavy imports at first use (`curvestudio.discovery`/`workflow`/`project`/`tools`,
NumPy, pyqtgraph), the two desktop package roots re-export lazily (PEP 562), and
`curvestudio_plotting`'s `__init__` resolves its Matplotlib half lazily (the plot's
colours come from the light palette and never needed it). The model library loads on a
background thread started when the window is shown, with a status-bar message while it
runs and a wait-or-says-still-loading guard on the commands that need it; the Auto-fit
pool pre-warms after the load — sequenced, because measured the other way (both at once)
the library load alone went from ~2.8 s to ~4.2 s while eleven workers spawned beside it.
A `--startup-timing` flag prints one line per phase (t0, app_module, qapp, window_built,
window_shown, model_list, pool_warm).

Why `model_list` misses. Measured apart, the library load is **2.82 s**: 0.87 s to import
the stack (`curvestudio.workflow.catalog` and everything under it) and **1.87 s for
`list_models()` itself** — the plugin entry-point load and registration of the 200-model
catalogue. The baseline hid that 1.87 s inside `window_built` (the old manual-fit view
called `list_models()` in its constructor, so Task 0's 0.17 s `model_list` gap was only
the warm re-list). Moving the library behind the window moved all of it there: the window
is 4.4 s faster and the catalogue 0.9 s faster (it no longer competes with the build), but
0.6 + 2.8 > 2.0. Reaching the target needs the catalogue itself to load faster — plugin
import and registration inside `curvestudio.models` — a model-library change, not a
launch-order one; recorded as the follow-up.

Server half, same machine, same day. In a process that has imported the fitting stack
(the shape of a warm job worker's forked child), one tiny Auto-fit run takes **1.810 s**
with the engine's default spawn-started pool and **0.719 s** with the job process's
fork-started warm pool — **1.09 s saved per Auto-fit job**, the pool's creation having
become a fork of already-imported memory (the engine is worker-count agnostic and a
candidate's seed depends only on the run seed and its id, so the results are identical —
also a test). The upload parser child costs **1.227 s** to start under `spawn` and
**0.023 s** forked from the preloaded forkserver the API now starts at start-up. Both
keep Task 5's isolation: the pool's workers die with their job child (PDEATHSIG plus the
manager's `killpg`), and the parser child is still one fresh, self-capped process per
upload (the timeout, memory-cap and crash tests run unchanged).
