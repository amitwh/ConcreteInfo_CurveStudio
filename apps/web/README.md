# curvestudio-web

The CurveStudio single-page app: Vite + React 19 + TypeScript (strict) + Plotly.js.
It talks only to the FastAPI backend in `apps/api` — same origin in production, proxied
in development — through a typed client generated from the committed OpenAPI contract.

## Develop

```
cd apps/web
npm ci
npm run dev          # http://localhost:5173, proxies /api → http://127.0.0.1:8000
```

Start the API next to it (`uv run curvestudio-api` from the repo root). Registration is
enabled by default in dev; create an account in the UI at `/register`.

## Build, test, check

```
npm run build        # production bundle into dist/ (served by the API, git-ignored)
npm run test         # vitest, jsdom + Testing Library
npm run lint         # eslint (flat config)
npm run typecheck    # tsc --noEmit
npm run gen:api      # regenerate src/api/schema.d.ts from ../api/openapi.json
```

`src/api/schema.d.ts` is generated and committed; rerun `npm run gen:api` whenever the
API's `openapi.json` changes (a test on the API side fails when that file is stale).

## Serving

The built SPA is served by the API itself: `CURVESTUDIO_WEB_DIST=apps/web/dist uv run
curvestudio-api`. Non-`/api` paths fall back to `index.html`, so deep links such as
`/about` work. All requests are same-origin relative `/api/v1/...`; the dev proxy exists
only for `npm run dev`.

## Structure

```
src/
├── api/          client.ts (fetch wrapper, auth, refresh single-flight), auth.tsx,
│                 schema.d.ts (generated types)
├── components/   PlotChart (the only Plotly import), CopyButton, ThemeToggle, States
├── format.ts     number/date formatting — "not defined" is n/a, never 0 or NaN
├── hooks/        useJobPolling (job state machine: backoff, cancel, error caps)
├── pages/        Login, Register, About, Workspace
├── workspace/    ModeSwitch (Auto-fit | Manual fit, remembered per user), mode (its
│                 persistence), DatasetsSidebar (upload + column pickers), ChartPanel
│                 (toggles), FitPanel (model picker, custom equations, bounds), traces
│                 (the no-curve invariant), spec (FitSpec assembly + validation)
│   ├── autofit/  useAutofit (the per-dataset Auto-fit state machine: auto-start,
│   │             poll 300 ms→1.5 s, cancel-keeps-partial), AutofitPanel (progress +
│   │             ranking + ordering selector + outcome sections + export),
│   │             RankingTable (keyboard-navigable rows), AutofitOptions (collapsed
│   │             disclosure), rowCurve (a stored row's curve → the chart's bundle)
│   └── results/  Parameters, Statistics, Residuals, Equation, Predict, History tabs
└── styles.css    the single stylesheet (CSS variables, light/dark, ≥400 px)
```

Auto-fit leads, per the product rule: selecting or uploading a dataset starts a run (or
shows the latest stored one) automatically, in either mode; Manual fit is the existing
FitPanel/tabs, shown only when selected.

## Notes and trade-offs

* **Plotly bundle**: `plotly.js-dist-min` (full) is kept because plotly.js 4.x ships no
  partial bundle with the WebGL `scattergl` trace, which 20 000-point datasets need;
  the basic bundle has only scatter/bar/pie (no `scattergl`, no `histogram`). It is
  loaded through a dynamic `import()` inside `PlotChart`, so it sits in its own lazily
  fetched chunk (~1.4 MB gzip) that /login, /register and /about never download — the
  entry bundle is ~98 kB gzip. CSP is `script-src 'self'`: nothing loads from a CDN.
* **Rendering fallback**: data/residual point clouds use plain SVG `scatter` up to 5 000
  points; above that, `scattergl` — but only after a runtime canvas probe confirms WebGL,
  otherwise the trace stays `scatter` on a stride-downsampled (≤ 5 000 points, extremes
  kept) copy. Software-rendered and headless browsers never see "WebGL is not supported".
* **Tokens**: the access token lives in memory only; the rotating refresh token is in
  `localStorage`. A refresh failure logs the user out; reuse detection on the server
  revokes the chain. Strict CSP is the mitigating control for XSS.
* **Chart honesty**: a fit that did not converge never draws a curve — see
  `workspace/traces.ts`; the invariant is unit-tested.
* **Dependencies**: exact-pinned in `package.json` (versions verified against OSV on
  2026-11-17); `package-lock.json` is committed. TypeScript is pinned to 5.9.x because
  typescript-eslint and openapi-typescript do not support TS 7 yet.
