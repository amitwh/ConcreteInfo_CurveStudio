# Changelog

All notable changes are listed here. Dates are IST. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.10.1] - 2026-11-22

### Fixed
- **"Open" on a ranked row now shows the curve it opened.** Opening a row loaded the
  full result but left the previous row selected, so the chart drew the opened model in
  the other row's colour at almost the same place — with two good fits the lines nearly
  coincide and the button looked dead. Opening now selects the row, so the chart, its
  rank colour and the ranking highlight all follow the model you opened.
- Clicking the row you just opened no longer throws its result away; only selecting a
  different row goes back to that row's stored curve.

## [0.10.0] - 2026-11-22

### Added
- **Desktop, a real product UI.** The brand theme (Fusion palette and stylesheet generated
  from the same tokens), bundled Inter and JetBrains Mono, dock title bars with collapse,
  float and close (remembered), View → Panels / Theme / Reset layout, and an icon toolbar
  with shortcut tooltips.
- **Details window.** The five-tab report (equation, parameters, statistics, diagnostics,
  predict) opens in one reusable, resizable, non-modal window with its geometry remembered;
  the dock keeps a compact fit summary. Enter, double-click, "Open full results" and Ctrl+D
  all reach it.
- **Help browser (F1).** The same eleven sections the web `/help` page shows, with a section
  index and a filtering search; Help → Keyboard shortcuts opens at that section, and Help →
  About Qt is there too.
- **A proper About.** ConcreteInfo logo on an always-white chip, live runtime versions, the
  website and source links, the licence and the credits.
- **Colour for comparisons.** One categorical palette in `curvestudio_plotting.palette`
  drives the desktop plot, the web chart and the exported figures: rank 1 is the brand
  accent, ranks 2-8 the design-system set, assigned by the row's position so tied ranks stay
  distinguishable. Confidence and prediction bands are neutral greys so a band can never be
  read as another model's curve, and a persisted "Show top 3" toggle overlays the three best
  models in both apps. Ranking rows carry a matching swatch.
- **Web:** a `/help` page with the canonical content, stable anchors and search; `/about`
  gains the logo, the website and the credits; `lucide-react` supplies the interface icons.
- Lucide icons in the desktop app, from the vendored SVG set (ISC).

### Changed
- Equations are shown and copied as `x^2`, never `x**2`: the engine keeps the parseable form
  while the equation tab, details window, ranking, ranking CSV and the XLSX/PDF reports use
  `^` (which Excel also evaluates). The JSON export keeps the stored form.
- The desktop ranking shows Rank, Model, R², adj. R², RMSE and k only, sized from the widest
  rendered value plus the stylesheet's own padding, so no numeric cell elides.
- The desktop's fit summary, ranking and disclosure area share the dock without overlapping a
  neighbouring dock at 1280×720 and above.

### Fixed
- Web chart bands were still filled with the abandoned `#0F6CBD` blue; they are neutral now.
- A near-perfect fit printed as "R² 1" in the metrics strips; it reads "R² 1.0000".
- The desktop analysis dock could be given more height than it had, painting its summary over
  the data-table dock; the ranking's own 275 px floor was the cause.
- The "Order by" control clipped a word; it elides with an ellipsis and keeps the full label
  in its tooltip.

## [0.9.2] - 2026-11-21

### Added
- Debian package (`.deb`) and AppImage desktop builds from the local build driver
  (`build.py deb|appimage|release`), alongside the existing tar.gz/zip archives.
- Brand icons and favicons in the web app, plus the ConcreteInfo mark in the header.

### Changed
- Web UI adopts the ConcreteInfo/cams2 theme: brand orange accent, warm greys, Inter for
  UI text and JetBrains Mono for equations and numbers (both self-hosted; the CSP now
  allows the inlined `data:` font faces).
- The Auto-fit ranking and fit results span the full workspace width instead of the
  middle column, with horizontal scrolling removed at desktop widths.
- The ranking paginates (10/25/50/100 rows per page) instead of rendering 100+ rows in
  one page; metrics are right-aligned, statuses are labelled badges, and row actions are
  compact ("Open", "Use manually") with full accessible names.
- Chart and about page: chart height adapts below 800 px, Plotly uses the UI typeface,
  and /about reports the real app version.

### Fixed
- Focus visibility on the custom dataset/history/tab/mode controls, `aria-current` /
  `aria-pressed` state cues, re-rank and fits-history failures surfaced instead of
  looking like an empty result, and X/Y column validation on upload.
- Compact "Copy equation" no longer collapses to two lines.

## [0.9.1] - 2026-11-21

### Fixed
- A worker crash during Auto-fit no longer aborts the whole run: `BrokenProcessPool`
  raised from a submit is isolated to the affected candidates and the pool is rebuilt.
- "Open full results" now reproduces the row's own numbers: the refit uses the row's
  `n_starts` and the same deterministic subsample for very large datasets.
- A missing static asset answers 404 instead of the SPA shell (stale chunks fail loudly).
- Web Auto-fit state races: re-selecting a running dataset no longer starts a second run,
  a replaced run's poll cannot overwrite the newer state, Cancel only targets a running
  job, and a failed opened-row job no longer leaves every Open button disabled.
- Cancelled runs' partial rows no longer offer an Open action that silently did nothing.

### Changed
- Version metadata is 0.9.1 everywhere (it still said 0.1.0.dev0 at the v0.9.0 release),
  and the committed OpenAPI document is regenerated.
- CI gains a web job (typecheck, lint, tests, build) — previously none of those ran.

## [0.9.0] - 2026-11-18 (public beta)

### Added
- **Auto-fit** as the primary workflow (web, desktop, CLI): every applicable model is screened and fitted in parallel, converged fits are ranked best to worst with their equations, R², adjusted R², RMSE, AICc, BIC and cross-validation error; selectable ordering ("Best overall" by default), exact fits and non-converged/skipped candidates listed separately with reasons, same-curve ties collapsed.
- Model library of 190+ ids: hand-written families (saturation, exponential, power/log, growth, sigmoid, dose-response, peaks, distributions, kinetics, physics, harmonic, rational), generated TableCurve-style families, and 31 concrete/civil-engineering models, each tested against an independent reference and for parameter recovery.
- Custom equations through a restricted expression grammar (no `eval`), with typo hints and an adversarial test corpus.
- FastAPI backend (accounts with Argon2 and RS256 JWT, datasets, fit and Auto-fit jobs with per-job processes, quotas, audit log) and React web app; container image, Docker Compose self-hosting stack with Caddy and Postgres, health-checked Coolify deployment.
- PySide6 desktop app: project tree, data table, plot, Auto-fit-first analysis, manual fit, results tabs, prediction, PNG/SVG/CSV/XLSX/PDF/JSON export, `.cstudio` projects.
- `.cstudio` project format with hardened loading; PDF/XLSX/CSV/JSON reports and Matplotlib figures.
- Cooperative time limits and multistart evidence for hard NIST problems.

### Security
- Upload parsing in isolated processes with time/memory caps; per-user quotas that survive dataset deletion; rate limiting keyed on trusted proxy hops; append-only audit log.

### Known limitations
- Billing and OAuth sign-in are not implemented. The API runs as a single process per database. Windows desktop builds are produced by CI and have had less manual testing than Linux.
