# Changelog

All notable changes are listed here. Dates are IST. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

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
