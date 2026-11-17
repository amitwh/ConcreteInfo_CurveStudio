# Changelog

All notable changes are listed here. Dates are IST. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

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
