# CurveStudio — Implementation Design

**Status:** Draft for approval · **Date:** 16/11/2026 · **Source of intent:** [`docs/intent.md`](../../intent.md) (the PRD; this document says *how* we build it, the PRD says *what*).

## 1. Intent and why

CurveStudio is a free, open-source alternative to CurveExpert: curve fitting, regression, model discovery and engineering analysis, with a native offline desktop app first and a browser app later. It is being built as an open-source project: the desktop app is free to users; the hosted web app may charge a very small subscription to cover hosting cost; anyone can self-host the web app with Docker at no cost.

What makes it worth building (PRD §27): trustworthy nonlinear fitting that never fails silently, ranked automatic model discovery, safe custom equations, reproducible project files, multivariable modelling, and optional engineering model libraries (concrete first).

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. A scientific core (`curvestudio-core`) usable on its own, validated against certified reference data.
- G2. Desktop app (Windows + Ubuntu) that works fully offline and meets the PRD MVP (§3.1).
- G3. Web app + API sharing the same core, with accounts and optional low-cost plans.
- G4. One-command self-hosting via Docker Compose, with billing disabled by default.
- G5. Every model is declarative, referenced and tested, so the library can grow to 100+ models safely.

**Non-goals (for v1.0)**
- Bayesian estimation, differential-equation models, GAMs/mixed-effects/GP, plugin marketplace (PRD §3.3 / §11 "future") — architecture must not preclude them, but they are not scheduled.
- Real-time collaboration, mobile apps, a distributed task queue (Celery/Kafka) before load justifies it.
- A desktop licence, activation or telemetry of any kind.

**Anti-goals (failure even if it "works")**
- A fit that fails to converge but is shown as a plausible result.
- A model "winning" on training R² alone, or a test certifying a model as scientifically valid.
- Different numbers on desktop vs web for the same project.
- Any use of `eval`/`exec`/`pickle` on user-supplied content.
- Self-hosters needing a payment provider, licence key or our servers to run.
- Billing logic leaking into the core or desktop app.

## 3. Constraints

- Python **3.14** floor (installed here: 3.14.4); `uv` for environments/lockfile; Ruff + MyPy (strict on core) + Pytest; GitHub Actions on Linux and Windows.
- Core imports only NumPy/SciPy/LMfit/Statsmodels/SymPy/Pandas/Pydantic — **never** PySide6, FastAPI or React. Enforced in CI by import-linter.
- Dependencies are pinned in `uv.lock`. Before adding or upgrading any dependency we check the latest stable version and OSV/GitHub advisories, and record source and date in the commit message.
- Web/API standards (project owner's conventions): `/api/v1/` prefix; error body `{"error","message","detail"}`; IST (+05:30) timestamps stored as timezone-aware ISO 8601, displayed `DD/MM/YYYY`; Indian digit grouping and `₹` for currency in the UI; tables snake_case plural with `id`, `created_at`, `updated_at`; one-change-per-file migrations with rollback; append-only `audit_logs` for state changes on accounts, plans and projects; secrets via env vars only, `.env.example` mandatory; no PII/tokens in logs.
- The sandbox for this work is a single developer machine; performance targets are benchmarked on documented hardware, not promised.

## 4. Approaches considered

| | Approach | Verdict |
|---|---|---|
| **A** | **Core-first uv monorepo** — core → desktop → api/web, strictly layered | **Chosen** |
| B | Desktop-first, extract core later | Rejected: fitting logic couples to UI; the web app would then need a rewrite |
| C | Web-first with desktop wrapper (Tauri/pywebview) | Rejected: contradicts native PySide6 / offline intent; weaker for dense data tables |

Chosen because it is the only approach that protects "same engine, same numbers, everywhere". Its cost (no clickable UI early) is reduced by a thin vertical slice — import → fit → plot in the desktop shell — at the end of the core phases.

## 5. Architecture

### 5.1 Repository layout (adapts PRD §4.3; names are `curvestudio`, not `curvelab`)

```text
ConcreteInfo_CurveStudio/
├── pyproject.toml            # uv workspace root, tool config (ruff, mypy, pytest, import-linter)
├── packages/
│   ├── core/                 # curvestudio-core   (MIT)
│   │   └── src/curvestudio/{data,models,fitting,optimization,statistics,diagnostics,
│   │                        discovery,prediction,interpolation,transformations,
│   │                        expressions,project,validation}/
│   ├── model_library/        # curvestudio-models (declarative model specs; subpackages by family)
│   ├── engineering/          # curvestudio-engineering (optional; concrete & civil models)
│   ├── reporting/            # curvestudio-reporting (PDF/XLSX/JSON/CSV; no GUI deps)
│   └── plotting/             # curvestudio-plotting (Matplotlib publication output; shared by desktop + reports)
├── apps/
│   ├── desktop/              # curvestudio-desktop (PySide6; AGPL-3.0)
│   ├── api/                  # curvestudio-api (FastAPI; AGPL-3.0)
│   └── web/                  # React + TypeScript + Plotly.js (AGPL-3.0)
├── deploy/                   # docker-compose, Dockerfiles, Caddyfile, backup scripts, .env.example
├── tests/{unit,integration,numerical,regression,e2e}/   # plus per-package tests
├── examples/  docs/  scripts/
```

Dependency direction (enforced): `apps/* → reporting, plotting, engineering, model_library → core`. Nothing points back up. `engineering` and `model_library` register models with core through an entry-point registry, so engineering libraries stay optional.

### 5.2 Core design

**Models.** A model is a frozen declarative `ModelSpec`: `id`, `name`, `category`, `equation` (SymPy string — doubles as LaTeX source and the evaluator), independent variables, `Parameter` list (name, description, unit, default bounds), `initial_guess(x, y)`, optional analytic Jacobian (otherwise SymPy-derived, otherwise numeric), validity rules (e.g. `x > 0` for `ln x`), references and assumptions. The PRD's `BaseModel` interface is satisfied by a thin adapter around `ModelSpec`. Most library models are one spec each — no per-model subclass boilerplate.

**Expressions (custom equations).** `curvestudio.expressions` parses user text with SymPy's parser in a restricted namespace (whitelist from PRD §10), rejects attribute access, names outside the whitelist and oversized expressions, then lambdifies to NumPy. It yields a `ModelSpec`, so custom models go through exactly the same fitting path as built-ins. A dedicated adversarial test suite (injection strings, `__import__`, huge exponents, deep nesting) gates every release.

**Data.** `Dataset` is immutable. Edits produce an append-only operations log (`DropRow`, `Transform`, `Mask`, …) layered over the untouched original, so every cleaning step is recorded and reversible (PRD §5.3). Validation returns structured issues (non-numeric, missing, infinite, duplicates, constant X, non-positive values for log transforms, invalid weights, too few points).

**Fitting.** One `fit()` entry point, returning an immutable, serializable `FitResult`: parameters, SEs, covariance, correlation, CIs, residuals, fit statistics, df, **optimizer, start values, final values, convergence status and termination reason, seeds, software versions, dataset fingerprint**. Methods: OLS/WLS (closed form), nonlinear LS via LMfit/SciPy (`lm`, `trf`, `dogbox`), robust loss (`soft_l1`, `huber`, `cauchy`), global (`differential_evolution`, `dual_annealing`), derivative-free (`nelder-mead`, `powell`, `bfgs`, `l-bfgs-b`). A compatibility table rejects invalid combinations (e.g. `lm` with bounds) with a clear error rather than silently switching. Multi-start uses a recorded `numpy.random.Generator` seed. A non-converged fit returns `status != "converged"` and the UIs show it prominently; no result is ever substituted.

**Orthogonal distance regression.** SciPy's ODR module is deprecated; before building this, verify its status for SciPy at our pinned version. If it's removed or unsafe, ODR becomes a separately reviewed implementation or a documented deferral. This decision is made in the statistics phase, not now.

**Statistics.** Covariance from the Jacobian at the solution (`s²·(JᵀJ)⁻¹`, weighted variant for WLS), delta-method confidence bands, prediction intervals (adds residual variance — kept distinct from mean-response CIs as the PRD requires), bootstrap and Monte-Carlo, profile likelihood where supported. Information criteria use one documented Gaussian-likelihood convention (`n·ln(SSE/n)` form, parameter count includes error variance); comparisons across different datasets or response transformations raise a warning.

**Diagnostics.** Residual set (vs fitted, vs X, Q-Q, histogram, scale-location, ACF, standardized) as plot-agnostic data; Shapiro-Wilk, Anderson-Darling, Breusch-Pagan, Ljung-Box, Durbin-Watson, Cook's distance, leverage. Every test carries a plain-language assumptions/limitations string; none returns "model valid".

**Model discovery.** Pipeline from PRD §9: validate → select applicable families (domain rules) → initial guesses → fit candidates in a process pool (multi-start for hard ones) → reject non-converged → metrics (AICc, BIC, k-fold CV RMSE, adj. R², residual-quality score, complexity) → configurable ranking with a high-order-polynomial guard. Output is the full ranked table; the user chooses.

**Auto-fit (headline feature).** One action, "Auto-fit", fits every model applicable to the data and returns **all converged fits ordered best to worst**, each row showing rank, model name, **equation with fitted numbers substituted**, **R²**, adjusted R², RMSE, AICc, BIC, CV-RMSE and convergence status; non-converged models are listed separately with their reason, never ranked. Default order is **"Best overall"** (AICc-led composite with CV and a complexity penalty), because ordering by R² alone always favours the model with the most parameters. A one-click column sort lets the user order by R² (or any column) when they want that, and the UI labels the active ordering. Results stream in as models finish, so a first ranking appears quickly; the run is cancellable and seeded for reproducibility. Models are screened by cheap rules first (domain: needs positive x, y monotonic, etc.) and then fitted in parallel with a per-model time limit, so a 300-model run stays interactive on a normal laptop (target to be benchmarked, not promised).

**Project format `.cstudio`.** ZIP with `manifest.json` (schema-versioned, Pydantic-validated; app + dependency versions), original and cleaned datasets as CSV (shortest-round-trip floats), operation logs, fit configs and results as JSON, plot configs. No pickle. Extraction is hardened (path traversal, size and file-count limits, zip-bomb ratio). Opening re-fits nothing automatically but can re-run any recorded analysis and compare to stored results.

### 5.2a Model library scale

Goal: **as many well-founded models as we can verify**, starting from the PRD's families and growing continuously. Three tiers, all declarative `ModelSpec`s sharing one test harness:
1. **Hand-written named models** (target 150+ at Phase 4): saturation, sigmoid/growth (logistic 3/4/5-parameter, Gompertz, Richards, Weibull, von Bertalanffy, Bleasdale-Nelder…), exponential/decay (single, double, triple, stretched), power/log variants, rational functions, peak/distribution shapes (Gaussian, Lorentzian, Voigt, pseudo-Voigt, lognormal, gamma, Weibull, Cauchy, Lévy…), dose-response (Hill, 4PL/5PL), kinetics, Arrhenius/Antoine/Sellmeier-style physics, harmonic/Fourier, and the concrete/engineering set in PRD §15.
2. **Generated families** (hundreds more): polynomial 1–10 (with/without intercept), rational polynomials `P_m(x)/Q_n(x)` for small m, n, inverse and reciprocal polynomials, log/exp/power transforms of x and y (TableCurve-style combinations), sums of 2–3 exponentials/Gaussians/power terms, Fourier series 1–6 terms. Generation is deterministic, each variant gets a stable id, and a parameter-count cap keeps overfit-prone variants out of default Auto-fit runs (opt-in "exhaustive" mode).
3. **User/community models** via custom equations and later plugins.

A model is only shipped once it passes the standard checks: equation matches reference, Jacobian matches finite differences, a seeded synthetic dataset recovers its own parameters, and its validity rules are exercised. This keeps the count honest — a model that cannot be shown to fit its own noiseless data is not added. Counts are tracked in CI so the library size is a measured number, not a claim.

### 5.3 Desktop app (PySide6)

MVC with controllers between Qt widgets and the core. Interactive plots via PyQtGraph; publication export via the shared Matplotlib module. Fits and discovery run in worker processes (`ProcessPoolExecutor`) with progress, cancel and timeout, so the UI never blocks. Layout follows PRD §16.1 (project tree, plot, analysis panel, data table, status bar) with tabbed results (§16.4). Packaging: PyInstaller → Windows installer + portable zip, Linux AppImage. No network access required, none attempted.

### 5.4 API and web app

- **API:** FastAPI + Pydantic, SQLAlchemy 2 (PostgreSQL in deployment; SQLite for tests and single-user mode). Routes under `/api/v1/{projects,datasets,models,fitting,discovery,statistics,predictions,optimization,reports}` plus `/auth`, `/me`, `/billing`. Long calls (fit, discovery, report) create a row in `jobs` and run in a worker process pool with timeouts; clients poll or stream status. Per-user quotas, upload validation and rate limits sit in middleware. The API contains no maths — it calls core.
- **Auth:** email + password with Argon2 hashing and JWT access/refresh tokens (RS256), refresh rotation; OAuth login (Google/GitHub) as an optional addition. AuthZ is enforced in every handler, scoped to the owning user.
- **Web UI:** React + TypeScript + Plotly.js (interactive), Vite, generated TypeScript client from the OpenAPI schema so the contract cannot drift.
- **Plans and billing (hosted service only):** a `plans` table (limits: datasets, rows per dataset, concurrent jobs, storage, discovery runs per day) and `subscriptions`. A provider-adapter interface keeps payments swappable; first adapter is **Razorpay** (INR, India-first) — to be re-verified against the current API at implementation time. Billing is behind `CURVESTUDIO_BILLING_ENABLED` (default **false**). When false, there is a single unlimited "self-hosted" plan, no billing routes, and no calls to any payment provider. Free tier remains useful, since the desktop app is the primary free product; paid tier is priced to cover hosting, not to profit.
- **Operational:** structured JSON logs with `request_id`, `audit_logs` table, health/readiness endpoints, DB backup script.

### 5.5 Self-hosting

`deploy/` ships: `docker-compose.yml` (reverse proxy with automatic HTTPS → web → api → worker → PostgreSQL, optional Redis profile only if later needed), multi-stage Dockerfiles (non-root users, pinned base images), `.env.example`, `Caddyfile`, `backup.sh` / `restore.sh`, healthchecks, a migration step on startup, and `docs/self-hosting.md`. Works on a plain VPS with `docker compose up -d`, and the same compose is Coolify-compatible. A CI job builds the images and runs a smoke test (register → upload → fit) against the compose stack.

### 5.6 Licensing (proposal — see §9)

Core, model_library, reporting, plotting, engineering: **MIT** (reusable libraries, per PRD §24). Desktop, API and web: **AGPL-3.0** (owner's decision, 16/11/2026). For the desktop app AGPL behaves like GPL-3.0 for distributed copies; its extra network clause only matters if someone offers the app as a hosted service, and PySide6 (LGPL) and the BSD scientific stack are compatible with it. Docs: CC BY 4.0. Open question for the owner: the MIT libraries can be embedded in closed products; switch them to AGPL too if that is not wanted.

## 6. Testing and scientific verification

- **Unit:** every model equation, Jacobian (against finite differences), bounds, initial guess, validation rule — one parametrised test generated from each `ModelSpec`.
- **Recovery:** synthetic data with known parameters and seeded noise; recovered parameters within stated tolerances (PRD §20).
- **Certified references:** the NIST StRD nonlinear-regression datasets (certified parameter values and SEs, with "lower/average/higher" difficulty and two start vectors) are the primary correctness benchmark for the fitting engine; linear datasets (e.g. Longley, Filip noted as hard) for OLS. Failures to converge on hard starts must be reported as non-convergence, never as success.
- **Cross-checks:** selected statistics compared to Statsmodels/R-computed values stored as fixtures.
- **Golden regression:** recorded results for fixed seeds; a change needs an explicit tolerance-aware update.
- **Security:** expression-parser fuzz and injection corpus, zip-slip/zip-bomb cases, upload limits, authZ tests per handler.
- **Integration/E2E:** CSV/XLSX → fit; project save → reload → identical numbers; API fit job; web smoke through compose; desktop UI tests with `pytest-qt`; PDF report generation; Linux + Windows CI matrix.
- **Performance:** benchmark at 100 / 1k / 10k / 100k points, documented hardware.

## 7. Delivery plan

Phases follow PRD §22, reordered so the first usable app lands early. Each phase gets its own implementation plan in `docs/superpowers/plans/`, written just before it starts, so plans never go stale. Every task ends with tests green, a commit and a push; each phase ends with a review gate.

| # | Phase | Deliverable | Gate |
|---|---|---|---|
| 0 | Foundation | uv workspace, CI, lint/type/test config, import-linter, CONTRIBUTING/CoC/templates, licences | CI green on Linux+Windows |
| 1 | Data | Import (CSV/TSV/TXT/JSON/XLSX/clipboard), `Dataset`, ops log, validation | Round-trip + validation tests |
| 2 | Fitting core | `ModelSpec`, registry, OLS/WLS/polynomial, nonlinear fit, `FitResult`, NIST StRD harness | NIST pass-rate documented |
| 3 | Statistics | Covariance, CIs, prediction bands, diagnostics, tests, IC/CV, bootstrap | Cross-check fixtures |
| 4 | Model library + expressions + Auto-fit engine | 150+ named models plus generated families; safe equation builder; multi-start; ranked Auto-fit with equations and R² | Every model recovers its params; Auto-fit ranks a known-truth dataset's true model in top 3 |
| 5 | Desktop MVP | Shell, data table, plot, fit panel, results tabs, PNG/SVG/CSV export, `.cstudio` save/load | **MVP gate (PRD §3.1)** |
| 6 | Discovery + visualization + reports | Discovery/ranking UI, residual plots, bands, PDF/XLSX/JSON reports, prediction studio | Alpha→Beta |
| 7 | Web + self-host | API, auth, jobs, React UI, plans/billing (flagged), Docker Compose, docs | RC |
| 8 | Hardening + release | Security tests, packaging (installer/AppImage), docs site, v1.0 | v1.0 |
| 9+ | Advanced | Multivariable/RSM/regularised, optimization studio, engineering libraries, global fitting | Post-1.0 roadmap |

Engineering (concrete) models begin as part of Phase 4's model library as an optional package, since they are the project's niche; each carries documented assumptions and is never presented as an IS-code acceptance equation (PRD §15).

## 8. Left to the builder

Concrete class/function signatures, exact SQL schemas and migrations, UI widget details, colour/theme (follows the owner's design tokens), the specific initial-guess heuristic per model, Plotly vs ECharts details, MkDocs theme, and the CI matrix mechanics. The builder confirms current latest-stable versions and advisories for each dependency when it is first added.

## 9. Decisions to confirm

These are my proposals; the first two change what is published, so I want explicit approval.
1. **Licence split** (MIT libraries / AGPL-3.0 desktop, API and web). Desktop AGPL is now your decision; still open: should the core libraries be MIT (embeddable anywhere) or AGPL?
2. **Billing provider:** Razorpay first, behind an adapter. Alternatives (Stripe, Lemon Squeezy) can be added later.
3. **Project extension:** `.cstudio` (the PRD's `.clab` predated the name).
4. **Quota defaults** for the hosted free/paid tiers are deferred to Phase 7 and priced against real hosting cost.
