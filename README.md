# CurveStudio

**Free, open-source curve fitting and engineering analysis. An alternative to CurveExpert.**

CurveStudio fits mathematical models to experimental data, ranks the candidates from best to worst with their equations and R², and reports the statistics you need to judge a fit. It is built for researchers, engineers, laboratories and students, with a particular focus on civil and concrete-materials engineering.

> [!WARNING]
> **Status: pre-alpha.** The repository currently contains the design and the project scaffold. The fitting engine, desktop app and web app are being built phase by phase (see [Roadmap](#roadmap)). Nothing here is ready to use for real analysis yet.

## What it will do

- **Auto-fit:** one click fits every applicable model and lists the converged fits best to worst, each with its equation (fitted numbers substituted), R², adjusted R², RMSE, AICc, BIC and cross-validation error. You choose the ordering: our recommended "Best overall" (which guards against overfitting) or plain R².
- **Honest fitting:** a fit that does not converge is reported as failed, never replaced by a plausible-looking result. Results are checked against NIST certified reference datasets.
- **Large model library:** 150+ named models plus generated families (polynomial, rational, sums of exponentials and peaks, and more), and a growing set of concrete and civil-engineering models.
- **Custom equations:** type your own equation; it is parsed safely into a restricted expression tree (never `eval`).
- **Full statistics:** parameter standard errors and confidence intervals, confidence and prediction bands, residual diagnostics, bootstrap uncertainty.
- **Reproducible projects:** portable `.cstudio` project files that record the original data, every cleaning step, the fit settings, the random seeds and the software versions.
- **Desktop and web:** a native offline desktop app (Windows and Linux) and a browser app that share the same engine and give the same numbers.

## Free, and self-hostable

- The **desktop app is free** and works fully offline, with no accounts or telemetry.
- The **hosted web app** may carry a very small subscription to cover hosting costs.
- **Self-hosting** is supported with Docker Compose, and billing is switched off by default, so a self-hosted install has no limits and never contacts a payment provider. Deployment scripts will live in `deploy/` (Phase 7).

## Architecture

```text
apps/desktop (PySide6)      apps/web (React) ── apps/api (FastAPI)
            \                         |
             +-------- packages/ ----+
   core · model_library · engineering · reporting · plotting
```

The scientific engine in `packages/core` never imports Qt, FastAPI or React, and a CI check enforces that, so desktop and web always run identical maths. Read the full design in [`docs/superpowers/specs/2026-11-16-curvestudio-design.md`](docs/superpowers/specs/2026-11-16-curvestudio-design.md) and the product requirements in [`docs/intent.md`](docs/intent.md).

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Foundation: workspace, CI, licence, community files | Done |
| 1 | Data: import, validation, reversible cleaning | Planned |
| 2 | Fitting core: linear, polynomial, nonlinear, NIST validation, CLI | Planned |
| 3 | Statistics and diagnostics | Planned |
| 4 | Model library, custom equations, Auto-fit engine | Planned |
| 5 | Desktop MVP | Planned |
| 6 | Model discovery UI, visualization, reports | Planned |
| 7 | Web app, accounts, self-hosting | Planned |
| 8 | Hardening and v1.0 release | Planned |

The detailed plan for Phases 0 to 2 is in [`docs/superpowers/plans/`](docs/superpowers/plans/2026-11-16-phase-0-2-foundation-data-fitting-core.md).

## Quickstart (for contributors)

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:amitwh/ConcreteInfo_CurveStudio.git
cd ConcreteInfo_CurveStudio
uv sync
uv run pytest
uv run ruff check .
uv run mypy packages/core/src
```

## Configuration

There is no runtime configuration yet. When the web app lands (Phase 7), its environment variables will be documented here and in `deploy/.env.example`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the licence of contributions and the model-contribution rules (every model needs its equation, parameter definitions, references, tests and assumptions), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. Security reports go through [SECURITY.md](SECURITY.md).

## Licence

Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See [LICENSE](LICENSE). In short: you may use, study, change and share CurveStudio freely, and if you run a modified version as a network service you must offer its source to the users of that service.
