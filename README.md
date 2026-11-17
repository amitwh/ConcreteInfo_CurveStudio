# CurveStudio

**Free, open-source curve fitting and engineering analysis. An alternative to CurveExpert.**

[![CI](https://github.com/amitwh/ConcreteInfo_CurveStudio/actions/workflows/ci.yml/badge.svg)](https://github.com/amitwh/ConcreteInfo_CurveStudio/actions/workflows/ci.yml)
[![Licence: AGPL v3+](https://img.shields.io/badge/licence-AGPL--3.0--or--later-blue.svg)](LICENSE)

CurveStudio fits mathematical models to experimental data, ranks the candidates from best to worst with their equations and R², and reports the statistics you need to judge a fit. It is built for researchers, engineers, laboratories and students, with a particular focus on civil and concrete-materials engineering.

> [!WARNING]
> **Status: pre-alpha. Phase 4 complete.** The foundation, data layer, fitting core (linear, polynomial and nonlinear least squares), statistics and diagnostics (bands, information criteria, residual diagnostics, tests, cross-validation, bootstrap, Monte-Carlo propagation, profile likelihood), the model library and **Auto-fit** are built: 168 hand-written models, 100 default generated model ids (346 with `--exhaustive`), 31 optional concrete and civil-engineering models, and a safe custom-equation builder. Auto-fit is available from the command line today (`curvestudio autofit`, see [Auto-fit quickstart](#auto-fit-quickstart)); the core is validated against the NIST StRD benchmarks and Statsmodels cross-checks, and every library model recovers its own parameters from noise-free data ([`tests/numerical/LIBRARY.md`](tests/numerical/LIBRARY.md)). The desktop app and the web app follow phase by phase (see [Roadmap](#roadmap)). Nothing here is ready for real analysis yet.
>
> **Numerical validation:** nonlinear single-start fits against the 12 NIST problems converge correctly on 23 of 24 starts (the other, MGH09 start 1, reports NOT_CONVERGED rather than a wrong answer), with zero wrong-but-CONVERGED results; linear problems pass 5/5. Full tables: [`tests/numerical/VALIDATION.md`](tests/numerical/VALIDATION.md).

## What it will do

- **Auto-fit.** One click fits every applicable model and lists the converged fits from best to worst, each with its equation (fitted numbers substituted), R², adjusted R², RMSE, AICc, BIC and cross-validation error. You choose the ordering: our recommended "Best overall" (which guards against overfitting) or plain R².
- **Honest fitting.** A fit that does not converge is reported as failed, never replaced by a plausible-looking result. Results are checked against NIST certified reference datasets.
- **Large model library.** 150+ named models plus generated families (polynomial, rational, sums of exponentials and peaks, and more), and a growing set of concrete and civil-engineering models.
- **Custom equations.** Type your own equation. It is parsed into a restricted expression tree, never run through `eval`.
- **Full statistics.** Parameter standard errors and confidence intervals, confidence and prediction bands, residual diagnostics and bootstrap uncertainty.
- **Reproducible projects.** Portable `.cstudio` files record the original data, every cleaning step, the fit settings, random seeds and software versions.
- **Desktop and web.** A native offline desktop app (Windows and Linux) and a browser app share one engine and give the same numbers.

## What exists today

| Area | State |
|---|---|
| Workspace, licence, CI on Linux and Windows, architecture guard | Done |
| Immutable `Dataset` with a reversible, fingerprinted operations log | Done (in review) |
| Validation, importers (CSV, TSV, Excel, clipboard) | In progress |
| Fitting engine, model library, desktop and web apps | Not started |

## Free, and self-hostable

- The **desktop app is free** and works fully offline, with no accounts or telemetry.
- The **hosted web app** may carry a very small subscription to cover hosting costs.
- **Self-hosting** is a one-command Docker Compose stack - see [`deploy/README.md`](deploy/README.md). Billing is off by default, so a self-hosted install has no limits and never contacts a payment provider.

## Architecture

```text
apps/desktop (PySide6)      apps/web (React) ── apps/api (FastAPI)
            \                         |
             +-------- packages/ ----+
   core · model_library · engineering · reporting · plotting
```

The scientific engine in `packages/core` never imports Qt, FastAPI, SQLAlchemy or React. Two independent checks enforce this in CI (an AST test and import-linter), so desktop and web always run identical maths.

Design principles:

- **Core first.** The engine is a plain Python library with a CLI; the apps are thin shells over it.
- **Declarative models.** A model is data (equation, parameters, initial guesses, references), parsed with SymPy into a whitelisted expression with an analytic Jacobian.
- **Immutable data.** Cleaning never edits the original data. It appends operations to a log that can be undone and is fingerprinted with SHA-256.
- **Fail loudly.** Missing values are reported with their row numbers, never silently dropped.

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Foundation: workspace, CI, licence, community files | Done |
| 1 | Data: import, validation, reversible cleaning | Done |
| 2 | Fitting core: linear, polynomial, nonlinear, NIST validation, CLI | Done |
| 3 | Statistics and diagnostics | Done |
| 4 | Model library, custom equations, Auto-fit engine | Done |
| 5 | Desktop MVP | Planned |
| 6 | Model discovery UI, visualization, reports | Planned |
| 7 | Web app, accounts, self-hosting | Planned |
| 8 | Hardening and v1.0 release | Planned |

## Auto-fit quickstart

`curvestudio autofit` fits every model the data can support and lists the converged fits from best to worst, each with its fitted equation, R², adjusted R², RMSE, AICc, BIC and cross-validation RMSE:

```bash
uv run curvestudio autofit examples/decay.csv --x x --y y --top 5
uv run curvestudio autofit examples/decay.csv --x x --y y --rank-by r2   # plain R², with the overfitting warning
uv run curvestudio autofit examples/decay.csv --x x --y y --json > result.json
```

```text
Ranked by: best overall (lowest AICc first; models within 2 AICc are ordered by cross-validation, then fewer parameters)
Rank  Model                   R²   Adj. R²       RMSE      AICc       BIC    CV-RMSE  k
   1  exp_decay_offset  0.996672  0.996603  0.0751714  -509.176  -499.176  0.0772958  3
      y = 5.00294*exp(-0.386077*x)+1.95385
```

The table is followed by the candidates that were skipped (with the reason: for example non-positive x for a power model), did not converge, or fit the data exactly; none of them is ranked or hidden. Rows that draw the same curve share a rank and name each other. Options: `--weights`, `--rank-by {best_overall,r2,adj_r2,rmse,aicc,bic,cv_rmse}`, `--top`, `--workers`, `--n-starts`, `--seed`, `--time-limit`, `--exhaustive`. Exit code 0 means the run completed, 1 that it could not run (for example `TOO_FEW_POINTS`). `curvestudio fit` remains for fitting one chosen model. Measured run times and the known-truth gate are in [`tests/numerical/AUTOFIT.md`](tests/numerical/AUTOFIT.md).

## Quickstart (for contributors)

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:amitwh/ConcreteInfo_CurveStudio.git
cd ConcreteInfo_CurveStudio
uv sync
uv run pytest -m "not slow"
uv run ruff check .
uv run ruff format --check .
uv run mypy packages/core/src
uv run lint-imports
```

## Configuration

There is no runtime configuration yet. When the web app lands (Phase 7), its environment variables will be documented here and in `deploy/.env.example`.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the licence of contributions and the model-contribution rules (every model needs its equation, parameter definitions, references, tests and stated assumptions). Community expectations are in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and security reports follow [SECURITY.md](SECURITY.md).

## Licence

Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See [LICENSE](LICENSE). In short: you may use, study, change and share CurveStudio freely, and if you run a modified version as a network service you must offer its source to the users of that service.
