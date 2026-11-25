# Home page claims

Every factual claim on the home page (`HomePage.tsx`) and what backs it. Paths are
relative to the repository root unless they start with `apps/web/`, which are relative
to `apps/web/`. Section rendering itself is pinned by `HomePage.test.tsx`.

## Hero

| Claim | Backing |
|---|---|
| "VALIDATED ON NIST STRD DATASETS" | `packages/core/src/curvestudio/testing/nist.py`; the harness runs in `tests/numerical/test_nist_linear.py`, `test_nist_nonlinear.py`, `test_nist_multistart.py`, `test_nist_report.py` |
| "fits every applicable model to your data, ranks them on the evidence" | `packages/core/src/curvestudio/discovery/engine.py`, `candidates.py` (candidate enumeration), `ranking.py` (AICc/BIC order) |
| "parameter uncertainty" | `packages/core/src/curvestudio/fitting/_uncertainty.py`; `statistics/bands.py`, `bootstrap.py`, `montecarlo.py` |
| "residual diagnostics you can defend" | `packages/core/src/curvestudio/diagnostics/hypothesis_tests.py`, `residuals.py`, `report.py` |

The hero eyebrow and sub-copy text are pinned verbatim (sub-copy prefix) by the
"opens with the research hero copy" test in `HomePage.test.tsx`.

## Trust strip

Pinned by the "keeps the trust strip pills and stats" test in `HomePage.test.tsx`.

| Claim | Backing |
|---|---|
| "NIST StRD reference sets" | As hero above |
| "~200 models" | The catalogue in `packages/model_library/src/curvestudio_models/` (`generated.py` plus `families/`). **Not yet pinned:** the count is pinned against the real catalogue by a test in `apps/web/src/api` that lands with Task 3 |
| "AICc / BIC ranking" | `packages/core/src/curvestudio/discovery/ranking.py` |
| "Confidence & prediction bands" | `packages/core/src/curvestudio/statistics/bands.py` |
| "AGPL-3.0" | `LICENSE` at the repository root |
| "Offline desktop app" | `apps/desktop/src/curvestudio_desktop/` (PySide6 app; the `curvestudio-core` engine runs in process, no server round trip) |
| "174 candidates ranked in one run" | `packages/core/src/curvestudio/discovery/candidates.py` (`enumerate_candidates`) feeding `engine.py`. **Not yet pinned:** the exact candidate count for the default plan is pinned by the same Task 3 test as the model count |
| "~200 models in the library" | As "~200 models" above |
| "6 export formats" | CSV and JSON: `apps/web/src/workspace/results/PredictTab.tsx` and the autofit panel, served by `apps/api/src/curvestudio_api/export.py`. PDF report: `apps/web/src/lib/report/export.ts`. PNG plot: `apps/web/src/components/PlotChart.tsx` (`downloadPlotPng`). XLSX report: `packages/reporting/src/curvestudio_reporting/xlsx_report.py` (desktop today). SVG plot: `packages/plotting/src/curvestudio_plotting/figures.py` (desktop today). **Timing note:** the web app gains the server-side XLSX report and SVG plot in Task 3, and Task 3 also lands the test that pins this count. Until then the count is six on the desktop and four live formats on the web |

## Who it is for

No numerical claims. The five cards and their bodies are pinned by the
"renders the who-it-is-for section with exactly five persona cards" test.

| Claim | Backing |
|---|---|
| "Fit sensor, materials and reliability data on your own machine" | The desktop app, `apps/desktop/` (`curvestudio-core` workspace dependency in `apps/desktop/pyproject.toml`) |
| "a high R² is not the end of the argument" | `discovery/ranking.py` reports adjusted R², AICc and BIC per candidate, not R² alone |

## Research requirements

The eight rows are pinned by the "renders the eight research requirements as
two-column rows" test.

| Requirement | Backing |
|---|---|
| Choose a model on evidence | `discovery/ranking.py` (AICc, BIC, adjusted R², cross-validated error as tie-breaker; models that draw the same curve share a rank); candidates from `discovery/candidates.py` |
| Report uncertainty honestly | `fitting/_uncertainty.py` (parameter standard errors, 95 % confidence intervals); `statistics/bands.py` (confidence and prediction bands); `statistics/bootstrap.py`; `statistics/montecarlo.py` |
| Show that the fit hides nothing | `diagnostics/hypothesis_tests.py` (Shapiro–Wilk, Anderson–Darling, Durbin–Watson, Ljung–Box, Breusch–Pagan); `diagnostics/residuals.py` (leverage and influence, flagged in `diagnostics/report.py`) |
| Trust the arithmetic | `testing/nist.py` against the certified StRD datasets (`tests/numerical/test_nist_*.py`); a fit that does not converge is never ranked as a plausible answer: `discovery/result.py` keeps `not_converged`, `failed`, `timed_out` and `skipped` outcomes out of the ranking and lists them with reasons |
| Use your own equation | `packages/core/src/curvestudio/expressions/grammar.py` (custom models with named parameters, start values and bounds) |
| Account for measurement error | Weighted least squares: `packages/core/src/curvestudio/fitting/api.py` (`data.weights` multiply the squared residuals) |
| Reproduce it next year | Desktop project files: `packages/core/src/curvestudio/project/model.py` and `archive.py` (the `.cstudio` archive carries the datasets, the recorded data operations, the fit configuration, the stored seeds, and a manifest with `app_version` and `versions`). The row says "Desktop project files" for exactly this reason |
| Publish and archive | Reports: `packages/reporting/src/curvestudio_reporting/` (`csv_report.py`, `json_report.py`, `xlsx_report.py`, `pdf_report.py`); plots: `packages/plotting/src/curvestudio_plotting/figures.py` and `apps/web/src/lib/report/export.ts`. **Timing note:** XLSX and SVG are live on the web from Task 3, as above |

## Use cases

Each discipline is pinned by the "replaces the concrete use cases with the seven
research ones" test; each maps to families that exist in the catalogue today.

| Use case | Backing |
|---|---|
| Reaction kinetics | `packages/model_library/src/curvestudio_models/families/kinetics.py` |
| Enzyme kinetics | `families/saturation.py` (the Michaelis–Menten family; the equation grid on the page shows `y = Vmax·S/(Km+S)`) |
| Growth curves | `families/growth.py` |
| Calibration and drift | Straight-line and polynomial fits: `packages/core/src/curvestudio/fitting/linear.py` (`fit_polynomial`) plus the generated linear-in-x families in `generated.py` |
| Spectra and chromatograms | `families/peaks.py` plus the generated `gauss_sum` sum-of-peaks models (`generated.py`) |
| Dose–response | `families/dose_response.py`, `families/sigmoid.py` |
| Teaching-lab physics | `families/physics.py`, `families/exponential.py`, `families/harmonic.py` (`harmonic_damped_sine` for damped oscillations) |

## Model gallery and statistics sections

| Claim | Backing |
|---|---|
| "The values below are illustrative runs, not live data" | True by construction: the gallery is hand-authored deterministic SVG (`FIT_MODELS` in `HomePage.tsx`), no engine in the browser; the "renders the six reproducible-fit chips" test pins the chips, equations and curve geometry |
| Equations shown are real library forms | Exponential and logistic families, Michaelis–Menten (`saturation.py`), Gaussian (`peaks.py`), power law (`power_log.py`), quadratic (`fitting/linear.py`) |
| The fit-statistics table and ΔAICc chart are an example, explained as such | Same illustrative construction; the "renders the five-model AICc comparison" test pins the bars |
| "174 candidates ranked in one run. AICc sets the order, not a hunch." | As trust strip above |
| "The export keeps the full ranking" | `apps/api/src/curvestudio_api/export.py` (`autofit_csv`) |

## Statistics

The section `#statistics` (the fit audit, the methods text and the list of the twenty-nine
statistical tools) is pinned by the "renders the Statistics section with its three sentences
and the record of the check" and "lists the twenty-nine statistical tools by family" tests in
`HomePage.test.tsx`. The test "claims no absence in the Statistics section" asserts that the
section, and the page's source, say none of "nowhere else", "unique" or "only tool".

| Claim | Backing |
|---|---|
| "After every fit, a fit audit checks the evidence for you: each check gets a verdict, the numbers behind it and what to do next." | `packages/core/src/curvestudio/tools/audit.py` (`audit_fit` runs `CHECKS` in a fixed order and returns each with its verdict, the statistic and threshold it turned on, its evidence and its next step); `GET /fits/{fit_id}/audit` in `apps/api/src/curvestudio_api/routers/fits.py`; the cross-validation check the Fit audit tab can queue is `POST /fits/{fit_id}/audit/cross-validation` in the same router |
| "A methods text writes the fit up for you: the model, the estimator, the estimates with their standard errors and the citations, as plain text, Markdown or LaTeX." | `packages/core/src/curvestudio/tools/methods.py` (`methods_text` and `all_methods_texts`: the paragraph, the equation, the table and the citations, in the three styles); `GET /fits/{fit_id}/methods` in the same router |
| "We looked for these two in the public documentation of CurveExpert and CurveExpert Pro, Origin, GraphPad Prism, LabPlot, SciDAVis, QtiPlot, jamovi and JASP in November 2026, and did not find either offered in its own right; those tools offer residual plots and tests, rankings, pairwise comparisons and equation export." | `docs/prior-art.md`, checked on 2026-11-24. No cell of its summary reads "Has" for fit audit or methods text in any of the nine products, and its "What this check supports" section names the nearest neighbours the last clause lists |
| "How we checked" (the link) | `docs/prior-art.md` on the repository's `main` branch, under the repository address the footer's "CurveStudio source" link uses |
| "Twenty-nine statistical tools, from summarising the columns of a dataset to planning the sample a study needs, compare means and ranks, trace relationships, count associations, compare two methods, check a score's discrimination, a scale's consistency and a process's stability, combine the effects of several studies, weigh a hypothesis against its alternative under a stated prior, and model grouped or time-to-event data — in the web app, the desktop app and the command line." | `packages/core/src/curvestudio/tools/stats/registry.py` (`FAMILIES` names the eleven families in the order the page lists them; each registered tool carries its title, and every tool but sample size and power declares `needs_data=True`). The web runs them in `apps/web/src/workspace/stats/StatsPanel.tsx` and keeps every result in `StatsHistory.tsx`; the desktop has a Statistics page in `apps/desktop/src/curvestudio_desktop/views/stats_view.py`; the command line is `packages/core/src/curvestudio/cli.py` (`curvestudio stats <tool>`, one subcommand per catalogue tool) |
| The eleven family titles and twenty-nine tool titles, in the catalogue's order | `packages/core/src/curvestudio/tools/stats/registry.py` (`FAMILIES`, and each tool spec's `title`); pinned against the page by the "lists the twenty-nine statistical tools by family" test in `HomePage.test.tsx` |

The tools list names the standard set, not a differentiating capability, so it carries no
prior-art claim: `docs/prior-art.md` covers only the five differentiating capabilities. The
list says what the tools are for and where they run, and says nothing about what other
products offer.

The six tools added in this batch — Principal component analysis, Factor analysis and
Logistic regression under Relationships between columns, and Receiver operating
characteristic curve, Reliability (Cronbach's alpha) and Control chart and process
capability under the new Measurement and classification family — are listed as capabilities
with no absence claim. `docs/prior-art.md` has no row for any of the six, and none was added:
the check there holds a claim the page makes, not a tool the page lists.

The four tools added in this batch — Linear mixed model and Survival analysis under the new
Advanced models family, Meta-analysis under the new Evidence synthesis family, and Bayesian
test under the new Bayesian tests family — are likewise listed as capabilities with no absence
claim. `docs/prior-art.md` has no row for any of the four, and none was added.

The third sentence is worded as found, never as proof, because the record says its check is
incomplete in four places:

- LabPlot: an earlier handbook page describing the fit dialog (its residuals tab and parameter
  confidence intervals) could not be reached; the old address redirects to the home page.
- SciDAVis: its results log is documented only by a screenshot.
- Origin: no version stated on the help pages read.
- QtiPlot: no version stated.

Before the sentence is changed, re-read `docs/prior-art.md`: its date is the date of the
check it records, and a later capability repeats the check for itself first.

## Desktop and web

| Claim | Backing |
|---|---|
| "Runs completely offline" | `apps/desktop/` runs locally against the in-process engine |
| "Same fitting engine" | Both apps depend on `curvestudio-core` (`apps/api/pyproject.toml`, `apps/desktop/pyproject.toml`) |
| "Your data stays in your account", "Nothing is sold or shared for advertising" | The privacy policy on `/about` (`AboutPage.tsx`, "Privacy and fair-use policy"); `apps/web` ships no third-party analytics or advertising scripts |

## More from ConcreteInfo

| Claim | Backing |
|---|---|
| The five entries, their summaries and links | `apps/web/src/content/other-projects.json`, which `src/content/other-projects.test.ts` pins byte-identical to the canonical `docs/other-projects.json`; rendering (open-source group first) pinned by the "renders the More from ConcreteInfo section" test |

## FAQ

| Claim | Backing |
|---|---|
| "Export the numbers as CSV, JSON, XLSX or PDF and the plots as PNG, SVG or PDF" | As "Publish and archive" above, with the same Task 3 timing note for web XLSX/SVG |
| "A desktop project file records the full analysis" | `packages/core/src/curvestudio/project/model.py` and `archive.py` |
| "Around 200 forms" | As trust strip above (Task 3 pin test_api_platform.py::test_catalogue_size_backs_the_home_page_claim (200 models)) |
| "The desktop app runs fully offline" | As "Runs completely offline" above |
| "CurveStudio is AGPL-3.0 open source" | `LICENSE` at the repository root |

## Footer

| Claim | Backing |
|---|---|
| Contact `mailto:amit@concreteinfo.in` | Owner-provided address; nothing to test beyond the link itself (pinned by the footer test) |
| Privacy `/about` | `/about` renders the privacy policy (`AboutPage.tsx`, "Privacy and fair-use policy" section). **Note:** that section has no HTML anchor yet, so the link opens the page rather than the section; add an `id` there to deep-link |
| Open source column | CurveStudio source plus the `open-source` entries of `other-projects.json`, pinned by the two footer tests in `HomePage.test.tsx` |

## Claims that are not yet fully backed

- "174 candidates" is the gallery demo run's candidate count and has no pinning test.
- "6 export formats", the requirements row "Publish and archive" and the FAQ export
  answer count XLSX and SVG, which the web app serves only after Task 3 wires the
  server-side export; the desktop already offers both today. Task 3 has landed, so
  both formats are live; this note stays until the deploy that carries them.
- The Privacy footer link targets the page, not a section anchor, because `/about`
  has no anchor on its privacy section yet.
