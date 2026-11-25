# Prior art: five curve-fitting and statistics capabilities in nine products

Checked on 2026-11-24 against public documentation. "Has" means the documentation describes the capability itself; "Part" means a neighbouring capability, named in the notes; "Not found" means nothing of the kind was found in the documentation read, which is not proof that the product lacks it; "Not checked" means the documentation could not be reached or searched properly.

Method: a read-only research pass over the vendors' own manuals, documentation sites, feature lists and release notes, found by web search and read in full; nothing was installed or signed up for. One of the cited pages (CurveExpert Professional's "Working with Results") was opened again independently and says what is attributed to it here. To repeat the check, start from each product's index or contents page named below.

What this record is for: the home page may say only what this check supports, worded as found. "Not found" is not "does not exist", so the page names the products that were checked and what they offer instead, with the date.

## Summary

| Product | Version checked | 1. Fit audit | 2. Methods text | 3. Discernibility report | 4. Feasibility simulator | 5. Next-measurement advisor |
|---|---|---|---|---|---|---|
| CurveExpert | Basic 2.2.3 | Part | Part | Part | Not found | Not found |
| CurveExpert Pro | 2.7.3 | Part | Part | Part | Not found | Not found |
| Origin | Not stated | Part | Part | Part | Part | Part |
| GraphPad Prism | 11 (guide); notes to 11.1.0 | Part | Part | Part | Part | Not found |
| LabPlot | 2.12 | Part | Not found | Not found | Not found | Not found |
| SciDAVis | 1.23 and newer | Part | Not found | Not found | Not found | Not found |
| QtiPlot | Not stated | Part | Not found | Not found | Not found | Not found |
| jamovi | 28.18 | Not found | Not found | Not found | Not found | Not found |
| JASP | 0.99.0 | Not found | Part | Not found | Not found | Part |

## Findings by product

### CurveExpert (Basic 2.2.3; pages read: contents, index, Working with Results, product page)
Index searched: https://www.curveexpert.net/docs/curveexpert/basic/html/genindex.html
- **Fit audit:** Part — only a colour code for converged and non-converged fits; no residual tests or advice. "A yellow color coding indicates that the result should only be used with caution." https://www.curveexpert.net/docs/curveexpert/basic/html/results.html
- **Methods text:** Part — a copyable numeric report and parameter list; no methods paragraph, version or citations. "copy a report about the selected result(s) to the clipboard" (same page)
- **Model discernibility report:** Part — models are ranked by score and the best is chosen; nothing on what would separate them. "letting CurveExpert Basic compare your data to each model to choose the best curve" https://www.curveexpert.net/products/curveexpert-basic
- **Experiment feasibility simulator:** Not found — manual index and contents list no simulation, Monte Carlo, bootstrap, sample size or power entry.
- **Next-measurement advisor:** Not found — manual index and contents list no design or optimal-design entry.

### CurveExpert Pro (2.7.3; pages read: contents, index, Working with Results, CurveFinder, Calculating Results, "Reporting on Results" article)
Index searched: https://www.curveexpert.net/docs/curveexpert/pro/html/genindex.html
- **Fit audit:** Part — residual plot with trend line, a runs test with a verdict, and a non-convergence colour code; no per-check audit. "CurveExpert Professional states that the run pattern of the residuals is unlikely" https://www.curveexpert.net/docs/curveexpert/pro/html/results.html
- **Methods text:** Part — formatted numeric report (print, HTML, clipboard); no methods paragraph. "This report can be printed directly, or can be exported to HTML." https://www.curveexpert.net/curveexpert-professional-helps-you-focus-on-results/
- **Model discernibility report:** Part — ranking by score and a pairwise AIC/F-test comparison; nothing on indistinguishable models or separating data. "The better of the two fits, as judged by these two tests, is shown at the top of the report" (results page above)
- **Experiment feasibility simulator:** Not found — index lists no simulation, Monte Carlo, bootstrap, sample size or power; only a Python sandbox.
- **Next-measurement advisor:** Not found — index lists no design or optimal-design entry.

### Origin (version not stated on help pages; app pages give minimum Origin 2027; pages read: Residual Plot Analysis, NLFit Settings tab, Fit Status Code, Rank Models, comparison algorithm, three app pages)
- **Fit audit:** Part — up to six residual plots, parameter dependency and a solver "Fit Status" with a suggested remedy; no residual, influence or cross-validation checks. "Try simplifying the function, or fixing several parameter values." https://docs.originlab.com/origin-help/nlfit-result-fitstatuscode
- **Methods text:** Part — customisable report tables and an "Equation with Values" note on the graph; no methods paragraph. "select Equation with Values for Equation in Notes" https://docs.originlab.com/quick-help/add-equations-to-graph
- **Model discernibility report:** Part — F-test, AIC with Akaike weights and BIC; Rank Models names the best fit only. "Results can be ranked by Akaike (AIC) and Bayesian Information Criterion (BIC) scores." https://docs.originlab.com/origin-help/fitrank-multi-func
- **Experiment feasibility simulator:** Part — official Monte Carlo app propagates distributions through a formula; a Bootstrap app covers column statistics and multiple linear regression. "Simulate your process by supplying a transform equation and distribution parameters for input variables." https://www.originlab.com/fileExchange/details.aspx?fid=1076
- **Next-measurement advisor:** Part — official factor-based DOE app with an optimal-design tool; nothing for curve fits. "Select optimal design subsets, augment designs with additional experimental runs, and evaluate design efficiency." https://www.originlab.com/fileExchange/details.aspx?fid=462

### GraphPad Prism (guide titled Prism 11; release notes to 11.1.0; pages read: Diagnostics tab, Interpreting results, both analysis checklists, Monte Carlo pages, notes 10.2.0–11.0.0)
- **Fit audit:** Part — runs, replicates, normality and weighting tests with P values, residual plots and an "Ambiguous" flag; the advice is in the guide, not the software. "here are some questions to ask yourself as you evaluate the fit" https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_analysischeck_nonlin_completed.htm
  - A "Narrative results" tab is mentioned only for two-way ANOVA and simple linear regression: https://www.graphpad.com/updates/prism-10-5-0-release-notes
- **Methods text:** Part — results export as text, CSV or XML and the TeX equation can be copied; no methods paragraph. "Allowed copying of TeX representation of equations from Prism to other apps" https://www.graphpad.com/updates/prism-10-3-0-release-notes
- **Model discernibility report:** Part — pairwise F-test or AICc, with a preferred model and the probability each is correct. "Finally, Prism reports the 'preferred' model." https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_analysischeck_nonlin_compare.htm
- **Experiment feasibility simulator:** Part — nearest found. Simulate XY Data takes a model, parameter values, X values and scatter; Monte Carlo then refits many data sets and tabulates chosen results and a hit fraction. Bias and interval coverage are not reported. "This analysis will repeat the simulations many times, and tabulate selected results." https://www.graphpad.com/guides/prism/latest/statistics/stat_how_to_monte_carlo_analyses.htm
- **Next-measurement advisor:** Not found — release notes 10.2.0–11.0.0 and the Monte Carlo pages, searched for "design" and "optimal", show power tools for tests only.

### LabPlot (2.12 manual and 2.12 release post; pages read: contents, index, Curve Fitting, Fitting, LaTeX Typesetting)
Contents searched: http://docs.labplot.org/en/
- **Fit audit:** Part — goodness-of-fit statistics per fit (SSE, RMSE, R², adjusted R²); no checks or verdicts. "LabPlot calculates several statistical measures that help to estimate the `goodness of the fit`." http://docs.labplot.org/en/data_analysis/data_analysis_curve_fitting.html
- **Methods text:** Not found — contents and index have no report, methods or citation entry; LaTeX typesetting renders worksheet labels.
- **Model discernibility report:** Not found — index has no AIC, F-test or model-comparison entry; contents have no comparison section.
- **Experiment feasibility simulator:** Not found — contents and index have no simulation, Monte Carlo or bootstrap entry; only random-value column functions and Python notebooks.
- **Next-measurement advisor:** Not found — contents and index have no design-of-experiments entry.
- Incomplete: an earlier handbook page describing the fit dialog (residuals tab, parameter confidence intervals) could not be reached; the old address redirects to the home page.

### SciDAVis (official handbook, "updated for version 1.23 and newer"; pages read: contents, index, fitting section. scidavis.org "is not affiliated with this codebase" per the project README and was not used)
Index searched: https://highperformancecoder.github.io/scidavis-handbook/manual-index.html
- **Fit audit:** Part — the results log gives parameters with errors, Chi²/dof, R² and a status line; no checks or advice. "Status = success" (Fig. 3.12, read from the screenshot) https://highperformancecoder.github.io/scidavis-handbook/sec-fitting.html
- **Methods text:** Not found — index has no report, copy, LaTeX, citation or methods entry.
- **Model discernibility report:** Not found — index has no AIC, F-test or model-comparison entry.
- **Experiment feasibility simulator:** Not found — index has no simulation, Monte Carlo or bootstrap entry; Python scripting is the only route.
- **Next-measurement advisor:** Not found — index has no design entry.

### QtiPlot (version not stated; handbook copyright to 2025; pages read: contents, Fit Wizard, Data Fitting, Normality Test)
Index searched: https://www.qtiplot.com/doc/manual-en/index.html
- **Fit audit:** Part — confidence and prediction bands, a residuals plot, parameter errors and limits, goodness-of-fit statistics; no verdicts or advice. "If the Residuals Plot option is checked the curve of the fit residuals is also displayed." https://www.qtiplot.com/doc/manual-en/x11110.html
- **Methods text:** Not found — index has no report or methods entry; only a results log, ASCII export and a TeX equation editor for plots.
- **Model discernibility report:** Not found — index has no AIC, F-test or model-comparison entry.
- **Experiment feasibility simulator:** Not found — index has no Monte Carlo or bootstrap entry; only "Simulate Surface" for matrices and Python scripting.
- **Next-measurement advisor:** Not found — index has no design or optimal-design entry.

### jamovi (releases page shows 28.18; pages read: modules page with 81 library modules, release notes, core package function index)
Modules searched: https://www.jamovi.org/modules.html
- **Fit audit:** Not found — no official curve fit; the core package index (https://search.r-project.org/CRAN/refmans/jmv/html/00Index.html) lists no curve fitting. Third-party module PGM offers growth-curve fitting with "calculation of fit quality".
- **Methods text:** Not found — nothing fit-specific; third-party SummaryTables makes publication-ready tables and two ClinicoPath modules write "natural language summaries" of data sets.
- **Model discernibility report:** Not found — the modules page describes no curve-fit model comparison.
- **Experiment feasibility simulator:** Not found — third-party jpower, pamlj and ssMP give power and sample size for common designs; BrawStats simulates for teaching; none simulates a curve fit.
- **Next-measurement advisor:** Not found — third-party Randomizer randomises factorial designs; no optimal design.

### JASP (0.99.0 per release notes, 21 Oct 2026; the features page is still labelled 0.19.3; pages read: features page, release notes, Power module README)
- **Fit audit:** Not found — the module list (https://jasp-stats.org/features) has no curve fitting or nonlinear regression.
- **Methods text:** Part — "JASP AI", a chat assistant needing the user's own model API, can generate reports; nothing is fit-specific. "an AI-powered assistant that lets users interact with JASP through natural language" https://jasp-stats.org/release-notes
- **Model discernibility report:** Not found — features page lists no curve-fit comparison (Bain and BFpack test informative hypotheses for standard models).
- **Experiment feasibility simulator:** Not found — Power module covers "classical power analyses for common t-tests, z-tests, proportion tests, and variance-ratio tests"; nothing is run through a fit. https://github.com/jasp-stats/jaspPower
- **Next-measurement advisor:** Part — Quality Control module has general factorial and response-surface design worksheets and analysis; not a curve-fit advisor. "Create Factorial Worksheet, Create Response Surface Worksheet, Analysis Design" (features page above)

## What this check supports

No product was found to offer any of the five capabilities in its own right, so no cell reads "Has". The nearest neighbours found are Prism's residual tests with short flags and Origin's solver status with remedies (fit audit); TeX equation and table export in Prism and Origin (methods text); pairwise or ranked AICc, AIC and F-test comparisons in Prism, Origin and both CurveExpert editions (discernibility); factor-based design tools in Origin and JASP (design); and Prism's Simulate XY Data plus Monte Carlo analysis, which refits simulated data sets and tabulates estimates and a hit fraction without reporting bias or coverage, so a uniqueness claim for the feasibility simulator would need that qualification. The check is incomplete for LabPlot (the fuller fit-dialog page was unreachable), for SciDAVis (its results log is documented only by a screenshot), and for Origin and QtiPlot (no version stated).

## How the home page may word it

- It may say, with the date, that these capabilities were looked for in the nine products above, in their public documentation, and were not found there in their own right; and what was found instead.
- It may not say "available nowhere else" or "unique" without the qualification above: a product that was not checked, or a capability that was not found, is not proof of absence.
- When a later batch ships one of the other three capabilities, its line on the home page repeats this check for that capability first, and carries the qualification for the feasibility simulator if Prism's Monte Carlo analysis has not changed.

## The three fit tools, shipped (2026-11-25)

The model discernibility report, the experiment feasibility simulator and the next-measurement advisor are now available in the app, from a fit's **More tools** tab and from the command line. Their capabilities are columns 3 to 5 of the check above, and that check already records their state:

- **What was looked for:** a report on whether the data can tell the leading models apart (column 3), a simulator of a planned design that refits simulated data (column 4), and a recommendation of where to measure next for a curve fit (column 5).
- **Where:** the public documentation of the same nine products, read in full on 2026-11-24; the per-product findings name the pages and the quotes.
- **What was found:** no cell reads "Has". The nearest neighbours are pairwise or ranked AICc, AIC and F-test comparisons in CurveExpert, Origin and Prism; Origin's Monte Carlo and Bootstrap apps and Prism's Simulate XY Data with Monte Carlo analysis, neither of which reports bias or interval coverage for a curve fit; and Origin's factor-based DOE app and JASP's Quality Control design worksheets, neither for a curve fit.
- **Not checked, as recorded above:** LabPlot's fuller fit-dialog page could not be reached, SciDAVis's results log is documented only by a screenshot, and Origin and QtiPlot state no version on the pages read.

The home page describes where the three run and says nothing about what other products offer. No "available nowhere else" or "unique" claim is published for them, and none should be until the check is complete and the feasibility simulator's Prism qualification is settled.
