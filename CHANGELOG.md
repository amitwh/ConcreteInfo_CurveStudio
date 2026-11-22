# Changelog

All notable changes are listed here. Dates are IST. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.10.3] - 2026-11-22

### Added
- **The selected model's equation is captioned on the chart**, in both apps. Select a
  ranking row (or fit a model manually) and the plot shows the model's name and its fitted
  equation, in that model's colour, with `x^2` rather than the engine's `x**2`. Long
  equations wrap over a few lines and end with an ellipsis (the complete text stays in the
  details window and the reports); the caption sits clear of the axis labels and never
  leaves the plot.
- **Screenshots in the help**, in both apps: the workspace with the equation caption, the
  spreadsheet paste grid, the Manual fit form and the details window (desktop); the
  workspace and Paste data (web). Each image carries a caption, and the desktop renders
  images from its packaged help folder (no network).
- **About now answers the questions people actually ask.** It lists the other free
  ConcreteInfo projects ([GitHub Desktop for Linux](https://github.com/amitwh/GitHubDesktopLinux),
  [Markdown Converter](https://github.com/amitwh/markdown-converter),
  [PC Power Calculator](https://pcpower.concreteinfo.co.in/)), explains why the desktop app
  is free and AGPL-3.0 (and why that licence: a hosted modification must stay open), and
  states plainly that the web app is free today with a possible small hosting subscription
  later, the desktop staying free. Contact: [www.concreteinfo.in](https://www.concreteinfo.in)
  · amit@concreteinfo.in.
- **A disclaimer and a privacy/fair-use policy**, on the same page in both apps: a fit
  describes agreement with your sample, not physical truth or code compliance; results are
  not engineering advice; the software is provided as-is with the AGPL's warranties
  disclaimed; desktop data never leaves the machine, web data is stored only to provide the
  service (Argon2 password hashes, no third-party analytics, no sale of data), with fair-use
  limits and a contact for access or deletion.

### Changed
- The help's last section is now "About, licence, projects and policy", mirroring the About
  page, and the web help's search also matches screenshot captions.

## [0.10.2] - 2026-11-22

### Added
- **A spreadsheet you can paste into, in both apps.** The desktop's *File → Import from
  clipboard* now opens an editable grid (column letters, row numbers) instead of a text box:
  copy a block of cells in Excel, LibreOffice Calc or Google Sheets and press Ctrl+V — the
  block lands at the focused cell, rows and columns grow to fit, and a paste into an empty
  grid replaces it. Cells stay editable, rows and columns can be added, quoted cells (a
  comma, a tab or a line break inside a value) survive the round trip, and the first row is
  the header. The web sidebar gains **Paste data** with the same grid, and the pasted sheet
  travels through the ordinary upload path (column pickers included) as a `.tsv` file.

### Fixed
- **The model picker no longer shows a half-height line.** An editable combo hosts a real
  text field, and the stylesheet's padding for text fields applied inside that field as well
  — 8 px of a 16 px line stayed visible. The field now draws no padding of its own; the combo
  around it provides the frame and the padding.
- **The panels can be resized again.** The Analysis dock's content demanded 503×546 px, so
  the chart/data-table divider had no travel and a floated panel could not be made smaller.
  The Auto-fit controls sit on two rows, the ranking's minimum is a small usable window
  (three rows; the default split still shows eight or more), the Manual fit form scrolls
  inside the dock instead of pinning its height, and a page that is not on screen no longer
  sets the dock's minimum. At 1280×820 the data table can now be dragged from 220 px to
  about 330 px, and further when the Analysis panel is collapsed or floated.
- **Floating panels move and resize like windows.** A custom title bar replaces the native
  one, so a press on it now asks the platform to start the window move (X11, Wayland and the
  native path) instead of doing nothing; a floating panel keeps its resize edges, and
  double-clicking its title docks it back.
- **View → Panels always returns a panel to its place.** Re-showing a panel re-docks it to
  its home area at its remembered size, even if it had been floated, closed while floating
  or dragged into a neighbour's area; the menu entry mirrors the dock without looping back
  into the layout, which previously re-docked a panel the instant it floated.

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
