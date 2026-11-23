# Changelog

All notable changes are listed here. Dates are IST. Format: [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- **The public surface presents CurveStudio as a scientific, research and statistical
  tool**: the home page states who it is for, carries a research-requirements table,
  links the owner's other open-source projects and commercial software, and drops the
  concrete-industry framing; the README, page metadata and installed-app manifest match.
- **The project list lives in one file** (`docs/other-projects.json`), shared by the
  home page, both About surfaces and the footer, pinned by tests.
- **The canonical address is https://curvestudio.concreteinfo.cloud**: the old addresses
  redirect there (see `deploy/redirect/`), and `apps/api/OAUTH_SETUP.md` gains a Microsoft
  error-code table.

### Added
- **XLSX fit reports and SVG plots on the web**: the fit export route serves
  `format=xlsx` (reporting package, plot sheet included) and the chart panel gains
  SVG beside PNG.
- **Two more example datasets** beside the bundled one: a first-order decay and an
  enzyme-kinetics set.
- **A profile page and an account menu on the web app**: the chip in the header opens a
  menu with Profile and Sign out. `/profile` changes the name and the password (an
  account created with Google or Microsoft sets one by e-mail), shows the plan's limits
  against today's usage, switches the name on PDF reports and error reports, signs out
  every other device, and deletes the account once the part of the e-mail before the `@`
  is typed.
- **A "Prepared by" line on reports**: the desktop's Edit → Preferences (Ctrl+,) stores a
  name and an organisation that PDF reports, exported plots and workbook print headers
  carry; the web PDF report prints the account's name when the profile's switch is on.
  Nothing set, nothing changes.
- **Administrator accounts**: an address listed in `CURVESTUDIO_ADMIN_EMAILS` becomes an
  administrator at its next sign-in, and the change is audited. The API refuses every
  admin route to everyone else. The console that uses this ships separately.

### Fixed
- **The OAuth callback log names the sign-in provider's error code** (Microsoft's AADSTS
  text), so a failed social sign-in says why.
- **Microsoft sign-in requests only the `openid`, `email` and `profile` scopes**: the
  profile read is unaffected and the consent screen is smaller.
- **The register page offers the same Google, Microsoft and LinkedIn buttons as the
  sign-in card**, and links styled as buttons are no longer underlined.

## [0.11.1] - 2026-11-22

### Added
- **A richer public home page**: a “Fits you can reproduce” gallery with six interactive
  curve models (animated draw, equation, R²/RMSE and where each is met), a ΔAICc bar
  chart, scroll reveals and count-up figures (all reduced-motion safe), and a footer with
  the ConcreteInfo logo, the website and the other open-source projects.
- **PDF fit report in the browser** (`@react-pdf/renderer` 4.9.0): PC Power-style A4 report
  with the chart captured at full scale, full-precision statistics, parameters, equation and
  the honest non-convergence banner. Dynamically loaded, so the app bundle is unchanged.
- **Social-login setup guide** (`apps/api/OAUTH_SETUP.md`): console steps for Google,
  Microsoft and LinkedIn, the exact redirect URIs and troubleshooting.

### Changed
- **Every chart carries a very faint “ConcreteInfo CurveStudio” watermark**: web Plotly
  charts (and their PNG exports), the desktop chart (and its PNG/SVG exports) and every
  matplotlib report figure.
- **Reports carry a watermark and a disclaimer**: a faint diagonal watermark plus
  “Results describe agreement with the sample, not compliance or engineering advice —
  verify before use.” on every PDF page, and the same line in the XLSX print footer and
  summary sheet.
- The web CSP now allows WebAssembly and blob workers for the client-side PDF renderer;
  JavaScript `eval` remains blocked.

## [0.11.0] - 2026-11-22

### Added
- **Email verification**: registering sends a branded verification link (24 h, single use)
  and sign-in is refused until the address is verified. Unverified sign-ins offer a resend;
  existing accounts are migrated as verified so nobody is locked out.
- **Password reset by email**: a “forgot password” link sends a one-hour, single-use reset
  link; setting a new password also verifies the address and signs out every other session.
- **Social sign-in**: Google, Microsoft and LinkedIn (OAuth 2.0 + PKCE) are supported and
  appear automatically once their client ID and secret are configured; accounts are matched
  by verified email.
- **Public home page**: a cams2-style landing page at `/` with the feature pillars, the
  four-step workflow, the model library, the statistics that matter, an Auto-fit ranking
  mock, use cases, FAQ and a working sign-in box (plus social buttons). The workspace now
  sits behind sign-in.

### Changed
- Registration no longer signs the user in; the page asks them to check their inbox.
- Anonymous unknown routes redirect to the home page instead of the sign-in form.

### Deployment
- The Hostinger VPS instance (shared Postgres, 2 GiB / 2 vCPU caps) hosts
  **https://curvestudio.concreteinfo.cloud**; the desktop app is unchanged.

## [0.10.8] - 2026-11-22

### Changed
- **Desktop: R² and adjusted R² are shown at full stored precision** (the shortest
  round-tripping form) in the ranking table, the fit summary, the details window and the
  self-test line. A near-perfect fit such as 0.9999999999999999 no longer displays as
  `1.0000`; an exact fit shows `1.0`. Numeric columns size to their values, so a narrow
  dock scrolls instead of rounding. RMSE keeps its four-significant-digit form.
- **Web: the selected dataset card no longer has a coloured bar down its left side**;
  selection is the accent tint plus the accent-tinted border.

### Fixed
- Registering with an email that already has an account now says so and points at sign-in,
  instead of “the account could not be created with these details”.
- A rate-limited sign-in now reports how long to wait (“try again in N s”) instead of
  “try again later”.

## [0.10.7] - 2026-11-22

### Changed
- **The web header follows the PC Power Calculator**: the full CurveStudio lockup sits on
  its own white chip, the sticky bar is translucent with a backdrop blur, and the pill rail
  is right-aligned beside the session controls. The lockup asset is cropped to its artwork,
  so the wordmark fills the chip instead of floating in white margins.
- **The datasets panel is redesigned and collapsible**: a title row with a count chip, a
  compact upload card (drag, choose file or paste a block of cells), compact two-column
  dataset rows with a hover/focus-revealed delete, and a collapse toggle whose state is
  remembered. Collapsing the rail hands its width to the chart.
- **The chart now follows its column**: Plotly resizes with its container rather than only
  on window resize, so collapsing the datasets rail grows the canvas instead of leaving it
  stretched. The Plotly touch toolbar is hidden below 640px, where the app's own Download
  PNG covers the need.

### Fixed
- **Mobile layout**: the header no longer overflows at 320–390px (the avatar chip hides
  below 480px while sign-out stays reachable), and the Help and About overlays fit a
  320px viewport.

## [0.10.6] - 2026-11-22

### Changed
- **The web navbar is one pill rail**, following the PC Power Calculator: the primary
  links sit in a rounded tray and the current page is a raised pill inside it (brand text,
  subtle shadow), with the active workspace keeping its accent-bordered tab below.
- The About page's link to the PC Power Calculator points at its new address,
  https://pcpower.concreteinfo.net/, in both apps.

## [0.10.5] - 2026-11-22

### Added
- **Multiple workspaces in the web app**, in a redesigned header. Each named workspace (the
  tabs under the navbar) remembers its dataset and its Auto-fit/Manual mode; switching tabs
  returns exactly where you were, and the list is stored per account. Double-click a tab to
  rename it, the × closes one (never the last), + starts another. The navbar itself gained a
  brand lockup, pill navigation with a current-page state, an account chip and a PWA install
  button.
- **Help and About open as overlays** in both the signed-in experience and the public pages:
  the workspace stays mounted underneath (a lookup no longer costs a selection, a running
  analysis or unsaved state), and their URLs still work for deep links. Escape, the close
  button or the backdrop returns to the workspace.
- **The web app is installable (PWA)**: a manifest (name, icons, shortcuts, theme colours),
  a service worker that is network-first for navigations and cache-first for hashed assets,
  never caches `/api/`, and an install button that appears only when the browser offers it.
  Offline, the app shell still loads.
- **Mobile layout**: icon-only navigation and a scrolling workspace bar on phones, stacked
  panels, 44 px-class touch targets, 16 px form text (no iOS zoom) and full-height sheets for
  Help/About.
- **Desktop: one window per project.** *File → New project*, *Open project…* and *Open recent*
  now open another window instead of replacing the project on screen (each with its own
  controller, jobs and unsaved-changes prompt); only the first window owns the stored layout.
- **Desktop: a real model picker.** The Manual fit model field is a searchable dropdown —
  type any part of a name, id or category to filter 200 models, or open the list; the value
  is always a catalogue entry (free text cannot become the model) and a chevron marks it as
  a dropdown.

### Fixed
- **Desktop: the model picker could fit a different model from the one it displayed.** The
  catalogue's category separators occupy item positions, so mapping an index into the model
  list silently shifted by the number of separators: the field read "Einstein heat capacity"
  while the form fitted `polynomial` (visible as "polynomial degree" next to that name). Items
  now carry their model id and every lookup goes through it; every catalogue item is checked
  against the model it names in a test.
- Web: a model search filter no longer hides the model that is actually selected, so the
  picker can never look empty while the panel would still fit the previous model.

## [0.10.4] - 2026-11-22

### Fixed
- **The app no longer opens with a panel floating over the workspace.** A panel floated in
  one session was saved as floating and reopened as a stranded window over the middle of the
  chart — usually at a position from a different screen size, which read as "the app opened
  broken". Floating is now a session state: every panel is restored **docked** in its own
  area (the Analysis panel on the right, the data table at the bottom, the project tree on
  the left) and the status bar says so. A closed panel still stays closed, and a docked,
  narrowed arrangement still restores exactly as saved.

### Added
- A GitHub Actions release workflow that builds and publishes every supported platform —
  Linux (`.deb`, `.AppImage`, `.tar.gz`) and Windows (`.zip`, Inno Setup installer when
  available) — on a `v*` tag, with a tag/version check, a fast test gate, checksum
  verification and release notes taken from this file. `workflow_dispatch` builds and
  uploads the artifacts without publishing.

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
