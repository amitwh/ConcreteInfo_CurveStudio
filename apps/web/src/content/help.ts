/**
 * The web /help page content. This is a transcription of docs/help-content.md — the
 * canonical text the desktop help browser renders too. Every claim there was checked
 * against the code; do not add a feature here that the engine does not have.
 *
 * The content is data rather than JSX so the page can render one heading per section,
 * give each a stable anchor and search the same text the reader sees.
 */

export type Inline =
  | string
  | { kind: "code"; text: string }
  | { kind: "strong"; text: string }
  | { kind: "em"; text: string }
  | { kind: "link"; text: string; href: string };

export type HelpBlock =
  | { kind: "p"; content: Inline[] }
  | { kind: "h3"; text: string }
  | { kind: "ul"; items: Inline[][] }
  | { kind: "ol"; items: Inline[][] }
  | { kind: "table"; headers: string[]; rows: Inline[][][] }
  | { kind: "figure"; src: string; alt: string; caption: string };

export interface HelpSection {
  /** Stable URL fragment: /help#keyboard-shortcuts */
  id: string;
  title: string;
  blocks: HelpBlock[];
}

const code = (text: string): Inline => ({ kind: "code", text });
const strong = (text: string): Inline => ({ kind: "strong", text });
const em = (text: string): Inline => ({ kind: "em", text });
const link = (text: string, href: string): Inline => ({ kind: "link", text, href });

export const HELP_SECTIONS: HelpSection[] = [
  {
    id: "getting-started",
    title: "Getting started",
    blocks: [
      {
        kind: "p",
        content: [
          "CurveStudio fits a curve to paired x/y measurements and tells you, honestly, how well it fits. It is the same engine on the web and on the desktop, so a fit made in one matches the other.",
        ],
      },
      {
        kind: "ol",
        items: [
          [
            strong("Open or create a project."),
            " The desktop app starts with an empty project; the web app keeps your datasets under your account.",
          ],
          [
            strong("Import your data"),
            " (CSV/TSV/TXT/JSON/XLSX, or paste from a spreadsheet).",
          ],
          [
            strong("Let Auto-fit run"),
            " — it starts by itself after the first import. It screens the model library, fits the promising candidates and ranks them.",
          ],
          [
            strong("Read the ranking."),
            " The top row is the best fit by the chosen criterion. Select it to draw its curve and see its headline numbers; open its details for the full report.",
          ],
          [strong("Export"), " the report, the ranking or the plot."],
        ],
      },
      {
        kind: "p",
        content: [
          "Three example datasets ship with the app: ",
          code("example.csv"),
          " (age in days vs. compressive strength in MPa), ",
          code("example-decay.csv"),
          " (a first-order decay) and ",
          code("example-kinetics.csv"),
          " (enzyme kinetics) — good data to explore with.",
        ],
      },
      {
        kind: "figure",
        src: "/help/workspace.png",
        alt: "The web workspace: datasets, the chart with the selected model's equation, and the Auto-fit ranking",
        caption:
          "The web workspace: the chart captions the selected model's fitted equation, and the ranking lists every converged model with its colour.",
      },
    ],
  },
  {
    id: "importing-data",
    title: "Importing data",
    blocks: [
      {
        kind: "ul",
        items: [
          [
            strong("Accepted files:"),
            " ",
            code(".csv"),
            ", ",
            code(".tsv"),
            ", ",
            code(".txt"),
            ", ",
            code(".json"),
            ", ",
            code(".xlsx"),
            ".",
          ],
          [
            strong("Paste data:"),
            " ",
            em("Paste data"),
            " in the sidebar opens an editable grid. Select the cells in Excel, LibreOffice Calc or",
            " Google Sheets, copy, click the grid and press ",
            code("Ctrl+V"),
            ": the whole block lands in place (a paste into an empty grid replaces it), the first",
            " row is the header, and the cells can be typed over before importing.",
          ],
          [
            strong("Columns:"),
            " you map the columns after import. A dataset needs an ",
            strong("x"),
            ", a ",
            strong("y"),
            ", and optionally a ",
            strong("weight"),
            " column (blank means weight 1) and an ",
            strong("included"),
            " flag for excluding rows without deleting them.",
          ],
          [
            strong("Refused input:"),
            " x and y must be different columns; a file with fewer than three usable rows cannot be fitted, and the import says so rather than producing a curve from nothing.",
          ],
          [
            strong("Missing values:"),
            " a row whose x or y is empty, or not a number, is reported and skipped; nothing is guessed.",
          ],
        ],
      },
      {
        kind: "figure",
        src: "/help/paste.png",
        alt: "The Paste data grid with a pasted table of age, strength and batch columns",
        caption:
          "Paste data: copy a block of cells from any spreadsheet app and drop it straight into the grid.",
      },
    ],
  },
  {
    id: "auto-fit",
    title: "Auto-fit",
    blocks: [
      {
        kind: "p",
        content: [
          "Auto-fit is the default workflow: it fits many candidate models and ranks them, so you can see whether a simpler model already explains the data.",
        ],
      },
      {
        kind: "p",
        content: [strong("Options"), " (collapsed under ", em("Options"), "):"],
      },
      {
        kind: "table",
        headers: ["Option", "Meaning"],
        rows: [
          [
            ["families (categories)"],
            ["Comma-separated families to include; blank means all."],
          ],
          [
            ["exhaustive"],
            [
              "Also include the long tail of generated families — slower, rarely changes the top.",
            ],
          ],
          [
            ["max parameters"],
            ["Cap on free parameters per model; ", code("auto"), " uses the engine's rule."],
          ],
          [
            ["multi-starts"],
            [
              "Number of starting points per nonlinear model (1–64). More starts cost time and reduce the chance that a fit lands on a poor optimum.",
            ],
          ],
          [
            ["CV folds"],
            ["K-fold cross-validation folds; ", code("off"), " disables cross-validation."],
          ],
          [
            ["seed"],
            [
              "Seed for the deterministic subsampling and start generation. The same seed, data and options reproduce the same ranking.",
            ],
          ],
        ],
      },
      { kind: "h3", text: "What the ranking columns mean" },
      {
        kind: "table",
        headers: ["Column", "Meaning"],
        rows: [
          [["Rank"], ["Position by the active criterion (best first)."]],
          [
            ["Model"],
            [
              "Catalogue name. ",
              code("✓"),
              " marks an exact fit, ",
              code("⚠"),
              " a degenerate fit (a parameter at a bound, no effect on the curve), ",
              code("≡"),
              " a model redundant with one ranked above it.",
            ],
          ],
          [
            ["R²"],
            [
              "Coefficient of determination. Higher is better, but it always rises with more parameters.",
            ],
          ],
          [
            ["adj. R²"],
            ["R² with the parameter count charged against it — the honest comparison."],
          ],
          [["RMSE"], ["Residual standard error in y units."]],
          [["k"], ["Number of fitted parameters."]],
          [
            ["AICc / BIC"],
            [
              "Information criteria (in the details report). Lower is better; both penalise parameters, AICc more strongly on small samples.",
            ],
          ],
        ],
      },
      {
        kind: "ul",
        items: [
          [
            strong("Overfit and collinearity warnings"),
            " appear above the ranking; read them. A model that collapses onto a simpler one is ranked below it, not above.",
          ],
          [
            strong("Non-converged and skipped candidates"),
            ' are listed in their own sections, never mixed into the ranking. "Not converged" means the optimiser could not satisfy its criteria; the model is not reported as a fit.',
          ],
          [
            strong("Cancel"),
            ' stops a run; the partial rows that did converge are still usable, but the run is labelled partial and "Open full results" is disabled for it.',
          ],
          [strong("Order by"), " re-sorts the same ranking without refitting."],
        ],
      },
    ],
  },
  {
    id: "manual-fit",
    title: "Manual fit and custom equations",
    blocks: [
      {
        kind: "p",
        content: ["Choose a model from the catalogue, or write an equation."],
      },
      { kind: "h3", text: "Custom equation grammar" },
      {
        kind: "ul",
        items: [
          [
            "Variables: the dataset's ",
            strong("x"),
            " and ",
            strong("y"),
            " names, plus any parameter names you declare.",
          ],
          ["Operators: ", code("+ - * / **"), " and parentheses."],
          ["Constants: ", code("pi"), ", ", code("E"), "."],
          [
            "Unary functions: ",
            code("exp"),
            ", ",
            code("log"),
            " and ",
            code("ln"),
            " (both natural log), ",
            code("log10"),
            ", ",
            code("sqrt"),
            ", ",
            code("sin"),
            ", ",
            code("cos"),
            ", ",
            code("tan"),
            ", ",
            code("sinh"),
            ", ",
            code("cosh"),
            ", ",
            code("tanh"),
            ", ",
            code("asin"),
            ", ",
            code("acos"),
            ", ",
            code("atan"),
            ", ",
            code("erf"),
            ", ",
            code("erfc"),
            ", ",
            code("Abs"),
            "/",
            code("abs"),
            ".",
          ],
          [
            "Two-argument functions: ",
            code("pow(base, exponent)"),
            ", ",
            code("min(a, b)"),
            ", ",
            code("max(a, b)"),
            ", ",
            code("atan2(y, x)"),
            ".",
          ],
          [
            "Three-argument function: ",
            code("voigt(x, sigma, gamma)"),
            " — the area-normalised Voigt profile (",
            code("scipy.special.voigt_profile"),
            ").",
          ],
          [
            "Limits: at most 300 expression nodes, 40 levels of nesting, and a constant exponent no larger than 1000. An expression that breaks these is refused with the reason.",
          ],
        ],
      },
      { kind: "h3", text: "Notes" },
      {
        kind: "ul",
        items: [
          [
            code("log(x)"),
            " is the natural logarithm, not base 10 — write ",
            code("log10(x)"),
            " for base 10.",
          ],
          [
            "Every parameter gets an initial value and optional lower/upper bounds. A blank bound is unbounded; when both are given they must satisfy ",
            strong("lower < upper"),
            " (equal bounds are refused, as is an initial value outside them), so a parameter cannot be pinned as a hidden constant.",
          ],
          [
            "An equation with no free parameter — one where every name is a variable, constant or function — is refused with a message saying there is nothing to fit.",
          ],
        ],
      },
    ],
  },
  {
    id: "understanding-the-numbers",
    title: "Understanding the numbers",
    blocks: [
      {
        kind: "ul",
        items: [
          [
            strong("Parameter table:"),
            " estimate, standard error and 95% confidence interval. ",
            code("n/a"),
            " means the fit could not determine the value (hovering the cell says so); it is never shown as zero.",
          ],
          [
            strong("Statistics tab:"),
            " fit statistics and the hypothesis tests the engine ran, each with a note saying what it tests.",
          ],
          [
            strong("Diagnostics tab:"),
            " residual plots and the checks that reveal a systematic misfit — a curve that passes through the middle of the residuals is doing less well than its R² suggests.",
          ],
          [
            strong("R² close to 1"),
            " is not proof of a good model: with many parameters it can be reached by fitting noise. Compare adj. R², AICc and the residual plots.",
          ],
        ],
      },
    ],
  },
  {
    id: "prediction",
    title: "Prediction",
    blocks: [
      {
        kind: "p",
        content: [
          "The Predict tab evaluates the fitted model at x values you type or paste (comma, space or newline separated) and can add a ",
          strong("confidence band"),
          " (the uncertainty of the mean response) or a ",
          strong("prediction band"),
          " (the uncertainty of a single new observation, always wider). The number of requested points is capped; past the cap the tab says so instead of silently truncating.",
        ],
      },
    ],
  },
  {
    id: "projects-desktop",
    title: "Projects (desktop)",
    blocks: [
      {
        kind: "p",
        content: [
          "A project holds its datasets, their fits and the Auto-fit results in one ",
          code(".cstudio"),
          " file (",
          code("Ctrl+S"),
          " saves, ",
          code("Ctrl+Shift+S"),
          " saves a copy, ",
          code("Ctrl+O"),
          " opens). Undo (",
          code("Ctrl+Z"),
          ") covers project edits. A project saved on one machine opens on another: it is a file on your computer, never uploaded.",
        ],
      },
    ],
  },
  {
    id: "exports",
    title: "Exports",
    blocks: [
      {
        kind: "table",
        headers: ["What", "Formats"],
        rows: [
          [
            ["Fit report"],
            [
              code(".json"),
              ", ",
              code(".xlsx"),
              ", ",
              code(".pdf"),
              ", ",
              code(".csv"),
              " (the CSV export writes the three result tables)",
            ],
          ],
          [["Auto-fit ranking"], [code(".csv")]],
          [["Plot"], [code(".png"), ", ", code(".svg"), ", ", code(".pdf")]],
        ],
      },
      {
        kind: "p",
        content: [
          "CSV cells that begin with ",
          code("="),
          ", ",
          code("+"),
          ", ",
          code("-"),
          " or ",
          code("@"),
          " are neutralised so a spreadsheet cannot execute imported data as a formula.",
        ],
      },
    ],
  },
  {
    id: "keyboard-shortcuts",
    title: "Keyboard shortcuts (desktop)",
    blocks: [
      {
        kind: "table",
        headers: ["Action", "Shortcut"],
        rows: [
          [
            ["New / Open / Save / Save as"],
            [code("Ctrl+N"), " / ", code("Ctrl+O"), " / ", code("Ctrl+S"), " / ", code("Ctrl+Shift+S")],
          ],
          [
            ["Import data / Import from clipboard"],
            [code("Ctrl+I"), " / ", code("Ctrl+Shift+V")],
          ],
          [["Undo"], [code("Ctrl+Z")]],
          [["Preferences… (report byline and theme)"], [code("Ctrl+,")]],
          [["Reset view (fit the plot to the data)"], [code("Ctrl+R")]],
          [
            ["Details (full result of the selected row or active fit)"],
            [code("Ctrl+D")],
          ],
          [["Export plot"], [code("Ctrl+E")]],
          [["Help"], [code("F1")]],
          [["Quit"], [code("Ctrl+Q")]],
        ],
      },
      {
        kind: "p",
        content: [
          "Panels can be moved, floated, tabbed together, collapsed with the chevron in their title bar, or closed. ",
          em("View → Reset layout"),
          " restores the default arrangement.",
        ],
      },
    ],
  },
  {
    id: "troubleshooting",
    title: "Troubleshooting",
    blocks: [
      {
        kind: "ul",
        items: [
          [
            strong('"Not converged" for everything:'),
            " the data may be too sparse or too noisy for the requested models, or every start may have hit the time limit. Try fewer families, more multi-starts, or a simpler model.",
          ],
          [
            strong("The curve looks right but the fit is not ranked:"),
            " a non-converged fit is never ranked, even when its curve looks plausible. Read the details report for the termination reason.",
          ],
          [
            strong("Undefined statistics:"),
            " ",
            code("n/a"),
            ' means "the fit could not determine this", which is information, not an error.',
          ],
          [
            strong("The desktop app reports an unexpected error:"),
            " a dialog shows the traceback with a copy button, and the same report is kept in a ",
            code("crash.log"),
            ' on your computer (its "Open log folder" button leads there); nothing is sent. Saved project files are unaffected. Report it with the copied text.',
          ],
          [
            strong("No network (desktop):"),
            " fitting is local, so the app keeps working in full, with no account. The only thing it sends is one anonymous install record (version, operating system, package type), on first launch and after an upgrade; without a connection it waits for a later launch, and Edit → Preferences → Privacy turns it off.",
          ],
          [
            strong("Web:"),
            " fitting needs a connection to the server. The desktop app is the offline story.",
          ],
        ],
      },
    ],
  },
  {
    id: "fit-audit-and-methods-text",
    title: "Fit audit and methods text",
    blocks: [
      {
        kind: "p",
        content: [
          "After a fit, the ",
          strong("fit audit"),
          " checks the evidence for it, check by check, and says what to do next. On the desktop it is the ",
          strong("Fit audit"),
          " tab of a fit's results, after Diagnostics; on the web it is the ",
          strong("Fit audit"),
          " tab; from the command line, ",
          code("curvestudio audit"),
          " fits a curve and prints the same checks.",
        ],
      },
      {
        kind: "p",
        content: [
          "At the top sit the overall verdict (",
          strong("Pass"),
          ", ",
          strong("Caution"),
          ", ",
          strong("Fail"),
          " or ",
          strong("Not applicable"),
          "), a summary in one to three sentences, the counts (",
          code("40 observations, 3 parameters"),
          ") and any rows that took no part, with the reason. Then comes one block per check: its verdict, its evidence, its statistic and the threshold it was judged by, what the check is about, and the next step when there is one. A fit that did not converge fails the first check, and every other check reads Not applicable.",
        ],
      },
      {
        kind: "table",
        headers: ["Check", "If it is flagged"],
        rows: [
          [
            ["The fit converged"],
            [
              "Check the starting values and the bounds, or try a model with fewer parameters.",
            ],
          ],
          [
            ["The model does not pass through every point"],
            [
              "There is no scatter to estimate uncertainty from, so report no intervals or p-values for this fit.",
            ],
          ],
          [
            ["Residuals are consistent with a normal distribution (Shapiro–Wilk)"],
            [
              "If a few points cause it, try a robust loss (Huber or soft-L1). Otherwise say in your write-up that the intervals rely on an approximation.",
            ],
          ],
          [
            ["Residuals show no autocorrelation in x order (Durbin–Watson)"],
            [
              "Plot the residuals against x in data order. If they wander, the model is missing a shape; if they alternate, look at how the data were taken.",
            ],
          ],
          [
            ["Residual variance is consistent with being constant (Breusch–Pagan)"],
            [
              "If the spread grows with the signal, use weights from replicates or a robust loss, and say so.",
            ],
          ],
          [
            ["No single point has a large influence on the fit (Cook's distance)"],
            [
              "Refit without the point and compare the estimates. If the conclusions differ, report both fits or justify leaving the point out. Do not delete data to improve a fit.",
            ],
          ],
          [
            ["Residuals show no trend in x"],
            [
              "The residuals show a curved trend, so the model is missing a shape the data contain. Try a model with one more parameter.",
            ],
          ],
          [
            ["No parameter sits at a bound or is poorly determined"],
            [
              "A parameter at a bound or with a wide interval is not pinned down by these data. Fix the bound if it is wrong, take more points where the model is sensitive, or use a model with fewer parameters.",
            ],
          ],
          [["There are at least 10 points per parameter"], ["Take more points, or use a simpler model."]],
          [
            ["Held-out error is close to what a correct model shows (cross-validation)"],
            [
              "The fit describes noise as well as signal. A simpler model, or one with fewer parameters, will predict better.",
            ],
          ],
          [
            ["No simpler model fits as well (from an Auto-fit ranking)"],
            ["Report the simpler model unless the extra parameter answers a question of your own."],
          ],
        ],
      },
      {
        kind: "p",
        content: [
          "The first column is how a check reads when it passes; the second is the audit's own next step. Three of the checks are read by more than a p-value. The autocorrelation check is decided by Durbin–Watson p-values that are exact for the fitted model and computed for at most 1,000 points: Ljung–Box is reported beside them and does not change the verdict, and where they are not computed it decides alone; with repeated x values the test is approximate. Cook's distance is set against the fit's own confidence region — a caution when one point's removal would move the estimates outside the 90 % region, a failure when outside the 99 % — and more points than chance crossing the 4/n line flag it too. A simpler model is one within 2 AICc of the best that is neither degenerate nor redundant.",
        ],
      },
      {
        kind: "p",
        content: [
          strong("Reading a verdict."),
          " A verdict is evidence, not proof. ",
          strong("Pass"),
          ": nothing in the data contradicts the assumption at the conventional level. ",
          strong("Caution"),
          ": for the statistical tests, evidence that would turn up by chance in 1 fit in 20 or fewer if the assumption held. ",
          strong("Fail"),
          ": for them, evidence that would turn up in 1 fit in 1,000 or fewer. For the rule-based checks — a parameter at a bound, fewer than 5 points per parameter, a point whose removal leaves the 99 % confidence region, a held-out error far above what a correct model shows — a failure means the rule was met outright. ",
          strong("Not applicable"),
          ": the check could not be made, and the block says why. The four residual tests work at the 5 % level, so on data that meet every assumption about one fit in twenty still shows a caution on a given test; a single caution is a prompt to look, not a verdict on the analysis.",
        ],
      },
      {
        kind: "p",
        content: [
          strong("Cross-validation."),
          " A stored fit does not carry its held-out error, so that check reads Not applicable until you run it. A control at the end of the tab runs 5, 10 or leave-one-out folds as a job you can cancel: a background job on the desktop, where the result is not saved with the project; on the web a job that uses none of your quota, whose result is kept with the fit. Leave-one-out refits once per point, so it runs on at most 2,000 points. A fit that did not converge cannot be cross-validated, and the control says so. On the command line, ",
          code("curvestudio audit --cv 5"),
          " (or ",
          code("--cv loo"),
          ") adds it, and ",
          code("--ranking"),
          " lets the simpler-model check run.",
        ],
      },
      {
        kind: "p",
        content: [
          strong("The methods text."),
          " It writes the fit up for a paper. Its paragraph names the model, the estimator, the observations used and those left out, and any bounds, and cites the software with its version and address; the equation gives each estimate with its standard error; the table adds 95 % confidence intervals. ",
          strong("Copy methods text"),
          " (desktop: the details window's footer and the Equation tab) and ",
          strong("Copy methods"),
          " (web) put the three on the clipboard as plain text, Markdown or LaTeX; the LaTeX table needs the ",
          code("booktabs"),
          " package and the equation ",
          code("amsmath"),
          ". ",
          code("curvestudio methods --style markdown"),
          " (or ",
          code("text"),
          " or ",
          code("latex"),
          ") prints them with the model's references, the three things only the author can write — the instrument and measurement method, how the sample size was chosen, how the data were cleaned — and whatever the fit did not record. Nothing is guessed, and there is no DOI to cite yet.",
        ],
      },
      {
        kind: "p",
        content: [
          "The audit is a screen for the common ways a fit misleads; it does not replace looking at the residuals and knowing the measurement.",
        ],
      },
    ],
  },
  {
    id: "statistics-tools",
    title: "Statistics tools",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Statistics page"),
          " on the desktop and the ",
          strong("Statistics mode"),
          " on the web run the same twenty-nine tools, in eleven families, on a dataset's columns. Pick a tool from the catalogue, choose the dataset it reads, fill in its fields and press ",
          strong("Run"),
          "; the command line runs the same tools with ",
          code("curvestudio stats --list"),
          " and ",
          code("curvestudio stats <tool> <file> …"),
          ".",
        ],
      },
      {
        kind: "p",
        content: [
          "Each field is one of four kinds: a ",
          strong("numeric column"),
          ", ",
          strong("several columns"),
          " (one to twenty, depending on the tool), a ",
          strong("group column"),
          " whose levels are the groups compared, or an ",
          strong("option"),
          " (a confidence level, a post-hoc test, a number of bins, a flag). The description under each field says what it expects. A required field left empty stops the run with its reason rather than guessing a column.",
        ],
      },
      {
        kind: "p",
        content: [
          strong("What a result holds."),
          " Every tool returns the same shape of result, and the desktop and web apps show it in the same order:",
        ],
      },
      {
        kind: "ul",
        items: [
          [
            strong("Estimates"),
            " — a label, the value, its standard error and the interval the level names.",
          ],
          [
            strong("Tests"),
            " — the test's name, its statistic, its degrees of freedom, its p-value and, where several comparisons were made, the adjusted p-value.",
          ],
          [
            strong("Assumptions"),
            " — one card per check the tool could make: what was read, the statistic, the rule it was judged by, and what to do when it is flagged.",
          ],
          [
            strong("Warnings"),
            " — anything the data forced, such as a column that does not vary.",
          ],
          [
            strong("Tables and plots"),
            " — the extra tables and pictures the tool draws.",
          ],
          [
            strong("Interpretation"),
            " — one to three plain sentences on what the result says.",
          ],
          [
            strong("Methods"),
            " — a paragraph that can be carried into a paper as it stands; copy it with ",
            strong("Copy methods text"),
            " on the desktop and on the web.",
          ],
          [
            strong("Counts and excluded rows"),
            " — how many rows were offered, used and excluded, with each excluded row's position and the reason it was left out (the web lists it under ",
            strong("Rows"),
            "). Nothing is dropped without being counted.",
          ],
        ],
      },
      {
        kind: "p",
        content: [
          strong("Reading a verdict."),
          " The assumption cards use four words. ",
          strong("Pass"),
          ": nothing in the data contradicts the assumption at the usual level. ",
          strong("Caution"),
          ": evidence that would turn up by chance in about 1 such check in 20 if the assumption held. ",
          strong("Fail"),
          ": evidence that would turn up in about 1 in 1,000. ",
          strong("Not applicable"),
          ": the check could not be made, and the card says why. A verdict is evidence, not proof: a caution is a prompt to look, and a pass does not prove the assumption holds. The same reading applies to every p-value: a small one is evidence against the null hypothesis, and a large one is not evidence that the null is true.",
        ],
      },
      {
        kind: "p",
        content: [
          "Points outside ",
          strong("Tukey's fences"),
          " are flagged in their own table and kept in every statistic: a summary describes the data it was given and does not decide which points to believe. No tool removes a point for being unusual.",
        ],
      },
    ],
  },
  {
    id: "summarising-columns",
    title: "Summarising columns",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Summarise"),
          " family describes one column: where its values sit and how far they spread. Both tools read a single numeric column, and the rows the column cannot use are counted and listed.",
        ],
      },
      { kind: "h3", text: "Descriptive statistics" },
      {
        kind: "p",
        content: [
          "Reads one column, optionally split by a group column. It reports the mean, the sample standard deviation, the quartiles, the coefficient of variation, the bias-corrected skewness and the excess kurtosis, and a t interval around each mean. A group column summarises each of its levels separately. Points beyond Tukey's fences (1.5 interquartile ranges past the quartiles) appear in their own table and are kept in every number above: the summary describes the data it was given. The methods fragment writes each definition once.",
        ],
      },
      { kind: "h3", text: "Distribution analysis" },
      {
        kind: "p",
        content: [
          "Reads one column and reports its mean with an interval and its standard deviation, then tests whether the values look normal with the Shapiro–Wilk and Anderson–Darling tests. A histogram with the fitted normal's expected count in each bin and a normal Q-Q plot show the shape the tests are reading; leave ",
          strong("Bins"),
          " empty and the Freedman–Diaconis rule picks the number. A small p-value is evidence that the values are not normal; a large one is not evidence that they are.",
        ],
      },
    ],
  },
  {
    id: "comparing-means",
    title: "Comparing means",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Compare means"),
          " family asks whether groups share a mean, by the t test and the analysis of variance. Every test reports the difference or the effect size with an interval beside its p-value, and every assumption the data can show is checked.",
        ],
      },
      { kind: "h3", text: "t tests" },
      {
        kind: "p",
        content: [
          "One tool with three kinds. ",
          strong("one_sample"),
          " tests a column's mean against the hypothesised value. ",
          strong("independent"),
          " compares two groups' means, given either two columns or one column and a group column with exactly two levels; Welch's test is the default and ",
          strong("Equal variances"),
          " switches to Student's test. ",
          strong("paired"),
          " compares the difference of two columns, or one column of differences, against the hypothesised difference. Each reports Hedges' g (Cohen's dz for the paired test) with its interval. Shapiro–Wilk checks normality, Levene's median-centred test checks equal variances under Student's test, and the paired test checks that every pair is complete.",
        ],
      },
      { kind: "h3", text: "One-way analysis of variance" },
      {
        kind: "p",
        content: [
          "Compares the mean of one column across the levels of a group column, and reports eta squared and omega squared for how much of the variation the groups account for. With more than two groups a post-hoc test follows: ",
          strong("Tukey"),
          " compares every pair, and ",
          strong("Dunnett"),
          " compares every group against the ",
          strong("Control"),
          " level; each p-value is already adjusted for the family. Welch's analysis of variance, which does not assume equal variances, is reported beside the ordinary one. Shapiro–Wilk checks each group's normality and Levene's test their equal variances.",
        ],
      },
      { kind: "h3", text: "Two-way analysis of variance" },
      {
        kind: "p",
        content: [
          "Splits a column's mean by two group columns and their interaction, and reports the main effects, the interaction and partial eta squared. An unbalanced design is read by Type II sums of squares, so the order the factors are given in does not change the table.",
        ],
      },
      { kind: "h3", text: "Repeated-measures analysis of variance" },
      {
        kind: "p",
        content: [
          "Compares the same subjects measured under two to twenty conditions, given one column per condition in wide form and optional condition ",
          strong("Names"),
          ". A row missing in any condition is left out of all of them. The F test assumes sphericity — that the conditions are equally correlated — so Mauchly's test checks it, and the Greenhouse–Geisser and Huynh–Feldt corrections scale the degrees of freedom when it is doubtful. Generalised eta squared is the effect size.",
        ],
      },
    ],
  },
  {
    id: "rank-based-comparisons",
    title: "Rank-based comparisons",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Rank-based comparisons"),
          " family reads the order of the values, never their size, so an outlier moves a rank no further than the next value would. None of them assumes the values are normally distributed; each reports an effect size or an interval beside its p-value. It holds the two-sample, paired and several-group comparisons.",
        ],
      },
      { kind: "h3", text: "Mann-Whitney U test" },
      {
        kind: "p",
        content: [
          "Compares two independent samples, given as two columns or as one column split by a group column with exactly two levels. It reports the Hodges–Lehmann shift with a distribution-free interval and the rank-biserial correlation. It answers whether one sample tends to be larger, not whether the means are equal.",
        ],
      },
      { kind: "h3", text: "Wilcoxon signed-rank test" },
      {
        kind: "p",
        content: [
          "Compares paired differences against the ",
          strong("Null value"),
          ": give two columns, or one column of differences. Zero differences are left out, as Wilcoxon's original rule does. It reports the pseudo-median — the median of the Walsh averages — with its interval and the matched-pairs rank-biserial correlation.",
        ],
      },
      { kind: "h3", text: "Sign test" },
      {
        kind: "p",
        content: [
          "Counts how many differences fall above and below the ",
          strong("Null value"),
          " and tests that count. It asks only that the differences mostly fall on one side, so it is used when their sizes are not trustworthy and only their direction is. It reports the median with the interval the sign test itself gives.",
        ],
      },
      { kind: "h3", text: "Kruskal–Wallis test" },
      {
        kind: "p",
        content: [
          "Compares the groups' mean ranks across the levels of a group column, gives tied values their average rank, and corrects for the ties. Dunn's test compares the pairs when there are more than two groups, with Holm's method adjusting its p-values. It makes no assumption of normality.",
        ],
      },
      { kind: "h3", text: "Friedman test" },
      {
        kind: "p",
        content: [
          "The rank-based counterpart of the repeated-measures analysis: it ranks each subject's conditions together and asks whether the mean ranks differ, with Kendall's W as its effect size. Durbin–Conover comparisons follow when there are more than two conditions, with Holm's method adjusting them. It needs at least three conditions.",
        ],
      },
      {
        kind: "p",
        content: [
          "None of these tests whether the groups share a shape; read the box plot for that. A small p-value is evidence that they differ in their typical value, and it does not prove by how much or why.",
        ],
      },
    ],
  },
  {
    id: "relationships-between-columns",
    title: "Relationships between columns",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Relationships between columns"),
          " family asks whether one column moves with another: the correlation coefficient for a pair, a regression line for a model, the components or factors behind a set of columns, and the odds of a two-level outcome.",
        ],
      },
      { kind: "h3", text: "Correlation" },
      {
        kind: "p",
        content: [
          "Takes two to twenty numeric columns and reports the coefficient between every pair, on the rows complete in both of that pair. ",
          strong("Pearson"),
          " measures a linear association; ",
          strong("Spearman"),
          " measures a monotone one and uses the t approximation; ",
          strong("Kendall"),
          " reports tau-b with the normal approximation. Each pair gets a test and an interval, and the whole matrix is shown with 1 on its diagonal. A pair with fewer than three complete rows, or with a column that does not vary, has no coefficient: the result says why and warns rather than showing a number. Scatter plots are drawn when there are at most four pairs. A strong curved relationship can read near 0, so a scatter plot is the first thing to read.",
        ],
      },
      { kind: "h3", text: "Linear regression" },
      {
        kind: "p",
        content: [
          "Explains one ",
          strong("Outcome"),
          " column from one to twenty ",
          strong("Predictors"),
          " by ordinary least squares. It reports each coefficient with its standard error, t statistic, p-value and interval, the standardised coefficients, each predictor's variance inflation factor, and prediction intervals at predictor values you give (up to 1,000 rows). One predictor is drawn as a scatter plot with the fitted line and the confidence band of the mean response; the residuals against the fitted values and a normal Q-Q plot of the residuals are always drawn. Shapiro–Wilk checks the residuals' normality, the studentized Breusch–Pagan test a constant variance, and the exact Durbin–Watson test independence in the order of the rows. Linearity is not one number: read the residuals against each predictor. A predictor that does not vary, or that is exactly collinear with the others, is refused, and so is a model with fewer than one residual degree of freedom.",
        ],
      },
      { kind: "h3", text: "Principal component analysis" },
      {
        kind: "p",
        content: [
          "Reads two to thirty numeric columns on the rows complete in all of them and turns them into components — orthogonal linear combinations, ordered by the variance they explain. ",
          strong("Standardise the columns"),
          ", the default, decomposes the correlation matrix, so a column in large units does not drown the others; turning it off decomposes the covariance matrix of the centred columns. ",
          strong("Most components to show"),
          " limits the tables and the plots, never the percentages, which are computed over every component. The estimates give the variance the first component and the first two components explain, and how many components reach 80 %. The components table carries each eigenvalue, its share of the variance, the cumulative share and its loadings; the scores table summarises the rows' scores. A scree plot shows the eigenvalues, and the scores on the first two components are drawn as a scatter plot when both are shown. Bartlett's test of sphericity asks whether the columns are uncorrelated at all, and the Kaiser-Meyer-Olkin measure how much variance they share: a small p-value and a high measure are what make the components worth reading. A constant column is refused under standardisation and left out with a warning otherwise, and fewer than four complete rows are refused. A component is a direction in the data, not a cause, and a loading is not proof that a column belongs to it.",
        ],
      },
      { kind: "h3", text: "Factor analysis" },
      {
        kind: "p",
        content: [
          "Reads three to thirty numeric columns and factors their correlation matrix by principal-axis factoring, with the communalities iterated from the squared multiple correlations. ",
          strong("Number of factors"),
          " runs from one to the smaller of 10 and one fewer than the columns, and ",
          strong("Rotation"),
          " turns them by Kaiser's normalised varimax or leaves them unrotated. The estimates give the Kaiser-Meyer-Olkin measure with its reading, the share of the variance the factors explain and Kaiser's criterion, the number of factors with an eigenvalue above one; Bartlett's test of sphericity is reported as a test. The loadings table carries each column's loadings, its communality and its uniqueness, the variance table each factor's share, and the scores table summarises the regression-method scores; the first two factors are drawn as a scatter plot when there are two or more. Each column's outliers are checked against Tukey's fences, and linearity is left to the scatter plots because no single number reads it. A communality at or above one is a Heywood case: it is warned about and the numbers are still shown. Fewer than ten complete rows, a constant column, a singular correlation matrix or a factor count the columns cannot carry is refused. A factor is a direction in the data, not a latent cause.",
        ],
      },
      { kind: "h3", text: "Logistic regression" },
      {
        kind: "p",
        content: [
          "Explains a two-level ",
          strong("Outcome"),
          " from one to twenty numeric ",
          strong("Predictors"),
          " by the log odds of the event, which is the outcome label that appears second, in the order of first appearance; the result names both labels. It reports each coefficient with its standard error, z statistic, p-value, Wald interval and odds ratio, the likelihood ratio test of the model, McFadden's and Nagelkerke's pseudo R squared, the AIC, the area under the ROC curve, and sensitivity, specificity and accuracy at the 0.5 cut-off. The coefficients table shows the fit, the classification table its agreement with the observed labels, and the calibration plot the fitted probabilities against the observed rates; predictions can be made at predictor values you give (up to 1,000 rows). The Hosmer-Lemeshow test checks calibration, while linearity of the logit and independence of the rows are stated as things no single number can read. Complete or quasi-complete separation is warned about with the coefficients still shown, and a fit that did not converge says so. A predictor that does not vary, or is exactly collinear with the others, is refused. An odds ratio measures association in these rows; it does not show that a predictor causes the outcome.",
        ],
      },
    ],
  },
  {
    id: "counts",
    title: "Counts",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Counts"),
          " family tabulates two label columns and tests whether they are associated.",
        ],
      },
      { kind: "h3", text: "Contingency table" },
      {
        kind: "p",
        content: [
          "Pairs a ",
          strong("Rows"),
          " label with a ",
          strong("Columns"),
          " label by row position and counts every combination. It reports the Pearson chi-square test of independence, Cramér's V as the effect size, and for a 2 x 2 table the odds ratio and the risk ratio with their intervals. The exact test replaces the chi-square test for a 2 x 2 table whose smallest expected count is below 5. Four tables show the counts, the row percentages, the column percentages and each cell's share of the total. A row without a label in either column is left out and listed.",
        ],
      },
      { kind: "h3", text: "McNemar's test" },
      {
        kind: "p",
        content: [
          "Reads the same subjects' two-level labels ",
          strong("Before"),
          " and ",
          strong("After"),
          " a change, paired by row position, and tests whether the two marginal proportions are equal. It uses the exact binomial distribution when the discordant pairs number at most 25, and the chi-square distribution with a continuity correction otherwise. The effect is the difference in the two marginal proportions, with Newcombe's method 10 interval for paired proportions. A level that the paired rows leave empty is dropped with a warning.",
        ],
      },
      {
        kind: "p",
        content: [
          "A small p-value is evidence of an association (or of a change), not proof of its cause, and a large one is not evidence that the columns are independent.",
        ],
      },
    ],
  },
  {
    id: "comparing-two-methods",
    title: "Comparing two methods",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Method comparison"),
          " family asks whether two methods that measure the same quantity agree closely enough to be used interchangeably. All three tools pair ",
          strong("Method A"),
          "'s readings with ",
          strong("Method B"),
          "'s by row position on the rows complete in both, and count and list every row left out.",
        ],
      },
      { kind: "h3", text: "Bland–Altman agreement" },
      {
        kind: "p",
        content: [
          "Takes the difference of each pair (method A minus method B) against the pair's mean, and reports the bias, the standard deviation of the differences and the limits of agreement at the bias plus or minus 1.96 standard deviations, each with an interval at the chosen level; the limits themselves stay at 95 %. Points beyond the limits are listed, never removed: they are where the methods disagree most. A slope away from zero in the regression of the difference on the mean is proportional bias — the difference changes with the size of the reading. Shapiro–Wilk checks the differences' normality and the studentized Breusch–Pagan test a constant variance. The difference-against-mean plot carries the bias, both limits and zero; a second plot puts method A against method B with the line of equality.",
        ],
      },
      { kind: "h3", text: "Deming regression" },
      {
        kind: "p",
        content: [
          "Fits the line relating method B to method A while allowing for measurement error in both, weighted by the ",
          strong("Error ratio"),
          " of their error variances. One means both methods carry the same error variance; a larger value means method B is the noisier one. That ratio is an assumption the fit makes, not a quantity these data estimate, and the result says so. The standard errors come from the delete-one jackknife.",
        ],
      },
      { kind: "h3", text: "Passing–Bablok regression" },
      {
        kind: "p",
        content: [
          "Fits the same line with no assumption about the distribution of the errors: the slope is the shifted median of the slopes between every pair of points, and the intercept the median of the intercepts through that slope. Its intervals are the rank-based intervals of Passing and Bablok. A straight-line relation is still assumed, and the plot against the line of identity is where a departure shows.",
        ],
      },
      {
        kind: "p",
        content: [
          "For all three, a slope away from one or an intercept away from zero is read at a glance against the line of identity. Agreement between two methods is evidence about these samples, not proof that the methods may be used interchangeably in every range.",
        ],
      },
    ],
  },
  {
    id: "measurement-and-classification",
    title: "Measurement and classification",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Measurement and classification"),
          " family reads a measurement against a standard: how well a score separates two classes, how consistently a set of items measures one thing, and whether a process stays inside its limits.",
        ],
      },
      { kind: "h3", text: "Receiver operating characteristic curve" },
      {
        kind: "p",
        content: [
          "Pairs a numeric ",
          strong("Score"),
          " with a two-level ",
          strong("Outcome"),
          " by row position on the rows complete in both, and reads the outcome label that appears second, in the order of first appearance, as the positive class; both labels and their counts are named. A higher score is read as pointing to the positive class unless ",
          strong("A higher score means positive"),
          " is turned off, when the scores are negated first. The curve is drawn at every distinct score, and the area under it is the probability that a randomly chosen positive row scores above a randomly chosen negative row, with ties counted as one half; its interval is DeLong's. ",
          strong("Best cut-off (Youden)"),
          " is the threshold that maximises sensitivity plus specificity minus one, and the sensitivity and specificity at that cut are reported beside it. A two-sided Mann-Whitney U test of the scores is reported with the tie-corrected normal approximation, a threshold table lists the curve's points, and the ROC curve is plotted. Fewer than six rows in either class, or fewer than three distinct scores, are refused. The curve ranks a score: it says how well the score orders the positive rows above the negative ones, not that the score diagnoses anything, and a cut chosen from these rows needs confirming on new ones.",
        ],
      },
      { kind: "h3", text: "Reliability (Cronbach's alpha)" },
      {
        kind: "p",
        content: [
          "Reads two to fifty numeric ",
          strong("Items"),
          " measured on the same rows and reports the internal consistency of their sum: Cronbach's alpha with Feldt's interval at the chosen level, the standardised alpha and the mean inter-item correlation. An F test of the reliability coefficient (Feldt) tests alpha against zero. Every item is listed with its mean, standard deviation, corrected item-total correlation and the alpha the scale would have with that item deleted. An item named in ",
          strong("Reverse-keyed items"),
          " is recoded as max + min − value on its own observed range and the result says which items were recoded; an item that correlates negatively with the rest and is not named is warned about. Fewer than three complete rows, a constant item or an item count outside the tool's range is refused. Alpha is a lower bound on reliability when the items are not tau-equivalent, and tau-equivalence cannot be tested from these rows; a high alpha is consistent with a set of items measuring one construct without establishing that they do.",
        ],
      },
      { kind: "h3", text: "Control chart and process capability" },
      {
        kind: "p",
        content: [
          "Reads one column of ",
          strong("Values"),
          " in row order as an individuals chart, or as an X-bar and R chart when ",
          strong("Chart"),
          " says so. The X-bar and R chart reads the ",
          strong("Subgroups"),
          " label column, takes each maximal run of one label in order and cuts it into consecutive subgroups of ",
          strong("Subgroup size"),
          " (two to ten, five when empty); a piece shorter than that size is left out, counted and listed. The individuals chart sets its centre line at the mean and its limits at the mean plus or minus 2.66 times the mean moving range, from d2 = 1.128; the X-bar and R chart sets its centre line at the grand mean and its limits from A2, D3 and D4 times the mean range, with the constants for subgroup sizes 2 to 10 taken from Montgomery's ",
          em("Introduction to Statistical Quality Control"),
          ". The estimates give the centre line, both control limits and the mean moving range (or the mean range), and the chart statistics table repeats them with the range chart's limits and the other figures; the control chart plot draws the points with the centre line and both limits. The Western Electric rules 1 (a point beyond a limit), 2 (two of three consecutive points beyond two standard deviations on the same side) and 4 (eight consecutive points on one side) name their points in a special-causes table when they fire, and a stability check reads the chart for special causes. At least 20 points are needed for the individuals chart and ten complete subgroups for the X-bar and R chart, and a constant series is refused. Both ",
          strong("Lower specification limit"),
          " and ",
          strong("Upper specification limit"),
          " add Cp, Cpk, Pp, Ppk and the estimated percentage of values outside the limits, after Shapiro-Wilk's W test checks normality. A capability index needs a stable, normal process: it describes the process the data came from only while the chart shows it in control and the values are close enough to normal.",
        ],
      },
    ],
  },
  {
    id: "sample-size-and-power",
    title: "Sample size and power",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Sample size and power"),
          " family needs no data. It answers one of two questions about a design: how many observations per group a target power needs, or what power a given number of observations reaches.",
        ],
      },
      { kind: "h3", text: "Sample size and power" },
      {
        kind: "p",
        content: [
          "Choose the ",
          strong("Design"),
          ": the one-sample, independent or paired t test, the one-way analysis of variance, the Pearson correlation, or the difference of two proportions. Give exactly one of ",
          strong("Power"),
          " and ",
          strong("N per group"),
          ", and the other is solved, with the sample size the smallest whole number that reaches the target. ",
          strong("Effect size"),
          " is Cohen's d for the one-sample and independent t tests, dz for the paired test, Cohen's f for the analysis of variance, and the correlation itself for the correlation test; the proportions design takes ",
          strong("First proportion"),
          " and ",
          strong("Second proportion"),
          " instead. ",
          strong("Alpha"),
          " is the significance level of the two-sided test. ",
          strong("Group size ratio"),
          " handles unequal groups, and ",
          strong("Number of groups"),
          " sets the groups of the analysis of variance. A table scans the power over sample size, from the first size that reaches 0.05 to the first that reaches 0.99, so you can see how quickly the design becomes useful.",
        ],
      },
      {
        kind: "p",
        content: [
          "Every number is a statement about the design and significance level you stated. It is not a p-value, and it does not prove that the study will find an effect or that the effect is real.",
        ],
      },
    ],
  },
  {
    id: "advanced-models",
    title: "Advanced models",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Advanced models"),
          " family fits models for data whose structure one line cannot hold: measurements that share a group effect, and time until an event that some rows never reach.",
        ],
      },
      { kind: "h3", text: "Linear mixed model" },
      {
        kind: "p",
        content: [
          "Explains a numeric ",
          strong("Outcome"),
          " from one to twenty ",
          strong("Predictors"),
          " with a fixed line, and gives every level of the ",
          strong("Group"),
          " column its own random intercept — an amount drawn from one normal distribution with mean zero and one shared variance, added to that group's rows. The fixed effects are estimated by restricted maximum likelihood (REML), each coefficient with its standard error, t statistic, p-value and interval, on Satterthwaite's degrees of freedom. The random intercept is tested against an ordinary least-squares line by a likelihood-ratio test whose null distribution is half chi-square with zero degrees of freedom and half with one. The estimates give the residual and between-group variances, the intraclass correlation, and Nakagawa and Schielzeth's marginal and conditional R squared; a group-effects table and a plot show each group's intercept shift, and predictions can be made at predictor values you give (up to 1,000 rows). A predictor that does not vary or is exactly collinear with the others is refused, as is a constant outcome, fewer than six rows or fewer than three groups, and every row left out is counted and listed.",
        ],
      },
      {
        kind: "p",
        content: [
          "A random intercept is not a random slope: the model assumes each predictor acts the same way in every group, and a group whose slope differs is not described by it. REML assumes the group effects and the errors are normally distributed, and whether the group effects are normal is something these data cannot show; with few groups, treat the group as a fixed factor instead. A random effect the test finds is evidence that the groups differ, not a measurement of how much of the variation lies between them.",
        ],
      },
      { kind: "h3", text: "Survival analysis" },
      {
        kind: "p",
        content: [
          "Reads one numeric ",
          strong("Time"),
          " column — the follow-up time to the event or to censoring — and a two-level ",
          strong("Event"),
          " column, whose level second in order of first appearance is the event unless ",
          strong("Event level"),
          " names it. A ",
          strong("Groups"),
          " column draws each group's Kaplan–Meier curve and reports its median with the Brookmeyer–Crowley interval and its restricted mean survival to the largest observed time with its standard error, and the log-rank test compares the groups' survival, with Holm's adjustment for the pairwise comparisons. ",
          strong("Predictors"),
          " fit a Cox model by Newton–Raphson on the partial likelihood with Breslow's handling of tied event times: each hazard ratio carries its Wald interval, the model a likelihood-ratio test, and Harrell's concordance index how well the risks order the event times. Proportional hazards is checked approximately by the correlation of the scaled Schoenfeld residuals with the event time, from ten events; a fit that does not converge is reported with its last change, never returned as if it had. A time must not be below zero, and at least five events and two distinct times are needed.",
        ],
      },
      {
        kind: "p",
        content: [
          "Censoring is assumed independent of the event time, and that is an assumption these data cannot show: it is argued from how the follow-up ended. A Cox hazard ratio is meaningful only while the hazards stay proportional, so read a coefficient the check finds changing as an average over the follow-up. A survival curve describes these rows; it does not predict any one person's time.",
        ],
      },
    ],
  },
  {
    id: "evidence-synthesis",
    title: "Evidence synthesis",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Evidence synthesis"),
          " family combines the results of several studies into one estimate and says how much the studies disagree.",
        ],
      },
      { kind: "h3", text: "Meta-analysis" },
      {
        kind: "p",
        content: [
          "Reads one ",
          strong("Effects"),
          " column and its ",
          strong("Standard errors"),
          ", one row per study, or, for a ratio measure, the four columns of each study's two-by-two table; ",
          strong("Study labels"),
          " name the rows and ",
          strong("Measure"),
          " says what the effects are — a generic effect, a mean difference, a standardised mean difference, a correlation, an odds ratio or a risk ratio. The studies are pooled by inverse variance, both as a fixed-effect mean and as a random-effects mean whose weights add the DerSimonian and Laird estimate of the between-study variance, each with its interval. Cochran's Q and I squared say how much of the variation between the studies is more than sampling error, and from three studies a prediction interval says where a new study's effect might fall. From ten studies Egger's test bears on publication bias. A forest plot draws the studies and both pools, a funnel plot sets each effect against its standard error with that axis inverted, and a two-by-two table with a zero or complete cell is refused rather than continuity-corrected. A row missing a value is left out and listed.",
        ],
      },
      {
        kind: "p",
        content: [
          "Pooling assumes the studies measure one quantity on the scale the measure names, and that they are independent, which a column cannot record. A pooled estimate cannot rule out publication bias: a funnel plot and Egger's test can suggest asymmetry, and only from ten studies, but neither establishes that small studies reporting nothing are absent. A high I squared says the studies disagree more than chance explains, and the random-effects pool averages that disagreement rather than resolving it.",
        ],
      },
    ],
  },
  {
    id: "bayesian-tests",
    title: "Bayesian tests",
    blocks: [
      {
        kind: "p",
        content: [
          "The ",
          strong("Bayesian tests"),
          " family answers the questions the t tests ask in the currency of evidence: how much more likely the data are under an alternative than under a null, and what the posterior says about the size of the effect.",
        ],
      },
      { kind: "h3", text: "Bayesian test" },
      {
        kind: "p",
        content: [
          "Choose the ",
          strong("Design"),
          ": one sample against a hypothesised mean, paired observations, two independent groups, or a single proportion from counts; a field the design does not take is refused. The three effect designs report the JZS Bayes factor of Rouder and colleagues, which integrates the likelihood ratio against a Cauchy prior on the standardised effect whose ",
          strong("Prior scale"),
          " is 0.707 unless you set another, and the posterior of that prior: its mean, its credible interval at ",
          strong("Level"),
          ", and the probability the effect falls inside ",
          strong("Null region"),
          ", the range you count as no practical effect. The proportion design takes ",
          strong("Successes"),
          " and ",
          strong("Trials"),
          " under a ",
          strong("Prior alpha"),
          " and ",
          strong("Prior beta"),
          ", and reports the beta posterior, the probability the proportion exceeds the hypothesised value, and its Bayes factor by the Savage–Dickey ratio. A posterior density plot is drawn.",
        ],
      },
      {
        kind: "p",
        content: [
          "A Bayes factor is evidence under the stated prior, not the probability that either hypothesis is true, and it is not a p-value: change the prior and the number moves, so report the prior with it. A credible interval is a statement about the effect given these data and that prior, and the null region is a choice of what counts as no practical effect, not a fact found in the data.",
        ],
      },
    ],
  },
  {
    id: "about-licence-credits",
    title: "About, licence, projects and policy",
    blocks: [
      {
        kind: "p",
        content: [
          "CurveStudio is built by ",
          strong("ConcreteInfo"),
          " — ",
          link("https://www.concreteinfo.in", "https://www.concreteinfo.in"),
          " · amit@concreteinfo.in. The ",
          link("About page", "/about"),
          " and the About window list the other tools we build and maintain, in two groups —",
          " open source (PC Power Calculator, Markdown Converter, GitHub Desktop for Linux) and",
          " commercial software (ConcreteInfo Mix Design Platform, CAMS2) — explain why the",
          " desktop app is free and AGPL, and carry the disclaimer and the privacy/fair-use",
          " policy in full. Three example datasets ship with the app: ",
          code("example.csv"),
          " (age against strength), ",
          code("example-decay.csv"),
          " (first-order decay) and ",
          code("example-kinetics.csv"),
          " (enzyme kinetics), so no single discipline leads.",
        ],
      },
      {
        kind: "ul",
        items: [
          [
            "The desktop app is free software under AGPL-3.0-or-later; the web app is free today",
            " and a small hosting subscription may be added later — the desktop stays free.",
          ],
          [
            "A fit describes agreement with ",
            strong("your"),
            " sample. It is not proof of a physical law, code compliance or safety — verify",
            " anything that matters against the standard and a second method.",
          ],
          [
            "Desktop data stays on your machine: the app works fully offline with no account,",
            " and on first launch and after an upgrade it sends one anonymous install record",
            " (version, operating system, package type) so we know how many installations",
            " exist — Edit → Preferences → Privacy turns it off.",
            " Nothing else ever leaves your computer.",
            " On the web your datasets are stored as the whole table you uploaded,",
            " and your fits too, in your account only to provide the service; deleting",
            " a dataset, a fit or your account removes them.",
          ],
        ],
      },
      {
        kind: "ul",
        items: [
          ["CurveStudio: AGPL-3.0-or-later."],
          ["Qt / PySide6 (desktop): LGPL-3.0."],
          ["NumPy, SciPy, SymPy, pandas: BSD-3-Clause."],
          ["pyqtgraph: MIT."],
          ["Inter and JetBrains Mono typefaces: SIL Open Font License 1.1."],
          ["Lucide icons: ISC."],
          [
            "The model library follows the published forms of the equations it implements; the details report names the source for each family.",
          ],
        ],
      },
      {
        kind: "p",
        content: [
          "Version and source-code links are on the ",
          link("About page", "/about"),
          ".",
        ],
      },
    ],
  },
];

/** All visible text of a section, lower-cased — what the search box filters on. */
export function sectionSearchText(section: HelpSection): string {
  const inlineText = (node: Inline): string => (typeof node === "string" ? node : node.text);
  const parts: string[] = [section.title];
  for (const block of section.blocks) {
    if (block.kind === "p") parts.push(...block.content.map(inlineText));
    else if (block.kind === "h3") parts.push(block.text);
    else if (block.kind === "ul" || block.kind === "ol") {
      for (const item of block.items) parts.push(...item.map(inlineText));
    } else if (block.kind === "figure") {
      parts.push(block.alt, block.caption);
    } else {
      parts.push(...block.headers);
      for (const row of block.rows) {
        for (const cell of row) parts.push(...cell.map(inlineText));
      }
    }
  }
  return parts.join(" ").toLowerCase();
}
