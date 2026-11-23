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
          ") covers project edits. A project saved on one machine opens on another; nothing is uploaded anywhere.",
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
            " a dialog shows the traceback with a copy button. Saved project files are unaffected. Report it with the copied text.",
          ],
          [
            strong("No network (desktop):"),
            " the status bar says so. Fitting is local, so the app keeps working in full — there is no account and no upload.",
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
            "Desktop data stays on your machine. On the web your datasets and fits are stored",
            " in your account only to provide the service; deleting a dataset, a fit or your",
            " account removes them.",
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
