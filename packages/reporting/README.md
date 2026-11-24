# curvestudio-reporting

Reports for a CurveStudio fit, rendered from a `curvestudio.workflow.FitBundle`:

- `fit_to_json` — the full bundle wrapped with a report header (tool, versions,
  IST timestamp, statistical convention, assumptions);
- `parameters_csv` / `observations_csv` / `statistics_csv` — CSV tables with
  `repr` floats, undefined statistics as "not defined" notes, and spreadsheet
  formula injection neutralised;
- `fit_to_xlsx` — a workbook (Summary, Parameters, Observations, Statistics,
  Plot) with the fit figure embedded;
- `fit_to_pdf` — a full report: identification, equation, parameter and
  statistics tables, fit and residual figures, assumptions and limitations; a
  fit that did not converge gets a first-page red warning and no fit figure.

`fit_to_xlsx` and `fit_to_pdf` also take an optional `byline`: the identity text
alone (a name, an organisation, or both). "Prepared by <byline>" prints under the
PDF's title, where the identity alone is also set as the author, and in the centre
of every sheet's print header; without one both files are as they have always been.

Both also take the fit's `methods` text and `audit` (`curvestudio.tools.methods_text`
and `audit_fit`). The PDF then carries two sections after the statistics table:
Methods (the paragraph, the audit's methods fragment, the equation with its
standard errors) and Fit audit (the audit's summary, then one paragraph per check
that reads caution or fail: title, evidence, and the next step after an arrow). The
workbook's Summary sheet carries the same under the line that names the convention,
with each flagged check as a row (title, evidence, next step). A check that passed
or could not run prints nothing of its own; without either argument both files are
as they have always been; a fit that did not converge gets neither.

Text safety: every cell that could be read as a spreadsheet formula
(`= + - @`, also behind leading whitespace) is neutralised with a leading
apostrophe, XML-illegal control characters are stripped, and long strings are
wrapped/truncated in the PDF tables. When a compact bundle's data snapshot is a
subsample, the Observations exports and the PDF say so explicitly ("showing N
of M observations (subsampled)").

Fonts: the PDF embeds the DejaVu Sans TrueType font that ships with matplotlib,
so Greek, Cyrillic and common symbols (σ², ≤, −) render as real glyphs and the
text stays extractable. Known limitation: DejaVu covers Latin/Greek/Cyrillic
and common symbols only — CJK characters are not in the font, and reportlab
does no complex-script shaping, so right-to-left scripts are not covered
either. Figures use matplotlib's own DejaVu Sans and have the same coverage.

Licence: AGPL-3.0-or-later, like the rest of CurveStudio.
