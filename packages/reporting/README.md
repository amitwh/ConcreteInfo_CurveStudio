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
