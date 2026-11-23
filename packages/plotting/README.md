# curvestudio-plotting

Publication-quality Matplotlib figures for CurveStudio fits.

Builds `matplotlib.figure.Figure` objects (never `pyplot` global state) from a
`curvestudio.workflow.FitBundle`:

- `fit_figure` — data points, the fitted curve with confidence/prediction bands,
  and an unambiguous banner when the fit did not converge (a curve is never drawn
  for a failed fit);
- `residual_figure` — the 2x2 residual panel set (vs fitted, vs x, normal Q-Q,
  histogram) built from the diagnostics data, with "n/a" panels where a piece is
  undefined;
- `save_figure` — deterministic PNG/SVG/PDF export (no timestamps in the output,
  SVG text as paths);
- `add_byline` — prints "Prepared by <identity>", small and muted, along the top
  edge of a figure from either builder (the argument is the identity text alone;
  nothing is drawn or reserved when it is blank).

Licence: AGPL-3.0-or-later, like the rest of CurveStudio.
