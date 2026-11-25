/**
 * Plotly built from its modules — the app's whole chart library surface. `plotly.js`
 * (the source distribution of the same locked 4.3.0 the app shipped as
 * `plotly.js-dist-min`) exposes `lib/core` plus one module per trace; registering only
 * what the app draws keeps the lazily-loaded chart chunk to a fraction of the full
 * bundle (docs/perf.md, "Web: the built bundles").
 *
 * The registered set is exactly the traces grep can find in the app:
 * - `bar` — the statistics histogram is drawn as bars on the server's bin edges, and the
 *   descriptives and comparison tools draw bar-style summaries;
 * - `box` — the statistics box plot, drawn from the server's quartiles;
 * - `scattergl` — the data trace above `SCATTER_MAX_SVG_POINTS` points (workspace/traces.ts);
 * - `scatter` needs no registration: it is part of `lib/core` itself and cannot be
 *   removed (the library's CUSTOM_BUNDLE.md).
 * The Plotly `histogram` trace is deliberately NOT registered: no chart builds one (the
 * statistics histogram bins server-side and draws `bar`), so it would be dead bytes.
 * plotly-bundle.test.ts pins this set; change it there first, with a reason.
 */

import Plotly from "plotly.js/lib/core";
import bar from "plotly.js/lib/bar";
import box from "plotly.js/lib/box";
import scattergl from "plotly.js/lib/scattergl";

Plotly.register([bar, box, scattergl]);

export default Plotly;
export type {
  DownloadImageOptions,
  PlotlyAnnotation,
  PlotlyConfig,
  PlotlyLayout,
  PlotlyTrace,
} from "plotly.js/lib/core";
