/**
 * Fonts embedded in the fit-report PDF. The body face is Inter (400/600/700) and the
 * numeric/equation face is JetBrains Mono (400), both shipped in
 * `public/fonts/report` with their OFL licences.
 *
 * `Font` is imported statically because it is a tiny registry entry point; the heavy
 * renderer is only ever reached through the dynamic import in `export.ts`, so this module
 * is itself part of the lazily-loaded report chunk.
 */

import { Font } from "@react-pdf/renderer";

export const REPORT_FONT_FAMILIES = {
  body: "Inter",
  numeric: "JetBrains Mono",
} as const;

let registered = false;

/**
 * Register the report's embedded faces. Idempotent: react-pdf keeps a process-global
 * registry, so calling this once per export (or from several exports) is safe.
 *
 * `base` is the public URL prefix in the browser (`/fonts/report`).
 */
export function registerReportFonts(base = "/fonts/report"): void {
  if (registered) return;

  Font.register({
    family: REPORT_FONT_FAMILIES.body,
    fonts: [
      { src: `${base}/Inter-Regular.ttf`, fontWeight: 400 },
      { src: `${base}/Inter-SemiBold.ttf`, fontWeight: 600 },
      { src: `${base}/Inter-Bold.ttf`, fontWeight: 700 },
    ],
  });

  Font.register({
    family: REPORT_FONT_FAMILIES.numeric,
    fonts: [{ src: `${base}/JetBrainsMono-Regular.ttf`, fontWeight: 400 }],
  });

  registered = true;
}
