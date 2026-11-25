/**
 * The chart bundle, prefetched while a signed-in visitor's browser is idle (P6 Task 3).
 *
 * After the route split a sign-in fetches its chunks in sequence — the signed-in shell,
 * the workspace page, then Plotly — because each fetch starts only when the previous
 * chunk's code has run. Starting the chart chunk's download once a session is authed and
 * the main thread goes idle overlaps its transfer with the shell and workspace fetches
 * and with the visitor picking a dataset, so the first chart waits on its render, not on
 * its download (docs/perf.md, "the first chart after sign-in").
 *
 * The prefetch is the same dynamic-import specifier PlotChart's loader uses, so the
 * browser's module map deduplicates: a chart mounted while the prefetch is still in
 * flight waits on the same fetch instead of starting a second one, and one mounted
 * after it resolves gets the module from memory. Only a signed-in visitor pays the
 * bytes; an anonymous one never schedules anything.
 */

import { useEffect } from "react";
import type { AuthStatus } from "../api/auth";

/**
 * The idle deadline: even a busy page prefetches within two seconds, and the timeout is
 * what guarantees the callback runs at all. The no-`requestIdleCallback` fallback (old
 * Safari) waits the same order of time, so it never competes with first paint.
 */
const IDLE_TIMEOUT_MS = 2_000;

/** The import itself: one specifier, resolved against this module (src/lib), which is
 *  the same file PlotChart's `../lib/plotly-bundle` resolves to. */
function prefetchChartBundle(): void {
  void import("./plotly-bundle");
}

export function useChartPrefetch(status: AuthStatus): void {
  useEffect(() => {
    if (status !== "authed") return;
    if (typeof window.requestIdleCallback === "function") {
      const handle = window.requestIdleCallback(() => prefetchChartBundle(), {
        timeout: IDLE_TIMEOUT_MS,
      });
      return () => window.cancelIdleCallback(handle);
    }
    const handle = window.setTimeout(prefetchChartBundle, IDLE_TIMEOUT_MS);
    return () => window.clearTimeout(handle);
  }, [status]);
}
