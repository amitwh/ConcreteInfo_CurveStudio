/**
 * The statistics catalogue, fetched once and kept for the session: the families a person
 * meets the tools in, and every tool with the fields a form shows. A failed fetch is
 * reported and retried on the next mount; a second mount during the first fetch shares it.
 */

import { useEffect, useState } from "react";
import { api, type ToolInfo } from "../../api/client";

export interface StatsTools {
  families: [string, string][];
  tools: ToolInfo[];
}

export interface StatsToolsState {
  families: [string, string][];
  tools: ToolInfo[];
  /** Why the catalogue could not be fetched; null while it has not failed. */
  error: unknown;
  /** True while no catalogue has arrived yet. */
  loading: boolean;
}

let cached: StatsTools | null = null;
let pending: Promise<StatsTools> | null = null;

/** The catalogue, from the session's cache or the one request in flight. */
function load(): Promise<StatsTools> {
  if (cached !== null) return Promise.resolve(cached);
  pending ??= api.statsTools().then(
    (out) => {
      const catalogue: StatsTools = { families: out.families, tools: out.tools };
      cached = catalogue;
      pending = null;
      return catalogue;
    },
    (error: unknown) => {
      pending = null;
      throw error;
    },
  );
  return pending;
}

export function useStatsTools(): StatsToolsState {
  const [state, setState] = useState<StatsToolsState>(() =>
    cached === null
      ? { families: [], tools: [], error: null, loading: true }
      : { families: cached.families, tools: cached.tools, error: null, loading: false },
  );

  useEffect(() => {
    if (!state.loading) return;
    let active = true;
    load().then(
      (catalogue) => {
        if (!active) return;
        setState({ ...catalogue, error: null, loading: false });
      },
      (error: unknown) => {
        if (!active) return;
        setState({ families: [], tools: [], error, loading: false });
      },
    );
    return () => {
      active = false;
    };
  }, [state.loading]);

  return state;
}
