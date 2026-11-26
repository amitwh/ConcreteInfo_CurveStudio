import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import {
  api,
  type DatasetDetailOut,
  type DatasetOut,
  type FitSummaryOut,
  type ModelInfo,
} from "../api/client";
import {
  DETAIL_CAP,
  METADATA_TTL_MS,
  clear,
  dropDataset,
  fetchDatasets,
  fetchDetail,
  fetchFits,
  fetchModels,
  getDatasets,
  getDetail,
  getFits,
  getModels,
  invalidateDatasets,
  invalidateFits,
} from "./workspaceCache";

const DATASET_A: DatasetOut = {
  id: "ds-1",
  name: "growth.csv",
  n_rows: 2,
  columns: ["x", "y"],
  x_column: "x",
  y_column: "y",
  weight_column: null,
  issues: [],
  created_at: "2026-11-17T05:00:00Z",
};

const DATASET_B: DatasetOut = { ...DATASET_A, id: "ds-2", name: "gen.csv" };

/** The first fetch answers the one-dataset list; every later one answers two, so a test
 *  can see which answer the cache actually holds. */
const LIST_FIRST = [DATASET_A];
const LIST_SECOND = [DATASET_B, DATASET_A];

const MODELS_FIRST = [{ name: "exp_decay" }] as unknown as ModelInfo[];
const MODELS_SECOND = [{ name: "quadratic" }] as unknown as ModelInfo[];

/** One stable detail object per dataset, so identity asserts what the cache holds. */
const DETAILS = new Map<string, DatasetDetailOut>();

function detailFor(id: string): DatasetDetailOut {
  const existing = DETAILS.get(id);
  if (existing !== undefined) return existing;
  const detail = {
    ...DATASET_A,
    id,
    points: null,
    table_stored: false,
    table_columns: [],
  } as DatasetDetailOut;
  DETAILS.set(id, detail);
  return detail;
}

const DETAIL_A = detailFor("ds-1");
const FITS_A = [
  {
    id: "fit-1",
    dataset_id: "ds-1",
    created_at: "2026-11-17T08:00:00Z",
    status: "converged",
    spec: {},
  },
] as FitSummaryOut[];

/** Per-path request log plus the answers: a path's first fetch answers the FIRST
 *  fixture, the later ones the SECOND (details and fits answer per dataset). */
let request: MockInstance;
let requests: string[];

beforeEach(() => {
  requests = [];
  const counts = new Map<string, number>();
  request = vi.spyOn(api, "request").mockImplementation((async (path: string) => {
    const target = String(path);
    requests.push(target);
    counts.set(target, (counts.get(target) ?? 0) + 1);
    if (target === "/datasets") return (counts.get(target) ?? 0) === 1 ? LIST_FIRST : LIST_SECOND;
    if (target === "/models") return (counts.get(target) ?? 0) === 1 ? MODELS_FIRST : MODELS_SECOND;
    const detail = /^\/datasets\/(ds-\d)\?points=\d+$/.exec(target);
    if (detail !== null) return detailFor(detail[1] as string);
    if (/^\/datasets\/ds-\d\/fits$/.test(target)) return FITS_A;
    throw new Error(`unexpected request: ${target}`);
  }) as unknown as typeof api.request);
  vi.useFakeTimers();
  // The cache is module-level: a test must not meet another's entries.
  clear();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("workspaceCache — the small metadata", () => {
  it("starts empty and an invalidation empties it again", async () => {
    expect(getDatasets()).toBeNull();
    expect(getModels()).toBeNull();
    await fetchDatasets();
    await fetchModels();
    expect(getDatasets()?.data).toBe(LIST_FIRST);
    expect(getModels()?.data).toBe(MODELS_FIRST);
    invalidateDatasets();
    expect(getDatasets()).toBeNull();
    // The catalogue has no invalidation: nothing in the app changes it.
    expect(getModels()?.data).toBe(MODELS_FIRST);
  });

  it("reports a list stale once it is older than the TTL, fresh before", async () => {
    await fetchDatasets();
    expect(getDatasets()).toEqual({ data: LIST_FIRST, stale: false });
    vi.advanceTimersByTime(METADATA_TTL_MS + 1);
    expect(getDatasets()).toEqual({ data: LIST_FIRST, stale: true });
  });

  it("a revalidation after the TTL replaces the cached answer with a fetched one", async () => {
    await fetchDatasets();
    vi.advanceTimersByTime(METADATA_TTL_MS + 1);
    await fetchDatasets();
    expect(getDatasets()?.data).toBe(LIST_SECOND);
    expect(requests.filter((path) => path === "/datasets")).toHaveLength(2);
  });

  it("reports the catalogue stale past the TTL", async () => {
    await fetchModels();
    vi.advanceTimersByTime(METADATA_TTL_MS + 1);
    expect(getModels()?.stale).toBe(true);
  });
});

describe("workspaceCache — the single flights", () => {
  it("one request however many callers share a metadata flight", async () => {
    const [first, second] = [fetchDatasets(), fetchDatasets()];
    await Promise.all([first, second]);
    expect(requests.filter((path) => path === "/datasets")).toHaveLength(1);
    expect(await second).toBe(LIST_FIRST);
  });

  it("one request per dataset and sample size for details, distinct keys fetch apart", async () => {
    const [first, second, other] = [
      fetchDetail("ds-1", 20000),
      fetchDetail("ds-1", 20000),
      fetchDetail("ds-1", 2000000),
    ];
    await Promise.all([first, second, other]);
    expect(requests).toHaveLength(2);
    expect(await second).toBe(await first);
  });

  it("one request per dataset for a fits list", async () => {
    await Promise.all([fetchFits("ds-1"), fetchFits("ds-1")]);
    expect(requests.filter((path) => path.endsWith("/fits"))).toHaveLength(1);
  });
});

describe("workspaceCache — the dataset details", () => {
  it("caches a detail by dataset and sample size", async () => {
    await fetchDetail("ds-1", 20000);
    expect(getDetail("ds-1", 20000)).toBe(DETAIL_A);
    expect(getDetail("ds-1", 2000000)).toBeNull();
    expect(getDetail("ds-2", 20000)).toBeNull();
  });

  it("keeps at most DETAIL_CAP details, evicting the least recently used", async () => {
    expect(DETAIL_CAP).toBe(3);
    await fetchDetail("ds-1", 20000);
    await fetchDetail("ds-2", 20000);
    await fetchDetail("ds-3", 20000);
    // The touch makes ds-2 the least recently used of the three.
    getDetail("ds-1", 20000);
    await fetchDetail("ds-4", 20000);
    expect(getDetail("ds-2", 20000)).toBeNull();
    expect(getDetail("ds-1", 20000)).not.toBeNull();
    expect(getDetail("ds-3", 20000)).not.toBeNull();
    expect(getDetail("ds-4", 20000)).not.toBeNull();
  });

  it("a failed detail fetch is stored no more than a failed list is", async () => {
    request.mockImplementation((async (path: string) => {
      const target = String(path);
      requests.push(target);
      if (target === "/datasets/ds-1?points=20000") {
        return requests.length === 1
          ? Promise.reject(new Error("detail failed"))
          : Promise.resolve(DETAIL_A);
      }
      throw new Error(`unexpected request: ${target}`);
    }) as unknown as typeof api.request);
    await expect(fetchDetail("ds-1", 20000)).rejects.toThrow("detail failed");
    expect(getDetail("ds-1", 20000)).toBeNull();
    await fetchDetail("ds-1", 20000);
    expect(getDetail("ds-1", 20000)).toBe(DETAIL_A);
    expect(requests).toHaveLength(2);
  });
});

describe("workspaceCache — the fits lists", () => {
  it("caches a dataset's fits list until it is invalidated", async () => {
    await fetchFits("ds-1");
    expect(getFits("ds-1")).toBe(FITS_A);
    expect(getFits("ds-2")).toBeNull();
    // A read while the list is cached serves it without a request.
    await expect(fetchFits("ds-1")).resolves.toBe(FITS_A);
    expect(requests.filter((path) => path.endsWith("/fits"))).toHaveLength(1);
    invalidateFits("ds-1");
    expect(getFits("ds-1")).toBeNull();
    await fetchFits("ds-1");
    expect(getFits("ds-1")).toBe(FITS_A);
    expect(requests.filter((path) => path.endsWith("/fits"))).toHaveLength(2);
  });
});

describe("workspaceCache — dropping a deleted dataset", () => {
  it("forgets the dataset's detail at every sample size and its fits, and nothing else", async () => {
    await fetchDetail("ds-1", 20000);
    await fetchDetail("ds-1", 2000000);
    await fetchDetail("ds-2", 20000);
    await fetchFits("ds-1");
    await fetchFits("ds-2");
    dropDataset("ds-1");
    expect(getDetail("ds-1", 20000)).toBeNull();
    expect(getDetail("ds-1", 2000000)).toBeNull();
    expect(getFits("ds-1")).toBeNull();
    expect(getDetail("ds-2", 20000)).not.toBeNull();
    expect(getFits("ds-2")).not.toBeNull();
  });
});

describe("workspaceCache — clearing at a session boundary", () => {
  it("forgets everything", async () => {
    await fetchDatasets();
    await fetchModels();
    await fetchDetail("ds-1", 20000);
    await fetchFits("ds-1");
    clear();
    expect(getDatasets()).toBeNull();
    expect(getModels()).toBeNull();
    expect(getDetail("ds-1", 20000)).toBeNull();
    expect(getFits("ds-1")).toBeNull();
  });

  it("a flight from before the clear stores nothing, and the new generation fetches its own", async () => {
    const resolvers: Array<(value: DatasetOut[]) => void> = [];
    request.mockImplementation((async (path: string) => {
      if (String(path) !== "/datasets") throw new Error(`unexpected request: ${path}`);
      requests.push(String(path));
      return new Promise<DatasetOut[]>((resolve) => {
        resolvers.push(resolve);
      });
    }) as unknown as typeof api.request);

    const stale = fetchDatasets();
    clear();
    // The cleared flight is forgotten: the new generation starts its own request.
    const fresh = fetchDatasets();
    expect(requests).toHaveLength(2);

    resolvers[0]?.(LIST_SECOND); // the stale flight's late answer
    await stale;
    expect(getDatasets()).toBeNull();

    resolvers[1]?.(LIST_FIRST); // the current flight's answer
    await fresh;
    expect(getDatasets()?.data).toBe(LIST_FIRST);
  });

  it("a flight from before an invalidation stores nothing, and a caller after the change does not share it", async () => {
    const resolvers: Array<(value: DatasetOut[]) => void> = [];
    request.mockImplementation((async (path: string) => {
      if (String(path) !== "/datasets") throw new Error(`unexpected request: ${path}`);
      requests.push(String(path));
      return new Promise<DatasetOut[]>((resolve) => {
        resolvers.push(resolve);
      });
    }) as unknown as typeof api.request);

    const stale = fetchDatasets();
    invalidateDatasets(); // an upload or a delete while the fetch was away
    const fresh = fetchDatasets(); // a caller after the change does not share the stale flight
    expect(requests).toHaveLength(2);

    resolvers[0]?.(LIST_SECOND); // the stale answer arrives, without the change
    await stale;
    expect(getDatasets()).toBeNull();

    resolvers[1]?.(LIST_FIRST); // the newer answer arrives
    await fresh;
    expect(getDatasets()?.data).toBe(LIST_FIRST);
  });
});
