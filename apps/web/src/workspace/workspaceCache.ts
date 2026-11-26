/**
 * The workspace's read-through cache: the GET answers a remount (every workspace tab
 * switch unmounts and remounts WorkspacePage) would otherwise download again. Two kinds
 * of entry, with different rules:
 *
 * - Small metadata — the dataset list and the model catalogue. Served from the cache at
 *   once; once older than METADATA_TTL_MS the next mount also revalidates it in the
 *   background, so a change made elsewhere arrives one paint late instead of blocking
 *   the first.
 * - Dataset details and fits lists. Cached hard — a remount serves them with no request
 *   at all — until a mutation invalidates them. Nothing expires them by time: only the
 *   workspace's mutations can change what they hold. A detail can be tens of megabytes
 *   for a large dataset (measured in docs/perf.md), so at most DETAIL_CAP details stay
 *   cached, the least recently used evicted first.
 *
 * The store belongs to one session; the auth context clears it at every session boundary
 * (sign-out, sign-in, an auth failure). Every invalidation also bumps the generation, so
 * a fetch that was already away when a change landed never stores its now-outdated
 * answer into the newer state — a missed write costs a refetch, never a wrong answer —
 * and callers after the change start their own request instead of sharing the stale
 * flight. A settling flight releases only a slot that still holds it.
 *
 * The invalidation map — every mutation point in apps/web that touches a cached
 * resource:
 *
 * - dataset upload (POST /datasets, the sidebar) — invalidates the datasets list;
 * - dataset delete (DELETE /datasets/{id}, the sidebar) — invalidates the list and drops
 *   the dataset's detail and fits (dropDataset);
 * - a manual-fit or opened-row job finishing (useJobPolling's callback; the job stored a
 *   fit) — invalidates the dataset's fits list;
 * - an Auto-fit run job finishing (useAutofit) — invalidates the dataset's fits list;
 * - fit delete (DELETE /fits/{id}) — invalidates the dataset's fits list.
 *
 * There is no dataset rename: the API has no PATCH /datasets/{id} and the UI has no
 * rename control. A completed fit job changes nothing the detail carries — the detail is
 * dataset data (columns, points, issues), which fits do not touch — so only the fits
 * list is invalidated. The statistical runs and the fit-scoped POSTs (predict, simulate,
 * advisor, discernibility, cross-validation audit) change neither the datasets, the
 * models, a detail, nor a fits list.
 */

import {
  api,
  type DatasetDetailOut,
  type DatasetOut,
  type FitSummaryOut,
  type ModelInfo,
} from "../api/client";

/** How long the dataset list and the model catalogue may serve without a revalidation. */
export const METADATA_TTL_MS = 30_000;
/** The cached details are the big ones: at most this many datasets keep theirs. */
export const DETAIL_CAP = 3;

interface TimedEntry<T> {
  data: T;
  fetchedAt: number;
}

/** A cached metadata answer: `stale` asks the caller to revalidate in the background. */
export interface CachedEntry<T> {
  data: T;
  stale: boolean;
}

/** One in-flight fetch, with the generation it started under. */
interface Flight<T> {
  gen: number;
  promise: Promise<T>;
}

let generation = 0;
let datasets: TimedEntry<DatasetOut[]> | null = null;
let models: TimedEntry<ModelInfo[]> | null = null;
/** Keyed by `${datasetId}|${points}`; Map iteration order is the recency order. */
const details = new Map<string, TimedEntry<DatasetDetailOut>>();
const fits = new Map<string, TimedEntry<FitSummaryOut[]>>();

let datasetsFlight: Flight<DatasetOut[]> | null = null;
let modelsFlight: Flight<ModelInfo[]> | null = null;
const detailsFlights = new Map<string, Flight<DatasetDetailOut>>();
const fitsFlights = new Map<string, Flight<FitSummaryOut[]>>();

function detailKey(datasetId: string, points: number): string {
  return `${datasetId}|${points}`;
}

function isStale(entry: TimedEntry<unknown>): boolean {
  return Date.now() - entry.fetchedAt > METADATA_TTL_MS;
}

/* ---- The datasets list ---- */

/** The cached dataset list, or null; `stale` asks for a background revalidation. */
export function getDatasets(): CachedEntry<DatasetOut[]> | null {
  if (datasets === null) return null;
  return { data: datasets.data, stale: isStale(datasets) };
}

/** Fetch the list (one request however many callers share the flight) and cache it. */
export function fetchDatasets(): Promise<DatasetOut[]> {
  const started = generation;
  if (datasetsFlight !== null && datasetsFlight.gen === started) return datasetsFlight.promise;
  const promise = api.request<DatasetOut[]>("/datasets").then(
    (out) => {
      if (datasetsFlight?.promise === promise) datasetsFlight = null;
      if (started === generation) datasets = { data: out, fetchedAt: Date.now() };
      return out;
    },
    (error: unknown) => {
      if (datasetsFlight?.promise === promise) datasetsFlight = null;
      throw error;
    },
  );
  datasetsFlight = { gen: started, promise };
  return promise;
}

/** The list changed (a dataset was uploaded or deleted): forget it. A fetch that was
 *  already away holds an answer without the change, so it must not store either. */
export function invalidateDatasets(): void {
  generation += 1;
  datasets = null;
}

/* ---- The model catalogue ---- */

/** The cached catalogue, or null; `stale` asks for a background revalidation. */
export function getModels(): CachedEntry<ModelInfo[]> | null {
  if (models === null) return null;
  return { data: models.data, stale: isStale(models) };
}

/** Fetch the catalogue (one request however many callers share the flight) and cache it. */
export function fetchModels(): Promise<ModelInfo[]> {
  const started = generation;
  if (modelsFlight !== null && modelsFlight.gen === started) return modelsFlight.promise;
  const promise = api.request<ModelInfo[]>("/models").then(
    (out) => {
      if (modelsFlight?.promise === promise) modelsFlight = null;
      if (started === generation) models = { data: out, fetchedAt: Date.now() };
      return out;
    },
    (error: unknown) => {
      if (modelsFlight?.promise === promise) modelsFlight = null;
      throw error;
    },
  );
  modelsFlight = { gen: started, promise };
  return promise;
}

/* ---- Dataset details ---- */

/** The cached detail for this dataset at this sample size, or null. Touches recency. */
export function getDetail(datasetId: string, points: number): DatasetDetailOut | null {
  const key = detailKey(datasetId, points);
  const entry = details.get(key);
  if (entry === undefined) return null;
  // Re-insert to move the entry to the end: Map order is the recency order the cap evicts by.
  details.delete(key);
  details.set(key, entry);
  return entry.data;
}

/** Fetch a detail (one flight per dataset and sample size) and cache it. */
export function fetchDetail(datasetId: string, points: number): Promise<DatasetDetailOut> {
  const key = detailKey(datasetId, points);
  const started = generation;
  const pending = detailsFlights.get(key);
  if (pending !== undefined && pending.gen === started) return pending.promise;
  const promise = api.request<DatasetDetailOut>(`/datasets/${datasetId}?points=${points}`).then(
    (out) => {
      if (detailsFlights.get(key)?.promise === promise) detailsFlights.delete(key);
      if (started === generation) storeDetail(key, out);
      return out;
    },
    (error: unknown) => {
      // A failed fetch is stored no more than a failed list is: the next mount retries.
      if (detailsFlights.get(key)?.promise === promise) detailsFlights.delete(key);
      throw error;
    },
  );
  detailsFlights.set(key, { gen: started, promise });
  return promise;
}

/** Store a detail, evicting the least recently used ones beyond the cap. */
function storeDetail(key: string, data: DatasetDetailOut): void {
  details.delete(key);
  details.set(key, { data, fetchedAt: Date.now() });
  while (details.size > DETAIL_CAP) {
    const oldest = details.keys().next();
    if (oldest.done) break;
    details.delete(oldest.value);
  }
}

/* ---- Fits lists ---- */

/** The cached fits list of this dataset, or null. */
export function getFits(datasetId: string): FitSummaryOut[] | null {
  return fits.get(datasetId)?.data ?? null;
}

/** The dataset's fits list: the cached one when it is there, else one fetched flight
 *  (single-flight per dataset) that caches it. */
export function fetchFits(datasetId: string): Promise<FitSummaryOut[]> {
  const cached = fits.get(datasetId);
  if (cached !== undefined) return Promise.resolve(cached.data);
  const started = generation;
  const pending = fitsFlights.get(datasetId);
  if (pending !== undefined && pending.gen === started) return pending.promise;
  const promise = api.request<FitSummaryOut[]>(`/datasets/${datasetId}/fits`).then(
    (out) => {
      if (fitsFlights.get(datasetId)?.promise === promise) fitsFlights.delete(datasetId);
      if (started === generation) fits.set(datasetId, { data: out, fetchedAt: Date.now() });
      return out;
    },
    (error: unknown) => {
      if (fitsFlights.get(datasetId)?.promise === promise) fitsFlights.delete(datasetId);
      throw error;
    },
  );
  fitsFlights.set(datasetId, { gen: started, promise });
  return promise;
}

/** The dataset's fit history changed (a fit stored or deleted): forget the list. A fetch
 *  that was already away holds an answer without the change, so it must not store
 *  either; the next read refetches. */
export function invalidateFits(datasetId: string): void {
  generation += 1;
  fits.delete(datasetId);
}

/* ---- Whole-dataset and whole-store invalidation ---- */

/** The dataset is gone: forget its detail (at any sample size) and its fits. A fetch
 *  that was already away for it must not store an answer for a deleted dataset. */
export function dropDataset(datasetId: string): void {
  generation += 1;
  const prefix = `${datasetId}|`;
  for (const key of [...details.keys()]) {
    if (key.startsWith(prefix)) details.delete(key);
  }
  fits.delete(datasetId);
}

/** The session ended or a new one began: forget everything, including in-flight answers. */
export function clear(): void {
  generation += 1;
  datasets = null;
  models = null;
  details.clear();
  fits.clear();
  datasetsFlight = null;
  modelsFlight = null;
  detailsFlights.clear();
  fitsFlights.clear();
}
