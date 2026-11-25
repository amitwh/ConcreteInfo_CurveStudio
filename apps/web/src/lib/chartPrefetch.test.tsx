import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AuthStatus } from "../api/auth";
import { useChartPrefetch } from "./chartPrefetch";

/**
 * The chart bundle's download is observed the way App.lazy.test.tsx observes a page
 * chunk's: the mock factory runs at the module's first import, so its counter says when
 * the dynamic import would have fetched the chunk. No real Plotly is ever loaded here.
 */
const chart = vi.hoisted(() => ({ imports: 0 }));

vi.mock("./plotly-bundle", () => {
  chart.imports += 1;
  return { default: {} };
});

type IdleCallback = (deadline: { didTimeout: boolean; timeRemaining: () => number }) => void;
interface QueuedIdle {
  id: number;
  callback: IdleCallback;
}

/** The browser side of requestIdleCallback, faked: queued callbacks, cancellable. */
const idle = vi.hoisted(() => ({
  queued: [] as QueuedIdle[],
  cancelled: [] as number[],
  nextId: 1,
}));

function stubIdleCallback(): void {
  vi.stubGlobal("requestIdleCallback", (callback: IdleCallback): number => {
    const id = idle.nextId;
    idle.nextId += 1;
    idle.queued.push({ id, callback });
    return id;
  });
  vi.stubGlobal("cancelIdleCallback", (id: number): void => {
    idle.cancelled.push(id);
  });
}

/** Runs every still-queued idle callback, exactly as the browser would dispatch them. */
function runIdle(): void {
  for (const { id, callback } of idle.queued.splice(0)) {
    if (!idle.cancelled.includes(id)) callback({ didTimeout: false, timeRemaining: () => 50 });
  }
}

function Probe({ status }: { status: AuthStatus }): null {
  useChartPrefetch(status);
  return null;
}

/** Awaits the dynamic import the callback started (it resolves on a microtask). */
async function awaitImportCount(count: number): Promise<void> {
  await vi.waitFor(() => expect(chart.imports).toBe(count));
}

beforeEach(() => {
  chart.imports = 0;
  idle.queued.length = 0;
  idle.cancelled.length = 0;
  idle.nextId = 1;
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("useChartPrefetch", () => {
  it("prefetches the chart bundle once a signed-in visitor's browser is idle", async () => {
    stubIdleCallback();
    const view = render(<Probe status="authed" />);
    // scheduled, but nothing downloaded while the browser is still busy
    expect(idle.queued.length).toBe(1);
    expect(chart.imports).toBe(0);
    runIdle();
    await awaitImportCount(1);
    view.unmount();
  });

  it("prefetches nothing for an anonymous or still-loading visitor", () => {
    stubIdleCallback();
    const anon = render(<Probe status="anon" />);
    const loading = render(<Probe status="loading" />);
    runIdle();
    expect(chart.imports).toBe(0);
    anon.unmount();
    loading.unmount();
  });

  it("cancels the pending prefetch when the visitor signs out before idle fires", () => {
    stubIdleCallback();
    const view = render(<Probe status="authed" />);
    view.unmount();
    expect(idle.cancelled.length).toBe(1);
    runIdle();
    expect(chart.imports).toBe(0);
  });

  it("falls back to a delayed timer when requestIdleCallback is unavailable", () => {
    vi.stubGlobal("requestIdleCallback", undefined);
    // Real timers and a spy, not a fake clock: the import the fallback starts resolves
    // on the module runner's own async hops, which a frozen fake clock holds forever.
    const setTimeoutSpy = vi.spyOn(window, "setTimeout");
    const view = render(<Probe status="authed" />);
    // One delayed timer — not an immediate prefetch, and long enough that it never
    // competes with first paint. That the scheduled function then imports the bundle
    // is the idle-path test's subject above: both branches run the same prefetch.
    expect(setTimeoutSpy).toHaveBeenCalledTimes(1);
    expect(setTimeoutSpy.mock.calls[0]?.[1]).toBe(2_000);
    expect(chart.imports).toBe(0);
    view.unmount(); // cancels the timer; nothing downloads
  });
});
