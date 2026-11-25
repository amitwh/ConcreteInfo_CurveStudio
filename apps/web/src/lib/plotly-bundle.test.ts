import { describe, expect, it, vi } from "vitest";

/** The mocked lib/core default: records what the bundle registers. */
const core = vi.hoisted(() => ({
  react: vi.fn(),
  register: vi.fn(),
}));

const bar = vi.hoisted(() => ({ moduleType: "trace", name: "bar" }));
const box = vi.hoisted(() => ({ moduleType: "trace", name: "box" }));
const scattergl = vi.hoisted(() => ({ moduleType: "trace", name: "scattergl" }));

vi.mock("plotly.js/lib/core", () => ({ default: core }));
vi.mock("plotly.js/lib/bar", () => ({ default: bar }));
vi.mock("plotly.js/lib/box", () => ({ default: box }));
vi.mock("plotly.js/lib/scattergl", () => ({ default: scattergl }));

/** The one thing this suite pins: the trace set the app's chart bundle carries. */
describe("the Plotly bundle", () => {
  it("registers exactly bar, box and scattergl on core (scatter is part of core)", async () => {
    const Plotly = (await import("./plotly-bundle")).default;
    expect(core.register).toHaveBeenCalledTimes(1);
    expect(core.register).toHaveBeenCalledWith([bar, box, scattergl]);
    expect(Plotly).toBe(core);
  });
});
