import { describe, expect, it } from "vitest";
import type { components } from "../../api/schema";
import { rowCurveBundle } from "./rowCurve";

type RowCurve = components["schemas"]["RowCurve"];

const curve: RowCurve = {
  x: [0, 1, 2],
  y: [1, 2.7, 7.4],
  confidence_lower: [0.9, 2.5, 6.8],
  confidence_upper: [1.1, 2.9, 8.0],
  prediction_lower: [0.5, 1.9, 5.5],
  prediction_upper: [1.5, 3.6, 9.3],
  bands_reason: null,
  notes: [],
};

describe("rowCurveBundle", () => {
  it("maps a RowCurve into the bundle shape the chart consumes", () => {
    const bundle = rowCurveBundle(curve);
    expect(bundle).not.toBeNull();
    // The chart's no-curve invariant reads result.status; an auto-fit row that has a
    // sampled curve was converged by construction.
    expect(bundle?.result.status).toBe("converged");
    expect(bundle?.curve?.x).toEqual([0, 1, 2]);
    expect(bundle?.curve?.y).toEqual([1, 2.7, 7.4]);
    expect(bundle?.confidence_band?.lower).toEqual([0.9, 2.5, 6.8]);
    expect(bundle?.confidence_band?.upper).toEqual([1.1, 2.9, 8.0]);
    expect(bundle?.prediction_band?.lower).toEqual([0.5, 1.9, 5.5]);
    expect(bundle?.prediction_band?.upper).toEqual([1.5, 3.6, 9.3]);
  });

  it("leaves bands null when the engine could not sample them", () => {
    const bundle = rowCurveBundle({
      ...curve,
      confidence_lower: null,
      confidence_upper: null,
      prediction_lower: null,
      prediction_upper: null,
      bands_reason: "the fit has no covariance matrix",
    });
    expect(bundle?.confidence_band).toBeNull();
    expect(bundle?.prediction_band).toBeNull();
    expect(bundle?.curve?.y).toEqual([1, 2.7, 7.4]);
  });
});
