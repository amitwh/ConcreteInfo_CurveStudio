/**
 * A stored Auto-fit row's sampled curve (RowCurve) into the FitBundle shape the chart
 * consumes. Only the fields `workspace/traces.ts` reads are populated (result.status,
 * curve.x/y and the bands' x/lower/upper); the cast documents that the bundle is a
 * chart source, not a fit result — the row already proved convergence by being ranked.
 */

import type { FitBundle, RowCurve } from "../../api/client";

export function rowCurveBundle(curve: RowCurve): FitBundle {
  const bundle: Record<string, unknown> = {
    result: { status: "converged" },
    curve: { x: curve.x, y: curve.y },
    confidence_band:
      curve.confidence_lower !== null && curve.confidence_upper !== null
        ? { x: curve.x, lower: curve.confidence_lower, upper: curve.confidence_upper }
        : null,
    prediction_band:
      curve.prediction_lower !== null && curve.prediction_upper !== null
        ? { x: curve.x, lower: curve.prediction_lower, upper: curve.prediction_upper }
        : null,
  };
  return bundle as unknown as FitBundle;
}
