/**
 * Prediction: enter x values, get ŷ with an optional confidence or prediction band,
 * plus CSV/JSON/XLSX export of the whole stored fit. The result table's headers reorder
 * the visible rows only (ascending → descending → the entered x order).
 */

import { useMemo, useState, type ReactElement, type SubmitEvent } from "react";
import { Download } from "lucide-react";
import { api, saveBlob, type FitBundle, type Prediction } from "../../api/client";
import { ErrorBox } from "../../components/States";
import { fmtNumber } from "../../format";
import { SortableTh, cycleSort, sortRows, type SortState } from "../sorting";

type BandChoice = "none" | "confidence" | "prediction";

export function PredictTab({
  fitId,
  bundle,
}: {
  fitId: string;
  bundle: FitBundle;
}): ReactElement {
  const [xText, setXText] = useState("");
  const [band, setBand] = useState<BandChoice>("none");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [exportError, setExportError] = useState<unknown>(null);
  const [sort, setSort] = useState<SortState | null>(null);

  const converged = bundle.result.status === "converged";

  const onPredict = async (event: SubmitEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setError(null);
    setPrediction(null);
    const values = xText
      .split(/[\s,;]+/)
      .map((token) => token.trim())
      .filter((token) => token !== "")
      .map(Number);
    if (values.length === 0) {
      setError("enter at least one x value (numbers, separated by spaces, commas or new lines)");
      return;
    }
    if (values.some((value) => !Number.isFinite(value))) {
      setError("every x value must be a finite number");
      return;
    }
    setBusy(true);
    try {
      setPrediction(
        await api.request<Prediction>(`/fits/${fitId}/predict`, {
          method: "POST",
          body: JSON.stringify({ x: values, band: band === "none" ? null : band }),
        }),
      );
      setSort(null); // a new result is shown in its own order
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  const onExport = async (format: "csv" | "json" | "xlsx"): Promise<void> => {
    setExportError(null);
    try {
      const { blob, filename } = await api.download(`/fits/${fitId}/export?format=${format}`);
      saveBlob(blob, filename ?? `fit.${format}`);
    } catch (error) {
      setExportError(error);
    }
  };

  // Display order over the result's row indices; a fresh prediction starts unsorted.
  const orderedIndices = useMemo(() => {
    if (prediction === null) return [];
    const keys: Record<string, (index: number) => number | string | null> = {
      x: (index) => prediction.x[index] ?? null,
      y: (index) => prediction.y[index] ?? null,
      lower: (index) => prediction.lower?.[index] ?? null,
      upper: (index) => prediction.upper?.[index] ?? null,
      extrapolated: (index) => (prediction.extrapolated[index] ? "yes" : "no"),
    };
    return sortRows(
      prediction.x.map((_, index) => index),
      sort,
      keys,
    );
  }, [prediction, sort]);

  const onSort = (column: string): void => {
    setSort((current) => cycleSort(current, column));
  };

  return (
    <div>
      {!converged && (
        <p className="alert alert--warning">
          The fit did not converge — predictions from it would be misleading, and the
          server refuses them. Refit with another model or method first.
        </p>
      )}
      <form onSubmit={(e) => void onPredict(e)}>
        <div className="field">
          <label htmlFor="predict-x">x values (separated by space, comma or new line)</label>
          <textarea
            id="predict-x"
            rows={2}
            placeholder="1 2 3&#10;4.5, 5.5"
            value={xText}
            onChange={(e) => setXText(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="predict-band">Band</label>
          <select
            id="predict-band"
            value={band}
            onChange={(e) => setBand(e.target.value as BandChoice)}
          >
            <option value="none">none</option>
            <option value="confidence">95% confidence (mean response)</option>
            <option value="prediction">95% prediction (new observation)</option>
          </select>
        </div>
        <button type="submit" className="btn btn--primary" disabled={busy}>
          {busy ? "Predicting…" : "Predict"}
        </button>
      </form>
      {error !== null && <ErrorBox error={error} />}
      {prediction !== null && (
        <div className="table-wrap">
          <table className="table">
            <caption className="sr-only">Predicted values</caption>
            <thead>
              <tr>
                <SortableTh column="x" state={sort} onSort={onSort} className="num">
                  x
                </SortableTh>
                <SortableTh column="y" state={sort} onSort={onSort} className="num">
                  ŷ
                </SortableTh>
                {prediction.band !== null && (
                  <SortableTh column="lower" state={sort} onSort={onSort} className="num">
                    lower
                  </SortableTh>
                )}
                {prediction.band !== null && (
                  <SortableTh column="upper" state={sort} onSort={onSort} className="num">
                    upper
                  </SortableTh>
                )}
                <SortableTh column="extrapolated" state={sort} onSort={onSort}>
                  extrapolated
                </SortableTh>
              </tr>
            </thead>
            <tbody>
              {orderedIndices.map((index) => (
                <tr key={index}>
                  <td className="num">
                    {prediction.x[index] === null
                      ? "n/a"
                      : fmtNumber(prediction.x[index], { digits: 8 })}
                  </td>
                  <td className="num">
                    {prediction.y[index] === null || prediction.y[index] === undefined
                      ? "n/a"
                      : fmtNumber(prediction.y[index], { digits: 8 })}
                  </td>
                  {prediction.band !== null && (
                    <td className="num">
                      {fmtNumber(prediction.lower?.[index] ?? null, { digits: 8 })}
                    </td>
                  )}
                  {prediction.band !== null && (
                    <td className="num">
                      {fmtNumber(prediction.upper?.[index] ?? null, { digits: 8 })}
                    </td>
                  )}
                  <td>{prediction.extrapolated[index] ? "yes" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {prediction.band_reason !== null && (
            <p className="hint">Band note: {prediction.band_reason}</p>
          )}
        </div>
      )}
      <h3>Export this fit</h3>
      <p className="btn-row">
        <button type="button" className="btn" onClick={() => void onExport("csv")}>
          <Download className="icon" aria-hidden="true" focusable="false" />
          Download CSV
        </button>
        <button type="button" className="btn" onClick={() => void onExport("json")}>
          <Download className="icon" aria-hidden="true" focusable="false" />
          Download JSON
        </button>
        <button type="button" className="btn" onClick={() => void onExport("xlsx")}>
          <Download className="icon" aria-hidden="true" focusable="false" />
          Download Excel workbook
        </button>
      </p>
      {exportError !== null && <ErrorBox error={exportError} />}
    </div>
  );
}
