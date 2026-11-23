/**
 * Prediction: enter x values, get ŷ with an optional confidence or prediction band,
 * plus CSV/JSON/XLSX export of the whole stored fit.
 */

import { useState, type ReactElement, type SubmitEvent } from "react";
import { Download } from "lucide-react";
import { api, saveBlob, type FitBundle, type Prediction } from "../../api/client";
import { ErrorBox } from "../../components/States";
import { fmtNumber } from "../../format";

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
                <th scope="col">x</th>
                <th scope="col">ŷ</th>
                {prediction.band !== null && <th scope="col">lower</th>}
                {prediction.band !== null && <th scope="col">upper</th>}
                <th scope="col">extrapolated</th>
              </tr>
            </thead>
            <tbody>
              {prediction.x.map((x, index) => (
                <tr key={index}>
                  <td className="num">{x === null ? "n/a" : fmtNumber(x, { digits: 8 })}</td>
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
