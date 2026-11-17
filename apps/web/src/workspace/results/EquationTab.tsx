/** The model equation, with fitted numbers substituted when the engine produced them. */

import type { ReactElement } from "react";
import { CopyButton } from "../../components/CopyButton";
import type { FitBundle } from "../../api/client";

export function EquationTab({ bundle }: { bundle: FitBundle }): ReactElement {
  return (
    <div>
      <h3>Model equation</h3>
      <p className="equation equation--large">{bundle.result.equation}</p>
      <p>
        <CopyButton text={bundle.result.equation} label="equation" />
      </p>
      <h3>With fitted numbers</h3>
      {bundle.result.equation_fitted !== null ? (
        <>
          <p className="equation equation--large">{bundle.result.equation_fitted}</p>
          <p>
            <CopyButton text={bundle.result.equation_fitted} label="fitted equation" />
          </p>
        </>
      ) : (
        <p className="muted">
          Not available — the fit did not converge, so no substituted equation is shown.
        </p>
      )}
      {bundle.equation_text !== bundle.result.equation && (
        <p className="hint">{bundle.equation_text}</p>
      )}
      <p className="hint">
        Optimiser: {bundle.result.method} · loss {bundle.result.loss} · starts{" "}
        {bundle.result.n_starts} · seed {bundle.result.provenance.seed}
      </p>
    </div>
  );
}
