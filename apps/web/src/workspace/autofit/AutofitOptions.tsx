/**
 * Auto-fit's advanced options in a collapsed disclosure: families (by category, from
 * the model catalogue), exhaustive generated families, parameter cap, multistarts, CV
 * folds and seed. Everything stays at its default until the user opens it and changes
 * something; "Run again" applies the changes. Custom equations remain manual-only.
 */

import { useMemo, type ReactElement } from "react";
import { ChevronDown } from "lucide-react";
import type { ModelInfo } from "../../api/client";
import { DEFAULT_AUTOFIT_OPTIONS, type AutoFitOptions } from "./useAutofit";

export interface AutofitOptionsProps {
  models: ModelInfo[];
  options: AutoFitOptions;
  onChange: (options: AutoFitOptions) => void;
  disabled?: boolean;
}

export function AutofitOptions({
  models,
  options,
  onChange,
  disabled = false,
}: AutofitOptionsProps): ReactElement {
  const categories = useMemo(
    () => [...new Set(models.map((model) => model.category))].sort((a, b) => a.localeCompare(b)),
    [models],
  );

  const toggleFamily = (category: string, checked: boolean): void => {
    const families = new Set(options.families);
    if (checked) families.add(category);
    else families.delete(category);
    onChange({ ...options, families: [...families] });
  };

  return (
    <details className="details autofit-options">
      <summary>
        <ChevronDown
          className="icon details__chevron"
          aria-hidden="true"
          focusable="false"
        />
        Options
      </summary>
      {/* The open content is one wrapper the stylesheet renders as a dropdown panel
          anchored under the summary: out of flow, height-capped and scrolled inside, so
          opening never grows the side column or moves the chart and ranking below. */}
      <div className="autofit-options__body">
        <p className="hint">
          Defaults suit most data. Changed options apply on the next “Run again”. Custom
          equations stay in Manual fit.
        </p>

        {categories.length > 0 && (
          <fieldset className="fieldset" disabled={disabled}>
            <legend>Families — none checked means every family</legend>
            <div className="autofit-options__families">
              {categories.map((category) => (
                <label key={category} className="check">
                  <input
                    type="checkbox"
                    checked={options.families.includes(category)}
                    onChange={(e) => toggleFamily(category, e.target.checked)}
                  />
                  {category}
                </label>
              ))}
            </div>
          </fieldset>
        )}

        <div className="field field--check">
          <input
            id="autofit-exhaustive"
            type="checkbox"
            checked={options.exhaustive}
            disabled={disabled}
            onChange={(e) => onChange({ ...options, exhaustive: e.target.checked })}
          />
          <label htmlFor="autofit-exhaustive">
            Include exhaustive generated families (slower; raises the parameter cap)
          </label>
        </div>

        <div className="field">
          <label htmlFor="autofit-maxparams">Max parameters (1–10)</label>
          <input
            id="autofit-maxparams"
            type="number"
            min={1}
            max={10}
            placeholder="auto"
            value={options.maxParameters}
            disabled={disabled}
            onChange={(e) => onChange({ ...options, maxParameters: e.target.value })}
          />
        </div>

        <div className="field">
          <label htmlFor="autofit-nstarts">Multistarts per model (1–8)</label>
          <input
            id="autofit-nstarts"
            type="number"
            min={1}
            max={8}
            value={options.nStarts}
            disabled={disabled}
            onChange={(e) => onChange({ ...options, nStarts: Number(e.target.value) || 1 })}
          />
        </div>

        <div className="field">
          <label htmlFor="autofit-cvfolds">Cross-validation folds (2–10)</label>
          <input
            id="autofit-cvfolds"
            type="number"
            min={2}
            max={10}
            value={options.cvFolds}
            disabled={disabled}
            onChange={(e) => onChange({ ...options, cvFolds: Number(e.target.value) || 2 })}
          />
        </div>

        <div className="field">
          <label htmlFor="autofit-seed">Seed</label>
          <input
            id="autofit-seed"
            type="number"
            min={0}
            placeholder={String(DEFAULT_AUTOFIT_OPTIONS.seed)}
            value={options.seed}
            disabled={disabled}
            onChange={(e) => onChange({ ...options, seed: e.target.value })}
          />
          <p className="hint">same seed, same numbers — the ranking is reproducible</p>
        </div>
      </div>
    </details>
  );
}
