/**
 * The plans and their limits. Changing a limit is not destructive, so editing asks for no
 * confirmation: the page itself refuses a limit below 1, and shows the server's own reason
 * when the server refuses.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState, type ChangeEvent, type FormEvent, type ReactElement } from "react";
import { Empty, apiErrorText } from "../components/States";
import { fmtBytes, fmtInt } from "../format";
import { plans, updatePlan, type PlanAdminOut, type PlanLimits } from "./adminApi";
import { QueryState } from "./QueryState";

const PLANS_KEY = ["admin", "plans"] as const;

type LimitKey = keyof PlanLimits;

/** The refusal the page makes itself. */
const TOO_LOW = "Each limit must be a whole number of at least 1.";

interface Limit {
  key: LimitKey;
  /** How the card names the limit. */
  label: string;
  /** How the form names its input, where a bare number needs its unit. */
  inputLabel: string;
  /** The limit as the card shows it. */
  inWords: (value: number) => string;
}

/** Storage is the plan's total across its datasets; the others are counts. */
const LIMITS: ReadonlyArray<Limit> = [
  { key: "max_datasets", label: "Datasets", inputLabel: "Datasets", inWords: fmtInt },
  { key: "max_dataset_bytes", label: "Storage", inputLabel: "Storage, in bytes", inWords: fmtBytes },
  { key: "max_fits", label: "Stored fits", inputLabel: "Stored fits", inWords: fmtInt },
  {
    key: "max_autofits_per_day",
    label: "Auto-fit runs a day",
    inputLabel: "Auto-fit runs a day",
    inWords: fmtInt,
  },
];

export function PlansPage(): ReactElement {
  const query = useQuery({ queryKey: PLANS_KEY, queryFn: plans });
  return (
    <>
      <h2>Plans</h2>
      <QueryState label="Loading the plans…" query={query}>
        {(list) => (list.length === 0 ? <Empty>There are no plans.</Empty> : <PlanList plans={list} />)}
      </QueryState>
    </>
  );
}

function PlanList({ plans }: { plans: PlanAdminOut[] }): ReactElement {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: ({ name, limits }: { name: string; limits: PlanLimits }) => updatePlan(name, limits),
    onSuccess: (updated) => {
      // The server's answer is the truth: show it at once, then refetch the list behind it.
      queryClient.setQueryData<PlanAdminOut[]>(PLANS_KEY, (current) =>
        current?.map((plan) => (plan.name === updated.name ? updated : plan)),
      );
      void queryClient.invalidateQueries({ queryKey: PLANS_KEY });
      setSaved(updated.name);
      setEditing(null);
    },
  });

  const startEditing = (name: string): void => {
    save.reset();
    setSaved(null);
    setEditing(name);
  };

  const cancelEditing = (): void => {
    save.reset();
    setEditing(null);
  };

  return (
    <div className="admin-grid">
      {plans.map((plan) => (
        <PlanCard
          key={plan.name}
          plan={plan}
          editing={editing === plan.name}
          saving={save.isPending}
          error={save.error}
          saved={saved === plan.name}
          onEdit={() => startEditing(plan.name)}
          onCancel={cancelEditing}
          onSave={(limits) => save.mutate({ name: plan.name, limits })}
        />
      ))}
    </div>
  );
}

interface PlanCardProps {
  plan: PlanAdminOut;
  editing: boolean;
  saving: boolean;
  error: Error | null;
  saved: boolean;
  onEdit: () => void;
  onCancel: () => void;
  onSave: (limits: PlanLimits) => void;
}

function PlanCard({
  plan,
  editing,
  saving,
  error,
  saved,
  onEdit,
  onCancel,
  onSave,
}: PlanCardProps): ReactElement {
  const headingId = useId();
  return (
    <article className="card admin-plan" aria-labelledby={headingId}>
      <h3 id={headingId}>{plan.name}</h3>
      <dl className="admin-limits">
        {LIMITS.map((limit) => (
          <div key={limit.key}>
            <dt>{limit.label}</dt>
            <dd>{limit.inWords(plan.limits[limit.key])}</dd>
          </div>
        ))}
      </dl>
      {editing ? (
        <LimitsForm plan={plan} saving={saving} error={error} onCancel={onCancel} onSubmit={onSave} />
      ) : (
        <button type="button" className="btn btn--small" onClick={onEdit}>
          Edit limits
        </button>
      )}
      {saved && (
        <p className="alert alert--success" role="status">
          Limits saved. The change is in the audit trail.
        </p>
      )}
    </article>
  );
}

type Draft = Record<LimitKey, string>;

/** The four inputs as limits, or null when one of them is not a whole number of at least 1. */
function parseLimits(draft: Draft): PlanLimits | null {
  const parsed: Partial<Record<LimitKey, number>> = {};
  for (const limit of LIMITS) {
    const text = draft[limit.key];
    if (!/^\d+$/.test(text) || Number(text) < 1) return null;
    parsed[limit.key] = Number(text);
  }
  const { max_datasets, max_dataset_bytes, max_fits, max_autofits_per_day } = parsed;
  if (
    max_datasets === undefined ||
    max_dataset_bytes === undefined ||
    max_fits === undefined ||
    max_autofits_per_day === undefined
  ) {
    return null;
  }
  return { max_datasets, max_dataset_bytes, max_fits, max_autofits_per_day };
}

interface LimitsFormProps {
  plan: PlanAdminOut;
  saving: boolean;
  /** The server's refusal of the last save, if it refused. */
  error: Error | null;
  onCancel: () => void;
  onSubmit: (limits: PlanLimits) => void;
}

/** Opens on the plan's limits; a refused save leaves what was typed in place. */
function LimitsForm({ plan, saving, error, onCancel, onSubmit }: LimitsFormProps): ReactElement {
  const formId = useId();
  const [draft, setDraft] = useState<Draft>({
    max_datasets: String(plan.limits.max_datasets),
    max_dataset_bytes: String(plan.limits.max_dataset_bytes),
    max_fits: String(plan.limits.max_fits),
    max_autofits_per_day: String(plan.limits.max_autofits_per_day),
  });
  const [refused, setRefused] = useState<string | null>(null);

  const submit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const limits = parseLimits(draft);
    if (limits === null) {
      setRefused(TOO_LOW);
      return;
    }
    setRefused(null);
    onSubmit(limits);
  };

  const message = refused ?? (error === null ? null : apiErrorText(error));

  return (
    <form
      className="admin-limits-form"
      aria-label={`Edit limits for ${plan.name}`}
      onSubmit={submit}
      noValidate
    >
      {LIMITS.map((limit) => (
        <div className="field" key={limit.key}>
          <label htmlFor={`${formId}-${limit.key}`}>{limit.inputLabel}</label>
          <input
            id={`${formId}-${limit.key}`}
            type="number"
            min={1}
            step={1}
            inputMode="numeric"
            value={draft[limit.key]}
            disabled={saving}
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              setDraft((current) => ({ ...current, [limit.key]: event.target.value }))
            }
          />
        </div>
      ))}
      {message !== null && (
        <p className="alert alert--error" role="alert">
          {message}
        </p>
      )}
      <div className="btn-row">
        <button type="submit" className="btn btn--primary" disabled={saving}>
          Save
        </button>
        <button type="button" className="btn" onClick={onCancel} disabled={saving}>
          Cancel
        </button>
      </div>
    </form>
  );
}
