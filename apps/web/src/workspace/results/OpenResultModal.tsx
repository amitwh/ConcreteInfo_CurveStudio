/**
 * A ranking row's opened result, presented where the user already is: the house modal
 * carries the same rendering the tabs below the ranking show — model equation,
 * goodness-of-fit statistics, parameter estimates — because the full results panel
 * renders below the fold, where it is scrolled past and missed. While the open request
 * fits (a new job) the modal says so; a refused or failed open shows its error here.
 */

import type { ReactElement } from "react";
import type { FitBundle } from "../../api/client";
import { Modal } from "../../components/Modal";
import { Loading } from "../../components/States";
import { EquationTab } from "./EquationTab";
import { ParametersTab } from "./ParametersTab";
import { StatisticsTab } from "./StatisticsTab";

export interface OpenResultModalProps {
  /** The opened row's display name (the ranking row's name). */
  name: string;
  /** The opened row's fit bundle once it is loaded; null while it is not. */
  bundle: FitBundle | null;
  /** True while the open request (POST or polled job) is in flight. */
  busy: boolean;
  /** Why the row could not be opened, when it could not. */
  error: string | null;
  onClose: () => void;
}

export function OpenResultModal({
  name,
  bundle,
  busy,
  error,
  onClose,
}: OpenResultModalProps): ReactElement {
  return (
    <Modal title={`Opened result — ${name}`} onClose={onClose}>
      {busy && <Loading label="Fitting this model — the result opens here when it is ready…" />}
      {error !== null && !busy && (
        <p className="alert alert--error" role="alert">
          {error}
        </p>
      )}
      {bundle !== null && !busy && (
        <>
          <EquationTab bundle={bundle} />
          <StatisticsTab bundle={bundle} />
          <ParametersTab bundle={bundle} />
        </>
      )}
      {!busy && error === null && bundle === null && (
        <p className="state state--empty">
          The result is not available right now — close this and open the row again.
        </p>
      )}
    </Modal>
  );
}
