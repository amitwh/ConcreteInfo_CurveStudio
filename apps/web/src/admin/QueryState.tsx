/**
 * The three states of one fetch, as every console page shows them: the loading line, the
 * error with a Retry that fetches again, or the content built from the answer.
 */

import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactElement, ReactNode } from "react";
import { ErrorBox, Loading } from "../components/States";

interface QueryStateProps<T> {
  /** What is being loaded, for the loading line: "Loading the overview…". */
  label: string;
  query: UseQueryResult<T, Error>;
  children: (data: T) => ReactNode;
}

export function QueryState<T>({ label, query, children }: QueryStateProps<T>): ReactElement {
  if (query.isPending) return <Loading label={label} />;
  if (query.isError) {
    return (
      <>
        <ErrorBox error={query.error} />
        <button type="button" className="btn" onClick={() => void query.refetch()}>
          Retry
        </button>
      </>
    );
  }
  return <>{children(query.data)}</>;
}
