import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactNode } from "react";

/**
 * Render a node inside a fresh QueryClientProvider, for the components that read
 * server state through TanStack Query. Every call gets its own client, so one test's
 * cache never leaks into the next. Retries are off by default: an error state must show
 * on the first failure instead of after a backoff the test would only wait on.
 */
export function renderWithQuery(ui: ReactNode): ReturnType<typeof render> & {
  queryClient: QueryClient;
} {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
  return { ...view, queryClient };
}
