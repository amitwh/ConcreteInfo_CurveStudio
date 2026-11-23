import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AuthProvider } from "./api/auth";
// Self-hosted variable fonts: the API serves a strict `default-src 'self'` CSP, so no
// CDN (Google Fonts etc.) may be used. Vite fingerprints and emits the woff2 files.
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles.css";

/**
 * Server state for the screens that read it through TanStack Query. One failed attempt is
 * retried once; the profile is not re-fetched whenever the window regains focus, and is
 * considered fresh for 30 s (a change made on a page writes its answer into the cache).
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 30_000 },
  },
});

const root = document.getElementById("root");
if (root === null) throw new Error("no #root element in index.html");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  </StrictMode>,
);

// The installable app: register the shell service worker in production builds only, so
// development is never served from a cache. The worker itself is network-first for
// navigations and never touches /api.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("/sw.js").catch(() => {
      // A blocked registration (private mode, an old browser) leaves the app fully usable.
    });
  });
}
