/**
 * Where the API lives. Same origin (`/api/v1`) by default; a static-only deployment can
 * point the built SPA at another host with `VITE_API_BASE` (for example
 * `VITE_API_BASE=https://api.example.com/api/v1`), and the API then lists that origin in
 * `CURVESTUDIO_CORS_ORIGINS`.
 *
 * A module of its own, with nothing behind it: the error reporter is installed as early as
 * an import can run, and it needs this before anything else has loaded.
 */
export const API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api/v1").replace(/\/+$/, "");
