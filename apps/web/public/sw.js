/*
 * CurveStudio's service worker: the app shell works offline, the API never does.
 *
 * Navigations are network-first (a fresh deployment is picked up immediately) with the
 * cached shell as the offline fallback; hashed assets are cache-first, since their URL
 * changes when their content does. /api/ requests are never cached or intercepted —
 * a stale fit or dataset served from a cache would be worse than an honest offline
 * error from the app.
 */

const CACHE = "curvestudio-shell-v1";
const SHELL = ["/", "/manifest.webmanifest", "/favicon-32.png", "/icon-192.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return; // the API is always live
  if (url.pathname === "/sw.js") return;

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).catch(() => caches.match("/")),
    );
    return;
  }

  const cacheable =
    url.pathname.startsWith("/assets/") ||
    /\.(?:png|svg|ico|webmanifest|woff2?|css|js)$/.test(url.pathname);
  if (!cacheable) return;

  event.respondWith(
    caches.match(request).then(
      (hit) =>
        hit ??
        fetch(request).then((response) => {
          if (response.ok) {
            const copy = response.clone();
            void caches.open(CACHE).then((cache) => cache.put(request, copy));
          }
          return response;
        }),
    ),
  );
});
