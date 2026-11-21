import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AuthProvider } from "./api/auth";
// Self-hosted variable fonts: the API serves a strict `default-src 'self'` CSP, so no
// CDN (Google Fonts etc.) may be used. Vite fingerprints and emits the woff2 files.
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles.css";

const root = document.getElementById("root");
if (root === null) throw new Error("no #root element in index.html");

createRoot(root).render(
  <StrictMode>
    <AuthProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AuthProvider>
  </StrictMode>,
);
