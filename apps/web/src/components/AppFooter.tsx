/**
 * The app footer: the ConcreteInfo attribution everywhere the signed-in app and the
 * account pages render (the home page keeps its richer marketing footer — this is the
 * slim sibling). The links are icons: the globe is Lucide's; the two brand marks are
 * Simple Icons' CC0 glyphs, inlined because Lucide dropped brand icons.
 */

import type { ReactElement } from "react";
import { Globe } from "lucide-react";

const WEBSITE_URL = "https://www.concreteinfo.in";
const LINKEDIN_URL = "https://in.linkedin.com/company/concreteinfo";
const SOURCE_URL = "https://github.com/amitwh/ConcreteInfo_CurveStudio";

/** Simple Icons' LinkedIn mark (CC0), filled with the current text colour. */
function LinkedInIcon(): ReactElement {
  return (
    <svg
      className="icon"
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      fill="currentColor"
    >
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );
}

/** Simple Icons' GitHub mark (CC0), filled with the current text colour. */
function GitHubIcon(): ReactElement {
  return (
    <svg
      className="icon"
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      fill="currentColor"
    >
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
}

export function AppFooter(): ReactElement {
  return (
    <footer className="app-footer" aria-label="About this app">
      <a className="app-footer__brand" href={WEBSITE_URL} target="_blank" rel="noopener noreferrer">
        <img src="/concreteinfo-logo.png" alt="ConcreteInfo" width={100} height={28} />
      </a>
      <p className="app-footer__note">A ConcreteInfo application.</p>
      <nav className="app-footer__links" aria-label="Site footer">
        <a href={WEBSITE_URL} target="_blank" rel="noopener noreferrer" aria-label="Website" title="ConcreteInfo website">
          <Globe className="icon" aria-hidden="true" focusable="false" />
        </a>
        <a href={LINKEDIN_URL} target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" title="ConcreteInfo on LinkedIn">
          <LinkedInIcon />
        </a>
        <a href={SOURCE_URL} target="_blank" rel="noopener noreferrer" aria-label="Source on GitHub" title="Source on GitHub">
          <GitHubIcon />
        </a>
        <span className="app-footer__legal">© {new Date().getFullYear()} ConcreteInfo</span>
      </nav>
    </footer>
  );
}
