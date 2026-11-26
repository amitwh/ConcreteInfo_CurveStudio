/**
 * The app footer: the ConcreteInfo attribution everywhere the signed-in app and the
 * account pages render (the home page keeps its richer marketing footer — this is the
 * slim sibling). One row: the ConcreteInfo mark links to the company site, and beside
 * it the company's LinkedIn and this application's source repository.
 */

import type { ReactElement } from "react";

const WEBSITE_URL = "https://www.concreteinfo.in";
const LINKEDIN_URL = "https://in.linkedin.com/company/concreteinfo";
const SOURCE_URL = "https://github.com/amitwh/ConcreteInfo_CurveStudio";

export function AppFooter(): ReactElement {
  return (
    <footer className="app-footer" aria-label="About this app">
      <a className="app-footer__brand" href={WEBSITE_URL} target="_blank" rel="noopener noreferrer">
        <img src="/concreteinfo-logo.png" alt="ConcreteInfo" width={100} height={28} />
      </a>
      <p className="app-footer__note">A ConcreteInfo application.</p>
      <nav className="app-footer__links" aria-label="Site footer">
        <a href={WEBSITE_URL} target="_blank" rel="noopener noreferrer">
          Website
        </a>
        <a href={LINKEDIN_URL} target="_blank" rel="noopener noreferrer">
          LinkedIn
        </a>
        <a href={SOURCE_URL} target="_blank" rel="noopener noreferrer">
          Source on GitHub
        </a>
        <span className="app-footer__legal">© {new Date().getFullYear()} ConcreteInfo</span>
      </nav>
    </footer>
  );
}
