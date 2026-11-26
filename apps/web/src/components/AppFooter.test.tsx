/**
 * The app footer: the ConcreteInfo attribution on every surface that is not the home
 * page's marketing footer — the links are the owner's list (website, LinkedIn, the
 * source repository).
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppFooter } from "./AppFooter";

function renderFooter() {
  return render(
    <MemoryRouter>
      <AppFooter />
    </MemoryRouter>,
  );
}

describe("AppFooter", () => {
  it("links the ConcreteInfo website, LinkedIn page and source repository", () => {
    renderFooter();
    const website = screen.getByRole("link", { name: "ConcreteInfo" }); // the brand mark
    expect(website.getAttribute("href")).toBe("https://www.concreteinfo.in");
    expect(screen.getByRole("link", { name: "Website" }).getAttribute("href")).toBe(
      "https://www.concreteinfo.in",
    );
    expect(screen.getByRole("link", { name: "LinkedIn" }).getAttribute("href")).toBe(
      "https://in.linkedin.com/company/concreteinfo",
    );
    expect(screen.getByRole("link", { name: "Source on GitHub" }).getAttribute("href")).toBe(
      "https://github.com/amitwh/ConcreteInfo_CurveStudio",
    );
  });

  it("says whose application it is, with the year", () => {
    renderFooter();
    expect(screen.getByText("A ConcreteInfo application.")).toBeTruthy();
    expect(screen.getByText(`© ${new Date().getFullYear()} ConcreteInfo`)).toBeTruthy();
  });
});
