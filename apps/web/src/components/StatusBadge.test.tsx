import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge, type StatusKind } from "./StatusBadge";

const KINDS: ReadonlyArray<[StatusKind, string, string]> = [
  ["new", "New", "badge--bad"],
  ["acknowledged", "Acknowledged", "badge--warning"],
  ["resolved", "Resolved", "badge--ok"],
  ["ignored", "Ignored", "badge--muted"],
  ["member", "Member", "badge--info"],
  ["admin", "Admin", "badge--accent"],
];

describe("StatusBadge", () => {
  it.each(KINDS)("%s reads %s in its %s tone", (kind, text, tone) => {
    render(<StatusBadge kind={kind} />);
    const badge = screen.getByText(text);
    expect(badge.className).toContain("badge");
    expect(badge.className).toContain(tone);
  });

  it("gives every kind its own tone", () => {
    const tones = new Set(KINDS.map(([, , tone]) => tone));
    expect(tones.size).toBe(KINDS.length);
  });
});
