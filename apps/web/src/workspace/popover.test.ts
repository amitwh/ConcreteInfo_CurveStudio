/**
 * The row menu's placement rules, unit-tested pure: jsdom has no layout, so the
 * clamping contract — on-screen for a last-row, right-edge kebab at every viewport
 * size — is verified against boxes and viewports directly.
 */

import { describe, expect, it } from "vitest";
import { placePopoverWithin } from "./popover";

/** A fixed 208×88 popover (the row menu's stylesheet width, two menu items). */
const MENU = { width: 208, height: 88 };

const withinViewport = (top: number, left: number, viewport: { width: number; height: number }): void => {
  expect(left).toBeGreaterThanOrEqual(0);
  expect(left + MENU.width).toBeLessThanOrEqual(viewport.width);
  expect(top).toBeGreaterThanOrEqual(0);
  expect(top + MENU.height).toBeLessThanOrEqual(viewport.height);
};

describe("placePopoverWithin", () => {
  it("opens below the trigger, right-aligned to its right edge", () => {
    const trigger = { top: 100, bottom: 132, left: 400, right: 428 };
    const viewport = { width: 1280, height: 800 };
    const { top, left } = placePopoverWithin(trigger, MENU, viewport);
    expect(top).toBe(136); // bottom + 4px gap
    expect(left).toBe(220); // 428 - 208
    withinViewport(top, left, viewport);
  });

  it("flips above the trigger when the space below is shorter than the menu", () => {
    // The last row's kebab: the button's bottom edge is 12px above the viewport floor.
    const trigger = { top: 600, bottom: 688, left: 1040, right: 1268 };
    const viewport = { width: 1280, height: 700 };
    const { top, left } = placePopoverWithin(trigger, MENU, viewport);
    expect(top).toBe(508); // trigger.top - gap - menu height
    expect(left).toBe(1060);
    withinViewport(top, left, viewport);
  });

  it("keeps the right-aligned menu inside the right edge for a rightmost kebab", () => {
    const trigger = { top: 100, bottom: 132, left: 1052, right: 1280 };
    const viewport = { width: 1280, height: 800 };
    const { top, left } = placePopoverWithin(trigger, MENU, viewport);
    expect(left).toBe(1072); // 1280 - 208 — flush with the right edge, never past it
    withinViewport(top, left, viewport);
  });

  it("left-aligns to the trigger when right-alignment would cross the left edge", () => {
    const trigger = { top: 100, bottom: 132, left: 24, right: 52 };
    const viewport = { width: 1280, height: 800 };
    const { top, left } = placePopoverWithin(trigger, MENU, viewport);
    expect(left).toBe(24); // the trigger's own left edge
    withinViewport(top, left, viewport);
  });

  it("pins the menu inside the bottom edge when it fits neither below nor above", () => {
    // A tall trigger filling the viewport: the menu overlaps the trigger, on-screen.
    const trigger = { top: 0, bottom: 696, left: 1040, right: 1268 };
    const viewport = { width: 1280, height: 700 };
    const { top, left } = placePopoverWithin(trigger, MENU, viewport);
    expect(top).toBe(612); // 700 - 88
    withinViewport(top, left, viewport);
  });

  it("stays on-screen on a small phone viewport", () => {
    const trigger = { top: 380, bottom: 412, left: 300, right: 328 };
    const viewport = { width: 360, height: 640 };
    const { top, left } = placePopoverWithin(trigger, MENU, viewport);
    withinViewport(top, left, viewport);
  });

  it("caps the menu at the viewport when it is wider or taller than it", () => {
    const trigger = { top: 50, bottom: 82, left: 10, right: 38 };
    const viewport = { width: 100, height: 40 };
    const { top, left } = placePopoverWithin(trigger, MENU, viewport);
    expect(left).toBe(0);
    expect(top).toBe(0);
  });
});
