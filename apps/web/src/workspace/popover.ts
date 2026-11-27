/**
 * Viewport-aware placement for a fixed popover anchored to a trigger (the ranking
 * table's row menu). The popover must stay fully on-screen at every scroll position:
 * it prefers opening just below the trigger, right-aligned to the trigger's right
 * edge, flips above when the space below is short, and clamps both axes inside the
 * viewport. Pure on purpose — jsdom has no layout, so the clamping rules are what the
 * unit tests verify; the component measures the real boxes around this helper.
 */

/** A trigger or popover box in viewport coordinates, as getBoundingClientRect returns. */
export interface Box {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

/** The popover's rendered size. */
export interface Size {
  width: number;
  height: number;
}

/** Gap between the trigger and the popover, matching the menu's old visual offset. */
const GAP = 4;

/**
 * Where to place a popover of `popover` size for a trigger at `trigger`, inside a
 * `viewport`. Returns viewport coordinates for CSS `top` and `left` (never `right`:
 * its origin is the opposite edge, which is how the menu used to end up off-screen).
 */
export function placePopoverWithin(
  trigger: Box,
  popover: Size,
  viewport: Size,
): { top: number; left: number } {
  const width = Math.min(popover.width, viewport.width);
  const height = Math.min(popover.height, viewport.height);

  // Right-align the popover to the trigger's right edge; a trigger nearer the left
  // border than the popover is wide left-aligns to it instead, and the result is
  // clamped so a popover at least as wide as the viewport still fits.
  let left = trigger.right - width;
  if (left < 0) left = trigger.left;
  left = Math.max(0, Math.min(left, viewport.width - width));

  // Below the trigger when it fits, above it when the whole menu fits there, and
  // pinned just inside the bottom edge when neither side has the room.
  let top = trigger.bottom + GAP;
  if (top + height > viewport.height) {
    const above = trigger.top - GAP - height;
    top =
      above >= 0 && above + height <= viewport.height
        ? above
        : Math.max(0, viewport.height - height);
  }
  return { top, left };
}
