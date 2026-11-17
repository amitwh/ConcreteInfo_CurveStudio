import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// jsdom lacks the APIs Plotly needs; every test that mounts a chart mocks the module.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});

// jsdom has no matchMedia; the theme toggle and responsive CSS rely on it.
if (typeof window !== "undefined" && !window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// jsdom has no ResizeObserver; Plotly's responsive mode attaches one.
if (typeof window !== "undefined" && !("ResizeObserver" in window)) {
  class FakeResizeObserver {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  Object.defineProperty(window, "ResizeObserver", {
    writable: true,
    value: FakeResizeObserver,
  });
}
