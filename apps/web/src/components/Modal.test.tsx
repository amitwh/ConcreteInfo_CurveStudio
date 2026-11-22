import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Modal } from "./Modal";

describe("Modal", () => {
  it("is a labelled dialog that closes on Escape and on the backdrop", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    const { container, rerender } = render(
      <Modal title="Help contents" onClose={onClose}>
        <p>Body</p>
      </Modal>,
    );
    const dialog = screen.getByRole("dialog", { name: "Help contents" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(screen.getByText("Body")).toBeTruthy();

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);

    const backdrop = container.querySelector(".modal") as HTMLElement;
    await user.pointer({ keys: "[MouseLeft]", target: backdrop });
    expect(onClose).toHaveBeenCalledTimes(2);

    // a click inside the panel never closes it
    rerender(
      <Modal title="Help contents" onClose={onClose}>
        <p>Body</p>
      </Modal>,
    );
    await user.click(screen.getByText("Body"));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("offers an explicit close button", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <Modal title="About CurveStudio" onClose={onClose}>
        <p>Body</p>
      </Modal>,
    );
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("locks page scrolling while open and gives it back on close", () => {
    const { unmount } = render(
      <Modal title="Help" onClose={() => {}}>
        <p>Body</p>
      </Modal>,
    );
    expect(document.body.style.overflow).toBe("hidden");
    unmount();
    expect(document.body.style.overflow).toBe("");
  });
});
