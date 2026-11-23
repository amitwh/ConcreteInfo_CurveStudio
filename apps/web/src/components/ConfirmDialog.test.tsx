import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ConfirmDialog } from "./ConfirmDialog";

function renderDialog(overrides: Partial<Parameters<typeof ConfirmDialog>[0]> = {}) {
  const onConfirm = vi.fn();
  const onClose = vi.fn();
  const props = {
    title: "Sign out everywhere?",
    message: "Every other device will be signed out.",
    confirmLabel: "Sign out everywhere",
    onConfirm,
    onClose,
    ...overrides,
  };
  const view = render(<ConfirmDialog {...props} />);
  return { ...view, onConfirm, onClose, props };
}

describe("ConfirmDialog", () => {
  it("is a labelled dialog that shows its title and message", () => {
    renderDialog();
    const dialog = screen.getByRole("dialog", { name: "Sign out everywhere?" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(screen.getByText("Every other device will be signed out.")).toBeTruthy();
  });

  it("without a phrase, Confirm is enabled and calls onConfirm", async () => {
    const user = userEvent.setup();
    const { onConfirm, onClose } = renderDialog();
    const confirm = screen.getByRole("button", { name: "Sign out everywhere" }) as HTMLButtonElement;
    expect(confirm.disabled).toBe(false);
    await user.click(confirm);
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onClose).not.toHaveBeenCalled();
  });

  it("with a phrase, Confirm stays disabled until the typed text matches", async () => {
    const user = userEvent.setup();
    const { onConfirm } = renderDialog({
      title: "Delete account?",
      message: "Your datasets, fits and jobs will be deleted.",
      confirmLabel: "Delete account",
      phrase: "amit.h",
    });
    const confirm = screen.getByRole("button", { name: "Delete account" }) as HTMLButtonElement;
    const input = screen.getByLabelText(/type amit\.h to confirm/i);
    // styles.css styles inputs by their type attribute: without it the field is unstyled
    expect(input.getAttribute("type")).toBe("text");
    expect(confirm.disabled).toBe(true);

    await user.type(input, "amit");
    expect(confirm.disabled).toBe(true);

    // The server compares the phrase case-insensitively; the dialog does not insist on more.
    await user.type(input, ".H");
    expect(confirm.disabled).toBe(false);
    await user.click(confirm);
    expect(onConfirm).toHaveBeenCalledWith("amit.H");
  });

  it("calls onClose from Escape and from Cancel, never onConfirm", async () => {
    const user = userEvent.setup();
    const { onConfirm, onClose } = renderDialog();
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalledTimes(2);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("closes from the backdrop", async () => {
    const user = userEvent.setup();
    const { container, onClose } = renderDialog();
    const backdrop = container.querySelector(".modal") as HTMLElement;
    await user.pointer({ keys: "[MouseLeft]", target: backdrop });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("uses the cancel label it is given", async () => {
    const user = userEvent.setup();
    const { onClose } = renderDialog({ cancelLabel: "Keep it" });
    await user.click(screen.getByRole("button", { name: "Keep it" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("while busy, both buttons are disabled and Escape does not dismiss it", async () => {
    const user = userEvent.setup();
    const { onConfirm, onClose } = renderDialog({ busy: true });
    const confirm = screen.getByRole("button", { name: "Sign out everywhere" }) as HTMLButtonElement;
    const cancel = screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement;
    expect(confirm.disabled).toBe(true);
    expect(cancel.disabled).toBe(true);
    await user.keyboard("{Escape}");
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Sign out everywhere?" })).toBeTruthy();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("shows an error and stays open", () => {
    renderDialog({ error: "This does not match the part of your e-mail address before the @" });
    expect(screen.getByRole("alert").textContent).toBe(
      "This does not match the part of your e-mail address before the @",
    );
    expect(screen.getByRole("dialog", { name: "Sign out everywhere?" })).toBeTruthy();
  });

  it("styles the confirm button as danger only when asked", () => {
    const common = {
      title: "A",
      message: "m",
      confirmLabel: "Yes",
      onConfirm: vi.fn(),
      onClose: vi.fn(),
    };
    const { rerender } = render(<ConfirmDialog {...common} />);
    const confirm = (): HTMLElement => screen.getByRole("button", { name: "Yes" });
    expect(confirm().className).not.toContain("btn--danger-solid");

    rerender(<ConfirmDialog {...common} danger />);
    expect(confirm().className).toContain("btn--danger-solid");
  });
});
