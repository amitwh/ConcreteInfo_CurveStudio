import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Pagination } from "./Pagination";

function button(name: "Previous" | "Next"): HTMLButtonElement {
  return screen.getByRole("button", { name }) as HTMLButtonElement;
}

describe("Pagination", () => {
  it("says what is showing, with en-IN digits, and names the ends", () => {
    render(<Pagination page={1} pageSize={25} total={128} onPageChange={vi.fn()} />);
    expect(screen.getByText("Showing 1–25 of 128")).toBeTruthy();
  });

  it("groups a large total the Indian way", () => {
    render(<Pagination page={4000} pageSize={25} total={100000} onPageChange={vi.fn()} />);
    expect(screen.getByText("Showing 99,976–1,00,000 of 1,00,000")).toBeTruthy();
  });

  it("disables Previous on the first page and enables Next", () => {
    render(<Pagination page={1} pageSize={25} total={128} onPageChange={vi.fn()} />);
    expect(button("Previous").disabled).toBe(true);
    expect(button("Next").disabled).toBe(false);
  });

  it("disables Next on the last page, whose range ends at the total", () => {
    render(<Pagination page={6} pageSize={25} total={128} onPageChange={vi.fn()} />);
    expect(screen.getByText("Showing 126–128 of 128")).toBeTruthy();
    expect(button("Next").disabled).toBe(true);
    expect(button("Previous").disabled).toBe(false);
  });

  it("disables both on the only page", () => {
    render(<Pagination page={1} pageSize={25} total={25} onPageChange={vi.fn()} />);
    expect(screen.getByText("Showing 1–25 of 25")).toBeTruthy();
    expect(button("Previous").disabled).toBe(true);
    expect(button("Next").disabled).toBe(true);
  });

  it("says nothing is showing when there is no total", () => {
    render(<Pagination page={1} pageSize={25} total={0} onPageChange={vi.fn()} />);
    expect(screen.getByText("Showing 0 of 0")).toBeTruthy();
    expect(button("Previous").disabled).toBe(true);
    expect(button("Next").disabled).toBe(true);
  });

  it("asks for the next and the previous page, never one past an end", async () => {
    const user = userEvent.setup();
    const onPageChange = vi.fn();
    const { rerender } = render(
      <Pagination page={2} pageSize={25} total={128} onPageChange={onPageChange} />,
    );
    await user.click(button("Next"));
    expect(onPageChange).toHaveBeenLastCalledWith(3);
    await user.click(button("Previous"));
    expect(onPageChange).toHaveBeenLastCalledWith(1);

    rerender(<Pagination page={1} pageSize={25} total={128} onPageChange={onPageChange} />);
    await user.click(button("Previous"));
    expect(onPageChange).toHaveBeenCalledTimes(2);
  });

  it("announces nothing on its own", () => {
    const { container } = render(
      <Pagination page={1} pageSize={25} total={128} onPageChange={vi.fn()} />,
    );
    expect(container.querySelector("[aria-live], [role='status'], [role='alert']")).toBeNull();
  });
});
