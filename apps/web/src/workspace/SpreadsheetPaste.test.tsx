import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SpreadsheetPaste } from "./SpreadsheetPaste";

function paste(element: HTMLElement, text: string): void {
  fireEvent.paste(element, { clipboardData: { getData: () => text } });
}

function cell(container: HTMLElement, label: string): HTMLInputElement {
  return container.querySelector(`input[aria-label="${label}"]`) as HTMLInputElement;
}

describe("SpreadsheetPaste", () => {
  it("fills the grid from a pasted block and imports it as TSV", async () => {
    const onImport = vi.fn();
    const { container } = render(<SpreadsheetPaste onImport={onImport} onClose={() => {}} />);

    paste(cell(container, "A1"), "time\tload\n0\t1.2\n1\t2.3\n2\t4.4\n");
    expect(cell(container, "A1").value).toBe("time");
    expect(cell(container, "B2").value).toBe("1.2");
    expect(cell(container, "A4").value).toBe("2");

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Import paste" }));
    expect(onImport).toHaveBeenCalledWith("time\tload\n0\t1.2\n1\t2.3\n2\t4.4\n");
  });

  it("grows rows and columns when a block is pasted beside existing cells", async () => {
    const user = userEvent.setup();
    const { container } = render(<SpreadsheetPaste onImport={() => {}} onClose={() => {}} />);
    await user.type(screen.getByLabelText("A1"), "kept"); // a blank grid is replaced, so seed one cell
    paste(cell(container, "C1"), "p\tq\tr\n1\t2\t3\n");
    expect(cell(container, "A1").value).toBe("kept");
    expect(cell(container, "E1").value).toBe("r");
    expect(cell(container, "C2").value).toBe("1");
    expect(container.querySelectorAll('th[scope="col"].spreadsheet__letter').length).toBe(5);
  });

  it("keeps a cell containing a comma intact on import", async () => {
    const onImport = vi.fn();
    const { container } = render(<SpreadsheetPaste onImport={onImport} onClose={() => {}} />);
    paste(cell(container, "A1"), 'name\tvalue\n"a,b"\t1\n');
    expect(cell(container, "A2").value).toBe("a,b");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Import paste" }));
    expect(onImport).toHaveBeenCalledWith('name\tvalue\n"a,b"\t1\n');
  });

  it("disables the import until something is entered", async () => {
    const user = userEvent.setup();
    render(<SpreadsheetPaste onImport={() => {}} onClose={() => {}} />);
    const button = screen.getByRole("button", { name: "Import paste" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    await user.type(screen.getByLabelText("A1"), "1");
    expect(button.disabled).toBe(false);
  });

  it("closes on Escape and on Cancel", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<SpreadsheetPaste onImport={() => {}} onClose={onClose} />);
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();

    onClose.mockClear();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalled();
  });
});
