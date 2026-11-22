import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import { DatasetsSidebar } from "./DatasetsSidebar";

function renderSidebar(): HTMLElement {
  const { container } = render(
    <DatasetsSidebar
      datasets={[]}
      selectedId={null}
      loading={false}
      loadError={null}
      onSelect={() => {}}
      onUploaded={() => {}}
      onDeleted={() => {}}
      collapsed={false}
      onToggleCollapse={() => {}}
    />,
  );
  return container;
}

/** Drive the hidden file input the way the OS file picker would. */
async function pickFile(container: HTMLElement, file: File): Promise<void> {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  await act(async () => {
    fireEvent.change(input, { target: { files: [file] } });
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("DatasetsSidebar — upload validation", () => {
  it("refuses the same X and Y column with a clear message", async () => {
    const upload = vi.spyOn(api, "uploadDataset").mockRejectedValue(
      new ApiError(422, "COLUMNS_REQUIRED", "choose x and y columns", {
        code: "COLUMNS_REQUIRED",
        columns: ["x", "y"],
      }),
    );
    const container = renderSidebar();
    await pickFile(container, new File(["x,y"], "a.csv", { type: "text/csv" }));

    expect(await screen.findByText("Choose columns")).toBeTruthy();
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Y column"), "x");
    await user.click(screen.getByRole("button", { name: "Upload with these columns" }));

    // The validation alert plus the server's COLUMNS_REQUIRED message both show.
    const alerts = screen.getAllByRole("alert").map((element) => element.textContent);
    expect(alerts).toContain("Choose two different columns");
    expect(alerts.join(" ")).toContain("choose x and y columns");
    // The first, rejected upload is the only call — the invalid choice never resends.
    expect(upload).toHaveBeenCalledTimes(1);
  });

  it("reports an oversized file in MB with one decimal, not raw bytes", async () => {
    const upload = vi.spyOn(api, "uploadDataset");
    const container = renderSidebar();
    const big = new File(["x"], "big.csv", { type: "text/csv" });
    Object.defineProperty(big, "size", { value: 25_000_000 });

    await pickFile(container, big);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("25.0 MB");
    expect(alert.textContent).toContain("20.0 MB");
    expect(upload).not.toHaveBeenCalled();
  });
});

describe("DatasetsSidebar — pasting a spreadsheet block", () => {
  it("uploads a pasted grid as a .tsv file", async () => {
    const upload = vi.spyOn(api, "uploadDataset").mockResolvedValue({
      id: "ds-1",
      name: "pasted",
      n_rows: 2,
      x_column: "x",
      y_column: "y",
      weight_column: null,
      issues: [],
      created_at: "2026-11-22T08:00:00+05:30",
    } as never);
    const user = userEvent.setup();
    render(
      <DatasetsSidebar
        datasets={[]}
        selectedId={null}
        loading={false}
        loadError={null}
        onSelect={() => {}}
        onUploaded={() => {}}
        onDeleted={() => {}}
        collapsed={false}
        onToggleCollapse={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Paste data" }));
    const grid = screen.getByRole("grid", { name: "Pasted data" });
    fireEvent.paste(grid.querySelector('input[aria-label="A1"]') as HTMLElement, {
      clipboardData: { getData: () => "x\ty\n1\t2.5\n2\t4.5\n" },
    });
    await user.click(screen.getByRole("button", { name: "Import paste" }));

    expect(upload).toHaveBeenCalledTimes(1);
    const file = upload.mock.calls[0]?.[0] as File;
    expect(file.name).toBe("pasted.tsv");
    expect(await file.text()).toBe("x\ty\n1\t2.5\n2\t4.5\n");
  });
});

describe("DatasetsSidebar — collapse", () => {
  it("shows a collapse control that hides the upload flow", async () => {
    const onToggleCollapse = vi.fn();
    const user = userEvent.setup();
    render(
      <DatasetsSidebar
        datasets={[]}
        selectedId={null}
        loading={false}
        loadError={null}
        onSelect={() => {}}
        onUploaded={() => {}}
        onDeleted={() => {}}
        collapsed={false}
        onToggleCollapse={onToggleCollapse}
      />,
    );

    expect(screen.getByText("Drop a data file here")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Collapse datasets" }));
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);
  });

  it("renders a narrow rail with the count when collapsed", async () => {
    const onToggleCollapse = vi.fn();
    const user = userEvent.setup();
    render(
      <DatasetsSidebar
        datasets={[]}
        selectedId={null}
        loading={false}
        loadError={null}
        onSelect={() => {}}
        onUploaded={() => {}}
        onDeleted={() => {}}
        collapsed
        onToggleCollapse={onToggleCollapse}
      />,
    );

    expect(screen.queryByText("Drop a data file here")).toBeNull();
    expect(screen.getByTitle("0 datasets")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Expand datasets" }));
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);
  });
});
