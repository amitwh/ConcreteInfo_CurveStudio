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
