import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { FitBundle } from "../../api/client";
import { PredictTab } from "./PredictTab";

const client = vi.hoisted(() => ({
  download: vi.fn(),
  request: vi.fn(),
  saveBlob: vi.fn(),
}));
vi.mock("../../api/client", () => ({
  api: { download: client.download, request: client.request },
  saveBlob: client.saveBlob,
}));

const converged = { result: { status: "converged" } } as unknown as FitBundle;

describe("PredictTab export", () => {
  beforeEach(() => {
    client.download.mockReset();
    client.saveBlob.mockReset();
  });

  it("offers CSV, JSON and the Excel workbook beside each other", () => {
    render(<PredictTab fitId="fit-1" bundle={converged} />);
    expect(screen.getByRole("button", { name: "Download CSV" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Download JSON" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Download Excel workbook" })).toBeTruthy();
  });

  it("downloads the workbook from the xlsx export route and saves it under the served name", async () => {
    const blob = new Blob(["PK"], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    client.download.mockResolvedValue({ blob, filename: "fit-1.xlsx" });
    render(<PredictTab fitId="fit-1" bundle={converged} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Download Excel workbook" }));
    expect(client.download).toHaveBeenCalledTimes(1);
    expect(client.download).toHaveBeenCalledWith("/fits/fit-1/export?format=xlsx");
    expect(client.saveBlob).toHaveBeenCalledTimes(1);
    expect(client.saveBlob).toHaveBeenCalledWith(blob, "fit-1.xlsx");
  });
});

describe("PredictTab table sorting", () => {
  beforeEach(() => {
    client.request.mockReset();
    client.request.mockResolvedValue({
      x: [3, 1, 2],
      y: [30, 10, 20],
      band: null,
      lower: null,
      upper: null,
      extrapolated: [false, false, true],
      band_reason: null,
    });
  });

  async function predict(user: ReturnType<typeof userEvent.setup>): Promise<void> {
    render(<PredictTab fitId="fit-1" bundle={converged} />);
    await user.type(screen.getByLabelText(/x values/i), "3 1 2");
    await user.click(screen.getByRole("button", { name: "Predict" }));
    await screen.findByRole("table");
  }

  function firstColumn(): string[] {
    return screen
      .getAllByRole("row")
      .slice(1)
      .map((row) => row.querySelector("td")?.textContent ?? "");
  }

  it("sorts by ŷ, reverses, then returns to the entered x order", async () => {
    const user = userEvent.setup();
    await predict(user);
    expect(firstColumn()).toEqual(["3", "1", "2"]); // the entered order
    await user.click(screen.getByRole("button", { name: "ŷ" }));
    expect(firstColumn()).toEqual(["1", "2", "3"]); // ascending ŷ
    await user.click(screen.getByRole("button", { name: "ŷ" }));
    expect(firstColumn()).toEqual(["3", "2", "1"]); // descending
    await user.click(screen.getByRole("button", { name: "ŷ" }));
    expect(firstColumn()).toEqual(["3", "1", "2"]); // back to the entered order
  });

  it("sorts the extrapolated flag", async () => {
    const user = userEvent.setup();
    await predict(user);
    await user.click(screen.getByRole("button", { name: "extrapolated" }));
    const rows = screen.getAllByRole("row").slice(1);
    // ascending "no" < "yes": the extrapolated row (x=2, the third entered value) is last
    expect(rows.map((row) => row.querySelector("td")?.textContent)).toEqual(["3", "1", "2"]);
    expect(rows[2]?.textContent).toContain("yes");
  });
});
