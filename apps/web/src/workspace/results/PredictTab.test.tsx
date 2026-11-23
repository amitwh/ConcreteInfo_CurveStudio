import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { FitBundle } from "../../api/client";
import { PredictTab } from "./PredictTab";

const client = vi.hoisted(() => ({
  download: vi.fn(),
  saveBlob: vi.fn(),
}));
vi.mock("../../api/client", () => ({
  api: { download: client.download },
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
