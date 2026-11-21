import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { api } from "../../api/client";
import type { components } from "../../api/schema";
import { AUTOFIT_IDLE, type AutofitState } from "./useAutofit";
import { AutofitPanel } from "./AutofitPanel";

type ProgressOut = components["schemas"]["ProgressOut"];
type AutofitOut = components["schemas"]["AutofitOut"];
type RankedFit = components["schemas"]["RankedFit"];

function row(modelId: string, overrides: Record<string, unknown> = {}): RankedFit {
  return {
    rank: 1,
    model_id: modelId,
    name: `Model ${modelId}`,
    category: "Exponential",
    equation_fitted: "y = 2.05 * exp(-0.32 * x)",
    metrics: { k: 2, n: 40, r2: 0.99, adj_r2: 0.98, rmse: 0.05, aicc: -120, bic: -117, cv_rmse: 0.06 },
    ...overrides,
  } as RankedFit;
}

function state(overrides: Partial<AutofitState> = {}): AutofitState {
  return { ...AUTOFIT_IDLE, ...overrides };
}

function stored(overrides: Record<string, unknown> = {}): AutofitOut {
  return {
    id: "af-1",
    dataset_id: "ds-1",
    created_at: "2026-11-17T10:00:00Z",
    best_model_id: "exp_decay",
    n_candidates: 12,
    rank_by: "best_overall",
    partial: false,
    result: {
      config: { rank_by: "best_overall" },
      dataset_fingerprint: "fp",
      n_active: 40,
      ranked: [row("exp_decay"), row("poly3", { rank: 2 })],
      outcomes: [
        {
          candidate_id: "exp_decay",
          outcome: "converged",
          name: "Exponential decay",
          category: "Exponential",
          reason: "",
        },
        {
          candidate_id: "stretched_exp",
          outcome: "not_converged",
          name: "Stretched exponential",
          category: "Exponential",
          reason: "no improvement over the last 200 iterations",
        },
        {
          candidate_id: "gauss4",
          outcome: "skipped",
          name: "4-term Gaussian",
          category: "Peak",
          reason: "",
        },
        {
          candidate_id: "poly5",
          outcome: "exact_fit",
          name: "Polynomial (degree 5)",
          category: "polynomial",
          reason: "",
        },
      ],
      skipped: { gauss4: ["needs more points than the dataset has (4 < 7)"] },
      unconverged: { stretched_exp: "no improvement over the last 200 iterations" },
      exact_fits: ["poly5"],
      warnings: ["cross-validation was incomplete for 1 candidate"],
      cancelled: false,
      elapsed_seconds: 8.4,
      curves: {},
      overfitting_note: null,
    },
    ...overrides,
  } as unknown as AutofitOut;
}

const models = [
  {
    id: "exp_decay",
    name: "Exponential decay",
    category: "Exponential",
    equation: "y = a * exp(-b * x)",
    latex: "",
    parameters: [],
    variables: ["x"],
    references: [],
    assumptions: "",
    is_family: false,
  },
] as unknown as Parameters<typeof AutofitPanel>[0]["models"];

function renderPanel(overrides: Partial<AutofitState>, props: Record<string, unknown> = {}) {
  const handlers = {
    onRun: vi.fn(),
    onCancel: vi.fn(),
    onRerank: vi.fn(),
    onOpenRow: vi.fn(),
    onUseManually: vi.fn(),
  };
  render(
    <AutofitPanel
      datasetId="ds-1"
      datasetName="growth.csv"
      state={state(overrides)}
      models={models}
      selectedRowId="exp_decay"
      onSelectRow={() => {}}
      openRowBusy={null}
      openRowError={null}
      {...handlers}
      {...props}
    />,
  );
  return handlers;
}

describe("AutofitPanel — progress", () => {
  it("shows a determinate bar, counts, elapsed time, live top rows and a cancel button", () => {
    const progress: ProgressOut = {
      done: 7,
      total: 12,
      converged: 5,
      failed: 1,
      skipped: 1,
      elapsed_s: 4.25,
      partial: false,
      top: [row("exp_decay")],
    };
    renderPanel({ phase: "running", progress });
    const bar = screen.getByRole("progressbar");
    expect(bar.getAttribute("value")).toBe("7");
    expect(bar.getAttribute("max")).toBe("12");
    expect(screen.getByText("7/12")).toBeTruthy();
    expect(screen.getByText(/converged 5/)).toBeTruthy();
    expect(screen.getByText(/not converged 1/)).toBeTruthy();
    expect(screen.getByText(/skipped 1/)).toBeTruthy();
    expect(screen.getByText(/4\.3\s*s/)).toBeTruthy();
    expect(screen.getByText("Model exp_decay")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
    expect(screen.getByRole("status").getAttribute("aria-live")).toBe("polite");
  });
});

describe("AutofitPanel — errors and edge states", () => {
  it("explains TOO_FEW_POINTS from detail.code", () => {
    renderPanel({ phase: "error", error: "the dataset cannot be auto-fitted", errorCode: "TOO_FEW_POINTS" });
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toMatch(/too few points/i);
  });

  it("explains a quota rejection", () => {
    renderPanel({ phase: "error", error: "too many auto-fits today", errorCode: "QUOTA_EXCEEDED" });
    expect(screen.getByRole("alert").textContent).toMatch(/quota/i);
  });

  it("shows a generic message for other failures", () => {
    renderPanel({ phase: "error", error: "lost contact with the server", errorCode: null });
    expect(screen.getByRole("alert").textContent).toContain("lost contact with the server");
  });

  it("shows the empty state when no dataset is selected", () => {
    render(
      <AutofitPanel
        datasetId={null}
        datasetName=""
        state={AUTOFIT_IDLE}
        models={[]}
        selectedRowId={null}
        onSelectRow={() => {}}
        onOpenRow={vi.fn()}
        onUseManually={() => {}}
        onRun={() => {}}
        onCancel={() => {}}
        onRerank={() => {}}
        openRowBusy={null}
        openRowError={null}
      />,
    );
    expect(screen.getByText(/select or upload a dataset/i)).toBeTruthy();
  });
});

describe("AutofitPanel — results", () => {
  it("renders the ranking table with the active ordering selected", () => {
    renderPanel({ phase: "ready", stored: stored() });
    expect(screen.getByText("Model exp_decay")).toBeTruthy();
    const order = screen.getByLabelText(/order by/i) as HTMLSelectElement;
    expect(order.value).toBe("best_overall");
  });

  it("re-ranks through the API when the ordering changes (no refit)", async () => {
    const handlers = renderPanel({ phase: "ready", stored: stored() });
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText(/order by/i), "aicc");
    expect(handlers.onRerank).toHaveBeenCalledWith("aicc");
  });

  it("shows the overfitting note banner when present", () => {
    const s = stored();
    (s.result as { overfitting_note: string | null }).overfitting_note =
      "poly2 fits nearly as well with two fewer parameters";
    renderPanel({ phase: "ready", stored: s });
    expect(screen.getByText(/poly2 fits nearly as well/i)).toBeTruthy();
  });

  it("lists exact fits with the information-criteria note", () => {
    renderPanel({ phase: "ready", stored: stored() });
    const heading = screen.getByText(/exact fits/i);
    const section = heading.closest("section") ?? heading.parentElement;
    expect(section?.textContent).toContain("Polynomial (degree 5)");
    expect(section?.textContent).toMatch(/information criteria are not defined/i);
  });

  it("collapses Not converged and Skipped with their counts; expanding shows model and reason", async () => {
    renderPanel({ phase: "ready", stored: stored() });
    // The count lives in the <summary> of a collapsed <details> (an outer element's
    // text content matches too, so pick the one inside a summary).
    const summaryText = (pattern: RegExp): HTMLElement => {
      const match = screen
        .getAllByText(pattern)
        .find((element) => element.closest("summary") !== null);
      if (match === undefined) throw new Error(`no summary matches ${String(pattern)}`);
      return match;
    };
    const unconverged = summaryText(/not converged \(1\)/i);
    const skipped = summaryText(/skipped \(1\)/i);
    expect(unconverged.tagName).not.toBe("DETAILS");
    expect(skipped.tagName).not.toBe("DETAILS");

    const user = userEvent.setup();
    await user.click(unconverged);
    expect(screen.getByText("Stretched exponential")).toBeTruthy();
    expect(screen.getByText(/no improvement over the last 200 iterations/i)).toBeTruthy();
    await user.click(skipped);
    expect(screen.getByText("4-term Gaussian")).toBeTruthy();
    expect(screen.getByText(/needs more points than the dataset has/i)).toBeTruthy();
  });

  it("shows run warnings", () => {
    renderPanel({ phase: "ready", stored: stored() });
    expect(screen.getByText(/cross-validation was incomplete/i)).toBeTruthy();
  });

  it("marks a partial stored run with a clear partial notice", () => {
    const partialStored = stored({ partial: true });
    (partialStored.result as { cancelled: boolean }).cancelled = true;
    renderPanel({ phase: "ready", stored: partialStored });
    expect(screen.getAllByText(/partial/i).length).toBeGreaterThan(0);
  });

  it("keeps what was ranked after a cancel, with a partial notice", () => {
    renderPanel({
      phase: "ready",
      stored: null,
      cancelled: true,
      partialRows: [row("exp_decay")],
    });
    expect(screen.getByText(/cancelled/i).textContent).toMatch(/partial|stopped/i);
    expect(screen.getByText("Model exp_decay")).toBeTruthy();
  });

  it("exports CSV and JSON", async () => {
    const download = vi.spyOn(api, "download").mockResolvedValue({
      blob: new Blob(["x"]),
      filename: "autofit.csv",
    });
    renderPanel({ phase: "ready", stored: stored() });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /export csv/i }));
    expect(download).toHaveBeenCalledWith("/autofits/af-1/export?format=csv");
    await user.click(screen.getByRole("button", { name: /export json/i }));
    expect(download).toHaveBeenCalledWith("/autofits/af-1/export?format=json");
    download.mockRestore();
  });

  it("opens the selected row and offers to use its model manually", async () => {
    const handlers = renderPanel({ phase: "ready", stored: stored() });
    const user = userEvent.setup();
    await user.click(screen.getAllByRole("button", { name: /open full results/i })[0]!);
    expect(handlers.onOpenRow).toHaveBeenCalledWith("exp_decay");
    await user.click(screen.getAllByRole("button", { name: /use this model manually/i })[0]!);
    expect(handlers.onUseManually).toHaveBeenCalled();
  });

  it("offers no Open full results control for a cancelled run's partial rows", () => {
    renderPanel({
      phase: "ready",
      stored: null,
      cancelled: true,
      partialRows: [row("exp_decay")],
    });
    // The rows are still shown…
    expect(screen.getByText("Model exp_decay")).toBeTruthy();
    // …but nothing can be opened from a run that stored nothing.
    expect(screen.queryAllByRole("button", { name: /open full results/i })).toHaveLength(0);
  });
});

describe("AutofitPanel — pagination", () => {
  function manyRows(count: number): RankedFit[] {
    return Array.from({ length: count }, (_, index) =>
      row(`model_${index + 1}`, { rank: index + 1 }),
    );
  }

  function manyStored(count: number): AutofitOut {
    const result = stored();
    (result.result as { ranked: RankedFit[] }).ranked = manyRows(count);
    result.n_candidates = count;
    return result;
  }

  it("shows the first page and the range for a 116-row run", () => {
    renderPanel({ phase: "ready", stored: manyStored(116) });
    expect(screen.getByText("1–10 of 116")).toBeTruthy();
    // header row + ten body rows
    expect(screen.getAllByRole("row")).toHaveLength(11);
    expect(screen.queryByText("Model model_11")).toBeNull();
    expect((screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it("moves to the next page", async () => {
    renderPanel({ phase: "ready", stored: manyStored(116) });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("11–20 of 116")).toBeTruthy();
    expect(screen.getByText("Model model_11")).toBeTruthy();
    expect(screen.queryByText("Model model_1")).toBeNull();
  });

  it("changes the page size from the footer select", async () => {
    renderPanel({ phase: "ready", stored: manyStored(116) });
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText(/rows per page/i), "25");
    expect(screen.getByText("1–25 of 116")).toBeTruthy();
    expect(screen.getAllByRole("row")).toHaveLength(26);
  });
});

describe("AutofitPanel — cancel availability", () => {
  it("disables Cancel until the run has a job id", () => {
    renderPanel({ phase: "starting", jobId: null });
    const cancel = screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement;
    expect(cancel.disabled).toBe(true);
    expect(cancel.getAttribute("title")).toMatch(/once the run (has )?start/i);
  });

  it("enables Cancel once the run has a job id", () => {
    renderPanel({ phase: "running", jobId: "job-1" });
    const cancel = screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement;
    expect(cancel.disabled).toBe(false);
  });
});
