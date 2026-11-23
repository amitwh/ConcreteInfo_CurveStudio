import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatTile } from "./StatTile";

describe("StatTile", () => {
  it("shows the label, the figure and its explanation as one titled group", () => {
    render(<StatTile label="Accounts" value="128" explanation="Registered accounts." />);
    const group = screen.getByRole("group", { name: "Accounts" });
    expect(within(group).getByText("Accounts")).toBeTruthy();
    expect(within(group).getByText("128")).toBeTruthy();
    expect(within(group).getByText("Registered accounts.")).toBeTruthy();
  });

  it("takes a figure made of several numbers", () => {
    render(
      <StatTile
        label="Active users"
        value={
          <>
            <span>3</span> <span>14</span> <span>62</span>
          </>
        }
        explanation="Accounts with a signed-in request in the last 1, 7 and 30 days."
      />,
    );
    const group = screen.getByRole("group", { name: "Active users" });
    expect(within(group).getByText("14")).toBeTruthy();
    expect(within(group).getByText("62")).toBeTruthy();
  });

  it("is still one group when two tiles carry the same figure", () => {
    render(
      <>
        <StatTile label="Accounts" value="1" explanation="Registered accounts." />
        <StatTile label="Installs" value="1" explanation="Distinct installations ever seen." />
      </>,
    );
    expect(screen.getAllByRole("group")).toHaveLength(2);
    expect(within(screen.getByRole("group", { name: "Installs" })).getByText("1")).toBeTruthy();
  });
});
