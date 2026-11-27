/**
 * The Auto-fit ranking table. Rows come pre-ordered from the server (the ordering is
 * the `RankBy` selector's business — the client never re-sorts the *ranking*); a metric
 * that is not defined renders as "n/a", never 0 or NaN. Rows with the same fitted curve
 * share their rank, and `same_curve_as` names the others. Keyboard: arrow keys move the
 * selection, Enter opens the selected row's full results.
 *
 * Column headers are a display ordering only: clicking one reorders the visible rows
 * client-side (ascending → descending → back to the ranked order) and the rank column
 * stays visible, so the true order is always readable. A row's kebab menu holds exactly
 * the actions the row's buttons used to offer — open the full results, use the model
 * manually — nothing more.
 */

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
} from "react";
import { MoreVertical } from "lucide-react";
import { CopyButton } from "../../components/CopyButton";
import { fmtEquation, fmtInt, fmtNumber } from "../../format";
import type { RankedFit } from "../../api/client";
import { modelColor } from "../traces";
import { SortableTh, cycleSort, sortRows, type SortKey, type SortState } from "../sorting";
import { placePopoverWithin } from "../popover";

/** Human label + semantic badge for a candidate's fit status. */
const STATUS_BADGE: Record<string, { label: string; tone: "ok" | "bad" }> = {
  converged: { label: "Converged", tone: "ok" },
  not_converged: { label: "Not converged", tone: "bad" },
};

function StatusBadge({ status }: { status: string }): ReactElement {
  const known = STATUS_BADGE[status];
  if (known !== undefined) {
    return (
      <span className={`badge badge--${known.tone}`}>{known.label}</span>
    );
  }
  return <span className="badge badge--muted">{status}</span>;
}

/** Display sort keys by column id; the ranked order itself is never rebuilt. */
const SORT_KEYS: Record<string, SortKey<RankedFit>> = {
  rank: (row) => row.rank,
  model: (row) => row.name,
  r2: (row) => row.metrics.r2,
  adj_r2: (row) => row.metrics.adj_r2,
  rmse: (row) => row.metrics.rmse,
  aicc: (row) => row.metrics.aicc,
  bic: (row) => row.metrics.bic,
  cv_rmse: (row) => row.metrics.cv_rmse,
  k: (row) => row.metrics.k,
  status: (row) => row.status,
};

/**
 * One row's overflow menu: every action the row's inline buttons used to offer, fired by
 * the same handlers. Opening moves focus to the first item; the arrow keys move between
 * items; Escape closes and returns focus to the kebab; a press anywhere outside — or a
 * scroll of the table under it — closes it too. The popover is `position: fixed` under
 * the kebab because the table's scroll container would clip an absolutely-positioned
 * one, and its placement goes through `placePopoverWithin` so it is always fully
 * inside the viewport — a last row's kebab opens the menu upwards, a rightmost one
 * never has it spill past the right or left edge.
 */

/** The menu's stylesheet size (13rem wide) for the first, pre-measurement placement. */
const MENU_ESTIMATE = { width: 208, height: 88 };

function RowActionMenu({
  row,
  onOpen,
  onUseManually,
  showUseManually,
  openRowBusy,
}: {
  row: RankedFit;
  onOpen: (modelId: string) => void;
  onUseManually: (row: RankedFit) => void;
  showUseManually: boolean;
  openRowBusy: string | null;
}): ReactElement | null {
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState<{ top: number; left: number } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const firstItemRef = useRef<HTMLButtonElement>(null);
  const buttonId = useId();
  const menuId = useId();

  const close = useCallback((restoreFocus: boolean): void => {
    setOpen(false);
    setAnchor(null);
    if (restoreFocus) buttonRef.current?.focus();
  }, []);

  const openAtTrigger = (): void => {
    const rect = buttonRef.current?.getBoundingClientRect();
    if (rect !== undefined) {
      setAnchor(
        placePopoverWithin(rect, MENU_ESTIMATE, {
          width: window.innerWidth,
          height: window.innerHeight,
        }),
      );
    }
    setOpen(true);
  };

  useEffect(() => {
    if (open) firstItemRef.current?.focus();
  }, [open]);

  // The first placement estimates the menu's size; once it is on the page its real
  // box replaces the estimate — before the browser paints, so the menu never shows
  // at an off-screen spot.
  useEffect(() => {
    if (!open || anchor === null) return;
    const menuRect = menuRef.current?.getBoundingClientRect();
    const buttonRect = buttonRef.current?.getBoundingClientRect();
    if (menuRect === undefined || buttonRect === undefined) return;
    const next = placePopoverWithin(
      buttonRect,
      { width: menuRect.width, height: menuRect.height },
      { width: window.innerWidth, height: window.innerHeight },
    );
    if (next.top !== anchor.top || next.left !== anchor.left) setAnchor(next);
  }, [open, anchor]);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent): void => {
      if (!wrapRef.current?.contains(event.target as Node)) close(false);
    };
    // capture: any scroll — the table's own or a page one — leaves the anchor behind
    const onScroll = (): void => close(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [open, close]);

  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === "Escape") {
      event.preventDefault();
      close(true);
      return;
    }
    if (event.key === "Tab") {
      close(false);
      return;
    }
    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLElement>('[role="menuitem"]'),
    );
    const index = items.indexOf(document.activeElement as HTMLElement);
    const last = items.length - 1;
    let target: number;
    switch (event.key) {
      case "ArrowDown":
        target = index < 0 || index === last ? 0 : index + 1;
        break;
      case "ArrowUp":
        target = index <= 0 ? last : index - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    items[target]?.focus();
  };

  // An open in flight blocks every row's actions, exactly as the inline buttons were
  // blocked; the busy row's kebab itself says what is happening.
  const busy = openRowBusy !== null;

  return (
    <div className="row-menu-wrap" ref={wrapRef}>
      <button
        ref={buttonRef}
        type="button"
        className="btn btn--small row-menu__toggle"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={`Row actions for ${row.name}`}
        disabled={busy}
        onClick={(event) => {
          event.stopPropagation();
          if (open) {
            close(false);
            return;
          }
          openAtTrigger();
        }}
      >
        <MoreVertical className="icon" aria-hidden="true" focusable="false" />
        {openRowBusy === row.model_id ? "Opening…" : "Actions"}
      </button>
      {open && anchor !== null && (
        <div
          className="row-menu"
          ref={menuRef}
          style={{ top: anchor.top, left: anchor.left }}
          onClick={(event) => event.stopPropagation()}
        >
          <div role="menu" id={menuId} aria-labelledby={buttonId} onKeyDown={onMenuKeyDown}>
            <button
              ref={firstItemRef}
              type="button"
              role="menuitem"
              className="row-menu__item"
              onClick={() => {
                close(false);
                onOpen(row.model_id);
              }}
            >
              Open full results
            </button>
            {showUseManually && (
              <button
                type="button"
                role="menuitem"
                className="row-menu__item"
                onClick={() => {
                  close(false);
                  onUseManually(row);
                }}
              >
                Use this model manually
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export interface RankingTableProps {
  rows: RankedFit[];
  selectedId: string | null;
  /** Model ids whose curve was sampled into the stored result. */
  curvesAvailable: ReadonlySet<string>;
  onSelect: (modelId: string) => void;
  onOpen: (modelId: string) => void;
  /** Shows per-row action menus (the progress view's live table has none). */
  actions?: boolean;
  onUseManually?: (row: RankedFit) => void;
  /** Model ids that can prefill the manual panel (registered models, polynomials). */
  manuallySelectable?: ReadonlySet<string>;
  openRowBusy?: string | null;
  /** Display position of the first row (the page's offset in the full ranking). */
  positionOffset?: number;
}

export function RankingTable({
  rows,
  selectedId,
  curvesAvailable,
  onSelect,
  onOpen,
  actions = true,
  onUseManually,
  manuallySelectable,
  openRowBusy = null,
  positionOffset = 0,
}: RankingTableProps): ReactElement {
  const rowRefs = useRef(new Map<string, HTMLTableRowElement>());
  const [sort, setSort] = useState<SortState | null>(null);

  const focusRow = (modelId: string): void => {
    rowRefs.current.get(modelId)?.focus();
  };

  // Display ordering only — `rows` (the served ranking) is never rebuilt or reordered.
  const ordered = useMemo(() => sortRows(rows, sort, SORT_KEYS), [rows, sort]);
  // The swatch follows the row's ordinal in the *ranking*, not its display position, so
  // it always matches the curve drawn for that row, whichever way the table is sorted.
  const rankedIndex = useMemo(() => {
    const map = new Map<string, number>();
    rows.forEach((row, index) => map.set(row.model_id, index));
    return map;
  }, [rows]);

  const onKeyDown = (event: KeyboardEvent<HTMLTableSectionElement>): void => {
    if (ordered.length === 0) return;
    const index = ordered.findIndex((row) => row.model_id === selectedId);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      const next =
        ordered[Math.min(index + 1, ordered.length - 1)] ?? ordered[ordered.length - 1];
      if (next !== undefined) {
        onSelect(next.model_id);
        focusRow(next.model_id);
      }
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      const previous = ordered[Math.max(index - 1, 0)] ?? ordered[0];
      if (previous !== undefined) {
        onSelect(previous.model_id);
        focusRow(previous.model_id);
      }
    } else if (event.key === "Enter" && selectedId !== null) {
      event.preventDefault();
      onOpen(selectedId);
    }
  };

  const onSort = (column: string): void => {
    setSort((current) => cycleSort(current, column));
  };

  return (
    <div className="table-wrap">
      <table className="table table--compact ranking-table" aria-label="Ranked candidates">
        <thead>
          <tr>
            <SortableTh column="rank" state={sort} onSort={onSort}>
              Rank
            </SortableTh>
            <SortableTh column="model" state={sort} onSort={onSort}>
              Model
            </SortableTh>
            <th scope="col">Equation (fitted)</th>
            <SortableTh column="r2" state={sort} onSort={onSort} className="num">
              R²
            </SortableTh>
            <SortableTh column="adj_r2" state={sort} onSort={onSort} className="num">
              adj. R²
            </SortableTh>
            <SortableTh column="rmse" state={sort} onSort={onSort} className="num">
              RMSE
            </SortableTh>
            <SortableTh column="aicc" state={sort} onSort={onSort} className="num">
              AICc
            </SortableTh>
            <SortableTh column="bic" state={sort} onSort={onSort} className="num">
              BIC
            </SortableTh>
            <SortableTh column="cv_rmse" state={sort} onSort={onSort} className="num">
              CV-RMSE
            </SortableTh>
            <SortableTh column="k" state={sort} onSort={onSort} className="num">
              k
            </SortableTh>
            <SortableTh column="status" state={sort} onSort={onSort}>
              Status
            </SortableTh>
            {actions && (
              <th scope="col" className="ranking-row__actions">
                Actions
              </th>
            )}
          </tr>
        </thead>
        <tbody onKeyDown={onKeyDown}>
          {ordered.map((row) => {
            const selected = row.model_id === selectedId;
            const swatchColor = modelColor(
              positionOffset + (rankedIndex.get(row.model_id) ?? 0) + 1,
            );
            const hasCurve = curvesAvailable.has(row.model_id);
            const equation =
              row.equation_fitted === null ? null : fmtEquation(row.equation_fitted);
            return (
              <tr
                key={row.model_id}
                ref={(element) => {
                  if (element === null) rowRefs.current.delete(row.model_id);
                  else rowRefs.current.set(row.model_id, element);
                }}
                tabIndex={selected ? 0 : -1}
                aria-selected={selected}
                className={selected ? "ranking-row ranking-row--selected" : "ranking-row"}
                onClick={() => onSelect(row.model_id)}
              >
                <td>
                  <span
                    className="rank-swatch"
                    style={{ backgroundColor: swatchColor }}
                    aria-hidden="true"
                  />
                  <span className="num">{fmtInt(row.rank)}</span>
                  {(row.same_curve_as ?? []).length > 0 && (
                    <span className="hint ranking-row__same">
                      same curve as: {(row.same_curve_as ?? []).join(", ")}
                    </span>
                  )}
                </td>
                <td>
                  <span className="ranking-row__name" title={row.name}>
                    {row.name}
                  </span>
                  {row.category !== "" && <span className="chip">{row.category}</span>}
                  {!hasCurve && <span className="hint ranking-row__nocurve">no stored curve</span>}
                </td>
                <td className="ranking-row__equation-cell">
                  {equation !== null && equation !== "" ? (
                    <span className="ranking-row__equation-line">
                      <code className="equation ranking-row__equation">{equation}</code>
                      <CopyButton text={equation} label="equation" compact />
                    </span>
                  ) : (
                    <span className="muted">n/a</span>
                  )}
                </td>
                <td className="num">{fmtNumber(row.metrics.r2)}</td>
                <td className="num">{fmtNumber(row.metrics.adj_r2)}</td>
                <td className="num">{fmtNumber(row.metrics.rmse)}</td>
                <td className="num">{fmtNumber(row.metrics.aicc)}</td>
                <td className="num">{fmtNumber(row.metrics.bic)}</td>
                <td className="num">{fmtNumber(row.metrics.cv_rmse)}</td>
                <td className="num">{fmtInt(row.metrics.k)}</td>
                <td>
                  <StatusBadge status={row.status} />
                </td>
                {actions && (
                  <td className="ranking-row__actions">
                    <RowActionMenu
                      row={row}
                      onOpen={onOpen}
                      onUseManually={onUseManually ?? (() => {})}
                      showUseManually={
                        onUseManually !== undefined &&
                        (manuallySelectable?.has(row.model_id) ?? false)
                      }
                      openRowBusy={openRowBusy}
                    />
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
