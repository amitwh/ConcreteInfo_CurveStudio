"""Auto-fit known-truth gate and benchmark; writes tests/numerical/AUTOFIT.md.

Two measurements, one report:

* **Known truth** (deterministic): seeded noisy datasets, one per true model across the model
  families. Each is run through ``autofit`` with the whole installed catalogue under
  ``RankBy.BEST_OVERALL``; the gate is that the true model, or a model whose fitted curve
  coincides with it, is among the top three rows. ``tests/numerical/test_autofit_known_truth.py``
  re-runs these cases and compares the ids of the top three with the committed table.
* **Benchmark** (not deterministic: wall times depend on the machine and on what else runs on
  it): the same synthetic curve at 100, 1000 and 10000 points, a default and an exhaustive run,
  with one worker and with ``cpu_count - 1`` workers. Each row records the load average when the
  run started, the candidate count and the time per stage (planning, the two-start screening
  stage with pool start-up, the full-multistart stage, the refit on all points of a large run).
  The numbers are measurements, not promises.

Usage: ``uv run python scripts/autofit_benchmark.py [--output PATH] [--skip-benchmark]
[--sizes 100 1000 10000] [--workers 1 7] [--modes default exhaustive]``.
Exit status 1 when a known-truth case misses its top three.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    CandidateOutcome,
    RankBy,
    autofit,
    get_shared_pool,
    plan_run,
    shutdown_shared_pools,
)
from curvestudio.models import get_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = _REPO_ROOT / "tests" / "numerical" / "AUTOFIT.md"

TOP = 3
"""The true model (or a same-curve model) must be among this many best rows."""

KNOWN_TRUTH_SECTION = "## Known truth"
BENCHMARK_SECTION = "## Benchmark"


# --- known truth ------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthCase:
    """One seeded dataset: the true model (its example parameters unless ``params``), the
    sampling grid and the noise as a fraction of the true curve's peak-to-peak range."""

    model_id: str
    noise: float
    seed: int
    n: int | None = None
    x_range: tuple[float, float] | None = None
    params: tuple[tuple[str, float], ...] | None = None


KNOWN_TRUTH: tuple[TruthCase, ...] = (
    TruthCase("michaelis_menten", 0.03, 101),
    TruthCase("logistic4", 0.03, 102),
    TruthCase("gompertz", 0.03, 103),
    TruthCase("gaussian", 0.03, 104),
    TruthCase("peak_lorentzian", 0.03, 105),
    TruthCase("exp_decay_offset", 0.05, 106),
    TruthCase("power", 0.05, 107),
    TruthCase("weibull_growth", 0.02, 108),
    # two well-separated rates (0.15 and 2.65): with the example's rates (0.4 and 2.0) a
    # three-parameter rational curve fits equally well at 2 % noise, so the data cannot tell
    TruthCase(
        "exponential_decay_double",
        0.02,
        109,
        n=120,
        x_range=(0.0, 12.0),
        params=(("a", 3.0), ("b", 2.5), ("r", 0.15), ("d", 2.5)),
    ),
    TruthCase("hill", 0.03, 110),
    TruthCase("rational_1_1", 0.02, 111),
    TruthCase("engineering_strength_logistic_age", 0.03, 112),
    TruthCase("engineering_strength_abrams", 0.04, 113),
)


@dataclass(frozen=True)
class TruthRow:
    case: TruthCase
    n: int
    top: tuple[str, ...]
    true_rank: int | None
    passed: bool
    how: str


def truth_data(case: TruthCase) -> XY:
    """The noisy dataset of ``case`` (a fresh seeded generator per case)."""
    spec = get_model(case.model_id)
    assert spec.example is not None, case.model_id
    low, high = case.x_range or spec.example.x_range
    n = case.n or spec.example.n
    params = dict(case.params) if case.params is not None else dict(spec.example.params)
    x = np.linspace(low, high, n)
    clean = spec.evaluate(x, **params)
    sd = case.noise * float(np.ptp(clean))
    y = clean + np.random.default_rng(case.seed).normal(0.0, sd, n)
    return XY(
        x=x,
        y=y,
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint=f"truth-{case.model_id}-{case.seed}",
    )


def judge(result: AutoFitResult, true_id: str) -> tuple[tuple[str, ...], int | None, bool, str]:
    """Top ids, the true model's rank, whether the gate passes, and why.

    Passes when ``true_id`` is in the top ``TOP`` rows, or a top row's fitted curve coincides
    with the true model's (``same_curve_as``), or the true model is flagged redundant with a
    top row (same curve, fewer parameters there).
    """
    top_rows = result.ranked[:TOP]
    top = tuple(row.model_id for row in top_rows)
    by_id = {row.model_id: row for row in result.ranked}
    true_row = by_id.get(true_id)
    true_rank = None if true_row is None else true_row.rank
    if true_id in top:
        return top, true_rank, True, "true model in the top three"
    if any(true_id in row.same_curve_as for row in top_rows):
        return top, true_rank, True, "a top row draws the same curve"
    if true_row is not None and true_row.redundant_with in top:
        return top, true_rank, True, f"redundant with top row {true_row.redundant_with}"
    return top, true_rank, False, "MISSED"


def run_known_truth(workers: int | None = None) -> list[TruthRow]:
    """Run every known-truth case on a shared warm pool and judge it."""
    pool = None if workers == 1 else get_shared_pool(workers)
    rows: list[TruthRow] = []
    for case in KNOWN_TRUTH:
        data = truth_data(case)
        config = AutoFitConfig(rank_by=RankBy.BEST_OVERALL, workers=1 if pool is None else None)
        result = autofit(data, config, pool=pool)
        top, true_rank, passed, how = judge(result, case.model_id)
        rows.append(TruthRow(case, int(data.x.size), top, true_rank, passed, how))
    return rows


def render_known_truth(rows: list[TruthRow]) -> list[str]:
    lines = [
        KNOWN_TRUTH_SECTION,
        "",
        f"Seeded noisy data, whole installed catalogue, `best_overall`. Gate: the true model, or a "
        f"model with the same fitted curve, is in the top {TOP}. The top-{TOP} ids are compared "
        "by `tests/numerical/test_autofit_known_truth.py`.",
        "",
        "| True model | n | Noise | Seed | Top 3 (best overall) | True model rank | Verdict |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for r in rows:
        top = ", ".join(f"`{i}`" for i in r.top)
        rank = "not ranked" if r.true_rank is None else str(r.true_rank)
        lines.append(
            f"| `{r.case.model_id}` | {r.n} | {r.case.noise:.0%} | {r.case.seed} | {top} | "
            f"{rank} | {r.how} |"
        )
    passed = sum(r.passed for r in rows)
    lines += ["", f"Passed: {passed}/{len(rows)}", ""]
    return lines


def parse_known_truth(text: str) -> dict[str, tuple[str, ...]]:
    """The ``{true model id: top ids}`` table of a rendered report (id lists only)."""
    table: dict[str, tuple[str, ...]] = {}
    section = text.split(KNOWN_TRUTH_SECTION, 1)[1].split("\n## ", 1)[0]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0].startswith("`") and cells[0] != "True model":
            table[cells[0].strip("`")] = tuple(i.strip().strip("`") for i in cells[4].split(","))
    return table


# --- benchmark --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchRow:
    n: int
    mode: str
    workers: int
    load_before: float
    wall: float
    candidates: int
    second_stage: int
    ranked: int
    plan: float
    screen: float
    full: float
    refit: float
    assemble: float


def benchmark_data(n: int) -> XY:
    """A four-parameter logistic with 3 % noise on ``n`` evenly spaced points."""
    x = np.linspace(0.0, 8.0, n)
    clean = 1.0 + 9.0 / (1.0 + np.exp(-1.2 * (x - 3.0)))
    y = clean + np.random.default_rng(2026).normal(0.0, 0.03 * float(np.ptp(clean)), n)
    return XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint=f"bench-{n}")


def measure(n: int, exhaustive: bool, workers: int) -> BenchRow:
    data = benchmark_data(n)
    config = AutoFitConfig(exhaustive=exhaustive, workers=workers)
    pool = get_shared_pool(workers) if workers > 1 else None
    if pool is not None:  # warm the workers first: start-up is reported by a cold run elsewhere
        autofit(
            benchmark_data(60), AutoFitConfig(workers=workers, families=("exponential",)), pool=pool
        )
    planned = time.perf_counter()
    plan = plan_run(data, config)
    t_plan = time.perf_counter() - planned
    stamps: list[tuple[float, CandidateOutcome]] = []
    load = os.getloadavg()[0]
    start = time.perf_counter()
    result = autofit(
        data,
        config,
        pool=pool,
        on_outcome=lambda o: stamps.append((time.perf_counter() - start, o)),
    )
    wall = time.perf_counter() - start
    jobs = len(plan.jobs)
    times = [t for t, _ in stamps]
    # the first `jobs` outcomes are the screening stage; later ones are the promoted candidates,
    # then (large runs only) the refits on all points, the only outcomes not marked on_subsample
    is_refit = [plan.subsampled and not o.on_subsample for _, o in stamps]
    screen_end = times[jobs - 1] if len(times) >= jobs else wall
    full_end = max((t for t, refit in zip(times, is_refit, strict=True) if not refit), default=0.0)
    full_end = max(full_end, screen_end)
    last = times[-1] if times else wall
    return BenchRow(
        n=n,
        mode="exhaustive" if exhaustive else "default",
        workers=workers,
        load_before=load,
        wall=wall,
        candidates=jobs,
        second_stage=len(stamps) - jobs - sum(is_refit),
        ranked=len(result.ranked),
        plan=t_plan,
        screen=max(0.0, screen_end - t_plan),
        full=max(0.0, full_end - screen_end),
        refit=last - full_end if any(is_refit) else 0.0,
        assemble=max(0.0, wall - last),
    )


def hardware_line() -> str:
    load = ", ".join(f"{v:.2f}" for v in os.getloadavg())
    return (
        f"{platform.platform()}; {platform.processor() or platform.machine()}; "
        f"os.cpu_count()={os.cpu_count()}, process_cpu_count()={os.process_cpu_count()}; "
        f"Python {platform.python_version()}; load average (1, 5, 15 min) when the benchmark "
        f"started: {load}"
    )


def render_benchmark(rows: list[BenchRow], header: str) -> list[str]:
    lines = [
        BENCHMARK_SECTION,
        "",
        "Measurements on the machine below, not promises: wall times depend on the hardware and "
        "on whatever else was running (see the load column). One synthetic four-parameter "
        "logistic with 3 % noise per size; whole installed catalogue; 8 starts per candidate; "
        "20 s per-candidate limit. Stage times: planning (candidate list and data screening); "
        "screening (every candidate from 2 starts, includes handing jobs to the pool); full "
        "(the promoted candidates from 8 starts); refit (large runs only: the best rows refitted "
        "on every point); assemble (ranking and result building after the last fit). Runs with "
        "several workers use an already warm pool: the roughly two seconds of worker start-up are "
        "not in the wall time.",
        "",
        f"Hardware: {header}",
        "",
        "| Points | Mode | Workers | Load before | Wall (s) | Candidates fitted | Promoted | "
        "Ranked | Plan (s) | Screen (s) | Full (s) | Refit (s) | Assemble (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.n} | {r.mode} | {r.workers} | {r.load_before:.2f} | {r.wall:.1f} | "
            f"{r.candidates} | {r.second_stage} | {r.ranked} | {r.plan:.2f} | {r.screen:.1f} | "
            f"{r.full:.1f} | {r.refit:.1f} | {r.assemble:.2f} |"
        )
    return [*lines, ""]


# --- report -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-benchmark", action="store_true", help="known truth only")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--workers", type=int, nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", choices=["default", "exhaustive"], default=None)
    args = parser.parse_args(argv)

    lines = [
        "# Auto-fit known-truth gate and benchmark",
        "",
        "Generated by `scripts/autofit_benchmark.py`; do not edit by hand.",
        "",
    ]
    truth = run_known_truth()
    lines += render_known_truth(truth)
    if not args.skip_benchmark:
        worker_counts = args.workers or [1, max(1, (os.process_cpu_count() or 2) - 1)]
        modes = args.modes or ["default", "exhaustive"]
        header = hardware_line()
        rows = []
        for n in args.sizes:
            for mode in modes:
                for workers in worker_counts:
                    row = measure(n, mode == "exhaustive", workers)
                    print(
                        f"n={n} {mode} workers={workers}: {row.wall:.1f} s "
                        f"(load before {row.load_before:.2f})",
                        flush=True,
                    )
                    rows.append(row)
        lines += render_benchmark(rows, header)
    shutdown_shared_pools()
    args.output.write_text("\n".join(lines), encoding="utf-8")
    missed = [r.case.model_id for r in truth if not r.passed]
    print(f"wrote {args.output} ({len(truth)} known-truth cases, {len(missed)} missed)")
    if missed:
        print("missed: " + ", ".join(missed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
