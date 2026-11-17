"""Tests for Auto-fit ranking (synthetic metrics, no fitting)."""

import itertools

import numpy as np
import pytest

from curvestudio.discovery import (
    CandidateMetrics,
    RankBy,
    RankedFit,
    best_overall_order,
    find_same_curves,
    overfitting_note,
    rank,
)


def row(
    model_id: str,
    k: int = 2,
    *,
    r2: float | None = 0.9,
    adj_r2: float | None = None,
    rmse: float | None = 1.0,
    aicc: float | None = None,
    bic: float | None = None,
    cv_rmse: float | None = None,
    same: tuple[str, ...] = (),
) -> RankedFit:
    return RankedFit(
        rank=0,
        model_id=model_id,
        name=model_id.upper(),
        metrics=CandidateMetrics(
            k=k, n=30, r2=r2, adj_r2=adj_r2, rmse=rmse, aicc=aicc, bic=bic, cv_rmse=cv_rmse
        ),
        same_curve_as=same,
    )


def ids(rows: tuple[RankedFit, ...] | list[RankedFit]) -> list[str]:
    return [r.model_id for r in rows]


def test_r2_order_prefers_more_parameters_and_attaches_overfitting_note() -> None:
    rows = [
        row("simple", 2, r2=0.950),
        row("complex", 6, r2=0.955),
        row("middle", 4, r2=0.900),
    ]
    ranked = rank(rows, RankBy.R2)
    assert ids(ranked) == ["complex", "simple", "middle"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    note = overfitting_note(ranked, RankBy.R2)
    assert note is not None
    assert "COMPLEX" in note and "complex" in note and "SIMPLE" in note and "simple" in note


def test_overfitting_note_conditions() -> None:
    near = rank([row("a", 6, r2=0.955), row("b", 3, r2=0.950)], RankBy.R2)
    assert overfitting_note(near, RankBy.R2) is not None  # gap of exactly 3 parameters
    assert overfitting_note(near, RankBy.AICC) is None  # only for R2 / adjusted R2
    two_more = rank([row("a", 5, r2=0.955), row("b", 3, r2=0.950)], RankBy.R2)
    assert overfitting_note(two_more, RankBy.R2) is None
    far = rank([row("a", 6, r2=0.99), row("b", 2, r2=0.90)], RankBy.R2)
    assert overfitting_note(far, RankBy.R2) is None
    assert overfitting_note([], RankBy.R2) is None
    none_top = rank([row("a", 6, r2=None), row("b", 2, r2=0.9)], RankBy.R2)
    assert overfitting_note(none_top, RankBy.R2) is None
    adj = rank(
        [row("a", 6, adj_r2=0.95), row("b", 2, adj_r2=0.945)],
        RankBy.ADJ_R2,
    )
    assert overfitting_note(adj, RankBy.ADJ_R2) is not None


def test_overfitting_note_picks_the_simplest_close_row() -> None:
    ranked = rank(
        [row("top", 8, r2=0.96), row("mid", 4, r2=0.955), row("lean", 2, r2=0.951)], RankBy.R2
    )
    note = overfitting_note(ranked, RankBy.R2)
    assert note is not None and "lean" in note and "mid" not in note


def test_best_overall_peels_tie_groups() -> None:
    # AICc 100 / 101.4 / 102.6 / 110: 101.4 is within 2 of the best, 102.6 is not (2.6),
    # even though it is within 2 of 101.4; the next group starts at 102.6.
    rows = [
        row("a", aicc=100.0, cv_rmse=0.5),
        row("b", aicc=101.4, cv_rmse=0.3),
        row("c", aicc=102.6, cv_rmse=0.1),
        row("d", aicc=110.0, cv_rmse=0.0),
    ]
    assert ids(best_overall_order(rows)) == ["b", "a", "c", "d"]
    ranked = rank(rows, RankBy.BEST_OVERALL)
    assert ids(ranked) == ["b", "a", "c", "d"]
    assert [r.rank for r in ranked] == [1, 2, 3, 4]


def test_best_overall_second_group_can_swallow_later_rows() -> None:
    rows = [
        row("a", aicc=0.0, cv_rmse=1.0),
        row("b", aicc=5.0, cv_rmse=0.9),
        row("c", aicc=6.9, cv_rmse=0.1),
        row("d", aicc=7.1, cv_rmse=0.05),
    ]
    # group 1 = {a}; best remaining is b (5.0) -> group {b, c} (6.9 - 5.0 = 1.9), d is 2.1 away
    assert ids(best_overall_order(rows)) == ["a", "c", "b", "d"]


def test_best_overall_boundary_delta_of_exactly_two_is_a_tie() -> None:
    rows = [row("a", aicc=10.0, cv_rmse=2.0), row("b", aicc=12.0, cv_rmse=1.0)]
    assert ids(best_overall_order(rows)) == ["b", "a"]


def test_best_overall_breaks_ties_by_cv_then_k_then_id() -> None:
    rows = [
        row("z", 3, aicc=50.0, cv_rmse=None),
        row("y", 3, aicc=50.5, cv_rmse=2.0),
        row("x", 2, aicc=50.2, cv_rmse=2.0),
        row("w", 3, aicc=50.1, cv_rmse=2.0),
        row("v", 4, aicc=50.3, cv_rmse=1.0),
    ]
    # cv 1.0 first; then cv 2.0 by k (x with 2) then id (w, y); cv None last
    assert ids(best_overall_order(rows)) == ["v", "x", "w", "y", "z"]


def test_rows_without_aicc_rank_after_rows_with_one() -> None:
    rows = [
        row("no_aicc_great_bic", aicc=None, bic=-100.0, cv_rmse=0.1),
        row("with_aicc", aicc=500.0, cv_rmse=9.0),
        row("no_aicc_ok_bic", aicc=None, bic=-90.0, cv_rmse=0.0),
        row("no_criteria", aicc=None, bic=None, cv_rmse=0.0),
        row("no_aicc_close_bic", aicc=None, bic=-99.0, cv_rmse=0.05),
    ]
    assert ids(best_overall_order(rows)) == [
        "with_aicc",
        "no_aicc_close_bic",
        "no_aicc_great_bic",
        "no_aicc_ok_bic",
        "no_criteria",
    ]


@pytest.mark.parametrize(
    ("rank_by", "expected"),
    [
        (RankBy.R2, ["c", "a", "b", "none"]),
        (RankBy.ADJ_R2, ["b", "c", "a", "none"]),
        (RankBy.RMSE, ["a", "b", "c", "none"]),
        (RankBy.AICC, ["b", "a", "c", "none"]),
        (RankBy.BIC, ["c", "a", "b", "none"]),
        (RankBy.CV_RMSE, ["a", "c", "b", "none"]),
    ],
)
def test_each_rank_by_orders_by_its_metric_and_none_last(
    rank_by: RankBy, expected: list[str]
) -> None:
    rows = [
        row("a", r2=0.8, adj_r2=0.70, rmse=1.0, aicc=20.0, bic=20.0, cv_rmse=1.0),
        row("b", r2=0.7, adj_r2=0.90, rmse=2.0, aicc=10.0, bic=30.0, cv_rmse=3.0),
        row("c", r2=0.9, adj_r2=0.80, rmse=3.0, aicc=30.0, bic=10.0, cv_rmse=2.0),
        row("none", r2=None, adj_r2=None, rmse=None, aicc=None, bic=None, cv_rmse=None),
    ]
    assert ids(rank(rows, rank_by)) == expected


def test_equal_metric_ties_go_to_fewer_parameters_then_id() -> None:
    rows = [row("b", 3, r2=0.9), row("a", 3, r2=0.9), row("c", 2, r2=0.9)]
    assert ids(rank(rows, RankBy.R2)) == ["c", "a", "b"]


@pytest.mark.parametrize("rank_by", list(RankBy))
def test_rank_is_pure_and_stable_under_input_order(rank_by: RankBy) -> None:
    rows = [
        row("a", 2, r2=0.9, adj_r2=0.88, rmse=1.0, aicc=10.0, bic=12.0, cv_rmse=1.0),
        row("b", 3, r2=0.9, adj_r2=0.89, rmse=1.0, aicc=10.5, bic=11.0, cv_rmse=0.5),
        row("c", 4, r2=0.95, adj_r2=0.9, rmse=0.5, aicc=14.0, bic=None, cv_rmse=None),
        row("d", 1, r2=None, adj_r2=None, rmse=None, aicc=None, bic=None, cv_rmse=None),
    ]
    before = [r.model_dump() for r in rows]
    reference = rank(rows, rank_by)
    for permutation in itertools.permutations(rows):
        assert rank(permutation, rank_by) == reference
    assert [r.model_dump() for r in rows] == before
    assert all(r.rank == 0 for r in rows)
    assert sorted(ids(reference)) == ["a", "b", "c", "d"]


def test_same_curve_rows_are_tied_not_dropped() -> None:
    y_range = 10.0
    base = np.linspace(0.0, y_range, 8)
    fitted = {
        "mm": base,
        "langmuir_copy": base + 5e-9 * y_range,  # within 1e-8 of the data range
        "other": base + 1e-6 * y_range,  # outside it
        "third": base[::-1].copy(),
    }
    same = find_same_curves(fitted, y_range)
    assert same == {"langmuir_copy": ("mm",), "mm": ("langmuir_copy",), "other": (), "third": ()}

    rows = [
        row("mm", 2, aicc=10.0, same=same["mm"]),
        row("other", 2, aicc=10.5, same=same["other"]),
        row("langmuir_copy", 3, aicc=14.0, same=same["langmuir_copy"]),
        row("third", 2, aicc=30.0, same=same["third"]),
    ]
    ranked = rank(rows, RankBy.AICC)
    assert len(ranked) == 4  # nothing dropped
    # the tied pair stays adjacent and shares the best member's rank; the next rank skips
    assert ids(ranked) == ["mm", "langmuir_copy", "other", "third"]
    assert [r.rank for r in ranked] == [1, 1, 3, 4]


def test_find_same_curves_edge_cases() -> None:
    assert find_same_curves({}, 1.0) == {}
    assert find_same_curves({"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]}, 1.0) == {"a": (), "b": ()}
    # a zero data range compares exactly
    assert find_same_curves({"a": [1.0, 1.0], "b": [1.0, 1.0 + 1e-12]}, 0.0) == {"a": (), "b": ()}
    assert find_same_curves({"a": [1.0, 1.0], "b": [1.0, 1.0]}, 0.0) == {"a": ("b",), "b": ("a",)}
