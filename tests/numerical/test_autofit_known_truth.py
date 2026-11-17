"""The Auto-fit gate: the true model of a seeded noisy dataset is found in the top three.

``scripts/autofit_benchmark.py`` defines the datasets (one true model per case, noise 2-5 % of
the curve's range, across the model families, one of them an engineering model) and writes
``tests/numerical/AUTOFIT.md``. This test re-runs every case on the whole installed catalogue
under ``RankBy.BEST_OVERALL`` and requires that the true model, or a model with the same fitted
curve, is among the top three rows, and that the ids of the top three still equal the committed
table (ids only; the report's wall times are measurements and are never compared).

The catalogue-size and per-model recovery claims of the plan are carried by the library tests
(``tests/unit/library``, ``tests/unit/engineering``) and ``tests/numerical/LIBRARY.md``; this
module only asserts the sizes from the live registry.
"""

import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from autofit_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT,
    KNOWN_TRUTH,
    TOP,
    parse_known_truth,
    run_known_truth,
)

from curvestudio.models.starter import STARTER_MODELS  # noqa: E402
from curvestudio_engineering import catalogue as engineering_catalogue  # noqa: E402
from curvestudio_models import catalogue as library_catalogue  # noqa: E402
from curvestudio_models.generated import ids as generated_ids  # noqa: E402

pytestmark = pytest.mark.real_plugins


def test_the_catalogue_is_as_large_as_the_plan_promises() -> None:
    assert len(library_catalogue()) + len(STARTER_MODELS) >= 150  # hand-written
    assert len(generated_ids()) >= 100  # default generated ids
    assert len(engineering_catalogue()) >= 30


def test_the_known_truth_cases_span_at_least_twelve_models() -> None:
    assert len({case.model_id for case in KNOWN_TRUTH}) >= 12
    assert any(case.model_id.startswith("engineering_") for case in KNOWN_TRUTH)
    assert all(0.02 <= case.noise <= 0.05 for case in KNOWN_TRUTH)


@pytest.mark.slow
def test_the_true_model_is_in_the_top_three_and_matches_the_committed_table() -> None:
    rows = run_known_truth()
    missed = [
        f"{r.case.model_id} (rank {r.true_rank}, top {list(r.top)})" for r in rows if not r.passed
    ]
    assert not missed, f"the true model is not in the top {TOP} for: {missed}"
    committed = parse_known_truth(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    fresh = {r.case.model_id: r.top for r in rows}
    assert fresh == committed, (
        "regenerate tests/numerical/AUTOFIT.md with scripts/autofit_benchmark.py"
    )
