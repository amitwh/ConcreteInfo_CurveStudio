"""The hand-written catalogue keeps its agreed minimum size."""

from collections import Counter

import pytest

from curvestudio.models.starter import STARTER_MODELS
from curvestudio_models import catalogue

pytestmark = pytest.mark.real_plugins

HAND_WRITTEN_FLOOR = 8
"""Raised to 150 once, in the merge step of the last hand-written family task."""


def test_hand_written_floor() -> None:
    assert len(catalogue()) + len(STARTER_MODELS) >= HAND_WRITTEN_FLOOR


def test_every_catalogue_model_is_in_exactly_one_category_of_the_vocabulary() -> None:
    from curvestudio.testing.model_contract import CATEGORY_VOCABULARY

    counts = Counter(m.category.partition(".")[0] for m in catalogue())
    assert set(counts) <= CATEGORY_VOCABULARY
    assert sum(counts.values()) == len(catalogue())
