"""The size harness: how often a tool rejects a null hypothesis that is true.

A test run at a level should reject in about that share of samples; an exact test, in very
nearly that share. The harness draws samples, runs the tool's p-value on each and measures
the share that falls below the level. These tests hold the harness itself to a case whose
size is known, the one-sample t test on standard normal data, and to what it must do with
its own settings.
"""

import numpy as np
import pytest
from scipy import stats

from curvestudio.tools.stats import assert_size, simulated_size

N = 20
LEVEL = 0.05


def _draw(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(N)


def _p(sample: np.ndarray) -> float:
    return float(stats.ttest_1samp(sample, 0.0).pvalue)


def test_the_one_sample_t_test_rejects_at_its_level() -> None:
    share = simulated_size(_draw, _p, alpha=LEVEL)

    assert 0.038 <= share <= 0.062
    assert_size(share)


def test_assert_size_rejects_a_share_outside_the_band_and_names_both() -> None:
    assert_size(0.038)
    assert_size(0.062)
    for share in (0.09, 0.02, 0.0379, 0.0621, float("nan")):
        with pytest.raises(AssertionError) as refused:
            assert_size(share)
        message = str(refused.value)
        assert "3.8 %" in message and "6.2 %" in message
    with pytest.raises(AssertionError, match=r"9\.0 %"):
        assert_size(0.09)


def test_an_exact_test_is_held_to_a_narrower_band() -> None:
    # Inside the wide band and outside the narrow one, on either side.
    for share in (0.04, 0.06):
        assert_size(share)
        with pytest.raises(AssertionError, match=r"4\.5 %"):
            assert_size(share, exact=True)
    assert_size(0.045, exact=True)
    assert_size(0.055, exact=True)
    with pytest.raises(AssertionError, match=r"9\.0 %"):
        assert_size(0.09, exact=True)


def test_a_missing_p_value_counts_as_no_rejection() -> None:
    assert simulated_size(_draw, lambda sample: None, draws=100) == 0.0


def test_size_honours_the_level_the_draws_and_the_seed() -> None:
    drawn: list[np.ndarray] = []

    def draw(rng: np.random.Generator) -> np.ndarray:
        drawn.append(rng.standard_normal(N))
        return drawn[-1]

    def by_first_value(sample: np.ndarray) -> float:
        return 0.01 if sample[0] > 0.0 else 0.5

    first = simulated_size(draw, by_first_value, draws=200, alpha=0.05, seed=1)
    again = simulated_size(draw, by_first_value, draws=200, alpha=0.05, seed=1)

    # The same seed draws the same samples, so it measures the same share.
    assert len(drawn) == 400
    assert again == first
    assert first == float(np.mean([sample[0] > 0.0 for sample in drawn[200:]]))
    assert first == float(np.mean([sample[0] > 0.0 for sample in drawn[:200]]))

    # A p-value is a rejection only when it falls below the level, never when it equals it.
    assert simulated_size(draw, by_first_value, draws=200, alpha=0.005, seed=1) == 0.0
    assert simulated_size(_draw, lambda sample: 0.05, draws=50, alpha=0.05) == 0.0
    assert simulated_size(_draw, lambda sample: 0.049, draws=50, alpha=0.05) == 1.0


def test_the_generator_is_numpy_s_default_for_the_seed() -> None:
    # What lets a tool's own test repeat a measurement: the samples are the first draws of
    # ``numpy.random.default_rng(seed)``, nothing else.
    seen: list[np.ndarray] = []

    def draw(rng: np.random.Generator) -> np.ndarray:
        seen.append(rng.standard_normal(N))
        return seen[-1]

    simulated_size(draw, lambda sample: None, draws=3, seed=20261124)

    reference = np.random.default_rng(20261124)
    for sample in seen:
        assert sample.tolist() == reference.standard_normal(N).tolist()


def test_size_refuses_to_draw_nothing() -> None:
    with pytest.raises(ValueError, match="draws"):
        simulated_size(_draw, _p, draws=0)
