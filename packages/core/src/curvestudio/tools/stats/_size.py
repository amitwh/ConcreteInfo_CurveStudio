"""The size harness: how often a tool rejects a null hypothesis that is true.

A test run at a level should reject in about that share of samples; an exact test, in very
nearly that share. A tool's size is measured, not argued: its test draws sample after sample
from a model where the null hypothesis holds and counts the rejections.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

_BAND = (0.038, 0.062)
"""What a test at the 5 % level may measure: 1.2 % either side of 5 %, about three and a half
standard errors over the default 4,000 draws."""

_EXACT_BAND = (0.045, 0.055)
"""The same for a test that is exact at its level. Over the default 4,000 draws 0.5 % is under
one and a half standard errors, so the test of an exact tool should draw more than that."""


def simulated_size(
    draw: Callable[[np.random.Generator], Any],
    run: Callable[[Any], float | None],
    *,
    draws: int = 4_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> float:
    """The share of samples on which ``run`` gives a p-value below ``alpha``.

    Args:
        draw: one sample, from the generator it is given.
        run: the tool's p-value for a sample, or ``None`` when it computed none, which counts
            as no rejection.
        draws: how many samples to draw.
        alpha: the level: a p-value below it is a rejection.
        seed: the seed of the generator, so a measurement can be repeated.

    Raises:
        ValueError: ``draws`` below one.
    """
    if draws < 1:
        raise ValueError(f"draws must be at least 1, not {draws}")
    rng = np.random.default_rng(seed)
    rejected = 0
    for _ in range(draws):
        p_value = run(draw(rng))
        if p_value is not None and p_value < alpha:
            rejected += 1
    return rejected / draws


def assert_size(share: float, *, exact: bool = False) -> None:
    """Hold a measured size to its band: 3.8 to 6.2 %, or 4.5 to 5.5 % for an exact test.

    Raises:
        AssertionError: naming the measured share and the band it falls outside.
    """
    low, high = _EXACT_BAND if exact else _BAND
    if not low <= share <= high:
        raise AssertionError(
            f"the measured size is {share * 100:.1f} %, outside the band "
            f"{low * 100:.1f} % to {high * 100:.1f} %"
        )
