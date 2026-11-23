"""The fixed advice of the checks and the small text helpers they share."""

from collections.abc import Sequence

from curvestudio.tools._thresholds import LISTED_MAX

RESTART = "Check the starting values and the bounds, or try a model with fewer parameters."
EXACT_NEXT_STEP = (
    "There is no scatter to estimate uncertainty from, so report no intervals or p-values "
    "for this fit."
)
NORMALITY_NEXT_STEP = (
    "If a few points cause it, try a robust loss (Huber or soft-L1). Otherwise say in your "
    "write-up that the intervals rely on an approximation."
)
INDEPENDENCE_NEXT_STEP = (
    "Plot the residuals against x in data order. If they wander, the model is missing a "
    "shape; if they alternate, look at how the data were taken."
)
VARIANCE_NEXT_STEP = (
    "If the spread grows with the signal, use weights from replicates or a robust loss, and say so."
)
TREND_NEXT_STEP = (
    "The residuals show a curved trend, so the model is missing a shape the data contain. "
    "Try a model with one more parameter."
)
IDENTIFIABLE_NEXT_STEP = (
    "A parameter at a bound or with a wide interval is not pinned down by these data. Fix "
    "the bound if it is wrong, take more points where the model is sensitive, or use a model "
    "with fewer parameters."
)
POINTS_NEXT_STEP = "Take more points, or use a simpler model."
CV_NEXT_STEP = (
    "The fit describes noise as well as signal. A simpler model, or one with fewer "
    "parameters, will predict better."
)
SIMPLER_NEXT_STEP = (
    "Report the simpler model unless the extra parameter answers a question of your own."
)

DEGREE_NAMES = {
    0: "constant",
    1: "linear",
    2: "quadratic",
    3: "cubic",
    4: "quartic",
    5: "quintic",
    6: "sextic",
}
"""Legendre degrees as a sentence names them; a higher one is ``degree 7`` and so on."""


def num(value: float) -> str:
    """A number for a sentence: at most three significant digits, never ``-0``."""
    text = f"{value:.3g}"
    return "0" if text == "-0" else text


def p_threshold(caution: float, fail: float) -> str:
    """The rule on a p-value, with its two cut-offs."""
    return (
        f"p >= {num(caution)} pass; {num(fail)} <= p < {num(caution)} caution; p < {num(fail)} fail"
    )


def joined(parts: Sequence[str]) -> str:
    """``a``, ``a and b`` or ``a, b and c``."""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def listing(items: Sequence[str], noun: str) -> str:
    """``row 13``, ``rows 13 and 41`` or ``rows 1, 2 and 3``."""
    return f"{noun} {items[0]}" if len(items) == 1 else f"{noun}s {joined(items)}"


def capped(numbers: Sequence[str], noun: str) -> str:
    """``rows 7 and 13``, or the first ``LISTED_MAX`` of a longer list and how many follow."""
    if len(numbers) <= LISTED_MAX:
        return listing(numbers, noun)
    return f"{listing(numbers[:LISTED_MAX], noun)}, and {len(numbers) - LISTED_MAX} more"


def subjects(count: int, noun: str, one: str, many: str) -> str:
    """``1 point has`` or ``3 points have``: the noun with its verb for ``count``."""
    return f"1 {noun} {one}" if count == 1 else f"{count} {noun}s {many}"


def named(names: Sequence[str], one: str, many: str) -> str:
    """``Parameter a sits`` or ``Parameters a and b sit``."""
    return f"{listing(names, 'Parameter')} {one if len(names) == 1 else many}"


def needs_look(count: int) -> str:
    """``1 needs a look`` or ``3 need a look``."""
    return "1 needs a look" if count == 1 else f"{count} need a look"


def degree_name(degree: int) -> str:
    return DEGREE_NAMES.get(degree, f"degree {degree}")


def terms(degrees: Sequence[int]) -> str:
    """``cubic and quartic``, or ``constant, cubic and quartic``."""
    return joined([degree_name(degree) for degree in degrees])
