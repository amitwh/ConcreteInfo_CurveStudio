"""The one categorical chart palette: model curves, neutral bands and data points.

The desktop app, the web app (mirrored in ``apps/web/src/workspace/traces.ts``) and the
exported Matplotlib figures all draw the same ranks in the same colours, so a model is
recognisable wherever it appears. Rank 1 is the ConcreteInfo brand accent; ranks 2-8 are
the design-system categorical set, chosen to stay distinguishable on both the light and
the dark plot backgrounds.

Bands and data points are deliberately *neutral* (equal red, green and blue channels) so a
grey band can never be mistaken for another model's curve. The prediction band is lighter
than the confidence band (lighter grey in the light theme, lower opacity in the dark one).

Fill colours are ``#RRGGBBAA`` strings: Matplotlib reads them directly and pyqtgraph's
``mkBrush`` parses the same order, so the two renderers share one definition. (``QColor``
itself uses ``#AARRGGBB`` for eight-digit hex, hence the brush, not the colour, is used.)
"""

from typing import Final, Literal

Theme = Literal["light", "dark"]

#: Categorical model colours by rank, best first. Exactly eight, all pairwise distinct.
MODEL_COLORS: Final[tuple[str, ...]] = (
    "#E5461F",  # 1 — brand accent
    "#3B82F6",  # 2 — blue
    "#10B981",  # 3 — emerald
    "#F59E0B",  # 4 — amber
    "#8B5CF6",  # 5 — violet
    "#EC4899",  # 6 — pink
    "#06B6D4",  # 7 — cyan
    "#84CC16",  # 8 — lime
)

#: The best fit is drawn slightly heavier so it still leads in a multi-curve overlay.
CURVE_WIDTH_LEAD: Final = 2.2
CURVE_WIDTH_OTHER: Final = 1.6

#: Neutral confidence-band fills (``#RRGGBBAA``) per theme.
BAND_CONFIDENCE_FILL: Final[dict[Theme, str]] = {
    "light": "#56565633",
    "dark": "#C8C8C83D",
}

#: Neutral prediction-band fills, lighter than the confidence band in both themes.
BAND_PREDICTION_FILL: Final[dict[Theme, str]] = {
    "light": "#8C8C8C1F",
    "dark": "#C8C8C81F",
}

#: Neutral data-point colour per theme (a grey never competes with a model colour).
DATA_COLOR: Final[dict[Theme, str]] = {
    "light": "#4B4B4B",
    "dark": "#B0B0B0",
}


def _theme(dark: bool) -> Theme:
    return "dark" if dark else "light"


def model_color(rank: int) -> str:
    """The categorical colour of a 1-based rank, wrapping every eight ranks.

    Ranks beyond eight reuse the palette in order, so an overlay of ten rows still draws
    every curve in a defined colour.

    Raises:
        ValueError: ``rank`` is below 1 (ranks are 1-based).
    """
    if rank < 1:
        raise ValueError(f"rank is 1-based; got {rank}")
    return MODEL_COLORS[(rank - 1) % len(MODEL_COLORS)]


def curve_style(rank: int) -> tuple[str, float]:
    """The ``(colour, width)`` a fitted curve of ``rank`` is drawn with."""
    width = CURVE_WIDTH_LEAD if rank == 1 else CURVE_WIDTH_OTHER
    return model_color(rank), width


def band_fill(kind: Literal["confidence", "prediction"], *, dark: bool) -> str:
    """The neutral ``#RRGGBBAA`` fill of a band in the light or dark theme."""
    fills = BAND_CONFIDENCE_FILL if kind == "confidence" else BAND_PREDICTION_FILL
    return fills[_theme(dark)]


def data_color(*, dark: bool) -> str:
    """The neutral data-point colour in the light or dark theme."""
    return DATA_COLOR[_theme(dark)]
