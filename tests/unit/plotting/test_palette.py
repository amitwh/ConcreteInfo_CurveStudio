"""The shared chart palette: model ranks, neutral bands and neutral data points."""

from __future__ import annotations

import pytest

from curvestudio_plotting.palette import (
    BAND_CONFIDENCE_FILL,
    BAND_PREDICTION_FILL,
    DATA_COLOR,
    MODEL_COLORS,
    band_fill,
    curve_style,
    data_color,
    model_color,
)


def _rgb(value: str) -> tuple[int, int, int]:
    """The RGB part of a ``#RRGGBB`` or ``#RRGGBBAA`` colour."""
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def _alpha(value: str) -> float:
    if len(value) == 9:
        return int(value[7:9], 16) / 255.0
    return 1.0


def _neutral(value: str) -> bool:
    red, green, blue = _rgb(value)
    return red == green == blue


def test_the_palette_is_eight_distinct_colours_with_the_brand_first() -> None:
    assert len(MODEL_COLORS) == 8
    assert len(set(MODEL_COLORS)) == 8
    assert MODEL_COLORS[0] == "#E5461F"  # the brand accent leads the ranking


def test_the_colours_stay_apart_from_each_other_and_both_plot_backgrounds() -> None:
    backgrounds = ("#FFFFFF", "#1E1E1E")

    def distance(a: str, b: str) -> float:
        return sum((x - y) ** 2 for x, y in zip(_rgb(a), _rgb(b), strict=True)) ** 0.5

    for i, first in enumerate(MODEL_COLORS):
        for second in MODEL_COLORS[i + 1 :]:
            assert distance(first, second) > 40, f"{first} and {second} are too close"
        for background in backgrounds:
            assert distance(first, background) > 60, f"{first} disappears on {background}"


def test_model_color_maps_a_1_based_rank() -> None:
    for rank, expected in enumerate(MODEL_COLORS, start=1):
        assert model_color(rank) == expected


def test_model_color_wraps_every_eight_ranks() -> None:
    assert model_color(9) == MODEL_COLORS[0]
    assert model_color(16) == MODEL_COLORS[7]
    assert model_color(17) == MODEL_COLORS[0]


def test_model_color_rejects_a_rank_below_one() -> None:
    with pytest.raises(ValueError):
        model_color(0)
    with pytest.raises(ValueError):
        model_color(-3)


def test_curve_style_makes_the_best_fit_heavier() -> None:
    colour, width = curve_style(1)
    assert colour == MODEL_COLORS[0]
    assert width > curve_style(2)[1]
    assert curve_style(2)[0] == MODEL_COLORS[1]
    assert curve_style(2)[1] == curve_style(3)[1]


@pytest.mark.parametrize("dark", [False, True])
def test_bands_are_neutral_in_both_themes(dark: bool) -> None:
    for kind in ("confidence", "prediction"):
        fill = band_fill(kind, dark=dark)
        assert _neutral(fill), f"{kind} fill {fill} has a hue"
        assert fill[:7].upper() not in {colour.upper() for colour in MODEL_COLORS}


def test_the_prediction_band_is_lighter_than_the_confidence_band() -> None:
    for dark in (False, True):
        confidence = band_fill("confidence", dark=dark)
        prediction = band_fill("prediction", dark=dark)
        assert _alpha(prediction) < _alpha(confidence)
        # In the light theme the prediction grey is also a lighter colour.
        if not dark:
            assert sum(_rgb(prediction)) > sum(_rgb(confidence))


def test_data_colour_is_neutral_and_belongs_to_no_model_rank() -> None:
    for dark in (False, True):
        colour = data_color(dark=dark)
        assert _neutral(colour)
        assert colour.upper() not in {model.upper() for model in MODEL_COLORS}
        assert colour == DATA_COLOR["dark" if dark else "light"]


def test_the_theme_fill_maps_cover_exactly_light_and_dark() -> None:
    assert set(BAND_CONFIDENCE_FILL) == {"light", "dark"}
    assert set(BAND_PREDICTION_FILL) == {"light", "dark"}
    assert set(DATA_COLOR) == {"light", "dark"}
