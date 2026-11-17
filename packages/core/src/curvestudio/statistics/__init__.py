"""Statistics on fitted curves: uncertainty, intervals, tests and resampling."""

from curvestudio.statistics.bands import Band, confidence_band, prediction_band
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.view import FitView, build_view

__all__ = [
    "Band",
    "FitView",
    "StatisticsError",
    "build_view",
    "confidence_band",
    "prediction_band",
]
