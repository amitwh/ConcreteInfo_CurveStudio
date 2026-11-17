"""Statistics on fitted curves: uncertainty, intervals, tests and resampling."""

from curvestudio.statistics.bands import Band, confidence_band, prediction_band
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.profile import ProfileInterval, ProfileResult, profile_likelihood
from curvestudio.statistics.view import FitView, build_view

__all__ = [
    "Band",
    "FitView",
    "ProfileInterval",
    "ProfileResult",
    "StatisticsError",
    "build_view",
    "confidence_band",
    "prediction_band",
    "profile_likelihood",
]
