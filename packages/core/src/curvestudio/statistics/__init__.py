"""Statistics on fitted curves: uncertainty, intervals, tests and resampling."""

from curvestudio.statistics.bands import Band, confidence_band, prediction_band
from curvestudio.statistics.criteria import (
    ComparisonRow,
    InformationCriteria,
    ModelComparison,
    compare_models,
    information_criteria,
)
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.montecarlo import (
    Propagation,
    monte_carlo_band,
    propagate,
    sample_parameters,
)
from curvestudio.statistics.profile import ProfileInterval, ProfileResult, profile_likelihood
from curvestudio.statistics.view import FitView, build_view

__all__ = [
    "Band",
    "ComparisonRow",
    "FitView",
    "InformationCriteria",
    "ModelComparison",
    "ProfileInterval",
    "ProfileResult",
    "Propagation",
    "StatisticsError",
    "build_view",
    "compare_models",
    "confidence_band",
    "information_criteria",
    "monte_carlo_band",
    "prediction_band",
    "profile_likelihood",
    "propagate",
    "sample_parameters",
]
