"""Statistics on fitted curves: uncertainty, intervals, tests and resampling."""

from curvestudio.statistics.criteria import (
    ComparisonRow,
    InformationCriteria,
    ModelComparison,
    compare_models,
    information_criteria,
)
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.view import FitView, build_view

__all__ = [
    "ComparisonRow",
    "FitView",
    "InformationCriteria",
    "ModelComparison",
    "StatisticsError",
    "build_view",
    "compare_models",
    "information_criteria",
]
