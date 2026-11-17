"""Statistics on fitted curves: uncertainty, intervals, tests and resampling."""

from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.view import FitView, build_view

__all__ = ["FitView", "StatisticsError", "build_view"]
