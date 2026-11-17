"""Statistics on fitted curves: uncertainty, intervals, tests and resampling."""

from curvestudio.statistics.bands import Band, confidence_band, prediction_band
from curvestudio.statistics.bootstrap import BootstrapParameter, BootstrapResult, bootstrap
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
from curvestudio.statistics.validation import (
    CrossValidation,
    Fitter,
    NonlinearFitter,
    PolynomialFitter,
    k_fold_cv,
    loo_cv,
)
from curvestudio.statistics.view import FitView, build_view

__all__ = [
    "Band",
    "BootstrapParameter",
    "BootstrapResult",
    "ComparisonRow",
    "CrossValidation",
    "FitView",
    "Fitter",
    "InformationCriteria",
    "ModelComparison",
    "NonlinearFitter",
    "PolynomialFitter",
    "ProfileInterval",
    "ProfileResult",
    "Propagation",
    "StatisticsError",
    "bootstrap",
    "build_view",
    "compare_models",
    "confidence_band",
    "information_criteria",
    "k_fold_cv",
    "loo_cv",
    "monte_carlo_band",
    "prediction_band",
    "profile_likelihood",
    "propagate",
    "sample_parameters",
]
