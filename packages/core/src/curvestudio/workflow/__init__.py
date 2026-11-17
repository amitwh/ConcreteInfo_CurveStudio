"""Headless service layer: import, fit and predict, shared by the desktop app and the API.

Both user interfaces call only this layer, so the numbers they show are identical. Every
result is a pydantic model that serialises to standard JSON (undefined values are ``None``,
never NaN or infinity); every user-caused problem is a :class:`WorkflowError` with a stable
code.
"""

from curvestudio.workflow.autofit import bundle_for_candidate
from curvestudio.workflow.bundle import (
    MAX_PREDICT_POINTS,
    CurveSamples,
    DataSnapshot,
    FitBundle,
    Prediction,
)
from curvestudio.workflow.catalog import ModelInfo, ParameterInfo, get_model, list_models
from curvestudio.workflow.dataset_import import (
    ImportIssue,
    ImportLimits,
    ImportOutcome,
    import_dataset,
)
from curvestudio.workflow.errors import WorkflowError
from curvestudio.workflow.service import fit_dataset, predict
from curvestudio.workflow.spec import CustomRef, FitSpec, ModelRef, PolynomialRef, RegisteredRef

__all__ = [
    "MAX_PREDICT_POINTS",
    "CurveSamples",
    "CustomRef",
    "DataSnapshot",
    "FitBundle",
    "FitSpec",
    "ImportIssue",
    "ImportLimits",
    "ImportOutcome",
    "ModelInfo",
    "ModelRef",
    "ParameterInfo",
    "PolynomialRef",
    "Prediction",
    "RegisteredRef",
    "WorkflowError",
    "bundle_for_candidate",
    "fit_dataset",
    "get_model",
    "import_dataset",
    "list_models",
    "predict",
]
